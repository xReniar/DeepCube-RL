import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_DIM = 320
MODEL_HEADS = 5
HEAD_DIM = MODEL_DIM // MODEL_HEADS
ROPE_THETA = 10_000.0
CONTEXT = 128
STATE_WIDTH = 32
INPUT_VOCAB = 19
MODEL_LAYERS = 12
MOVES = 18
DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else
          "cpu")


def apply_rope(x, cos, sin):
    batch, heads, length, width = x.shape
    pair = x.float().reshape(batch, heads, length, width // 2, 2)
    even, odd = pair.unbind(-1)
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    )
    return rotated.flatten(-2).to(x.dtype)

class KVCache:
    def __init__(self, layers, batch, capacity, dtype, device):
        shape = (layers, batch, MODEL_HEADS, capacity, HEAD_DIM)
        self.keys = torch.empty(shape, dtype=dtype, device=device)
        self.values = torch.empty(shape, dtype=dtype, device=device)
        self.length = 0
        self.capacity = capacity

    def gather(self, index):
        gathered = KVCache(
            self.keys.shape[0], len(index), self.capacity,
            self.keys.dtype, self.keys.device,
        )
        if self.length:
            for layer in range(self.keys.shape[0]):
                gathered.keys[layer, :, :, :self.length].copy_(
                    self.keys[layer, :, :, :self.length].index_select(0, index)
                )
                gathered.values[layer, :, :, :self.length].copy_(
                    self.values[layer, :, :, :self.length].index_select(0, index)
                )
        gathered.length = self.length
        return gathered

class RotaryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(MODEL_DIM, 3 * MODEL_DIM)
        self.output = nn.Linear(MODEL_DIM, MODEL_DIM)
        inverse_frequency = 1.0 / (
            ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)
        )
        angle = torch.outer(torch.arange(CONTEXT).float(), inverse_frequency)
        self.register_buffer("rope_cos", angle.cos(), persistent=False)
        self.register_buffer("rope_sin", angle.sin(), persistent=False)

    def forward(self, x, layer, cache=None):
        batch, length, _ = x.shape
        start = 0 if cache is None else cache.length
        end = start + length
        qkv = self.qkv(x).view(
            batch, length, 3, MODEL_HEADS, HEAD_DIM
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        cos = self.rope_cos[start:end][None, None]
        sin = self.rope_sin[start:end][None, None]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is None:
            keys, values, causal = k, v, True
        else:
            cache.keys[layer, :, :, start:end].copy_(k)
            cache.values[layer, :, :, start:end].copy_(v)
            keys = cache.keys[layer, :, :, :end]
            values = cache.values[layer, :, :, :end]
            causal = start == 0
        attended = F.scaled_dot_product_attention(
            q, keys, values, is_causal=causal
        )
        return self.output(
            attended.transpose(1, 2).reshape(batch, length, MODEL_DIM)
        )

class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.color = nn.Embedding(6, STATE_WIDTH)
        self.position = nn.Parameter(torch.empty(54, STATE_WIDTH))
        self.network = nn.Sequential(
            nn.Linear(54 * STATE_WIDTH, 2 * MODEL_DIM),
            nn.SiLU(),
            nn.Linear(2 * MODEL_DIM, MODEL_DIM),
        )

    def forward(self, state):
        embedded = self.color(state.long()) + self.position
        return self.network(embedded.flatten(-2))

class AdaLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(MODEL_DIM)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(MODEL_DIM, 2 * MODEL_DIM)
        )

    def forward(self, x, condition):
        gamma, beta = self.modulation(condition).chunk(2, -1)
        return self.norm(x) * (1 + gamma) + beta

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = AdaLayerNorm()
        self.attention = RotaryAttention()
        self.mlp_norm = AdaLayerNorm()
        self.mlp = nn.Sequential(
            nn.Linear(MODEL_DIM, 4 * MODEL_DIM),
            nn.GELU(),
            nn.Linear(4 * MODEL_DIM, MODEL_DIM),
        )

    def forward(self, x, condition, layer, cache=None):
        x = x + self.attention(
            self.attention_norm(x, condition), layer, cache
        )
        return x + self.mlp(self.mlp_norm(x, condition))

class DirectStatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(INPUT_VOCAB, MODEL_DIM)
        self.state_encoder = StateEncoder()
        self.blocks = nn.ModuleList(
            TransformerBlock() for _ in range(MODEL_LAYERS)
        )
        self.norm = nn.LayerNorm(MODEL_DIM)
        self.policy = nn.Linear(MODEL_DIM, MOVES)
        self.apply(self._initialize)
        nn.init.normal_(self.state_encoder.position, std=0.02)
        for module in self.modules():
            if isinstance(module, AdaLayerNorm):
                nn.init.normal_(module.modulation[-1].weight, std=1e-3)
                nn.init.zeros_(module.modulation[-1].bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def new_cache(self, batch, dtype):
        return KVCache(MODEL_LAYERS, batch, CONTEXT, dtype, DEVICE)

    def hidden(self, token, state, cache=None):
        length = token.shape[1]
        start = 0 if cache is None else cache.length
        if start + length > CONTEXT:
            raise ValueError("sequence exceeds context")
        condition = self.state_encoder(state)
        x = self.token_embedding(token) + condition
        for layer, block in enumerate(self.blocks):
            x = block(x, condition, layer, cache)
        if cache is not None:
            cache.length += length
        return self.norm(x)

    def forward(self, token, state, cache=None):
        return self.policy(self.hidden(token, state, cache))