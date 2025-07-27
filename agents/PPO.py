import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .agent import Agent


class Network(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()

        pass

    def forward(self, x: torch.Tensor):
        pass


class PPO(Agent):
    def __init__(self, args):
        super().__init__(args)
