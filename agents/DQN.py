import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from torch import optim
from .agent import Agent
from collections import namedtuple, deque
import random
import math


class DeepQNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)

        # Testa finale: Q-values per le 12 mosse
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        root_index = 0
        root_feat = x[root_index]

        q_values = self.head(root_feat)
        return q_values


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(Agent):
    def __init__(self, args):
        super().__init__(args)
        self.steps = 0
        self.gamma: float = float(args["gamma"])
        self.batch_size: int = int(args["batch_size"])
        self.eps_start: float = float(args["eps_start"])
        self.eps_end: float = float(args["eps_end"])
        self.eps_decay: int = int(args["eps_decay"])
        self.tau: float = float(args["tau"])
        self.lr: float = float(args["lr"])
        self.n_actions: int = int(args["n_actions"])
        self.num_episodes: int = int(args["num_episodes"])

        self.policy_net = DeepQNet(1, 64, self.n_actions).to(self.device)
        self.target_net = DeepQNet(1, 64, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(int(args["mem_capacity"]))

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states_batch = Batch.from_data_list(non_final_next_states).to(self.device)

        state_batch = Batch.from_data_list(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states_batch).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps / self.eps_decay)
        self.steps += 1

        if sample > eps_threshold:
            with torch.no_grad():
                result = self.policy_net.forward(state).argmax().unsqueeze(0)
                return result
        else:
            result = torch.tensor(
                data=[random.randint(0, self.n_actions - 1)],
                device=self.device
            )
            return result