import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x:torch.Tensor):
        x = F.relu(self.fc1(x))
        return self.fc2(x)