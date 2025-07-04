import torch
import torch.nn as nn
import torch.nn.functional as F
from .agent import Agent


class Actor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x
    

class A2C(Agent):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

    def action(self, state: str):
        pass