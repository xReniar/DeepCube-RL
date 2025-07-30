from .agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc4 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        distribution = Categorical(F.softmax(x, dim=-1))
        return distribution
    
class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc4 = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        return value

class A2C(Agent):
    def __init__(self, args):
        super().__init__(args)
        self.num_episodes = int(args["num_episodes"])

        self.actor_net = Actor(54, 128, 12).to(self.device)
        self.actor_optim = optim.Adam(self.actor_net.parameters())
        self.critic_net = Critic(54, 128).to(self.device)
        self.critic_optim = optim.Adam(self.critic_net.parameters())

    def compute_returns(
        self,
        next_value,
        rewards,
        masks, 
        gamma=0.99
    ):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns
    
    def optimize(self):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        #actor_loss.backward()
        #critic_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
    
    def action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor_net(state)
        value = self.critic_net(state)

        return dist, value