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

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


class A2C(Agent):
    def __init__(self, args):
        super().__init__(args)
        self.total_timesteps = int(args["total_timesteps"])
        self.timesteps_per_batch = int(args["timesteps_per_batch"])
        self.max_timesteps_per_episode = int(args["max_timesteps_per_episode"])
        self.n_updates_per_iteration = int(args["n_updates_per_iteration"])

        self.lr = float(args["lr"])
        self.gamma = float(args["gamma"])
        self.clip = float(args["clip"])
        self.n_actions = int(args["n_actions"])

        self.actor = Network(54, 128, self.n_actions)
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=self.lr)
        self.critic = Network(54, 128, 1)
        self.critic_optim = optim.AdamW(self.actor.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def rollout(self):
        pass

