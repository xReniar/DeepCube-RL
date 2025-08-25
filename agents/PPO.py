import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .agent import Agent
from environment import Environment


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
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        pass

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class PPO(Agent):
    def __init__(
        self,
        env: Environment,
        args: dict
    ) -> None:
        super().__init__(env, args)
        self.total_timesteps = int(args["total_timesteps"])
        self.timesteps_per_batch = int(args["timesteps_per_batch"])
        self.max_timesteps_per_episode = int(args["max_timesteps_per_episode"])

        # environment info
        self.obs_dim = 54

        # initialize actor and critic networks
        self.actor = Network(self.obs_dim, 64, self.env.action_space.shape[0])
        self.critic = Network(self.obs_dim, 1)

        # covariance matrice for get_action
        self.cov_var = torch.full(size=(self.self.env.action_space.shape,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def train(self) -> None:
        t = 0

        while t < self.total_timesteps:
            ep_rews = []

            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                #batch_obs.append()

    def rollout(self):
        # Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

