import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .agent import Agent
from environment import Environment
from torch.distributions import MultivariateNormal
import numpy as np


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
        self.gamma = float(args["gamma"])

        # environment info
        self.obs_dim = 54
        self.act_dim = self.env.action_space.shape[0]

        # initialize actor and critic networks
        self.actor = Network(self.obs_dim[1], 64, self.act_dim)
        self.critic = Network(self.obs_dim[1], 1)

        # covariance matrice for get_action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def compute_rtgs(self, batch_rews: list) -> torch.Tensor:
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def rollout(self) -> tuple:
        # Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []

            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(self.state_to_tensor(obs))
                action, log_prob = self.get_action(self.state_to_tensor(obs))
                obs, reward, done = self.env.step(action)

                ep_rews.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    def get_action(self, obs: torch.Tensor) -> tuple[str, any]:
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def train(self) -> None:
        t = 0

        while t < self.total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

