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
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 26, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to("cuda" if torch.cuda.is_available() else "cpu")
        '''

        embedded = self.embedding(x)
        attn_output, _ = self.attention(embedded, embedded, embedded)
        attn_output = self.layer_norm(embedded + attn_output)
        ffn_output = self.ffn(attn_output)
        output = self.layer_norm(attn_output + ffn_output)

        flattened = output.view(x.size(0), -1)
        output = self.output_layer(flattened)
        return output

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
        self.lr = float(args["lr"])
        self.n_updates_per_iteration  = int(args["n_updates_per_iteration"])
        self.clip = float(args["clip"])

        # environment info
        self.obs_dim = 54
        self.act_dim = self.env.action_space.shape[0]

        # initialize actor and critic networks)
        self.actor = Network(5, 128, self.act_dim).to(self.device)
        self.critic = Network(5, 128, 1).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        # covariance matrice for action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

    def compute_rtgs(self, batch_rews: list) -> torch.Tensor:
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)
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
            obs = self.env.state2
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(self.state_to_tensor(obs))
                action, log_prob = self.action(self.state_to_tensor(obs))
                obs, reward, done = self.env.step(self.action_to_move(action.argmax().item()))
                obs = self.env.state2

                ep_rews.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.stack(batch_obs).float().squeeze(1).to(self.device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float, device=self.device).squeeze(1)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


    def action(self, obs: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        
        action: torch.Tensor = dist.sample()
        log_prob: torch.Tensor = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob.detach()
    
    def evaluate(self, batch_obs, batch_acts) -> tuple:
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def train(self) -> None:
        t = 0

        while t < self.total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            
            t += np.sum(batch_lens)
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1+ self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()