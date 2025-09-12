import torch
import torch.nn as nn
from torch import optim
from .agent import Agent
from collections import namedtuple, deque
from itertools import count
import random
import numpy as np
import math
import os


class DeepQNet(nn.Module):
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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        embedded: torch.Tensor = self.embedding(state)
        attn_output, _ = self.attention(embedded, embedded, embedded)
        attn_output: torch.Tensor = self.layer_norm(embedded + attn_output)
        ffn_output: torch.Tensor = self.ffn(attn_output)
        output: torch.Tensor = self.layer_norm(attn_output + ffn_output)

        flattened: torch.Tensor = output.view(state.size(0), -1)
        output: torch.Tensor = self.output_layer(flattened)
        return output


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
    def __init__(
        self,
        env,
        phase: str,
        args: dict
    ) -> None:
        super().__init__(env, phase)
        args = args["DQN"][phase]
        self.steps = 0
        self.gamma: float = float(args["gamma"])
        self.batch_size: int = int(args["batch_size"])
        self.eps_start: float = float(args["eps_start"])
        self.eps_end: float = float(args["eps_end"])
        self.eps_decay: int = int(args["eps_decay"])
        self.tau: float = float(args["tau"])
        self.lr: float = float(args["lr"])
        self.num_episodes: int = int(args["num_episodes"])
        self.policy_net = DeepQNet(5, 128, self.env.action_space.shape[0]).to(self.device)
        self.target_net = DeepQNet(5, 128, self.env.action_space.shape[0]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(int(args["mem_capacity"]))

        self.episode_durations = []

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def action(self, state: np.ndarray) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps / self.eps_decay)
        self.steps += 1

        result = None
        if sample > eps_threshold:
            state = self.state_to_tensor(state)
            with torch.no_grad():
                result = self.policy_net.forward(state).argmax().unsqueeze(0)
        else:
            result = torch.tensor(
                data=[random.randint(0, self.env.action_space.shape[0] - 1)],
                device=self.device
            )
        return result.unsqueeze(0)
        
    def train(self) -> None:
        for _ in range(self.num_episodes):
            rewards = {}
            state = self.env.reset()
            state = self.env.state2

            current_reward = self.env.algorithm.reward()
            for t in count():
                action: torch.Tensor = self.action(state)
                move: str = self.action_to_move(action.item())
                obs, reward, done = self.env.step(move)
                obs = self.env.state2

                if reward not in rewards:
                    rewards[reward] = 1
                else:
                    rewards[reward] += 1
                print(rewards)

                current_reward = reward
                torch_current_reward = torch.tensor([current_reward], device=self.device)

                if done:
                    next_state = None
                else:
                    next_state = obs
                
                self.memory.push(
                    self.state_to_tensor(state),
                    action,
                    self.state_to_tensor(next_state) if next_state is not None else None,
                    torch_current_reward
                )
                state = next_state

                self.optimize()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    break
        
        os.makedirs("models/DQN", exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"models/DQN/{self.phase}_net.pth")