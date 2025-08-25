import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        self.fc1 = F.relu(self.fc1(x))
        self.fc2 = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        env: object,
        args: dict
    ) -> None:
        super().__init__(env, args)
        self.steps = 0
        self.gamma: float = float(args["gamma"])
        self.batch_size: int = int(args["batch_size"])
        self.eps_start: float = float(args["eps_start"])
        self.eps_end: float = float(args["eps_end"])
        self.eps_decay: int = int(args["eps_decay"])
        self.tau: float = float(args["tau"])
        self.lr: float = float(args["lr"])
        self.num_episodes: int = int(args["num_episodes"])

        self.policy_net = DeepQNet(1, 64, self.env.action_space.shape[0]).to(self.device)
        self.target_net = DeepQNet(1, 64, self.env.action_space.shape[0]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayMemory(int(args["mem_capacity"]))

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
                data=[random.randint(0, self.env.action_space.shape[0] - 1)],
                device=self.device
            )
            return result
        
    def train(self) -> None:
        pass