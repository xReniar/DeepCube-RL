import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from .agent import Agent
from collections import namedtuple, deque
import random
import math


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
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
        self.gamma: float = args["gamma"]
        self.batch_size: int = args["batch_size"]
        self.eps_start: float = args["eps_start"]
        self.eps_end: float = args["eps_end"]
        self.eps_decay: int = args["eps_decay"]
        self.tau: float = args["tau"]
        self.lr: float = args["lr"]
        self.n_actions: int = args["n_actions"]
        self.num_episodes: int = args["num_episodes"]

        self.policy_net = DeepQNet(54, 128, self.n_actions).to(self.device)
        self.target_net = DeepQNet(54, 128, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.parameters())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(args["mem_capacity"])

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def action(self, state: str):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1 * self.steps / self.eps_decay)
        self.steps += 1

        state_tensor = self.convert_state_to_tensor(state)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state_tensor).max(1).indices.view(1,1)
        else:
            return torch.tensor(
                [[random.randint(0, self.n_actions - 1)]],
                device = self.device,
                dtype = torch.long
            )