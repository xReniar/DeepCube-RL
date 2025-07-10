import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.normal as Normal
import numpy as np
from .agent import Agent


class ReplayBuffer():
    def __init__(
        self,
        max_size: int,
        input_shape,
        n_actions: int
    ) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(
        self,
        state,
        action,
        reward: int,
        state_,
        done
    ) -> None:
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(
        self,
        batch_size: int
    ) -> tuple:
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones 

class CriticNet(nn.Module):
    def __init__(
        self,
        beta: float,
        input_dims: int,
        n_actions: int,
        device: str
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(
        self,
        state,
        action
    ) -> torch.Tensor:
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q
    
class ValueNet(nn.Module):
    def __init__(
        self,
        beta: float,
        input_dims,
        device: str
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.fc1 = nn.Linear(*self.input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self,state) -> torch.Tensor:
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v
    
class ActorNet(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims,
        max_action,
        n_actions,
        device: str
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-6
        self.device = device

        self.fc1 = nn.Linear(*self.input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, self.n_actions)
        self.sigma = nn.Linear(256, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma
    
    def sample_normal(self, state, reparametrize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparametrize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

class SA2C(Agent):
    def __init__(self, args: dict):
        super().__init__(args)
        self.gamma = float(args["gamma"])
        self.tau = float(args["tau"])
        self.n_actions = int(args["n_actions"])
        self.memory = ReplayBuffer(
            max_size = int(args["max_size"]),
            input_shape = [54],
            n_actions = self.n_actions
        )
        self.batch_size = int(args["batch_size"])

        self.actor = ActorNet()

    

