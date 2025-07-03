import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from .agent import Agent
import numpy as np


class DeepQNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        alpha: float
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
    

class DQN(Agent):
    def __init__(self, args):
        super().__init__(args)
        self.GAMMA = args["gamma"]
        self.EPSILON = args["epsilon"]
        self.EPS_END = args["eps_end"]
        self.action_space = args["actionSpace"]
        self.memSize = args["maxMemorySize"]
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = args["replace"]
        self.Q_eval = DeepQNet(54, 128, 12, )
        self.T_eval = DeepQNet(54, 128, 12)

    def store_transition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr % self.memSize] = [state, action, reward, state_]
        self.memCntr += 1

    def chooseAction(self, state: str):
        rand = np.random.random()
        actions = self.Q_eval(state)

        if rand < 1 - self.EPSILON:
            action = torch.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.action_space)
        self.steps += 1
        
        return action
    
    def learn(self, batch_size):
        pass

    def predict(self, state: str):
        pass