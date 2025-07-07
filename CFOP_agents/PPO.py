import torch
import torch.nn as nn
import torch.nn.functional as F
from .agent import Agent


class PPO(Agent):
    def __init__(self, args):
        super().__init__(args)

