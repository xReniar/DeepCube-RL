import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment import cube_graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE


class GCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(out_channels * 2, out_channels, cached=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
class GAEModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
        super().__init__()
        self.model = GAE(GCNEncoder(in_channels, out_channels))

    def forward(self, x):
        return self.model(x)