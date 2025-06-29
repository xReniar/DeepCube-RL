import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data


# Define the edge indices (source, target)
edge_index = torch.tensor([
    [0, 1, 1, 2, 3, 3],
    [1, 0, 2, 1, 2, 0]
], dtype=torch.long)

# Define node features (4 nodes, each with 3 features)
x = torch.tensor([
    [1],
    [4],
    [7],
    [10]
], dtype=torch.float)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 4)  # First layer: 3 input features, 4 output features
        self.conv2 = GCNConv(4, 2)  # Second layer: 4 input features, 2 output features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    

model = GCN()


conv1 = GCNConv(1, 2)
conv2 = GCNConv(2, 4)
conv3 = GCNConv(4, 6)
y = conv1(x, edge_index)
y = conv2(y, edge_index)
print(y)