from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
import torch.nn as nn
from magiccube import Cube
from cube2graph import cube2graph


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 54)
        self.conv2 = GCNConv(54, 108)
        self.fc1 = nn.Linear(108, 54)
        self.fc2 = nn.Linear(54, 26)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = global_mean_pool(x, data.batch)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        return x


cube = Cube()
G = from_networkx(cube2graph(cube))

model = BaseModel()
output = model(G)

print(output.shape)