import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment import Cube

from cube_encoding import GAEModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch


model = GAEModel(1, 3)


def train(loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        # batch.x, batch.edge_index, batch.train_pos_edge_index
        z = model.model.encode(batch.x, batch.edge_index)
        loss = model.model.recon_loss(z, batch.edge_index)
        # if args.variational:
        #     loss = loss + (1 / batch.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * batch.num_graphs
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    dataset = open("scrambles.txt", "r").readlines()
    dataset = list(map(lambda x:x.strip("\n"), dataset))

    data_list = []
    for scramble in dataset:
        cube = Cube()
        cube.rotate(scramble)

        x, edge_index = cube.graph_state()

        data_list.append(Data(
            x = x,
            edge_index = edge_index
        ))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    split = int(0.8 * len(data_list))
    train_data = data_list[:split]
    test_data = data_list[split:]

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    for epoch in range(100):
        loss = train(train_loader, optimizer)
        print(f"Epoch {epoch}: Loss={loss:.4f}")

    