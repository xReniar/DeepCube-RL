import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from magiccube import Cube


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
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x
    
def convert_move_to_tensor(move: str):
    moves = [
        "#",
        "U", "U'",
        "D", "D'",
        "F", "F'",
        "R", "R'",
        "B", "B'",
        "L", "L'"
    ]

    index = moves.index(move)
    one_hot = torch.zeros(len(moves), dtype=torch.float32)
    one_hot[index] = 1.0
    return one_hot.unsqueeze(0)

def state_to_tensor(state: str) -> torch.Tensor:
    '''
    state_for_tensor = []
    for s in state:
        state_for_tensor.append(color[s])
    '''
    color = {
        "U": 0,"D": 1,"F": 2,
        "R": 3,"B": 4,"L": 5,
        ".": 7
    }
    faces = []


    for i in range(0, 6):
        faces.append(state[9*i: 9 + 9*i])

    top = faces[0]
    bottom = faces[3]
    front = faces[2]
    right = faces[1]
    back = faces[5]
    left = faces[4]
    state_for_tensor = []

    for i in range(9):
        state_for_tensor.append(color[top[i]])
        state_for_tensor.append(color[front[i]])
        state_for_tensor.append(color[right[i]])
        state_for_tensor.append(color[back[i]])
        state_for_tensor.append(color[left[i]])
        state_for_tensor.append(color[bottom[i]])

    return torch.tensor(state_for_tensor, dtype=torch.float).unsqueeze(0)


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.cuda.is_available() else
    "cpu"
)

dataset = json.load(open("data/dataset.json", "r"))
model = DeepQNet(54, 512, 13).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for data in dataset:
    sample: dict = dataset[data]
    cross_solution: str = sample["solution"]["cross"]

    cube = Cube()
    cube.rotate(sample["scramble"])

    for move in cross_solution.split():
        cube.rotate(move)
        label = convert_move_to_tensor(move).to(device)
        input = state_to_tensor(cube.get_kociemba_facelet_positions()).to(device)

        output = model(input)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)