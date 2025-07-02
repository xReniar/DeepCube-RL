import torch
import torch.nn as nn


class DeepQNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, output_dim * 2)
        self.layer2 = nn.Linear(output_dim * 2, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        return x