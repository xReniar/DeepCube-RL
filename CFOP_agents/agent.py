from abc import ABC, abstractmethod
import torch


class Agent(ABC):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

    def convert_state_to_tensor(self, state: str) -> torch.Tensor:
        colors = ['U', 'D', 'F', 'R', 'B', 'L']
        color_to_idx = {color: idx for idx, color in enumerate(colors)}
        num_colors = len(colors)

        one_hot = torch.zeros(54, num_colors, dtype=torch.float32)
        for i, color in enumerate(state):
            one_hot[i, color_to_idx[color]] = 1.0

        return one_hot.flatten().to(self.device)
    
    @abstractmethod
    def action(self, state: str):
        pass