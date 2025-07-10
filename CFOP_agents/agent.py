from abc import ABC, abstractmethod
import torch


class Agent(ABC):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.mps.is_available() else
            "cpu"
        )
    
    @abstractmethod
    def action(self, state: torch.Tensor):
        pass