from abc import ABC, abstractmethod
from environment import Environment
import torch


class Agent(ABC):
    def __init__(
        self,
        env: Environment,
        args: dict
    ) -> None:
        super().__init__()
        self.env = env
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.mps.is_available() else
            "cpu"
        )
    
    @abstractmethod
    def action(self, state: torch.Tensor):
        pass

    @abstractmethod
    def train(self) -> None:
        pass