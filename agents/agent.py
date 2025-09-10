from abc import ABC, abstractmethod
from environment import Environment
import torch
import numpy as np


class Agent(ABC):
    def __init__(
        self,
        env: Environment,
        phase: str
    ) -> None:
        super().__init__()
        self.env = env
        self.phase = phase
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.mps.is_available() else
            "cpu"
        )

    def action_to_move(self, move_id: int) -> str:
        mapper = {i: action.item() for i, action in enumerate(self.env.action_space)}
        return mapper[move_id]
    
    def state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        '''
        mapper = {action: i for i, action in enumerate(["U", "D", "F", "R", "B", "L"])}
        tensor_state = torch.from_numpy(np.vectorize(mapper.get)(state)).to(self.device).float()
        return tensor_state.unsqueeze(0)
        '''
        return torch.from_numpy(state).float().to(self.device)
    
    @abstractmethod
    def action(self, state: torch.Tensor):
        pass

    @abstractmethod
    def train(self) -> None:
        pass