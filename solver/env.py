from abc import ABC, abstractmethod
from magiccube import Cube, Color


class Environment(ABC):
    def __init__(self):
        self.color = [Color.B, Color.R, Color.G, Color.O]

    @abstractmethod
    def evaluate(self, cube: Cube):
        pass

    @abstractmethod
    def is_terminated(self, cube: Cube) -> bool:
        pass