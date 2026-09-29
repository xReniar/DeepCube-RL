from abc import ABC, abstractmethod
from magiccube import Cube


class Environment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, cube: Cube):
        pass

    @abstractmethod
    def is_terminated(self, cube: Cube) -> bool:
        pass