from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        pass