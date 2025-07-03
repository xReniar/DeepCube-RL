from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args

    @abstractmethod
    def predict(self, state: str):
        pass