from .env import Environment
from magiccube import Cube


class CrossEnv(Environment):
    def __init__(self):
        super().__init__()

    def evaluate(self, cube: Cube) -> int:
        pass

    def is_terminated(self, cube: Cube) -> bool:
        pass