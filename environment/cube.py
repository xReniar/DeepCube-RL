import magiccube
import torch


class Environment:
    def __init__(
        self,
        state: str = None
    ) -> None:
        self.cube = magiccube.Cube(
            size = 3,
            state = state
        )

    def reset(
        self
    ) -> tuple:
        pass

    def step(
        self,
        action: str
    ) -> tuple:
        self.cube.rotate(action)

    