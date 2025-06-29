from magiccube import cube
from .cube_graph import CubeGraph
import torch


class Cube:
    def __init__(
        self,
        state: str = None
    ) -> None:
        self.cube = cube.Cube(state)
        self.cube

    def graph_state(self) -> torch.Tensor:
        pass

    def rotate(
        self,
        moves: str
    ) -> None:
        self.cube.rotate(moves)

    