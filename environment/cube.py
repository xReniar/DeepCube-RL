from magiccube import cube
import torch


class Cube:
    def __init__(
        self,
        state: str = None
    ) -> None:
        self.cube = cube.Cube(state)

    def graph_state(self) -> torch.Tensor:
        faces = {
            "U": [],
            "D": [],
            "F": [],
            "R": [],
            "B": [],
            "L": []
        }

    def rotate(
        self,
        moves: str
    ) -> None:
        self.cube.rotate(moves)

    