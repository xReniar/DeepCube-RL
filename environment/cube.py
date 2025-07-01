import magiccube
from .cube_graph import CubeGraph
import torch


class Cube:
    def __init__(
        self,
        state: str = None
    ) -> None:
        self.cube = magiccube.Cube(
            size = 3,
            state = state
        )

    def graph_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.cube.get_kociemba_facelet_positions()
        faces = []
        for i in range(0, 6):
            faces.append(positions[9*i: 9 + 9*i])

        cGraph = CubeGraph(faces)

        return cGraph.graph_state()

    def rotate(
        self,
        moves: str
    ) -> None:
        self.cube.rotate(moves)

    