from abc import ABC, abstractmethod
from magiccube import Cube


class Algorithm(ABC):
    def __init__(self) -> None:
        super().__init__()

    def cube_faces(
        self,
        cube: Cube
    ) -> dict[str, list]:
        positions = cube.get_kociemba_facelet_positions()
        faces = []
        for i in range(0, 6):
            faces.append(positions[9*i: 9 + 9*i])

        faces_dict = dict(
            top = faces[0],
            right = faces[1],
            front = faces[2],
            bottom = faces[3],
            left = faces[4],
            back = faces[5]
        )

        return faces_dict

    def solved(
        self,
        cube: Cube
    ) -> bool:
        return cube.is_done()

    @abstractmethod
    def status(
        self,
        cube: Cube
    ) -> int:
        pass