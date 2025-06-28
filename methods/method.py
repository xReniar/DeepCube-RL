from abc import ABC, abstractmethod
from magiccube import cube


class Method(ABC):
    def __init__(
        self,
        cube:  cube.Cube
    ) -> None:
        super().__init__()
        self.cube = cube

    def cube_faces(self) -> dict[str, list]:
        positions = self.cube.get_kociemba_facelet_positions()
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

    def solved(self) -> bool:
        return self.cube.is_done()

    @abstractmethod
    def cube_status() -> int:
        pass