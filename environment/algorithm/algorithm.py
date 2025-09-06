from abc import ABC, abstractmethod
from magiccube import Cube


class Algorithm(ABC):
    def __init__(self, cube: Cube) -> None:
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
    
    def status(self):
        # cube_status will be a variable that flags 1 for every correct piece that is inserted
        cube_status = [

        ]

    @abstractmethod
    def reward(self) -> int:
        pass