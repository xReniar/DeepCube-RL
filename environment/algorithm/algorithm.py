from abc import ABC, abstractmethod
from magiccube import Cube
import numpy as np


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
    
    def status(self) -> np.ndarray:
        # cube_status will be a variable that flags 1 for every correct piece that is inserted
        faces = self.cube_faces()

        cube_status = np.array([
            [
                int((faces["front"][4] == faces["front"][7]) and (faces["bottom"][4] == faces["bottom"][1])),
                int((faces["right"][4] == faces["right"][7]) and (faces["bottom"][4] == faces["bottom"][5])),
                int((faces["back"][4] == faces["back"][7]) and (faces["bottom"][4] == faces["bottom"][7])),
                int((faces["left"][4] == faces["left"][7]) and (faces["bottom"][4] == faces["bottom"][3]))
            ],
            [
                int((faces["front"][8] == faces["front"][4]) and (faces["right"][6] == faces["right"][4])),
                int((faces["right"][8] == faces["right"][4]) and (faces["back"][6] == faces["back"][4])),
                int((faces["back"][8] == faces["back"][4]) and (faces["left"][6] == faces["left"][4])),
                int((faces["left"][8] == faces["left"][4]) and (faces["front"][6] == faces["front"][4])),
            ],
            [
                int((faces["front"][5] == faces["front"][4]) and (faces["right"][3] == faces["right"][4])),
                int((faces["right"][5] == faces["right"][4]) and (faces["back"][3] == faces["back"][4])),
                int((faces["back"][5] == faces["back"][4]) and (faces["left"][3] == faces["left"][4])),
                int((faces["left"][5] == faces["left"][4]) and (faces["front"][3] == faces["front"][4])),
            ]
        ])

        return cube_status

    @abstractmethod
    def reward(self) -> int:
        pass