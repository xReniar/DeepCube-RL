from ..env_base import EnvBase
from ..algorithm import Algorithm
import numpy as np


@EnvBase.register_model("f2l")
class F2L_Env(EnvBase):
    def __init__(self, args: dict):
        self.action_space = np.array([
            "L' U L", "R U' R'", "F' U F", "B U' B'",                   # insert moves good orientation case 1 (first layer)
            "R' U R", "L U' L'", "B' U B", "F U' F'",                   # insert moves good orientation case 2 (first layer)
            "F' U2 F U2", "R' U2 R U2", "B' U2 B U2", "L' U2 L U2",     # orient moves bad orientation (first layer)
            "R U' R' U'", "F' U F U", "B U' B' U'", "R' U R U",         # insert moves (second layer) 
            "L U' L' U'", "B' U B U", "F U' F' U'", "L' U L U"          # insert moves (second layer)
            "U", "U'"
        ])
        super().__init__(args)
        self.algorithm: Algorithm = Algorithm(self.cube, "f2l")

    def is_terminated(self) -> bool:
        faces = self.algorithm.cube_faces()
        
        first_layer = [
            int(faces["front"][8] == faces["front"][4]) and (faces["right"][6] == faces["right"][4]),
            int(faces["right"][8] == faces["right"][4]) and (faces["back"][6] == faces["back"][4]),
            int(faces["back"][8] == faces["back"][4]) and (faces["left"][6] == faces["left"][4]),
            int(faces["left"][8] == faces["left"][4]) and (faces["front"][6] == faces["front"][4])
        ]
        second_layer = [
            int(faces["front"][5] == faces["front"][4]) and (faces["right"][3] == faces["right"][4]),
            int(faces["right"][5] == faces["right"][4]) and (faces["back"][3] == faces["back"][4]),
            int(faces["back"][5] == faces["back"][4]) and (faces["left"][3] == faces["left"][4]),
            int(faces["left"][5] == faces["left"][4]) and (faces["front"][3] == faces["front"][4])
        ]

        return sum(first_layer) + sum(second_layer) == 8

    def scramble(self):
        pass