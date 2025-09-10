from ..env_base import EnvBase
from ..algorithm import Algorithm
import numpy as np
import random


@EnvBase.register_model("cross")
class Cross_Env(EnvBase):
    def __init__(self, args: dict):
        self.action_space = np.array([
            "U", "D", "F","R", "B", "L",
            "U'", "D'", "F'", "R'", "B'", "L'"
        ])
        super().__init__(args)
        self.algorithm: Algorithm = Algorithm(self.cube, "cross")

    def is_terminated(self) -> bool:
        faces = self.algorithm.cube_faces()
        
        condition = [
            int(faces["front"][4] == faces["front"][7] and faces["bottom"][4] == faces["bottom"][1]),
            int(faces["right"][4] == faces["right"][7] and faces["bottom"][4] == faces["bottom"][5]),
            int(faces["back"][4] == faces["back"][7] and faces["bottom"][4] == faces["bottom"][7]),
            int(faces["left"][4] == faces["left"][7] and faces["bottom"][4] == faces["bottom"][3])
        ]

        return sum(condition) == 4

    def scramble(self) -> None:
        self.cube.rotate(' '.join(random.choices(self.action_space, k=self._scramble_moves)))
        self.state = np.array(list(self.cube.get_kociemba_facelet_positions()))
        self.state2 = self._get_state()