from ..env_base import EnvBase
from ..reward import Reward
import numpy as np
import json
import random


@EnvBase.register_model("f2l")
class F2L_Env(EnvBase):
    def __init__(self, args: dict):
        self.action_space = np.array([
            "L' U L", "R U' R'", "F' U F", "B U' B'",                   # insert moves good orientation case 1 (first layer)
            "R' U R", "L U' L'", "B' U B", "F U' F'",                   # insert moves good orientation case 2 (first layer)
            "F' U2 F U2 R U' R'", "R' U2 R U2 B U' B'",                 # insert moves bad orientation (first layer)
            "B' U2 B U2 L U' L'", "L' U2 L U2 F U' F'",                 # insert moves bad orientation (first layer)
            "R U' R' U'", "F' U F U", "B U' B' U'", "R' U R U",         # insert moves (second layer) 
            "L U' L' U'", "B' U B U", "F U' F' U'", "L' U L U",         # insert moves (second layer)
            "U", "U'"
        ])
        super().__init__(args)
        self.algorithm: Reward = Reward(self.cube, "f2l")

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
        dataset: dict = json.load(open("data/dataset.json", "r"))
        choice = str(random.randint(1, len(dataset.keys())))

        selection = dataset[choice]

        '''reverse_history = self.cube.reverse_history()
        print(len(reverse_history))
        self.cube.rotate(reverse_history)'''
        self.cube.reset()

        self.cube.rotate(selection["scramble"])
        self.cube.rotate(selection["solution"]["cross"])

        self.state = np.array(list(self.cube.get_kociemba_facelet_positions()))
        self.state2 = self._get_state()