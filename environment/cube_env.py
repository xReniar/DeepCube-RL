from .algorithm import Algorithm, init_algo
from magiccube import Cube
from .dummy_cube import DummyCube
import itertools
import numpy as np
import random


class Environment:
    def __init__(
        self,
        method: str,
        size: int,
        args: dict
    ) -> None:
        self._scramble_moves = int(args["scramble_moves"])
        self.cube = Cube(size=size)
        self.algorithm: Algorithm = init_algo(method)

        self._colors_to_positions = {"U": "W", "D": "Y", "F": "G", "R": "R", "B": "B", "L": "O"}
        self.action_space = np.array(["U", "D", "F", "R", "B", "L","U'", "D'", "F'", "R'", "B'", "L'"])

        self._piece_mapper = {
            0:  (1, 3, 4), 1:  (1, 3, -1), 2:  (1, 3, 5), 3:  (1, -1, 4), 4:  (1, -1, -1), 5:  (1, -1, 5),
            6:  (1, 2, 4), 7:  (1, 2, -1), 8:  (1, 2, 5), 9:  (-1, 3, 4), 10: (-1, 3, -1), 11: (-1, 3, 5),
            12: (-1, -1, 4), 13: (-1, -1, 5), 14: (-1, 2, 4), 15: (-1, 2, -1), 16: (-1, 2, 5), 17: (0, 3, 4),
            18: (0, 3, -1), 19: (0, 3, 5), 20: (0, -1, 4), 21: (0, -1, -1), 22: (0, -1, 5), 23: (0, 2, 4),
            24: (0, 2, -1), 25: (0, 2, 5)
        }
        
        self.scramble() # start with a scrambled cube
        self._start_state = np.array(list(self.cube.get_kociemba_facelet_positions()))
        self.state = self._start_state
        self.state2 = self._get_state()

    def _color_to_id(self, color) -> int:
        return color.value if color != None else -1

    def _get_state(self) -> np.ndarray:
        state = []
        for i, data in enumerate(self.cube.get_all_pieces().items()):
            #coord = data[0]
            piece = data[1]

            x = self._color_to_id(piece.get_piece_color(0))
            y = self._color_to_id(piece.get_piece_color(1))
            z = self._color_to_id(piece.get_piece_color(2))

            piece_id = None
            for id in self._piece_mapper.keys():
                permutations = list(itertools.permutations(self._piece_mapper[id]))
                if (x, y, z) in permutations:
                    piece_id = id

            state.append([x, y, z, i, piece_id])
        return np.array([state])

    def reset(self) -> np.ndarray:
        '''
        Reset the environment to get the first observation
        '''
        self.state = self._start_state
        positions = "".join(self.state)
        faces = []
        for i in range(0, 6):
            faces.append(positions[9*i: 9 + 9*i])

        top = "".join([self._colors_to_positions[face] for face in faces[0]])
        right = "".join([self._colors_to_positions[face] for face in faces[1]])
        front = "".join([self._colors_to_positions[face] for face in faces[2]])
        bottom = "".join([self._colors_to_positions[face] for face in faces[3]])
        left = "".join([self._colors_to_positions[face] for face in faces[4]])
        back = "".join([self._colors_to_positions[face] for face in faces[5]])
        self.cube = Cube(state=f"{top}{left}{front}{right}{back}{bottom}")
        self.state2 = self._get_state()
        return self.state
    
    def is_terminated(self) -> bool:
        '''
        Checks if all the facelets of the same color are in the same face
        '''
        positions = self.cube.get_kociemba_facelet_positions()
        faces = []
        for i in range(0, 6):
            faces.append(positions[9*i: 9 + 9*i])

        top = all([face == "U" for face in faces[0]])
        right = all([face == "R" for face in faces[1]])
        front = all([face == "F" for face in faces[2]])
        bottom = all([face == "D" for face in faces[3]])
        left = all([face == "L" for face in faces[4]])
        back = all([face == "B" for face in faces[5]])

        return self.algorithm.status(self.cube) == 50
        #return top and right and front and bottom and left and back
    
    def scramble(self) -> None:
        '''
        Scrambles the cube
        '''
        self.cube.rotate(' '.join(random.choices(self.action_space, k=self._scramble_moves)))
        self.state = np.array(list(self.cube.get_kociemba_facelet_positions()))
        self.state2 = self._get_state()

    def step(
        self,
        action: str
    ) -> tuple[np.ndarray, float, bool]:
        # makes action
        if action not in self.action_space:
            raise ValueError(f"Unrecognized '{action}' move")
        
        # update state
        self.cube.rotate(action)
        self.state = np.array(list(self.cube.get_kociemba_facelet_positions()))
        self.state2 = self._get_state()

        # calculate reward
        reward = self.algorithm.status(self.cube)

        return (self.state, reward, self.is_terminated())