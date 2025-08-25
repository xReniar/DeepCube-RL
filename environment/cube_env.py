from .algorithm import Algorithm, init_algo
from magiccube import Cube
from .dummy_cube import DummyCube
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
        
        self.scramble() # start with a scrambled cube
        self._start_state = np.array(list(self.cube.get_kociemba_facelet_positions()))
        self.state = self._start_state

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

        return self.algorithm.status(self.cube) == 12
        #return top and right and front and bottom and left and back
    
    def scramble(self) -> None:
        '''
        Scrambles the cube
        '''
        self.cube.rotate(' '.join(random.choices(self.action_space, k=self._scramble_moves)))
        self.state = np.array(list(self.cube.get_kociemba_facelet_positions()))

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

        # calculate reward
        reward = self.algorithm.status(self.cube)

        return (self.state, reward, self.is_terminated())