from .algorithm import Algorithm, init_algo
from magiccube import Cube
import random


moves = ['R', "R'", 'L', "L'", 'U', "U'", 'D', "D'", 'F', "F'", 'B', "B'"]

class Environment:
    def __init__(
        self,
        method: str,
        state: str = None
    ) -> None:
        self.cube = Cube(
            size = 3,
            state = state
        )
        self.algorithm: Algorithm = init_algo(method)
        
        self.scramble() # to start with a scrambled cube
        self.start_state: str = self.cube.get_kociemba_facelet_positions()
        self.state: str = self.start_state

    def reset(self) -> str:
        '''
        Reset the environment to get the first observation
        '''
        self.state = self.start_state
        return self.state
    
    def is_terminated(self) -> bool:
        '''
        Checks if all the facelets of the same color are in the same face
        '''
        positions = self.cube.get_kociemba_facelet_positions()
        faces = []
        for i in range(0, 6):
            faces.append(positions[9*i: 9 + 9*i])

        top = any([face == "U" for face in faces[0]])
        right = any([face == "R" for face in faces[1]])
        front = any([face == "F" for face in faces[2]])
        bottom = any([face == "D" for face in faces[3]])
        left = any([face == "L" for face in faces[4]])
        back = any([face == "B" for face in faces[5]])

        return top and right and front and bottom and left and back
    
    def scramble(self) -> None:
        '''
        Scrambles the cube
        '''
        self.cube.rotate(' '.join(random.choices(moves, k=20)))
        self.state = self.cube.get_kociemba_facelet_positions()

    def step(
        self,
        action: str
    ) -> tuple:
        # makes action and update state
        self.cube.rotate(action)
        self.state = self.cube.get_kociemba_facelet_positions()

        # calculate reward
        reward = self.algorithm.status()

        return (self.state, reward, self.is_terminated())