from .algorithm import Algorithm, init_algo
from magiccube import Cube
from .dummy_cube import DummyCube
import random
import torch


moves = ["U", "D", "F", "R", "B", "L",
         "U'", "D'", "F'", "R'", "B'", "L'"]
color = {
    "U": 0,"D": 1,"F": 2,
    "R": 3,"B": 4,"L": 5,
    ".": 7
}

class Environment:
    def __init__(
        self,
        method: str,
        size: int,
        device: str
    ) -> None:
        self.cube = DummyCube()
        #self.cube = Cube(size=size)
        self.algorithm: Algorithm = init_algo(method)
        self.device = device
        
        self.scramble() # to start with a scrambled cube
        self.start_state: torch.Tensor = self.__state_to_tensor(self.cube.get_kociemba_facelet_positions())
        self.state: torch.Tensor = self.start_state

    def __state_to_tensor(self, state: str) -> torch.Tensor:
        state_for_tensor = []
        for s in state:
            state_for_tensor.append(color[s])

        return torch.tensor(state_for_tensor, device=self.device, dtype=torch.float).unsqueeze(0)

    def reset(self) -> torch.Tensor:
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

        top = all([face == "U" for face in faces[0]])
        right = all([face == "R" for face in faces[1]])
        front = all([face == "F" for face in faces[2]])
        bottom = all([face == "D" for face in faces[3]])
        left = all([face == "L" for face in faces[4]])
        back = all([face == "B" for face in faces[5]])

        return self.algorithm.status(self.cube) == 100.0
        #return top and right and front and bottom and left and back
    
    def scramble(self) -> None:
        '''
        Scrambles the cube
        '''
        self.cube.rotate(' '.join(random.choices(moves, k=20)))
        self.state = self.__state_to_tensor(self.cube.get_kociemba_facelet_positions())

    def step(
        self,
        move_id: int
    ) -> tuple:
        # makes action
        move = moves[move_id]
        self.cube.rotate(move)

        # update state
        self.state = self.__state_to_tensor(self.cube.get_kociemba_facelet_positions())

        # calculate reward
        reward = self.algorithm.status(self.cube)
        print(reward)

        return (self.state, reward, self.is_terminated())