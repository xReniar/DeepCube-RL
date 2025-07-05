from .algorithm import Algorithm, init_algo
from magiccube import Cube
import random
import torch


moves = ["U", "D", "F", "R", "B", "L", "U'", "D'", "F'", "R'", "B'", "L'"]

class Environment:
    def __init__(
        self,
        method: str,
        size: int,
        device: str
    ) -> None:
        self.cube = Cube(size=size)
        self.algorithm: Algorithm = init_algo(method)
        self.device = device
        
        self.scramble() # to start with a scrambled cube
        self.start_state: torch.Tensor = self.__state_to_tensor(self.cube.get_kociemba_facelet_positions())
        self.state: torch.Tensor = self.start_state

    def __state_to_tensor(self, state: str) -> torch.Tensor:
        state_for_tensor = []
        for s in state:
            state_for_tensor.append(state[s])

        return torch.tensor(state_for_tensor, device=self.device, dtype=torch.float)
    
    def __convert_tensor_to_move(self, tensor_action: torch.Tensor) -> str:
        return moves[tensor_action.item()]

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

        return top and right and front and bottom and left and back
    
    def scramble(self) -> None:
        '''
        Scrambles the cube
        '''
        self.cube.rotate(' '.join(random.choices(moves, k=20)))
        self.state = self.__state_to_tensor(self.cube.get_kociemba_facelet_positions())

    def step(
        self,
        action_tensor: torch.Tensor
    ) -> tuple:
        # makes action and update state
        move = self.__convert_tensor_to_move(action_tensor)
        self.cube.rotate(move)
        self.state = self.__state_to_tensor(self.cube.get_kociemba_facelet_positions())

        # calculate reward
        reward = self.algorithm.status(self.cube)

        return (self.state, reward, self.is_terminated())