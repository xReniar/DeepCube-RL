import magiccube
import torch


class Environment:
    def __init__(
        self,
        state: str
    ) -> None:
        self.cube = magiccube.Cube(
            size = 3,
            state = state
        )

        self.start_state: str = self.cube.get_kociemba_facelet_positions()
        self.state: str = self.start_state

    def reset(self) -> str:
        self.state = self.start_state
        return self.state
    
    def get_action(self) -> None:
        pass

    def step(
        self,
        action: str
    ) -> tuple:
        self.cube.rotate(action)

    