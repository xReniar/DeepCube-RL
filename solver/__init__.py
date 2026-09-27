from .env import Environment
from .cross import CrossEnv
from magiccube import Cube
from copy import deepcopy


MOVES = ["U", "U'", "F", "F'", "R", "R'",
         "D", "D'", "B", "B'", "L", "L'"]


_env = {
    "cross": CrossEnv,
    "f2l": None,
    "oll": None,
    "pll": None
}


class Solver:
    def __init__(self, env: str, cube: Cube):
        self.cube = cube
        self.env: Environment = _env[env]()

    def solve(self) -> list[str]:
        moves = []

        while not self.env.is_terminated(self.cube):
            _curr_reward = self.env.evaluate(self.cube)
            neighbor_reward_list = []
            for i, neighbor in enumerate(self._generate_neighbors()):
                neighbor_reward_list.append([
                    MOVES[i],
                    self.env.evaluate(neighbor),
                    neighbor
                ])

            neighbor_reward_list = list(sorted(
                neighbor_reward_list,
                key=lambda x: x[1],
                reverse=True
            ))

            best_move, best_value, best_cube = neighbor_reward_list[0]
            if best_value > _curr_reward:
                self.cube = best_cube
                _curr_reward = best_value
                moves.append(best_move)
            else:
                break

        return moves
        

    def _generate_neighbors(self) -> list[Cube]:
        neighbors = []
        for move in MOVES:
            # apply move
            self.cube.rotate(move)

            # add to neighbors
            neighbors.append(deepcopy(self.cube))

            # undo cube for next moves, if present
            self.cube.rotate(move[:-1] if move.endswith("'") else move + "'")

        return neighbors


__all__ = [
    "Solver"
]