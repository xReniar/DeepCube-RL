import conditions
from typing import Callable
from collections import deque
from magiccube import Cube, CubeMove
from magiccube.solver.basic.basic_solver import BasicSolver
import json


def split_solution_steps(
    scrambled_state: str,
    solution: list[CubeMove],
    stages: list
) -> dict:
    current = {
        "cube": Cube(state=scrambled_state),
        "solution": deque(solution)
    }

    result = []
    for stage_check in stages:
        applied = []
        
        while not stage_check(current["cube"]):
            move = current["solution"].popleft()
            current["cube"].rotate([move])
            applied.append(str(move.type.value) + ("'" if move.is_reversed else ""))
    
        result.append(" ".join(applied))

    return {
        "w_cross": result[0],
        "f2l_steps": result[1],
        #"last_cross": parsers[2],
        #"permute_edges": parsers[3],
        #"permute_corners": parsers[4],
        #"orient_corners": parsers[5]
    }

def generate_data(
    num_steps: int,
    stages: list[Callable]
) -> dict:
    cube = Cube()
    cube.rotate("x2")

    cube.scramble(num_steps=num_steps)
    scrambled_state = cube.get()

    solver = BasicSolver(Cube(state=scrambled_state))
    solution = solver.solve()
    
    return {
        "inital_state": scrambled_state,
        "solution": split_solution_steps(
            scrambled_state=scrambled_state,
            solution=solution,
            stages = stages
        )
    }

def generate_scrambles():
    pass


if __name__ == "__main__":
    stages = [
        conditions.is_first_cross_solved,
        conditions.is_f2l_solved
    ]

    x = generate_data(num_steps=15, stages=stages)
    print(json.dumps(x, indent=4))