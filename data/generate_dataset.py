import conditions
from scrambler import generate_scrambles
from typing import Callable
from collections import deque
from magiccube import Cube, CubeMove
from magiccube.solver.basic.basic_solver import BasicSolver
from multiprocessing import Pool, cpu_count
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
    scramble: str,
    stages: list[Callable]
) -> dict:
    cube = Cube()
    cube.rotate("x2")
    cube.rotate(scramble)
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


if __name__ == "__main__":
    stages = [
        conditions.is_first_cross_solved,
        conditions.is_f2l_solved
    ]

    dataset = []
    scrambles = generate_scrambles()

    print("Generating solution for each scramble...")
    with Pool(cpu_count()) as pool:
        dataset += pool.starmap(
            generate_data,
            list(map(lambda x: (x, stages), scrambles))
        )

    print(len(dataset))
    print(dataset[0])