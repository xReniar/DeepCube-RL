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

    return dict(zip(
        list(map(lambda func: func.__name__, stages)),
        result
    ))

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
        "initial_state": scrambled_state,
        "solution": split_solution_steps(
            scrambled_state=scrambled_state,
            solution=solution,
            stages = stages
        )
    }


if __name__ == "__main__":
    stages = [
        conditions.w_cross,
        conditions.f2l,
        conditions.last_cross,
        conditions.permute_edges
    ]

    dataset = []
    scrambles = generate_scrambles()

    print("Generating solution for each scramble...")
    with Pool(cpu_count()) as pool:
        dataset += pool.starmap(
            generate_data,
            list(map(lambda x: (x, stages), scrambles))
        )

    print("Saving dataset...")
    with open("dataset.txt", "w") as f:
        dataset = list(map(lambda x: f"{json.dumps(x)}\n", dataset))
        f.writelines(dataset)

        #for data in dataset:
        #    f.write(f"{json.dumps(data)}\n")