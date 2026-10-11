from typing import Callable
from collections import deque
from magiccube import Cube, CubeMove, Face
from magiccube.solver.basic.basic_solver import BasicSolver


def get_faces(cube: Cube) -> dict:
    return {
        "F": cube.get_face(Face.F),
        "R": cube.get_face(Face.R),
        "B": cube.get_face(Face.B),
        "L": cube.get_face(Face.L),
        "U": cube.get_face(Face.U),
        "D": cube.get_face(Face.D)
    }

def is_first_cross_solved(cube: Cube) -> bool:
    f = get_faces(cube)
    d_center = f["D"][1][1]

    return all([
        f["F"][2][1] == f["F"][1][1] and f["D"][0][1] == d_center,
        f["R"][2][1] == f["R"][1][1] and f["D"][1][2] == d_center,
        f["B"][2][1] == f["B"][1][1] and f["D"][2][1] == d_center,
        f["L"][2][1] == f["L"][1][1] and f["D"][1][0] == d_center,
    ])

def is_f2l_solved(cube: Cube) -> bool:
    f = get_faces(cube)
    d_center = f["D"][1][1]
    f_center = f["F"][1][1]
    r_center = f["R"][1][1]
    b_center = f["B"][1][1]
    l_center = f["L"][1][1]

    first_layer = all([
        f["F"][2][2] == f_center and f["D"][0][2] == d_center,
        f["R"][2][2] == r_center and f["D"][2][2] == d_center,
        f["B"][2][2] == b_center and f["D"][2][0] == d_center,
        f["L"][2][2] == l_center and f["D"][0][0] == d_center,
    ])

    second_layer = all([
        f["F"][1][2] == f_center and f["R"][1][0] == r_center,
        f["R"][1][2] == r_center and f["B"][1][0] == b_center,
        f["B"][1][2] == b_center and f["L"][1][0] == l_center,
        f["L"][1][2] == l_center and f["F"][1][0] == f_center,
    ])

    return (first_layer and second_layer)

def is_last_cross_solved(cube: Cube) -> bool:
    pass

def split_solution_steps(
    scrambled_state: str,
    solution: list[CubeMove],
    parsers: list
) -> dict:
    current = {
        "cube": Cube(state=scrambled_state),
        "solution": deque(solution)
    }

    result = []
    for parser in parsers:
        applied = []
        
        while not parser(current["cube"]):
            move = current["solution"].popleft()
            current["cube"].rotate([move])
            applied.append(move)
    
        result.append(applied)

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
    parsers: list[Callable]
) -> dict:
    cube = Cube()
    cube.rotate("x2")

    cube.scramble(num_steps=num_steps)
    scrambled_state = cube.get()

    solver = BasicSolver(Cube(state=scrambled_state))
    solution = solver.solve()

    steps = split_solution_steps(
        scrambled_state=scrambled_state,
        solution=solution,
        parsers = parsers
    )
    
    return {
        "inital_state": scrambled_state,
        "solution": steps
    }

def generate_scrambles():
    pass


if __name__ == "__main__":
    parsers = [
        is_first_cross_solved,
        is_f2l_solved
    ]

    x = generate_data(num_steps=15, parsers=parsers)
    print(x)