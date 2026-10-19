from magiccube import Cube, Face


def get_faces(cube: Cube) -> dict:
    return {
        "F": cube.get_face(Face.F),
        "R": cube.get_face(Face.R),
        "B": cube.get_face(Face.B),
        "L": cube.get_face(Face.L),
        "U": cube.get_face(Face.U),
        "D": cube.get_face(Face.D)
    }

# function that check if the first-cross-step is done
def is_first_cross_solved(cube: Cube) -> bool:
    f = get_faces(cube)
    d_center = f["D"][1][1]

    return all([
        f["F"][2][1] == f["F"][1][1] and f["D"][0][1] == d_center,
        f["R"][2][1] == f["R"][1][1] and f["D"][1][2] == d_center,
        f["B"][2][1] == f["B"][1][1] and f["D"][2][1] == d_center,
        f["L"][2][1] == f["L"][1][1] and f["D"][1][0] == d_center,
    ])

# function that check if the f2l-step is done
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

# function that check if the last-cross-step is done
def is_last_cross_solved(cube: Cube) -> bool:
    pass