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
def d_cross(cube: Cube) -> bool:
    f = get_faces(cube)
    d_center = f["D"][1][1]

    return all([
        f["F"][2][1] == f["F"][1][1] and f["D"][0][1] == d_center,
        f["R"][2][1] == f["R"][1][1] and f["D"][1][2] == d_center,
        f["B"][2][1] == f["B"][1][1] and f["D"][2][1] == d_center,
        f["L"][2][1] == f["L"][1][1] and f["D"][1][0] == d_center,
    ])

# function that check if the first layer is done
def first_layer(cube: Cube) -> bool:
    f = get_faces(cube)
    d_center = f["D"][1][1]
    f_center = f["F"][1][1]
    r_center = f["R"][1][1]
    b_center = f["B"][1][1]
    l_center = f["L"][1][1]

    return all([
        f["F"][2][2] == f_center and f["D"][0][2] == d_center,
        f["R"][2][2] == r_center and f["D"][2][2] == d_center,
        f["B"][2][2] == b_center and f["D"][2][0] == d_center,
        f["L"][2][2] == l_center and f["D"][0][0] == d_center,
    ])

def second_layer(cube: Cube) -> bool:
    f = get_faces(cube)
    f_center = f["F"][1][1]
    r_center = f["R"][1][1]
    b_center = f["B"][1][1]
    l_center = f["L"][1][1]

    return all([
        f["F"][1][2] == f_center and f["R"][1][0] == r_center,
        f["R"][1][2] == r_center and f["B"][1][0] == b_center,
        f["B"][1][2] == b_center and f["L"][1][0] == l_center,
        f["L"][1][2] == l_center and f["F"][1][0] == f_center,
    ])

# function that check if the last-cross-step is done
def u_cross(cube: Cube) -> bool:
    f = get_faces(cube)
    u_center = f["U"][1][1]

    return all([
        f["U"][0][1] == u_center,
        f["U"][1][0] == u_center,
        f["U"][1][2] == u_center,
        f["U"][2][1] == u_center
    ])

def permute_edges(cube: Cube) -> bool:
    f = get_faces(cube)
    f_center = f["F"][1][1]
    r_center = f["R"][1][1]
    b_center = f["B"][1][1]
    l_center = f["L"][1][1]

    return all([
        f["F"][0][1] == f_center,
        f["R"][0][1] == r_center,
        f["B"][0][1] == b_center,
        f["L"][0][1] == l_center,
        u_cross(cube)
    ])

def permute_corners(cube: Cube) -> bool:
    f = get_faces(cube)
    pairs = [("F", "R", 2, 2), ("R", "B", 0, 2), ("B", "L", 0, 0), ("L", "F", 2, 0)]
    status = []

    for l, r, index_a, index_b in pairs:
        colors = [
            f[l][0][2],
            f[r][0][0],
            f["U"][index_a][index_b]
        ]
        labels = [
            f[l][1][1],
            f[r][1][1],
            f["U"][1][1]
        ]

        status.append(all(color in colors for color in labels))

    return all(status)
        