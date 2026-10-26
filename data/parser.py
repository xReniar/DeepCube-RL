from magiccube import Cube
from conditions import get_faces


def get_reverse(applied: list[str]) -> list[str]:
    reverse = []
    for move in applied:
        if "\'" in move:
            reverse.append(move[:1])
        else:
            reverse.append(f"{move}\'")

    reverse.reverse()
    return " ".join(reverse)

def d_cross(cube: Cube, applied: list[str]):
    tokens = set([
        "<|f_piece_end|>","<|r_piece_end|>",
        "<|b_piece_end|>","<|l_piece_end|>"
    ])
    new_applied = ["<|d_cross_start|>"]

    # check which piece is already inserted
    f = get_faces(cube)
    d_center = f["D"][1][1]
    if (f["F"][2][1] == f["F"][1][1] and f["D"][0][1] == d_center):
        tokens.remove("<|f_piece_end|>")
    if (f["R"][2][1] == f["R"][1][1] and f["D"][1][2] == d_center):
        tokens.remove("<|r_piece_end|>")
    if (f["B"][2][1] == f["B"][1][1] and f["D"][2][1] == d_center):
        tokens.remove("<|b_piece_end|>")
    if (f["L"][2][1] == f["L"][1][1] and f["D"][1][0] == d_center):
        tokens.remove("<|l_piece_end|>")

    # add end-tokens where needed
    for move in applied:
        cube.rotate(move)
        f = get_faces(cube)

        new_applied.append(move)

        if (f["F"][2][1] == f["F"][1][1] and f["D"][0][1] == d_center) and ("<|f_piece_end|>" in tokens):
            new_applied.append("<|f_piece_end|>")
            tokens.remove("<|f_piece_end|>")
        if (f["R"][2][1] == f["R"][1][1] and f["D"][1][2] == d_center) and ("<|r_piece_end|>" in tokens):
            new_applied.append("<|r_piece_end|>")
            tokens.remove("<|r_piece_end|>")
        if (f["B"][2][1] == f["B"][1][1] and f["D"][2][1] == d_center) and ("<|b_piece_end|>" in tokens):
            new_applied.append("<|b_piece_end|>")
            tokens.remove("<|b_piece_end|>")
        if (f["L"][2][1] == f["L"][1][1] and f["D"][1][0] == d_center) and ("<|l_piece_end|>" in tokens):
            new_applied.append("<|l_piece_end|>")
            tokens.remove("<|l_piece_end|>")
    new_applied.append("<|d_cross_end|>")

    # add start-tokens where needed
    applied = [new_applied[0]]
    block = []
    for element in new_applied[1:-1]:
        block.append(element)
        if element.endswith("_piece_end|>"):
            applied.append(element.replace("_end", "_start"))
            applied.extend(block)
            block = []

    # moves after the last piece end belong to no piece
    applied.extend(block)
    applied.append(new_applied[-1])

    # replace an empty <|x_piece_start|> <|x_piece_end|> pair with <|x_piece_done|>
    result = []
    for element in applied:
        if element.endswith("_piece_end|>") and result[-1] == element.replace("_end", "_start"):
            result[-1] = element.replace("_end", "_done")
        else:
            result.append(element)

    return result

def first_layer(cube: Cube, applied: list[str]):
    pass

def second_layer(cube: Cube, applied: list[str]):
    pass

def u_cross(cube: Cube, applied: list[str]):
    pass

def permute_edges(cube: Cube, applied: list[str]):
    pass

def permute_corners(cube: Cube, applied: list[str]):
    pass


_functions = [d_cross, first_layer, second_layer, u_cross, permute_edges, permute_corners]
_functions = {func.__name__: func for func in _functions}


def parser(cube: Cube, applied: list[str], func_name: str) -> list[str]:
    reverse = get_reverse(applied)
    cube.rotate(reverse)

    new_applied = _functions[func_name](cube, applied)
    return new_applied