from magiccube import Cube
import json
import time


def mov(moveset: list, c: int):
    moves = ["F", "R", "B", "L"]
    copy = moves
    for _ in range(c):
        first = copy[0]
        other = copy[1:]
        other.append(first)

        copy = other

    new_moveset = []
    for m in moveset:
        if m == "F":
            new_moveset.append(copy[0])
        elif m == "F'":
            new_moveset.append(f"{copy[0]}'")
        elif m == "R":
            new_moveset.append(copy[1])
        elif m == "R'":
            new_moveset.append(f"{copy[1]}'")
        elif m == "B":
            new_moveset.append(copy[2])
        elif m == "B'":
            new_moveset.append(f"{copy[2]}'")
        elif m == "L":
            new_moveset.append(copy[3])
        elif m == "L'":
            new_moveset.append(f"{copy[3]}'")
        else:
            new_moveset.append(m)

    return new_moveset


cube = Cube()

moves = "F' U F U R U' R'"
for i in range(1, 4):
    print(" ".join(mov(moves.split(), i)))