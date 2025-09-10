from .lbl import LBL
from .algorithm import Algorithm
from magiccube import Cube


def init_algo(name: str, cube: Cube) -> Algorithm:
    method = None
    if name == "LBL":
        method = LBL(cube)
    else:
        raise ValueError(f"No {name} method implemented!!")

    return method 


__all__ = [
    "init_algo"
    "Algorithm"
]