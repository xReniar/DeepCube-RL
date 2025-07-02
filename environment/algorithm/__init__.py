from .cfop import CFOP
from .lbl import LBL
from algorithm import Algorithm


def init_algo(name: str) -> Algorithm:
    method = None
    if name == "CFOP":
        method = CFOP()
    elif name == "LBL":
        method = LBL()
    else:
        raise ValueError(f"No {name} method implemented!!")

    return method 


__all__ = [
    "init_algo"
    "Algorithm"
]