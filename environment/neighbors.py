import networkx as nx
from magiccube import Cube
from collections import deque
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from .algorithm import init_algo


moves = ["U", "D", "F", "R", "B", "L",
         "U'", "D'", "F'", "R'", "B'", "L'"]


def convert_to_color(kociemba_state: str) -> str:
    state = list(kociemba_state)
    new_state = []

    kociemba_to_color = {
        "U": "W",
        "D": "Y",
        "F": "G",
        "R": "R",
        "B": "B",
        "L": "O"
    }

    for x in state:
        new_state.append(kociemba_to_color[x])

    new_state = "".join(new_state)
    faces = [new_state[i:i+9] for i in range(0, 54, 9)]

    return f"{faces[0]}{faces[4]}{faces[2]}{faces[1]}{faces[5]}{faces[3]}"


def _neighbors(
    cube: Cube
) -> nx.DiGraph:
    global moves
    G = nx.DiGraph()

    algo = init_algo("LBL")

    for move in moves:
        current_k_state = cube.get_kociemba_facelet_positions()
        current_reward = algo.status(cube)
        cube.rotate(move)
        new_k_state = cube.get_kociemba_facelet_positions()
        new_reward = algo.status(cube)
        inverse_move = move[:1] if "'" in move else f"{move}'"

        G.add_node(current_k_state, reward=current_reward)
        G.add_node(new_k_state, reward=new_reward)
        G.add_edge(current_k_state, new_k_state, m=move)
        #G.add_edge(new_k_state, current_k_state, m=inverse_move)
        
        cube.rotate(inverse_move)

    return G

def generate_neighbors(kociemba_state: str, depth: int):
    G = nx.DiGraph()
    cube = Cube(state=convert_to_color(kociemba_state))

    start_k_state = cube.get_kociemba_facelet_positions()

    queue = deque()
    queue.append((start_k_state, 0))

    while queue:
        current_k_state, d = queue.popleft()

        if d >= depth:
            continue

        neighbors = _neighbors(cube = Cube(state=convert_to_color(current_k_state)))

        for node in neighbors.nodes():
            queue.append((node, d + 1))
        G = nx.compose(G, neighbors)

    return G

c = Cube()
G = generate_neighbors(c.get_kociemba_facelet_positions(), depth=3)
#data = from_networkx(G)

print(len(list(G.nodes(data=True))))