import networkx as nx
import matplotlib.pyplot as plt
from magiccube import Cube
import hashlib
import itertools as it


def hash5(s: str) -> int:
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h, 16) % 100000

def draw_labeled_multigraph(G, attr_name, ax=None):
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]

    pos = nx.shell_layout(G)

    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=14, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
    )

    labels = {
        tuple(edge): f"{attr_name}={attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )


moves = ["U", "D", "F", "R", "B", "L",
         "U'", "D'", "F'", "R'", "B'", "L'"]

childrens = []

G = nx.MultiDiGraph()
cube = Cube()

visited = set()
fringe = [cube.get_kociemba_facelet_colors()]

while len(visited) < 5:
    current = fringe.pop(0)

    cube = Cube(state=current)
    for move in moves:
        current_k_state = cube.get_kociemba_facelet_positions()
        current_c_state = cube.get_kociemba_facelet_colors()
        cube.rotate(move)

        new_k_state = cube.get_kociemba_facelet_positions()
        new_c_state = cube.get_kociemba_facelet_colors()

        visited.add(new_c_state)
        if new_c_state not in visited:
            fringe.append(new_c_state)

        inverse_move = None
        if "'" in move:
            inverse_move = move[:1]
        else:
            inverse_move = f"{move}'"

        G.add_edge(hash5(current_k_state), hash5(new_k_state), m=move)
        G.add_edge(hash5(new_k_state), hash5(current_k_state), m=inverse_move)
        
        cube.rotate(inverse_move)
    

fig, ax = plt.subplots(figsize=(6, 5))
draw_labeled_multigraph(G, "m", ax)
ax.set_title("Graph")
plt.tight_layout()
plt.show()