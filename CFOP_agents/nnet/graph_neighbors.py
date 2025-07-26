import networkx as nx
import matplotlib.pyplot as plt
from magiccube import Cube
import hashlib
import itertools as it
from collections import deque


def hash5(s: str) -> int:
    digits = 5
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h, 16) % (pow(10, digits))

def draw_labeled_multigraph_tree(G, attr_name="m", ax=None, show_labels=False):
    if ax is None:
        ax = plt.gca()

    root = next(iter(G.nodes))

    levels = {}
    queue = deque([(root, 0)])
    visited = set()

    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        levels[node] = depth
        for neighbor in G.successors(node):
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))

    pos = {}
    nodes_by_level = {}
    for node, level in levels.items():
        nodes_by_level.setdefault(level, []).append(node)

    for level, nodes in nodes_by_level.items():
        for i, node in enumerate(nodes):
            pos[node] = (i - len(nodes) / 2, -level)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="grey",
        arrows=True,
        arrowstyle="-|>",
        ax=ax,
    )

    if show_labels:
        labels = {
            tuple(edge): f"{attr_name}={attrs[attr_name]}"
            for *edge, attrs in G.edges(keys=True, data=True)
        }

        nx.draw_networkx_edge_labels(
            G,
            pos,
            labels,
            label_pos=0.3,
            font_color="blue",
            bbox={"alpha": 0},
            ax=ax,
        )

def generate_state_graph(cube: Cube, moves: list, depth: int):
    G = nx.MultiDiGraph()
    visited = set()

    start_k_state = cube.get_kociemba_facelet_colors()
    start_hash = hash5(start_k_state)
    
    queue = deque()
    queue.append((start_hash, start_k_state, 0))

    visited.add(start_hash)

    while queue:
        current_hash, current_k_state, d = queue.popleft()

        if d >= depth:
            continue

        for move in moves:
            cube = Cube(state=current_k_state)
            cube.rotate(move)
            new_k_state = cube.get_kociemba_facelet_colors()
            new_hash = hash5(new_k_state)

            G.add_edge(current_hash, new_hash, m=move)

            inverse_move = move[:-1] if "'" in move else move + "'"
            G.add_edge(new_hash, current_hash, m=inverse_move)

            if new_hash not in visited:
                visited.add(new_hash)
                queue.append((new_hash, new_k_state, d + 1))

    return G

moves = ["U", "D", "F", "R", "B", "L",
         "U'", "D'", "F'", "R'", "B'", "L'"]

G = generate_state_graph(cube=Cube(), moves=moves, depth=2)

print(len(G.nodes))
print(len(G.edges))

'''
fig, ax = plt.subplots(figsize=(6, 5))
draw_labeled_multigraph_tree(G, "m", ax, show_labels=True)
ax.set_title("Graph")
plt.tight_layout()
plt.show()
'''