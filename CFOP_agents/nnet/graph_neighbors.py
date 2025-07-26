import networkx as nx
import matplotlib.pyplot as plt
from magiccube import Cube
import hashlib
import itertools as it


def hash5(s: str) -> int:
    digits = 5
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h, 16) % (pow(10, digits))

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

def draw_labeled_multigraph2(G, attr_name, ax=None):
    if ax is None:
        ax = plt.gca()

    # Identifica il nodo iniziale (primo stato del cubo)
    initial_node = next(iter(G.nodes()))  # Prendi il primo nodo aggiunto

    # Usa spring_layout con un posizionamento fisso per il nodo iniziale
    pos = nx.spring_layout(G, k=0.3, seed=42, pos={initial_node: (0, 0)}, fixed=[initial_node])

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    # Disegna archi con curvature diverse per evitare sovrapposizioni
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * len(G.edges()))]

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="grey",
        connectionstyle=connectionstyle,
        arrows=True,
        arrowstyle="-|>",
        ax=ax,
    )

    # Aggiungi etichette agli archi
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

G = nx.MultiDiGraph()
cube = Cube()

for move in moves:
    current_k_state = cube.get_kociemba_facelet_positions()
    current_c_state = cube.get_kociemba_facelet_colors()
    cube.rotate(move)

    new_k_state = cube.get_kociemba_facelet_positions()
    new_c_state = cube.get_kociemba_facelet_colors()

    inverse_move = None
    if "'" in move:
        inverse_move = move[:1]
    else:
        inverse_move = f"{move}'"

    G.add_edge(hash5(current_k_state), hash5(new_k_state), m=move)
    G.add_edge(hash5(new_k_state), hash5(current_k_state), m=inverse_move)
    
    cube.rotate(inverse_move)

fig, ax = plt.subplots(figsize=(6, 5))
draw_labeled_multigraph2(G, "m", ax)
ax.set_title("Graph")
plt.tight_layout()
plt.show()