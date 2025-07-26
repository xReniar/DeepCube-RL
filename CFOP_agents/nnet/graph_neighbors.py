import networkx as nx
import matplotlib.pyplot as plt
from magiccube import Cube


# Crea un grafo diretto
G = nx.DiGraph()

# Aggiungi nodi e archi
G.add_edge("RU", "UUR", attr="U")
G.add_edge("RA", "RU")

# Disegna il grafo
pos = nx.spring_layout(G)  # Layout per posizionare i nodi
nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True, node_size=2000, font_size=15)

edge_labels = nx.get_edge_attributes(G, 'attr')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Grafo Diretto")
plt.show()
