from environment import Environment, generate_neighbors
from magiccube import Cube
from torch_geometric.utils import from_networkx


c = Cube()
c.scramble()
G = generate_neighbors(c.get_kociemba_facelet_positions(), depth=2)
#data = from_networkx(G)

data = from_networkx(G)
'''
for node in G.nodes(data=True):
    print(node)
'''