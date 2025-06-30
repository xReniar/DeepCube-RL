import plotly.graph_objs as go
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from cube_graph import CubeGraph
from magiccube import Cube


obj = Cube()
positions = obj.get_kociemba_facelet_positions()
faces = []
for i in range(0, 6):
    faces.append(positions[9*i: 9 + 9*i])

cGraph = CubeGraph(faces)

data = Data(x=cGraph.get_nodes(), edge_index=cGraph.get_edges())

G = to_networkx(data, to_undirected=True)

pos = nx.spring_layout(G, dim=3, seed=42)
xyz = np.array([pos[i] for i in range(len(pos))])
x_nodes, y_nodes, z_nodes = xyz[:,0], xyz[:,1], xyz[:,2]

edge_x = []
edge_y = []
edge_z = []

for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]


edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='gray', width=2),
    hoverinfo='none'
)

node_trace = go.Scatter3d(
    x=x_nodes, y=y_nodes, z=z_nodes,
    mode='markers+text',
    marker=dict(symbol='circle', size=8, color='skyblue'),
    text=[str(i) for i in range(len(x_nodes))],
    textposition="top center",
    hoverinfo='text'
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Grafo 3D',
                    showlegend=False,
                    margin=dict(l=0, r=0, b=0, t=40),
                    scene=dict(
                        xaxis=dict(showbackground=False),
                        yaxis=dict(showbackground=False),
                        zaxis=dict(showbackground=False)
                    )
                ))

fig.show()