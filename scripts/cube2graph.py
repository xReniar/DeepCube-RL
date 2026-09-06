from magiccube import Cube
import networkx as nx
import matplotlib.pyplot as plt

COLORS = {
    'W': 1,
    'Y': -1,
    'O': -3,
    'G': 2,
    'R': 3,
    'B': -2
}

def _connect_face(G: nx.Graph, face: list[str, int]):
    for color, node_id in face:
        G.add_node(node_id, x=[color])

    # horizontal connections
    for row in range(3):
        for col in range(2):
            idx1 = row * 3 + col
            idx2 = row * 3 + col + 1
            G.add_edge(face[idx1][1], face[idx2][1])

    # vertical connections
    for col in range(3):
        for row in range(2):
            idx1 = row * 3 + col
            idx2 = (row + 1) * 3 + col
            G.add_edge(face[idx1][1], face[idx2][1])

    for idx in range(0, 3, 2):
        G.add_edge(face[idx][1], face[4][1])
        G.add_edge(face[idx + 6][1], face[4][1])

def cube2graph(cube: Cube) -> nx.Graph:
    G = nx.Graph()

    _faces = {}
    _id = 0
    # get all the faces
    for face, colors in cube.get_all_faces().items():
        face_with_ids = []
        for sublist in colors:
            for item in sublist:
                face_with_ids.append((COLORS[item.name], _id))
                _id += 1
        _faces[face.name] = face_with_ids

    # create connections for each face
    for _, face in _faces.items():
        _connect_face(G, face)

    # separate vertical and horizontal faces
    _horizontal = [_faces['L'], _faces['F'], _faces['R'], _faces['B']]

    # connect _horizontal faces
    for i in range(len(_horizontal)):
        lx = _horizontal[i]
        rx = _horizontal[(i + 1) % (len(_horizontal))]

        for i in range(0, 7, 3):
            G.add_edge(lx[i + 2][1], rx[i][1])

    # connect _vertical faces with _horizontal faces
    G.add_edge(_faces['F'][0][1], _faces['U'][6][1])
    G.add_edge(_faces['F'][1][1], _faces['U'][7][1])
    G.add_edge(_faces['F'][2][1], _faces['U'][8][1])
    G.add_edge(_faces['F'][6][1], _faces['D'][0][1])
    G.add_edge(_faces['F'][7][1], _faces['D'][1][1])
    G.add_edge(_faces['F'][8][1], _faces['D'][2][1])

    G.add_edge(_faces['R'][0][1], _faces['U'][8][1])
    G.add_edge(_faces['R'][1][1], _faces['U'][5][1])
    G.add_edge(_faces['R'][2][1], _faces['U'][2][1])
    G.add_edge(_faces['R'][6][1], _faces['D'][2][1])
    G.add_edge(_faces['R'][7][1], _faces['D'][5][1])
    G.add_edge(_faces['R'][8][1], _faces['D'][8][1])

    G.add_edge(_faces['B'][0][1], _faces['U'][2][1])
    G.add_edge(_faces['B'][1][1], _faces['U'][1][1])
    G.add_edge(_faces['B'][2][1], _faces['U'][0][1])
    G.add_edge(_faces['B'][6][1], _faces['D'][8][1])
    G.add_edge(_faces['B'][7][1], _faces['D'][7][1])
    G.add_edge(_faces['B'][8][1], _faces['D'][6][1])

    G.add_edge(_faces['L'][0][1], _faces['U'][0][1])
    G.add_edge(_faces['L'][1][1], _faces['U'][3][1])
    G.add_edge(_faces['L'][2][1], _faces['U'][6][1])
    G.add_edge(_faces['L'][6][1], _faces['D'][6][1])
    G.add_edge(_faces['L'][7][1], _faces['D'][3][1])
    G.add_edge(_faces['L'][8][1], _faces['D'][0][1])

    return G

def visualize_graph(G: nx.Graph):
    pos = nx.spring_layout(G, dim=3, k=0.3, iterations=100)

    x_vals = [pos[node][0] for node in G.nodes()]
    y_vals = [pos[node][1] for node in G.nodes()]
    z_vals = [pos[node][2] for node in G.nodes()]

    node_colors = []
    for node in G.nodes():
        feature = G.nodes[node]['x']
        if feature == [1]:
            node_colors.append('white')
        elif feature == [-1]:
            node_colors.append('yellow')
        elif feature == [-3]:
            node_colors.append('orange')
        elif feature == [2]:
            node_colors.append('green')
        elif feature == [3]:
            node_colors.append('red')
        elif feature == [-2]:
            node_colors.append('blue')

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color='gray', alpha=0.3, linewidth=1)

    scatter = ax.scatter(x_vals, y_vals, z_vals, 
                        c=node_colors, 
                        s=50, 
                        edgecolors='black', 
                        linewidth=0.5,
                        alpha=0.8)

    ax.set_title("3D Rubik's Cube", fontsize=16, pad=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True, alpha=0.3)

    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cube = Cube()
    G = cube2graph(cube)
    visualize_graph(G)