from magiccube import Cube
from magiccube.cube_base import Face, Color
import networkx as nx
import matplotlib.pyplot as plt


def _connect_face(G: nx.Graph, face_ids):
    for row in range(3):
        for col in range(2):
            G.add_edge(face_ids[row * 3 + col], face_ids[row * 3 + col + 1])
    for col in range(3):
        for row in range(2):
            G.add_edge(face_ids[row * 3 + col], face_ids[(row + 1) * 3 + col])
    for idx in range(0, 3, 2):
        G.add_edge(face_ids[idx], face_ids[4])
        G.add_edge(face_ids[idx + 6], face_ids[4])


def _build_topology_and_id_map(cube):
    G = nx.Graph()
    faces_ids, faces_coords = {}, {}
    next_id = 0
    for face in Face:
        face_indexes = cube._cube_face_indexes[face.value]
        coords_flat = [c for line in face_indexes for c in line]
        ids_flat = list(range(next_id, next_id + 9))
        faces_ids[face.name] = ids_flat
        faces_coords[face.name] = coords_flat
        G.add_nodes_from(ids_flat)
        next_id += 9

    for face_name, ids in faces_ids.items():
        _connect_face(G, ids)

    _horizontal = ['L', 'F', 'R', 'B']
    for i in range(len(_horizontal)):
        lx = faces_ids[_horizontal[i]]
        rx = faces_ids[_horizontal[(i + 1) % 4]]
        for j in range(0, 7, 3):
            G.add_edge(lx[j + 2], rx[j])

    G.add_edge(faces_ids['F'][0], faces_ids['U'][6])
    G.add_edge(faces_ids['F'][1], faces_ids['U'][7])
    G.add_edge(faces_ids['F'][2], faces_ids['U'][8])
    G.add_edge(faces_ids['F'][6], faces_ids['D'][0])
    G.add_edge(faces_ids['F'][7], faces_ids['D'][1])
    G.add_edge(faces_ids['F'][8], faces_ids['D'][2])
    G.add_edge(faces_ids['R'][0], faces_ids['U'][8])
    G.add_edge(faces_ids['R'][1], faces_ids['U'][5])
    G.add_edge(faces_ids['R'][2], faces_ids['U'][2])
    G.add_edge(faces_ids['R'][6], faces_ids['D'][2])
    G.add_edge(faces_ids['R'][7], faces_ids['D'][5])
    G.add_edge(faces_ids['R'][8], faces_ids['D'][8])
    G.add_edge(faces_ids['B'][0], faces_ids['U'][2])
    G.add_edge(faces_ids['B'][1], faces_ids['U'][1])
    G.add_edge(faces_ids['B'][2], faces_ids['U'][0])
    G.add_edge(faces_ids['B'][6], faces_ids['D'][8])
    G.add_edge(faces_ids['B'][7], faces_ids['D'][7])
    G.add_edge(faces_ids['B'][8], faces_ids['D'][6])
    G.add_edge(faces_ids['L'][0], faces_ids['U'][0])
    G.add_edge(faces_ids['L'][1], faces_ids['U'][3])
    G.add_edge(faces_ids['L'][2], faces_ids['U'][6])
    G.add_edge(faces_ids['L'][6], faces_ids['D'][6])
    G.add_edge(faces_ids['L'][7], faces_ids['D'][3])
    G.add_edge(faces_ids['L'][8], faces_ids['D'][0])

    id_to_coord = {}
    for face_name, ids in faces_ids.items():
        for nid, coord in zip(ids, faces_coords[face_name]):
            id_to_coord[nid] = coord

    return G, id_to_coord


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
    G, _ = _build_topology_and_id_map(cube)
    visualize_graph(G)