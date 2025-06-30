import torch
from magiccube import cube


class Face():
    def __init__(
        self,
        face_str: str,
        index: int
    ) -> None:
        self.start_index = index

        self.facelet = face_str
        self.index_facelet = [i for i in range(index, index + 9)]


class CubeGraph():
    def __init__(
        self,
        faces: list[str]
    ) -> None:
        self.faces = {
            "U": Face(faces[3], 9),
            "D": Face(faces[0], 45),
            "F": Face(faces[5], 27),
            "R": Face(faces[1], 36),
            "B": Face(faces[2][::-1], 0),
            "L": Face(faces[4], 18)
        }

    def get_nodes(self) -> torch.Tensor:
        nodes = []

        for side in ["B", "U", "L", "F", "R", "D"]:
            for f in self.faces[side].facelet:
                nodes.append([f])

        return torch.tensor(
            data = nodes,
            dtype = torch.float
        )
    
    def __add_edges(
        self,
        edge: tuple,
        source: list,
        target: list
    ) -> None:
        s, t = edge
        source.append(s)
        target.append(t)
        source.append(t)
        target.append(s)

    def get_edges(self) -> torch.Tensor:
        source = []
        target = []

        # adding face edges
        for side in ["B", "U", "L", "F", "R", "D"]:
            index_facelet = self.faces[side].index_facelet
            grid = [index_facelet[i:i+3] for i in range(0, len(index_facelet), 3)]

            rows = len(grid)
            cols = len(grid[0])
            edges = set()

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            for i in range(rows):
                for j in range(cols):
                    current = grid[i][j]
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbor = grid[ni][nj]
                            edges.add((current, neighbor))

            for (s, t) in list(sorted(edges)):
                source.append(s)
                target.append(t)

        # vertical edges
        for i in range(6, 9):
            self.__add_edges((i, i + 3), source, target)       # B to U
            self.__add_edges((i + 9, i + 21), source, target)  # U to F
            self.__add_edges((i + 27, i + 39), source, target) # F to D

        # horizontal edges
        for i in range(20, 27, 3):
            self.__add_edges((i, i + 7), source, target)      # L to F
            self.__add_edges((i + 9, i + 16), source, target) # F to R

        # lateral edges
        for i in range(0, 3):
            self.__add_edges((i + 36, 17 - (i * 3)), source, target) # U to R
            self.__add_edges((i + 42, 47 + (i * 3)), source, target) # D to R
            self.__add_edges((i + 24, 51 - (i * 3)), source, target) # L to D
            self.__add_edges((i + 18, 9 + (i * 3)), source, target)  # L to U

            # extra case
            self.__add_edges((i, i + 51), source, target) # B to D

        # back edges
        for i in range(0, 7, 3):
            self.__add_edges((i, 24 - i), source, target)     # L to B
            self.__add_edges((i + 2, 44 - i), source, target) # B to R
            self.__add_edges((i, i + 18), source, target) 
            

        return torch.tensor([source, target], dtype=torch.long)

    def graph_state(
        self
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.get_nodes(), self.get_edges())
    

obj = cube.Cube()
positions = obj.get_kociemba_facelet_positions()
faces = []
for i in range(0, 6):
    faces.append(positions[9*i: 9 + 9*i])

cGraph = CubeGraph(faces)
print(cGraph.get_edges())