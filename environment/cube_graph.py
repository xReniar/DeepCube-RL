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
            "B": Face(faces[2], 0),
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

    def get_edges(self) -> torch.Tensor:
        source = []
        target = []

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

        for side in ["B", "U", "L", "F", "R", "D"]:
            facelet = self.faces[side].facelet
            index_facelet = self.faces[side].index_facelet

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