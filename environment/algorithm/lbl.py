from .algorithm import Algorithm
from magiccube import Cube


class LBL(Algorithm):
    def __init__(self) -> None:
        super().__init__()

    def bottom_cross(self, cube: Cube) -> int:
        '''
        Checks if bottom cross pieces are inserted correctly
        '''
        faces = self.cube_faces(cube)

        sides = ["front", "right", "back", "left"]
        adjacency = {}
        for i, direction in enumerate(sides):
            left_adj = sides[(i - 1) % len(sides)]
            right_adj = sides[(i + 1) % len(sides)]
            adjacency[direction] = [left_adj, right_adj]

        cross_status = 0.0
        for face in adjacency.keys():
            if ((faces[face][3] == faces[face][4]) and (faces[adjacency[face][0]][5] == faces["bottom"][4])) or \
               ((faces[face][5] == faces[face][4]) and (faces[adjacency[face][1]][3] == faces["bottom"][4])):
                cross_status += 2
            else:
                if face == "front":
                    cross_status += int((faces[face][1] == faces[face][4]) and (faces["top"][7] == faces["bottom"][4])) * 1 + \
                                    int((faces[face][7] == faces[face][4]) and (faces["bottom"][1] == faces["bottom"][4])) * 3
                elif face == "right":
                    cross_status += int((faces[face][1] == faces[face][4]) and (faces["top"][5] == faces["bottom"][4])) * 1 + \
                                    int((faces[face][7] == faces[face][4]) and (faces["bottom"][5] == faces["bottom"][4])) * 3
                elif face == "back":
                    cross_status += int((faces[face][1] == faces[face][4]) and (faces["top"][1] == faces["bottom"][4])) * 1 + \
                                    int((faces[face][7] == faces[face][4]) and (faces["bottom"][7] == faces["bottom"][4])) * 3
                else:
                    cross_status += int((faces[face][1] == faces[face][4]) and (faces["top"][3] == faces["bottom"][4])) * 1 + \
                                    int((faces[face][7] == faces[face][4]) and (faces["bottom"][3] == faces["bottom"][4])) * 3
                    
        if cross_status == 12:
            cross_status = 50
        if cross_status == 0:
            cross_status = -1
        
        return cross_status

    def first_layer(self, cube: Cube) -> int:
        faces = self.cube_faces(cube)

        piece_1 = (faces["front"][8] == faces["front"][4]) and (faces["right"][6] == faces["right"][4])
        piece_2 = (faces["right"][8] == faces["right"][4]) and (faces["back"][6] == faces["back"][4])
        piece_3 = (faces["back"][8] == faces["back"][4]) and (faces["left"][6] == faces["left"][4])
        piece_4 = (faces["left"][8] == faces["left"][4]) and (faces["front"][6] == faces["front"][4])

        sum_reward = piece_1 + piece_2 + piece_3 + piece_4
        if sum_reward == 4:
            return sum_reward * 10
        else:
            return sum_reward

    def second_layer(self, cube: Cube) -> int:
        faces = self.cube_faces(cube)

        piece_1 = (faces["front"][5] == faces["front"][4]) and (faces["right"][3] == faces["right"][4])
        piece_2 = (faces["right"][5] == faces["right"][4]) and (faces["back"][3] == faces["back"][4])
        piece_3 = (faces["back"][5] == faces["back"][4]) and (faces["left"][3] == faces["left"][4])
        piece_4 = (faces["left"][5] == faces["left"][4]) and (faces["front"][3] == faces["front"][4])

        sum_reward = piece_1 + piece_2 + piece_3 + piece_4
        if sum_reward == 4:
            return sum_reward * 10
        else:
            return sum_reward

    def top_cross(self, cube: Cube) -> int:
        faces = self.cube_faces(cube)

    def top_edge(self, cube: Cube) -> int:
        faces = self.cube_faces(cube)

    def top_corners(self, cube: Cube) -> int:
        faces = self.cube_faces(cube)
    
    def status(self, cube: Cube) -> int:
        return self.bottom_cross(cube)