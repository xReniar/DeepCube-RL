from .algorithm import Algorithm
from magiccube import Cube


class LBL(Algorithm):
    def __init__(self, cube: Cube) -> None:
        super().__init__(cube)

    def bottom_cross(self) -> int:
        '''
        Checks if bottom cross pieces are inserted correctly
        '''
        faces = self.cube_faces()

        sides = ["front", "right", "back", "left"]
        adjacency = {}
        for i, direction in enumerate(sides):
            left_adj = sides[(i - 1) % len(sides)]
            right_adj = sides[(i + 1) % len(sides)]
            adjacency[direction] = [left_adj, right_adj]

        cross_reward = 0.0
        for face in adjacency.keys():
            # piece on the side
            if ((faces[face][3] == faces[face][4]) and (faces[adjacency[face][0]][5] == faces["bottom"][4])) or \
               ((faces[face][5] == faces[face][4]) and (faces[adjacency[face][1]][3] == faces["bottom"][4])):
                cross_reward += 2

            # first condition: piece on top correctly positioned
            # second condition: piece inserted
            # thid condition: piece bad orientation correctly positioned
            if face == "front":
                cross_reward += int((faces[face][1] == faces[face][4]) and (faces["top"][7] == faces["bottom"][4])) * 1 + \
                                int((faces[face][7] == faces[face][4]) and (faces["bottom"][1] == faces["bottom"][4])) * 3
                cross_reward += int((faces[adjacency[face][0]][1] == faces["bottom"][4]) and (faces["top"][3] == faces[face][4])) + \
                                int((faces[adjacency[face][1]][1] == faces["bottom"][4]) and (faces["top"][5] == faces[face][4]))
            elif face == "right":
                cross_reward += int((faces[face][1] == faces[face][4]) and (faces["top"][5] == faces["bottom"][4])) * 1 + \
                                int((faces[face][7] == faces[face][4]) and (faces["bottom"][5] == faces["bottom"][4])) * 3
                cross_reward += int((faces[adjacency[face][0]][1] == faces["bottom"][4]) and (faces["top"][7] == faces[face][4])) + \
                                int((faces[adjacency[face][1]][1] == faces["bottom"][4]) and (faces["top"][1] == faces[face][4]))
            elif face == "back":
                cross_reward += int((faces[face][1] == faces[face][4]) and (faces["top"][1] == faces["bottom"][4])) * 1 + \
                                int((faces[face][7] == faces[face][4]) and (faces["bottom"][7] == faces["bottom"][4])) * 3
                cross_reward += int((faces[adjacency[face][0]][1] == faces["bottom"][4]) and (faces["top"][5] == faces[face][4])) + \
                                int((faces[adjacency[face][1]][1] == faces["bottom"][4]) and (faces["top"][3] == faces[face][4]))
            else:
                cross_reward += int((faces[face][1] == faces[face][4]) and (faces["top"][3] == faces["bottom"][4])) * 1 + \
                                int((faces[face][7] == faces[face][4]) and (faces["bottom"][3] == faces["bottom"][4])) * 3
                cross_reward += int((faces[adjacency[face][0]][1] == faces["bottom"][4]) and (faces["top"][1] == faces[face][4])) + \
                                int((faces[adjacency[face][1]][1] == faces["bottom"][4]) and (faces["top"][7] == faces[face][4]))
                    
        if cross_reward == 12:
            cross_reward = 100
        if cross_reward == 0:
            cross_reward = -10
        
        return cross_reward

    def f2l(self) -> int:
        faces = self.cube_faces()

        first_layer = [
            int(faces["front"][8] == faces["front"][4]) and (faces["right"][6] == faces["right"][4]),
            int(faces["right"][8] == faces["right"][4]) and (faces["back"][6] == faces["back"][4]),
            int(faces["back"][8] == faces["back"][4]) and (faces["left"][6] == faces["left"][4]),
            int(faces["left"][8] == faces["left"][4]) and (faces["front"][6] == faces["front"][4])
        ]
        second_layer = [
            int(faces["front"][5] == faces["front"][4]) and (faces["right"][3] == faces["right"][4]),
            int(faces["right"][5] == faces["right"][4]) and (faces["back"][3] == faces["back"][4]),
            int(faces["back"][5] == faces["back"][4]) and (faces["left"][3] == faces["left"][4]),
            int(faces["left"][5] == faces["left"][4]) and (faces["front"][3] == faces["front"][4])
        ]
        if sum(first_layer) < 4:
            return sum(first_layer)
        elif sum(first_layer) + sum(second_layer) == 8:
            return 100
        else:
            return sum(first_layer) * 50 + sum(second_layer)

    def top_cross(self) -> int:
        faces = self.cube_faces()

    def top_edge(self) -> int:
        faces = self.cube_faces()

    def top_corners(self) -> int:
        faces = self.cube_faces()
    
    def reward(self, **kwargs) -> int:
        if kwargs["lbl_phase"] == 0:
            return self.bottom_cross()
        if kwargs["lbl_phase"] == 1:
            return self.f2l()