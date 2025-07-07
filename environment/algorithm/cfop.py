from .algorithm import Algorithm
from magiccube import Cube


class CFOP(Algorithm):
    def __init__(self) -> None:
        super().__init__()

    def cross(self, cube: Cube) -> float:
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
                cross_status += 10.0
            else:
                if face == "front":
                    cross_status += int((faces[face][1] == faces[face][4]) and (faces["top"][7] == faces["bottom"][4])) * 5.0 + \
                                    int((faces[face][7] == faces[face][4]) and (faces["bottom"][1] == faces["bottom"][4])) * 25.0
                elif face == "right":
                    cross_status += int((faces[face][1] == faces[face][4]) and (faces["top"][5] == faces["bottom"][4])) * 5.0 + \
                                    int((faces[face][7] == faces[face][4]) and (faces["bottom"][5] == faces["bottom"][4])) * 25.0
                elif face == "back":
                    cross_status += int((faces[face][1] == faces[face][4]) and (faces["top"][1] == faces["bottom"][4])) * 5.0 + \
                                    int((faces[face][7] == faces[face][4]) and (faces["bottom"][7] == faces["bottom"][4])) * 25.0
                else:
                    cross_status += int((faces[face][1] == faces[face][4]) and (faces["top"][3] == faces["bottom"][4])) * 5.0 + \
                                    int((faces[face][7] == faces[face][4]) and (faces["bottom"][3] == faces["bottom"][4])) * 25.0

        return cross_status


    def F2L(self, cube: Cube) -> float:
        faces = self.cube_faces(cube)

        sides = ["front", "right", "back", "left"]
        status = 0
        for (l_side, r_side) in list(zip(sides, sides[1:] + sides[:1])):
            corner = (faces[l_side][8] == faces[l_side][4]) and (faces[r_side][6] == faces[r_side][4])
            edge = (faces[l_side][5] == faces[l_side][4]) and (faces[r_side][3] == faces[r_side][4])

            status += float(all([corner, edge]))

        return status


    def OLL(self, cube: Cube) -> float:
        '''
        Checks if OLL step is done
        '''
        faces = self.cube_faces(cube)

        return float(all(c == "U" for c in faces["top"]))

    def PLL(self, cube: Cube) -> float:
        '''
        Checks if PLL step is done
        '''
        faces = self.cube_faces(cube)

        front = int(all((c == faces["front"][4]) for c in faces["front"][:3]))
        right = int(all((c == faces["right"][4]) for c in faces["right"][:3]))
        back = int(all((c == faces["back"][4]) for c in faces["back"][:3]))
        left = int(all((c == faces["left"][4]) for c in faces["left"][:3]))

        return float(all([front, right, back, left]))
        
    def status(
        self,
        cube: Cube
    ) -> float:
        '''
        Return the number of completed steps of the cube
        '''
        weights = [10, 10, 10, 10]
        cross_state = self.cross(cube) * weights[0]
        f2l_state = self.F2L(cube) * weights[0]
        oll_state = self.OLL(cube) * weights[0]
        pll_state = self.PLL(cube) * weights[0]

        if cross_state < 4:
            return cross_state
        elif f2l_state < 4:
            return cross_state + f2l_state
        elif oll_state == 1:
            return cross_state + f2l_state + oll_state
        else:
            return cross_state + f2l_state + oll_state + pll_state
        
        return cross_state + f2l_state + oll_state + pll_state


'''
algo = CFOP()
cube = Cube()
cube.rotate("R F")

print(algo.cross(cube))
'''
