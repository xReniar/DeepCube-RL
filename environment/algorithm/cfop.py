from .algorithm import Algorithm
from magiccube import Cube


class CFOP(Algorithm):
    def __init__(self) -> None:
        super().__init__()

    def cross(self, cube: Cube) -> int:
        '''
        Checks if bottom cross pieces are inserted correctly
        '''
        faces = self.cube_faces(cube)
        
        piece_1 = (faces["front"][7] == faces["front"][4]) and (faces["bottom"][1] == faces["bottom"][4])
        piece_4 = (faces["left"][7] == faces["left"][4]) and (faces["bottom"][3] == faces["bottom"][4])
        piece_2 = (faces["right"][7] == faces["right"][4]) and (faces["bottom"][5] == faces["bottom"][4])
        piece_3 = (faces["back"][7] == faces["back"][4]) and (faces["bottom"][7] == faces["bottom"][4])

        return piece_1 + piece_2 + piece_3 + piece_4


    def F2L(self, cube: Cube) -> int:
        faces = self.cube_faces(cube)

        sides = ["front", "right", "back", "left"]
        status = 0
        for (l_side, r_side) in list(zip(sides, sides[1:] + sides[:1])):
            corner = (faces[l_side][8] == faces[l_side][4]) and (faces[r_side][6] == faces[r_side][4])
            edge = (faces[l_side][5] == faces[l_side][4]) and (faces[r_side][3] == faces[r_side][4])

            status += int(all([corner, edge]))

        return status


    def OLL(self, cube: Cube) -> int:
        '''
        Checks if OLL step is done
        '''
        faces = self.cube_faces(cube)

        return int(all(c == "U" for c in faces["top"]))

    def PLL(self, cube: Cube) -> int:
        '''
        Checks if PLL step is done
        '''
        faces = self.cube_faces(cube)

        front = int(all((c == faces["front"][4]) for c in faces["front"][:3]))
        right = int(all((c == faces["right"][4]) for c in faces["right"][:3]))
        back = int(all((c == faces["back"][4]) for c in faces["back"][:3]))
        left = int(all((c == faces["left"][4]) for c in faces["left"][:3]))

        return int(all([front, right, back, left]))
        
    def status(
        self,
        cube: Cube
    ) -> int:
        '''
        Return the number of completed steps of the cube
        '''
        weights = [10, 10, 10, 10]
        cross_state = self.cross(cube) * weights[0]
        f2l_state = self.F2L(cube) * weights[1]
        oll_state = self.OLL(cube) * weights[2]
        pll_state = self.PLL(cube) * weights[3]

        if cross_state != 4:
            return cross_state
        elif f2l_state != 4:
            return cross_state + f2l_state
        elif oll_state == 1:
            return cross_state + f2l_state + oll_state
        else:
            return cross_state + f2l_state + oll_state + pll_state
        
        return cross_state + f2l_state + oll_state + pll_state