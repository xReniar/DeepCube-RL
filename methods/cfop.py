from .method import Method
from magiccube import cube


class CFOP(Method):
    def __init__(
        self,
        cube: cube.Cube
    ) -> None:
        super().__init__(cube)
        self.cube = cube

    def cross(self) -> int:
        '''
        Checks if bottom cross pieces are inserted correctly
        '''
        faces = self.cube_faces()
        
        piece_1 = (faces["front"][7] == faces["front"][4]) and (faces["bottom"][1] == faces["bottom"][4])
        piece_4 = (faces["left"][7] == faces["left"][4]) and (faces["bottom"][3] == faces["bottom"][4])
        piece_2 = (faces["right"][7] == faces["right"][4]) and (faces["bottom"][5] == faces["bottom"][4])
        piece_3 = (faces["back"][7] == faces["back"][4]) and (faces["bottom"][7] == faces["bottom"][4])

        return piece_1 + piece_2 + piece_3 + piece_4


    def F2L(self) -> int:
        faces = self.cube_faces()

        sides = ["front", "right", "back", "left"]
        status = 0
        for (l_side, r_side) in list(zip(sides, sides[1:] + sides[:1])):
            corner = (faces[l_side][8] == faces[l_side][4]) and (faces[r_side][6] == faces[r_side][4])
            edge = (faces[l_side][5] == faces[l_side][4]) and (faces[r_side][3] == faces[r_side][4])

            status += int(all([corner, edge]))


    def OLL(self) -> int:
        '''
        Checks if OLL step is done
        '''
        faces = self.cube_faces()

        return int(all(c == "U" for c in faces["top"]))

    def PLL(self) -> int:
        '''
        Checks if PLL step is done
        '''
        faces = self.cube_faces()

        front = int(all(c == faces["front"][4]) for c in faces["front"][:3])
        right = int(all(c == faces["right"][4]) for c in faces["right"][:3])
        back = int(all(c == faces["back"][4]) for c in faces["back"][:3])
        left = int(all(c == faces["left"][4]) for c in faces["left"][:3])

        return int(all([front, right, back, left]))
        
    def cube_status(self) -> int:
        return self.cross() + self.F2L() + self.OLL() + self.PLL()