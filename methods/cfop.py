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

    def OLL(self) -> int:
        faces = self.cube_faces()

        return int(all(c == "U" for c in faces["top"]))

    def PLL(self) -> int:
        pass
        
    def cube_status(self) -> int:
        return self.cross() + self.F2L() + self.OLL() + self.PLL()