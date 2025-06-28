from .method import Method
from magiccube import cube


class LBL(Method):
    def __init__(
        self,
        cube: cube.Cube
    ) -> None:
        self.cube = cube

    def bottom_cross(self) -> int:
        '''
        Checks if bottom cross pieces are inserted correctly
        '''
        faces = self.cube_faces()
        
        piece_1 = (faces["front"][7] == faces["front"][4]) and (faces["bottom"][1] == faces["bottom"][4])
        piece_2 = (faces["right"][7] == faces["right"][4]) and (faces["bottom"][5] == faces["bottom"][4])
        piece_3 = (faces["back"][7] == faces["back"][4]) and (faces["bottom"][7] == faces["bottom"][4])
        piece_4 = (faces["left"][7] == faces["left"][4]) and (faces["bottom"][3] == faces["bottom"][4])

        return piece_1 + piece_2 + piece_3 + piece_4

    def first_layer(self) -> int:
        faces = self.cube_faces()

        piece_1 = (faces["front"][8] == faces["front"][4]) and (faces["right"][6] == faces["right"][4])
        piece_2 = (faces["right"][8] == faces["right"][4]) and (faces["back"][6] == faces["back"][4])
        piece_3 = (faces["back"][8] == faces["back"][4]) and (faces["left"][6] == faces["left"][4])
        piece_4 = (faces["left"][8] == faces["left"][4]) and (faces["front"][6] == faces["front"][4])

        return piece_1 + piece_2 + piece_3 + piece_4

    def second_layer(self) -> int:
        faces = self.cube_faces()

        piece_1 = (faces["front"][5] == faces["front"][4]) and (faces["right"][3] == faces["right"][4])
        piece_2 = (faces["right"][5] == faces["right"][4]) and (faces["back"][3] == faces["back"][4])
        piece_3 = (faces["back"][5] == faces["back"][4]) and (faces["left"][3] == faces["left"][4])
        piece_4 = (faces["left"][5] == faces["left"][4]) and (faces["front"][3] == faces["front"][4])

        return piece_1 + piece_2 + piece_3 + piece_4

    def top_cross(self) -> int:
        faces = self.cube_faces()

    def top_edge(self) -> int:
        faces = self.cube_faces()

    def top_corners(self) -> int:
        faces = self.cube_faces()
    
    def cube_status(self) -> int:
        return self.bottom_cross() \
            + self.first_layer() \
            + self.second_layer() \
            + self.top_cross() \
            + self.top_edge() \
            + self.top_corners()