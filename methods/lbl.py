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

    def second_layer(self) -> int:
        faces = self.cube_faces()

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