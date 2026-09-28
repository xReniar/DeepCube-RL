from .env import Environment
from magiccube import Cube, Face, Color


class CrossEnv(Environment):
    def __init__(self):
        self.color = [Color.B, Color.R, Color.G, Color.O]

    def evaluate(self, cube: Cube) -> int:
        value = 0
        
        for i in range(4):
            # front face base cases
            value += (cube.get_face(Face.F)[2][1] == self.color[i] and cube.get_face(Face.D)[0][1] == Color.W) * 25
            value += ((cube.get_face(Face.F)[1][0] == self.color[i] and cube.get_face(Face.L)[1][2] == Color.W)
                       or
                      (cube.get_face(Face.F)[1][2] == self.color[i] and cube.get_face(Face.R)[1][0] == Color.W)) * 20
            value += (cube.get_face(Face.F)[0][1] == self.color[i] and cube.get_face(Face.U)[2][1] == Color.W) * 10
    
            # front face base cases (1 move away)
            value += ((cube.get_face(Face.U)[1][0] == Color.W and cube.get_face(Face.L)[0][1] == self.color[i])
                       or
                      (cube.get_face(Face.U)[1][2] == Color.W and cube.get_face(Face.R)[0][1] == self.color[i])) * 7
            value += ((cube.get_face(Face.U)[1][0] == self.color[i] and cube.get_face(Face.L)[0][1] == Color.W)
                       or
                      (cube.get_face(Face.U)[1][2] == self.color[i] and cube.get_face(Face.R)[0][1] == Color.W)) * 7
    
            # front face base cases (2 move away)
            value += (cube.get_face(Face.F)[0][1] == Color.W and cube.get_face(Face.U)[2][1] == self.color[i]) * 5
            value += ((cube.get_face(Face.F)[1][0] == Color.W and cube.get_face(Face.L)[1][2] == self.color[i])
                       or
                      (cube.get_face(Face.F)[1][2] == Color.W and cube.get_face(Face.R)[1][0] == self.color[i])) * 3
    
            # case when piece is in the back
            value += ((cube.get_face(Face.B)[1][2] == Color.W and cube.get_face(Face.L)[1][0] == self.color[i])
                       or
                      (cube.get_face(Face.B)[1][0] == Color.W and cube.get_face(Face.R)[1][2] == self.color[i])) * 2
            value += ((cube.get_face(Face.B)[1][2] == self.color[i] and cube.get_face(Face.L)[1][0] == Color.W)
                       or
                      (cube.get_face(Face.B)[1][0] == self.color[i] and cube.get_face(Face.R)[1][2] == Color.W)) * 2

            # rotate cube
            cube.rotate("y")

        # TODO Extend this with all cases
        # manage special cases when inserting

        return value

    def is_terminated(self, cube: Cube) -> bool:
        return self.evaluate(cube) == 100