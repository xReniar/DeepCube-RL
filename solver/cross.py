from .env import Environment
from magiccube import Cube, Face, Color


class CrossEnv(Environment):
    def __init__(self):
        self.color = [Color.B, Color.R, Color.G, Color.O]

    def evaluate(self, cube: Cube) -> int:
        value = 0
        
        for i in range(4):
            faces = {
                "F": cube.get_face(Face.F),
                "R": cube.get_face(Face.R),
                "B": cube.get_face(Face.B),
                "L": cube.get_face(Face.L),
                "U": cube.get_face(Face.U),
                "D": cube.get_face(Face.D)
            }

            # front face base cases
            value += (faces["F"][2][1] == self.color[i] and faces["D"][0][1] == Color.W) * 25
            value += ((faces["F"][1][0] == self.color[i] and faces["L"][1][2] == Color.W)
                       or
                      (faces["F"][1][2] == self.color[i] and faces["R"][1][0] == Color.W)) * 21
            value += (faces["F"][0][1] == self.color[i] and faces["U"][2][1] == Color.W) * 19
    
            # front face base cases (1 move away)
            value += ((faces["U"][1][0] == Color.W and faces["L"][0][1] == self.color[i])
                       or
                      (faces["U"][1][2] == Color.W and faces["R"][0][1] == self.color[i])) * 7
            value += ((faces["U"][1][0] == self.color[i] and faces["L"][0][1] == Color.W)
                       or
                      (faces["U"][1][2] == self.color[i] and faces["R"][0][1] == Color.W)) * 7
    
            # front face base cases (2 move away)
            value += (faces["F"][0][1] == Color.W and faces["U"][2][1] == self.color[i]) * 5
            value += ((faces["F"][1][0] == Color.W and faces["L"][1][2] == self.color[i])
                       or
                      (faces["F"][1][2] == Color.W and faces["R"][1][0] == self.color[i])) * 3
    
            # case when piece is in the back
            value += ((faces["B"][1][2] == Color.W and faces["L"][1][0] == self.color[i])
                       or
                      (faces["B"][1][0] == Color.W and faces["R"][1][2] == self.color[i])) * 2
            value += ((faces["B"][1][2] == self.color[i] and faces["L"][1][0] == Color.W)
                       or
                      (faces["B"][1][0] == self.color[i] and faces["R"][1][2] == Color.W)) * 2

            # special cases when inserting
            if(faces["F"][1][0] == self.color[i] and faces["L"][1][2] == Color.W):
                pass
                #value -= (faces["U"][2][1] == Color.W and faces["F"][0][1] != self.color[i - 1])

            if(faces["F"][1][2] == self.color[i] and faces["R"][1][0] == Color.W):
                pass

            # rotate cube
            cube.rotate("y")

        # TODO Extend this with all cases
        # manage special cases when inserting

        return value

    def is_terminated(self, cube: Cube) -> bool:
        return self.evaluate(cube) == 100