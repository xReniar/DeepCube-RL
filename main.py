import magiccube
import torch


cube = magiccube.Cube(3)


#print(cube.get_kociemba_facelet_positions())
#print(cube.get_all_pieces())

cube.rotate("R")
print(cube)

print(cube.get_kociemba_facelet_colors())
print(cube.get_kociemba_facelet_positions())