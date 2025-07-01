import magiccube
import torch
from environment import Cube


#print(cube.get_kociemba_facelet_positions())
#print(cube.get_all_pieces())
'''
print(cube.get_kociemba_facelet_colors())
print(cube.get_kociemba_facelet_positions())
'''

env_obj = Cube()
print(env_obj.graph_state())