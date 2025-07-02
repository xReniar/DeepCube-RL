from environment import Environment
from CFOP_agents import create_agent


#print(cube.get_kociemba_facelet_positions())
#print(cube.get_all_pieces())
'''
print(cube.get_kociemba_facelet_colors())
print(cube.get_kociemba_facelet_positions())
'''

env = Environment()
agent = create_agent("DQN")