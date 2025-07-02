from environment import Environment
from CFOP_agents import create_agent


#print(cube.get_kociemba_facelet_positions())
#print(cube.get_all_pieces())
'''
print(cube.get_kociemba_facelet_colors())
print(cube.get_kociemba_facelet_positions())
'''


if __name__ == "__main__":
    env = Environment()
    env.scramble()
    
    model = create_agent("DQN")
    
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict()
        obs, reward, done = env.step(action)