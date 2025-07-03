from environment import Environment
from CFOP_agents import init_agent, Agent


#print(cube.get_kociemba_facelet_positions())
#print(cube.get_all_pieces())
'''
print(cube.get_kociemba_facelet_colors())
print(cube.get_kociemba_facelet_positions())
'''


if __name__ == "__main__":
    env = Environment(method="CFOP")
    env.scramble()
    
    args = dict()
    agent: Agent = init_agent("DQN", args)
    
    obs = env.reset()
    done = False
    while not done:
        action, _ = agent.predict("")
        obs, reward, done = env.step(action)