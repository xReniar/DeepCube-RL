from environment import Environment
from CFOP_agents import DQN
import yaml
from itertools import count
import torch


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def convert_tensor_to_move(tensor: torch.Tensor) -> str:
    moves = {
        0: "U",1: "D",2: "F",3: "R",4: "B",5: "L",
        6: "U'",7: "D'",8: "F'",9: "R'",10: "B'",11: "L'",
    }

    return moves[tensor.item()]


if __name__ == "__main__":
    env = Environment(method="CFOP")
    env.scramble()
    
    args = yaml.safe_load(open("config.yaml", "r"))
    agent: DQN = DQN(args["DQN"])

    for i_episode in range(agent.num_episodes):
        state = env.reset()
        
        for t in count():
            action = agent.action(state)
            obs, reward, done = env.step(action)

            reward = torch.tensor([reward], device=device)

            agent.memory.push(state, action, obs, reward)
            state = obs

            agent.optimize_model()
            agent.soft_update()