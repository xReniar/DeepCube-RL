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

if __name__ == "__main__":
    env = Environment(
        method="CFOP",
        size=3,
        device=device
    )
    env.scramble()
    
    args = yaml.safe_load(open("config.yaml", "r"))
    agent: DQN = DQN(args["DQN"])

    for episode in range(agent.num_episodes):
        state = env.reset()

        current_reward = env.algorithm.status(env.cube)

        for t in count():
            action = agent.action(state)
            obs, reward, done = env.step(action.item())

            current_reward = reward

            torch_current_reward = torch.tensor([current_reward], device=device)

            if done:
                next_state = None
            else:
                next_state = obs

            agent.memory.push(state, action, next_state, torch_current_reward)
            state = next_state

            agent.optimize()

            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*agent.tau + target_net_state_dict[key]*(1-agent.tau)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                with open(f"episode{episode}-{t}.txt", "w") as f:
                    f.write(env.cube.__str__())
                break