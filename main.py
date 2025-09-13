from environment import Environment
from agents import A2C, DQN, PPO
import json
import torch
import yaml


def load_experience(agent: DQN, env):
    if agent.phase != "cross" and isinstance(agent, DQN):
        return
    
    dataset = json.load(open("data/dataset.json", "r"))
    for _, data in enumerate(list(map(lambda x: x[1], dataset.items()))[:6644]):
        env.cube.reset()
        env.cube.rotate(data["scramble"])

        cross_solution = data["solution"]["cross"]
        for move in cross_solution.split():
            state = env.state2
            next_state, reward, done = env.step(move)
            next_state = env.state2

            agent.memory.push(
                agent.state_to_tensor(state),
                torch.tensor(list(map(lambda x: int(x == move), env.action_space))).unsqueeze(0),
                agent.state_to_tensor(next_state) if next_state is not None else None,
                torch.tensor([reward], device=agent.device)
            )

if __name__ == "__main__":
    args = yaml.safe_load(open("config.yaml", "r"))

    phase = "cross"

    env = Environment(
        phase=phase,
        args=args["environment"]
    )

    agent = DQN(env, phase, args)
    print(len(agent.memory))
    #agent.train()