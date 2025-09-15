from environment import Environment
from agents import A2C, DQN, PPO
from agents import DeepQNet
import json
import torch
import numpy as np
import yaml


def load_experience(agent: DQN, env):
    if agent.phase == "cross":
        dataset = json.load(open("data/dataset.json", "r"))
        for _, data in enumerate(list(map(lambda x: x[1], dataset.items()))[:6644]):
            env.cube.reset()
            env.cube.rotate(data["scramble"])

            cross_solution = data["solution"]["cross"]
            for move in cross_solution.split():
                state = env.state2
                obs, reward, _ = env.step(move)
                obs = env.state2

                if done:
                    next_state = None
                else:
                    next_state = obs

                agent.memory.push(
                    agent.state_to_tensor(state),
                    torch.tensor([np.where(env.action_space == move)[0][0]], device=agent.device).unsqueeze(0),
                    agent.state_to_tensor(next_state) if next_state is not None else None,
                    torch.tensor([reward], device=agent.device).float()
                )

    if agent.phase == "f2l":
        # load trained model
        net = DeepQNet(5, 128, env.action_space.shape[0])
        net.load_state_dict(torch.load("models/DQN/dqn_policy_net(f2l).pth"))
        net.eval()

        dataset = json.load(open("data/dataset.json", "r"))
        for _, data in enumerate(list(map(lambda x: x[1], dataset.items()))[:6644]):
            env.cube.reset()
            env.cube.rotate(data["scramble"])
            env.cube.rotate(data["solution"]["cross"])
            
            done = False
            while not done:
                with torch.no_grad():
                    action = agent.action(env.state2)
                move = agent.action_to_move(action.item())
                state = env.state2
                obs, reward, done = env.step()
                obs = env.state2

                if done:
                    next_state = None
                else:
                    next_state = obs

                agent.memory.push(
                    agent.state_to_tensor(state),
                    torch.tensor(),
                    agent.state_to_tensor(next_state) if next_state is not None else None,
                    torch.tensor([reward], device=agent.device).float()
                )


if __name__ == "__main__":
    args = yaml.safe_load(open("config.yaml", "r"))

    phase = "cross"

    env = Environment(
        phase=phase,
        args=args["environment"]
    )

    agent = DQN(env, phase, args)
    #load_experience(agent, env)
    #agent.train()