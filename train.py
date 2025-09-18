from environment import Environment
from agents import A2C, DQN, PPO
from agents import DeepQNet
from magiccube import Cube, CubeMove
import json
import torch
import numpy as np
import yaml
import argparse


def load_experience(agent: DQN, env):
    if agent.phase == "cross":
        dataset = json.load(open("data/dataset.json", "r"))
        for _, data in enumerate(list(map(lambda x: x[1], dataset.items()))[:6644]):
            env.cube.reset()
            env.cube.rotate(data["scramble"])

            cross_solution = data["solution"]["cross"]
            for move in cross_solution.split():
                state = env.state2
                obs, reward, done = env.step(move)
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
        def reverse_move(moveset: str):
            moveset: list = moveset.split()
            moveset.reverse()
            new_moveset = []
            for move in moveset:
                if len(move) == 1:
                    new_moveset.append(f"{move}'")
                else:
                    new_moveset.append(move if move[1].isdigit() else f"{move[0]}")
                    '''if move[1].isdigit():
                        new_moveset.append(move)
                    else:
                        new_moveset.append(f"{move[0]}")'''
            return " ".join(new_moveset)

        action_space: np.ndarray = env.action_space

        dataset = json.load(open("data/dataset.json", "r"))
        for i, data in enumerate(list(map(lambda x: x[1], dataset.items()))):
            if i - 1 not in set([516, 1227, 4361]):
                env.cube.reset()
                env.cube.rotate(data["scramble"])
                env.cube.rotate(data["solution"]["cross"])

                curr_rwrd = env.algorithm.reward()
                solution = []
                while curr_rwrd < 100:
                    reward_list = []
                    state = env.state2
                    for action in action_space:
                        _, reward, _ = env.step(action)
                        reward_list.append((f"{action}", reward))
                        env.cube.rotate(reverse_move(action))
                    reward_list = sorted(reward_list, key=lambda x: x[1], reverse=True)
                    solution.append(reward_list[0][0])

                    obs, reward, done = env.step(reward_list[0][0])
                    obs = env.state2
                    next_state = None if done else obs

                    agent.memory.push(
                        agent.state_to_tensor(state),
                        torch.tensor([np.where(env.action_space == action)[0][0]], device=agent.device).unsqueeze(0),
                        agent.state_to_tensor(next_state) if next_state is not None else None,
                        torch.tensor([reward], device=agent.device).float()
                    )
                    curr_rwrd = reward

                #print(env.cube)
                #print(solution)
            
            if len(agent.memory) > 90000:
                break
            
def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN agent for Rubik\'s Cube')
    parser.add_argument(
        "--phase",
        required=True,
        choices=["cross", "f2l"],
        help="Phase to train"
    )
    parser.add_argument(
        "--load-experience",
        default=None,
        help="Wheter to load experience",
        action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args_cmd = parse_args()
    args = yaml.safe_load(open("config.yaml", "r"))

    phase = args_cmd.phase

    env = Environment(
        phase=phase,
        args=args["environment"]
    )

    agent = DQN(env, phase, args)

    if args_cmd.load_experience:
        load_experience(agent, env)
    agent.train()