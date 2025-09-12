from environment import Environment
from agents import A2C, DQN, PPO
import yaml


if __name__ == "__main__":
    args = yaml.safe_load(open("config.yaml", "r"))

    phase = "cross"

    env = Environment(
        phase=phase,
        args=args["environment"]
    )

    agent = PPO(env, phase, args)
    agent.train()