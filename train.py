from environment import Environment
from agents import A2C, DQN, PPO
import yaml


if __name__ == "__main__":
    args = yaml.safe_load(open("config.yaml", "r"))

    env = Environment(
        method="LBL",
        size=3,
        args=args["environment"]
    )

    agent = DQN(env, args["DQN"])
    agent.train()