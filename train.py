from environment import Environment
from agents import A2C, DQN, PPO
import yaml
from itertools import count
import torch


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.mps.is_available() else
    "cpu"
)

if __name__ == "__main__":
    args = yaml.safe_load(open("config.yaml", "r"))

    env = Environment(
        method="LBL",
        size=3,
        device=device,
        args=args["environment"]
    )

    agent = DQN(env, args["DQN"])
    agent.train()