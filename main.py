from environment import Environment, Phase
from agents import A2C, DQN, PPO
import yaml


if __name__ == "__main__":
    args = yaml.safe_load(open("config.yaml", "r"))

    phase = Phase.CROSS

    env = Environment(
        method="LBL",
        phase=phase,
        size=3,
        args=args["environment"]
    )

    agent = DQN(env, args[phase.value])
    agent.train()