from .A2C import A2C
from .DQN import DQN
from .agent import Agent


def init_agent(agent_type: str, args: dict) -> Agent:
    if agent_type == "A2C":
        return A2C(args)
    elif agent_type == "DQN":
        return DQN(args)
    else:
        raise ValueError(f"No {agent_type} agent type exists!!!")


__all__ = [
    "init_agent",
    "Agent"
]