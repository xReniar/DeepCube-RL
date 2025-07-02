from .A2C import *
from .DQN import *


def create_agent(agent_type: str):
    if agent_type == "A2C":
        pass
    elif agent_type == "DQN":
        pass
    else:
        raise ValueError(f"No {agent_type} agent type exists!!!")


__all__ = [
    "create_agent"
]