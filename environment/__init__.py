from .env_base import EnvBase
from .envs import *
import numpy as np


class Environment:
    def __init__(
        self,
        phase: str,
        args = dict
    ) -> None:
        if phase not in ["cross", "f2l", "oll", "pll"]:
            raise ValueError(f"Unrecognized {phase} phase!!!")

        MODEL_CLASS = EnvBase.get_model_class(phase)
        self._env: EnvBase = MODEL_CLASS(args)

    def scramble(self) -> None:
        self._env.scramble()

    def reset(self) -> np.ndarray[str]:
        self._env.reset()

    def step(self, action: str) -> tuple[np.ndarray, float, bool]:
        return self._env.step(action)
    
    def __getattr__(self, name: str):
        return getattr(self._env, name)


__all__ = [
    "Environment"
]