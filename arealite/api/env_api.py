import abc
from dataclasses import dataclass
from typing import Any

from gymnasium import Env

from arealite.api.cli_args import EnvConfig, TrainingArgs


# Re-export the gymnasium environment class
class Environment(abc.ABC, Env):
    def __init__(self, args: TrainingArgs, config: Any):
        self.args = args
        self.config = config


@dataclass
class EnvFactory:
    args: TrainingArgs

    def make_env(self, config: EnvConfig) -> Environment:
        if config.type == "math-code-single-step":
            from arealite.impl.environment.math_code_single_step_env import (
                MathCodeSingleStepEnv,
            )

            return MathCodeSingleStepEnv(self.args, config.math_code_single_step)
        else:
            raise NotImplementedError(f"Unknown env type: {config.type}")
