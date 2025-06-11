import abc
from dataclasses import dataclass

from gymnasium import Env

from refactoring.api.cli_args import EnvConfig, TrainingArgs


# Re-export the gymnasium environment class
class Environment(abc.ABC, Env):
    def __init__(self, args: TrainingArgs, config: EnvConfig):
        self.args = args
        self.config = config


@dataclass
class EnvFactory:
    args: TrainingArgs

    def make_env(self, config: EnvConfig) -> Environment:
        if config.type == "math-code-single-step":
            from xxx import MathCodeSingleStepEnv

            return MathCodeSingleStepEnv(self.args, config)
        else:
            raise NotImplementedError(f"Unknown env type: {config.type}")
