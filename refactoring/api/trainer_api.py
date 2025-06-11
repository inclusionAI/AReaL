import abc
from dataclasses import dataclass
from typing import Any, Optional, Union

from refactoring.api.cli_args import TrainerConfig, TrainingArgs


class Trainer(abc.ABC):
    def __init__(self, args: TrainingArgs, config: TrainerConfig):
        self.args = args
        self.config = config

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        raise NotImplementedError()


@dataclass
class TrainerFactory:
    args: TrainingArgs

    def make_trainer(self, config: TrainerConfig) -> Trainer:
        if config.type == "ppo":
            from xxx import PPOTrainer

            return PPOTrainer(self.args, config.ppo)
        else:
            raise NotImplementedError(f"Unknown trainer type: {config.type}")
