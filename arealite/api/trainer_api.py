import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import TrainerConfig, TrainingArgs

if TYPE_CHECKING:
    from arealite.impl.rollout_controller import RolloutController


class Trainer(abc.ABC):
    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional["RolloutController"] = None,
    ):
        self.args = args
        self.trainer_config = trainer_config

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.rollout_controller = rollout_controller

    def create_train_dataloader(self):
        cfg = self.args.train_dataset
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers,
            drop_last=True,
        )

    def create_valid_dataloader(self):
        cfg = self.args.valid_dataset
        self.valid_dataloader = StatefulDataLoader(
            dataset=self.valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers,
            drop_last=True,
        )

    # TODO: check HF trainer signature
    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        raise NotImplementedError()

    def save_checkpoint(self):
        raise NotImplementedError()


@dataclass
class TrainerFactory:
    args: TrainingArgs

    def make_trainer(
        self,
        config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional["RolloutController"] = None,
    ) -> Trainer:
        if config.type == "ppo-rlvr":
            from arealite.impl.trainer.ppo import SpmdRlvrPPOTrainer

            return SpmdRlvrPPOTrainer(
                self.args,
                config,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                rollout_controller=rollout_controller,
            )
        else:
            raise NotImplementedError(f"Unknown trainer type: {config.type}")
