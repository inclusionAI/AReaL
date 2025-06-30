# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import os
import sys

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from arealite.api.cli_args import TrainingArgs, prepare_training_args
from arealite.api.dataset_api import DatasetFactory
from arealite.api.rollout_api import RolloutCollectorFactory
from arealite.api.trainer_api import TrainerFactory
from arealite.system.rollout_controller import RolloutController
from realhf.base import seeding


@record
def main():
    """Main entry point for launching the trainer."""
    cfg: TrainingArgs = prepare_training_args(sys.argv[1:])[0]
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    seeding.set_random_seed(cfg.seed, f"trainer{rank}")

    # Initialize the global pytorch distributed communication group.
    dist.init_process_group("nccl")

    # Load and split dataset
    dataset_factory = DatasetFactory(cfg)
    train_dataset = dataset_factory.make_dataset(cfg.train_dataset, rank, world_size)
    valid_dataset = None
    if cfg.valid_dataset is not None:
        valid_dataset = dataset_factory.make_dataset(
            cfg.valid_dataset, rank, world_size
        )

    # Create rollout controller for online training and evaluation.
    rollout_controller = None
    if cfg.rollout is not None:
        rollout_factory = RolloutCollectorFactory(cfg)
        collector = rollout_factory.make_collector(cfg.rollout.collector)
        rollout_controller = RolloutController(cfg, cfg.rollout, collector=collector)

    # If trainer is given, run RL or offline training.
    if cfg.trainer is not None:
        trainer_factory = TrainerFactory(cfg)
        trainer = trainer_factory.make_trainer(
            cfg.trainer,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            rollout_controller=rollout_controller,
        )
        trainer.train()


if __name__ == "__main__":
    main()
