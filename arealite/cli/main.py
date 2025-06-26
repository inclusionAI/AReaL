# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import os

import hydra
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from arealite.api.cli_args import DatasetConfig, TrainingArgs
from arealite.api.dataset_api import DatasetFactory
from arealite.api.rollout_api import RolloutWorkflowFactory
from arealite.api.trainer_api import TrainerFactory
from arealite.system.rollout_controller import RolloutController


@record
@hydra.main(version_base=None, config_path="../config")
def main(args: TrainingArgs):
    dist.init_process_group("nccl")

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Load and split dataset
    dataset_factory = DatasetFactory(args)
    train_dataset = dataset_factory.make_dataset(args.train_dataset, rank, world_size)
    valid_dataset = None
    if args.valid_dataset is not None:
        valid_dataset = dataset_factory.make_dataset(
            args.valid_dataset, rank, world_size
        )

    # Create rollout controller for online training and evaluation.
    rollout_controller = None
    if args.rollout is not None:
        rollout_factory = RolloutWorkflowFactory(args)
        workflow = rollout_factory.make_workflow(args.rollout.workflow)
        rollout_controller = RolloutController(args, args.rollout, workflow=workflow)

    # If trainer is given, run RL or offline training.
    if args.trainer is not None:
        trainer_factory = TrainerFactory(args)
        trainer = trainer_factory.make_trainer(
            args.trainer,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            rollout_controller=rollout_controller,
        )
        trainer.train()


if __name__ == "__main__":
    main()
