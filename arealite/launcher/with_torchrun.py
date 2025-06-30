# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import os
import sys

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from arealite.api.cli_args import TrainingArgs, prepare_training_args
from arealite.api.dataset_api import DatasetFactory
from arealite.api.io_struct import AllocationMode
from arealite.api.llm_server_api import LLMServerFactory
from arealite.api.rollout_api import RolloutCollectorFactory
from arealite.api.trainer_api import TrainerFactory
from arealite.system.rollout_controller import RolloutController
from realhf.base import logging, seeding

logger = logging.getLogger(__file__, "system")


@record
def main():
    cfg: TrainingArgs = prepare_training_args(sys.argv[1:])[0]
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    seeding.set_random_seed(cfg.seed, f"worker{rank}")

    if cfg.mode != "torchrun":
        logger.warning("Using the torchrun script, CLI arg `mode` is ignored.")

    alloc_mode = AllocationMode.from_str(cfg.allocation_mode)
    if rank < alloc_mode.gen_world_size:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("LOCAL_RANK", "0")
        server = LLMServerFactory(cfg).make_server(cfg.rollout.llm_service)
        server.start()
        return

    os.environ["RANK"] = str(rank - alloc_mode.gen_world_size)
    os.environ["WORLD_SIZE"] = str(world_size - alloc_mode.gen_world_size)
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
