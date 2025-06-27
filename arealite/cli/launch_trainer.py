# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import argparse
import os
from pathlib import Path

import torch.distributed as dist
from hydra import compose as hydra_compose
from hydra import initialize as hydra_init
from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

from arealite.api.cli_args import DatasetConfig, TrainingArgs
from arealite.api.dataset_api import DatasetFactory
from arealite.api.rollout_api import RolloutCollectorFactory
from arealite.api.trainer_api import TrainerFactory
from arealite.system.rollout_controller import RolloutController
from realhf.base import constants, name_resolve, seeding


@record
def main():
    """Main entry point for launching the trainer."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="The path of the main configuration file", required=True
    )
    args, overrides = parser.parse_known_args()

    # Initialize hydra config
    config_file = Path(args.config).absolute()
    assert config_file.exists()
    relpath = Path(os.path.relpath(str(config_file), Path(__file__).parent.absolute()))
    hydra_init(config_path=str(relpath.parent), job_name="app", version_base=None)
    cfg = hydra_compose(
        config_name=str(relpath.name).rstrip(".yaml"),
        overrides=overrides,
    )

    # Merge with the default configuration
    default_cfg = OmegaConf.structured(TrainingArgs)
    cfg = OmegaConf.merge(default_cfg, cfg)
    cfg: TrainingArgs = OmegaConf.to_object(cfg)

    seeding.set_random_seed(cfg.seed, "llm_server")
    constants.set_experiment_trial_names(cfg.experiment_name, cfg.trial_name)
    name_resolve.reconfigure(cfg.cluster.name_resolve)

    # Initialize the global pytorch distributed communication group.
    dist.init_process_group("nccl")

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

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
