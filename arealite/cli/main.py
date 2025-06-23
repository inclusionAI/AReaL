import os

import hydra
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.elastic.multiprocessing.errors import record

from arealite.api.cli_args import DatasetConfig, TrainingArgs
from arealite.api.rollout_api import RolloutWorkflowFactory
from arealite.api.trainer_api import TrainerFactory
from arealite.impl.rollout_controller import RolloutController


def create_distributed_dataset(cfg: DatasetConfig):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dataset = load_dataset(
        cfg.path,
        # name=cfg.name,
        # split=cfg.split,
        # data_files=cfg.data_files,
    )
    dataset = split_dataset_by_node(dataset["train"], rank=rank, world_size=world_size)
    return dataset


@record
@hydra.main(version_base=None, config_path="../config")
def main(args: TrainingArgs):
    dist.init_process_group("nccl")

    # Load and split dataset
    train_dataset = create_distributed_dataset(args.train_dataset)
    valid_dataset = None
    if args.valid_dataset is not None:
        valid_dataset = create_distributed_dataset(args.valid_dataset)

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

    # After training, run rollout over the entire dataset
    if valid_dataset and rollout_controller:
        rollout_controller.eval_dataset(valid_dataset)


if __name__ == "__main__":
    main()
