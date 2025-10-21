from typing import Callable

from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import DatasetConfig


def create_dataloader(
    dataset,
    rank: int,
    world_size: int,
    dataset_config: DatasetConfig,
    collate_fn: Callable | None = None,
) -> StatefulDataLoader:
    """Create a stateful dataloader for a dataset with distributed sampler.

    Args:
        dataset: The dataset to create a dataloader for.
        rank: The rank of the process.
        world_size: The world size.
        dataset_config: The dataset config.
        collate_fn: The collate function to use.
    """
    if dataset_config.batch_size % world_size != 0:
        raise ValueError(
            f"batch size({dataset_config.batch_size}) must be divisible by world_size({world_size})!"
        )
    return StatefulDataLoader(
        dataset,
        batch_size=dataset_config.batch_size if dataset_config.single_rank_load else dataset_config.batch_size // world_size,
        sampler=DistributedSampler(
            dataset,
            world_size,
            rank,
            shuffle=dataset_config.shuffle,
            drop_last=dataset_config.drop_last,
        ),
        num_workers=dataset_config.num_workers,
        collate_fn=collate_fn or (lambda x: x),
    )
