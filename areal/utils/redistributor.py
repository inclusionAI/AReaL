from dataclasses import dataclass
from typing import List

import torch.distributed as dist
from tensordict import TensorDict

from areal.utils.data import all_gather_tensor_container, concat_padded_tensors
from areal.utils.datapack import ffd_allocate, flat2d


@dataclass
class RedistributedData:
    all_data: List[TensorDict]
    data: TensorDict
    rank: int
    group_indices: List[List[int]]


def redistribute(data: TensorDict, group=None) -> RedistributedData:
    """Redistribute a batch across a process group.

    This function only accepts padded data which must have an "attention_mask" field,
    Each tensor should have shape [bs, seqlen, *] or [bs].

    This function does not respect the boundary of grouped responses (aka responses
    with the same prompt).
    """
    all_data = all_gather_tensor_container(data, group=group)
    all_data = flat2d(
        [
            [data[i : i + 1] for i in range(data["attention_mask"].shape[0])]
            for data in all_data
        ]
    )
    seqlens = [d["attention_mask"].sum().item() for d in all_data]

    # Remove pad positions
    for d, seqlen in zip(all_data, seqlens):
        for k, v in d.items():
            if v.shape[:2] == d["attention_mask"].shape[:2]:
                d[k] = v[:, :seqlen]

    # No capacity limit leads to balanced partition across this group
    group_indices = ffd_allocate(
        seqlens, capacity=int(1e12), min_groups=dist.get_world_size(group)
    )
    local_indices = group_indices[dist.get_rank(group=group)]

    data = concat_padded_tensors([all_data[i] for i in local_indices])
    return RedistributedData(
        all_data=all_data,
        data=data,
        rank=dist.get_rank(group=group),
        group_indices=group_indices,
    )
