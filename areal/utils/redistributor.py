from dataclasses import dataclass
from typing import List

import torch.distributed as dist
from tensordict import TensorDict

from areal.utils.data import all_gather_tensor_container, concat_padded_tensors
from areal.utils.datapack import ffd_allocate


@dataclass
class RedistributedData:
    all_data: List[TensorDict]
    data: TensorDict
    rank: int
    group_indices: List[List[int]]


def redistribute(
    data: TensorDict, granularity: int = 1, group=None
) -> RedistributedData:
    """Redistribute a batch across a process group.

    This function only accepts padded data which must have an "attention_mask" field,
    Each tensor should have shape [bs, seqlen, *] or [bs].

    This function will divide the global batch into segments each with consecutive
    `granularity` sequences, and then redistribute the segments (e.g., for GRPO).
    """
    all_gathered = all_gather_tensor_container(data, group=group)

    all_data = []
    for data in all_gathered:
        bs = data["attention_mask"].shape[0]
        assert bs % granularity == 0
        all_data += [data[i : i + granularity] for i in range(0, bs, granularity)]

    seqlens = [d["attention_mask"].sum().item() for d in all_data]

    # Remove pad positions
    for d in all_data:
        l = d["attention_mask"].sum(-1).max().item()
        attn_mask_shape = d["attention_mask"].shape
        for k, v in d.items():
            if v.shape[:2] == attn_mask_shape[:2]:
                d[k] = v[:, :l]

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
