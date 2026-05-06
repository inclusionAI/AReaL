# SPDX-License-Identifier: Apache-2.0

"""Core operations for training engines.

This module provides stateless utility functions that are shared across
different training engine implementations (FSDP, Megatron, etc.).
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils.data import (
    MicroBatchList,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
)
from areal.utils.logging import getLogger as _getLogger

_SPLIT_DIAG_LOGGER = _getLogger("R3SplitDiag")

__all__ = [
    "compute_total_loss_weight",
    "aggregate_eval_losses",
    "reorder_and_pad_outputs",
]


def compute_total_loss_weight(
    mb_list: MicroBatchList,
    loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    dp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Compute total loss weight and all_reduce across data parallel group.

    This aggregates the loss weights from all micro-batches and reduces
    them across the data parallel group to get a global normalization factor.

    Parameters
    ----------
    mb_list : MicroBatchList
        The list of micro-batches.
    loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor]
        Function to compute loss weight for each micro-batch.
    dp_group : dist.ProcessGroup
        The data parallel process group for all_reduce.

    Returns
    -------
    torch.Tensor
        The total loss weight (scalar tensor) after all_reduce.
    """
    total_weight = (
        torch.stack([loss_weight_fn(mb) for mb in mb_list.mbs])
        .sum()
        .detach()
        .clone()
        .to(dtype=torch.float32)
    )
    dist.all_reduce(total_weight, group=dp_group)
    assert total_weight > 0, (
        "Global total loss weight must be positive after all_reduce"
    )
    return total_weight


def aggregate_eval_losses(
    losses: list[torch.Tensor] | None,
    dp_group: dist.ProcessGroup,
    is_pp_last_stage: bool = True,
    pp_group: dist.ProcessGroup | None = None,
    pp_src_rank: int | None = None,
) -> torch.Tensor:
    """Aggregate evaluation losses from micro-batches.

    Parameters
    ----------
    losses : list[torch.Tensor] | None
        List of loss tensors from each micro-batch. None on non-last PP stages.
    dp_group : dist.ProcessGroup
        The data parallel process group for all_reduce.
    is_pp_last_stage : bool
        Whether this rank is the last PP stage. True by default.
    pp_group : dist.ProcessGroup | None
        Pipeline parallel group for broadcast. None if PP broadcast is not required.
    pp_src_rank : int | None
        Global rank of last PP stage (required if pp_group is set).

    Returns
    -------
    torch.Tensor
        The aggregated loss after summing and all_reduce.
    """
    if is_pp_last_stage:
        assert losses is not None, "losses required on last PP stage"
        loss = torch.stack(losses).sum(dtype=torch.float32)
        dist.all_reduce(loss, group=dp_group)
    else:
        device = current_platform.current_device()
        loss = torch.empty(1, device=device, dtype=torch.float32)

    if pp_group is not None:
        assert pp_src_rank is not None, "pp_src_rank required when pp_group is set"
        dist.broadcast(loss, src=pp_src_rank, group=pp_group)

    return loss


def reorder_and_pad_outputs(
    outputs: list[torch.Tensor],
    output_seqlens: list[int],
    mb_list: MicroBatchList,
    aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
) -> torch.Tensor:
    """Aggregate, reorder, and pad forward outputs from micro-batches.

    This handles the output post-processing for forward_batch:
    1. Aggregate outputs from all micro-batches
    2. Unpack by sequence lengths
    3. Reorder to match original input order
    4. Pad and stack along batch dimension

    Parameters
    ----------
    outputs : list[torch.Tensor]
        List of output tensors from each micro-batch.
    output_seqlens : list[int]
        Sequence lengths for unpacking.
    mb_list : MicroBatchList
        The micro-batch list containing reordering indices.
    aggregate_fn : Callable[[list[Any]], Any], optional
        Function to aggregate outputs, by default torch.cat.

    Returns
    -------
    torch.Tensor
        The processed outputs, padded and stacked along batch dimension.
    """
    res = aggregate_fn(outputs)
    seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
    # [SPLIT_MISMATCH_DIAG] Log EVERYTHING needed to root-cause the
    # `split_with_sizes` mismatch reported during compute_logp.
    try:
        _rank = None
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                _rank = _dist.get_rank()
        except Exception:
            _rank = None
        _out_shapes = [tuple(o.shape) for o in outputs]
        _out_sum0 = [int(o.shape[0]) for o in outputs if o.ndim >= 1]
        _sum_seqlens = int(sum(seqlens))
        _res_shape = tuple(res.shape)
        _res_dim0 = int(res.shape[0]) if res.ndim >= 1 else -1
        _fwd_idx = list(mb_list.forward_indices)
        _bwd_idx = list(mb_list.backward_indices)
        _mbs_lens = None
        try:
            _mbs_lens = [
                int(mb.get("cu_seqlens", torch.empty(0))[-1].item())
                if isinstance(mb.get("cu_seqlens", None), torch.Tensor)
                and mb["cu_seqlens"].numel() > 0
                else None
                for mb in getattr(mb_list, "mbs", [])
            ]
        except Exception:
            _mbs_lens = "ERR"
        _padded_lens = getattr(mb_list, "padded_to_lengths", None)
        _group_lens = getattr(mb_list, "group_lens", None)
        _padding_lens = getattr(mb_list, "padding_lengths", None)
        _SPLIT_DIAG_LOGGER.info(
            "[SPLIT_MISMATCH_DIAG][reorder_and_pad_outputs] rank=%s "
            "n_outputs=%d out_shapes=%s out_sum_dim0=%s sum_out_dim0=%d "
            "res.shape=%s res.dim0=%d "
            "len(output_seqlens)=%d sum(output_seqlens)=%d "
            "len(seqlens_reordered)=%d sum(seqlens_reordered)=%d "
            "forward_indices=%s backward_indices=%s "
            "mb_real_total_lens=%s padded_to_lengths=%s "
            "group_lens=%s padding_lengths=%s "
            "match=%s output_seqlens_head=%s output_seqlens_tail=%s",
            _rank,
            len(outputs),
            _out_shapes,
            _out_sum0,
            int(sum(_out_sum0)),
            _res_shape,
            _res_dim0,
            len(output_seqlens),
            int(sum(output_seqlens)),
            len(seqlens),
            _sum_seqlens,
            _fwd_idx,
            _bwd_idx,
            _mbs_lens,
            _padded_lens,
            _group_lens,
            _padding_lens,
            (_res_dim0 == _sum_seqlens),
            list(output_seqlens[:16]),
            list(output_seqlens[-16:]),
        )
    except Exception:
        _SPLIT_DIAG_LOGGER.exception(
            "[SPLIT_MISMATCH_DIAG][reorder_and_pad_outputs] log-emit failed"
        )
    unpacked = unpack_sequence(res, lens=seqlens, dim=0)
    reordered = reorder_list(unpacked, mb_list.backward_indices)
    return pad_and_stack_tensors_along_first_dim(reordered)
