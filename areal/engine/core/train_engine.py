# SPDX-License-Identifier: Apache-2.0

"""Core operations for training engines.

This module provides stateless utility functions that are shared across
different training engine implementations (FSDP, Megatron, etc.).
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from areal.api.loss import LossNormalizer, LossValue, get_named_loss_names
from areal.infra.platforms import current_platform
from areal.utils.data import (
    TRANSPORT_DUMMY_KEY,
    MicroBatchList,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    tensor_container_to,
    unpack_sequence,
)

__all__ = [
    "compute_microbatch_loss_weight",
    "compute_total_loss_weight",
    "normalize_loss",
    "aggregate_eval_losses",
    "reorder_and_pad_outputs",
    "stage_batch_for_engine",
]


def compute_microbatch_loss_weight(
    microbatch: dict[str, Any],
    loss_weight_fn: LossNormalizer,
) -> LossValue:
    """Validate real masses or construct schema-shaped zeros for transport data."""
    names = get_named_loss_names(loss_weight_fn)
    if microbatch.get(TRANSPORT_DUMMY_KEY) is True:
        reference = next(
            (value for value in microbatch.values() if isinstance(value, torch.Tensor)),
            None,
        )
        if reference is None:
            raise ValueError("Transport micro-batch does not contain a tensor")
        zero = torch.zeros((), dtype=torch.float32, device=reference.device)
        return {name: zero.clone() for name in names} if names is not None else zero
    weight = loss_weight_fn(microbatch)
    if names is not None:
        if not isinstance(weight, dict) or tuple(sorted(weight)) != names:
            raise ValueError("Loss normalizer keys must match its declared names")
        for mass in weight.values():
            _validate_mass(mass)
    elif isinstance(weight, dict):
        raise ValueError("Named losses require @named_loss_normalizer(names=...)")
    return weight


def compute_total_loss_weight(
    mb_list: MicroBatchList,
    loss_weight_fn: LossNormalizer,
    dp_group: dist.ProcessGroup,
    device: torch.device | str | int | None = None,
) -> LossValue:
    """Reduce optimizer-batch masses over DP (or the engine's DP+CP group).

    Named terms use one vector collective in declared name order. Names must
    match across ranks; static schemas also cover ranks with only transport
    data. Globally empty named terms retain zero mass. Scalar behavior is
    unchanged, including the requirement of a positive global weight.
    """
    weights = [compute_microbatch_loss_weight(mb, loss_weight_fn) for mb in mb_list.mbs]
    names = get_named_loss_names(loss_weight_fn)
    if names is not None:
        total = torch.stack(
            [
                torch.stack([weight[name].float() for name in names])
                for weight in weights
            ]
        ).sum(dim=0)
        total = total.detach().to(device=device, dtype=torch.float32)
        dist.all_reduce(total, group=dp_group)
        torch._assert_async(
            torch.all(torch.isfinite(total) & (total >= 0)),
            "Global loss normalizers must be finite and nonnegative",
        )
        return dict(zip(names, total.unbind(), strict=True))
    total_weight = torch.stack(weights).sum()
    total_weight = total_weight.detach().clone().to(device=device, dtype=torch.float32)
    dist.all_reduce(total_weight, group=dp_group)
    assert total_weight > 0, (
        "Global total loss weight must be positive after all_reduce"
    )
    return total_weight


def _validate_mass(value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError("Each loss normalizer must be a scalar tensor")
    if value.requires_grad:
        raise ValueError("Loss normalizers must be batch-derived, without gradients")
    torch._assert_async(
        torch.isfinite(value) & (value >= 0),
        "Loss normalizers must be finite and nonnegative",
    )


def normalize_loss(
    loss: LossValue,
    local_weight: LossValue,
    total_weight: LossValue,
    multiplier: float = 1.0,
) -> torch.Tensor:
    """Combine named numerators, or rescale a legacy microbatch mean.

    Each named term is independently normalized over the optimizer batch and
    the engine's existing data-parallel group. Fractional masses must not be
    clamped to one. A globally empty term contributes a graph-connected zero.
    """
    if isinstance(loss, dict):
        if not isinstance(local_weight, dict) or not isinstance(total_weight, dict):
            raise TypeError("Named loss numerators require named normalizers")
        if (
            not loss
            or loss.keys() != local_weight.keys()
            or loss.keys() != total_weight.keys()
        ):
            raise ValueError("Loss numerator and normalizer names must match")
        normalized = []
        for name in sorted(loss):
            numerator = loss[name]
            if not isinstance(numerator, torch.Tensor) or numerator.ndim != 0:
                raise ValueError(f"Loss numerator {name!r} must be a scalar tensor")
            numerator = numerator.float()
            torch._assert_async(
                torch.isfinite(numerator),
                f"Loss numerator {name!r} must be finite, including empty terms",
            )
            _validate_mass(local_weight[name])
            _validate_mass(total_weight[name])
            mass = total_weight[name].to(device=numerator.device, dtype=torch.float32)
            denominator = torch.where(mass > 0, mass, 1.0)
            normalized.append(
                torch.where(mass > 0, numerator.float() / denominator, numerator * 0)
            )
        return torch.stack(normalized).sum() * multiplier
    if isinstance(local_weight, dict) or isinstance(total_weight, dict):
        raise TypeError("Scalar losses require scalar normalizers")
    return loss * (local_weight / total_weight * multiplier)


def stage_batch_for_engine(data: dict[str, Any], engine: Any) -> dict[str, Any]:
    """Move a transient trainer batch to the engine's staging device in-place.

    Megatron streams microbatches from CPU. Replacing the values in the
    existing dictionary is important for RPC calls: the RPC argument container
    aliases this dictionary, so a simple local rebind would keep the original
    full-batch accelerator tensors alive for the duration of training.
    Other engines retain their existing input placement.
    """
    if not getattr(engine, "stream_microbatches_from_cpu", False):
        return data
    staged = tensor_container_to(data, "cpu")
    data.clear()
    data.update(staged)
    return data


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
    semantic_batch_size = len(output_seqlens)
    output_seqlens = [*output_seqlens, *([1] * mb_list.transport_dummy_count)]
    seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
    unpacked = unpack_sequence(res, lens=seqlens, dim=0)
    reordered = reorder_list(unpacked, mb_list.backward_indices)
    if mb_list.transport_dummy_count:
        reordered = reordered[:semantic_batch_size]
    return pad_and_stack_tensors_along_first_dim(reordered)
