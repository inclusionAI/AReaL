"""Colocation weight synchronization via direct tensor passing.

In colocation mode, training and inference processes share the same GPU.
NCCL cannot be used for inter-process communication on the same device.
Instead, this module passes tensors directly to the inference engine,
bypassing NCCL entirely.

The approach:
1. Training process gets full tensors (handling DTensor/FSDP sharding)
2. Tensors are chunked and passed directly to the inference engine
3. Inference engine loads weights from the received tensors

References:
- verl (https://github.com/volcengine/verl) - tensor-based weight sync
- slime (https://github.com/THUDM/slime) - offload/onload patterns
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from areal.api.io_struct import WeightUpdateMeta
from areal.infra.platforms import current_platform
from areal.utils import logging
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine

logger = logging.getLogger("ColocationSync")


def _get_full_tensor(param: nn.Parameter | torch.Tensor) -> torch.Tensor:
    """Get full tensor from a parameter, handling DTensor and CPU offload."""
    tensor = param.data if isinstance(param, nn.Parameter) else param
    if isinstance(tensor, DTensor):
        if tensor.device.type != "cpu":
            return tensor.full_tensor()
        return DTensor.from_local(
            tensor.to_local(),
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
        ).full_tensor()
    else:
        if tensor.device.type == "cpu":
            tensor = tensor.to(current_platform.device_type)
        return tensor


@trace_perf("colocation.update_weights_from_tensor", category="comm")
def update_weights_from_tensor(
    model: nn.Module,
    meta: WeightUpdateMeta,
    rollout_engine: InferenceEngine,
    cpu_group: dist.ProcessGroup,
    use_lora: bool = False,
    get_model_name_parameters=None,
) -> None:
    """Update inference engine weights by direct tensor passing (colocation mode).

    This bypasses NCCL entirely. On rank 0, tensors are exported and passed
    directly to the local inference engine's update_weights_from_tensor method.

    Parameters
    ----------
    model : nn.Module
        The training model whose weights to export.
    meta : WeightUpdateMeta
        Weight update metadata.
    rollout_engine : InferenceEngine
        The inference engine to update.
    cpu_group : dist.ProcessGroup
        CPU process group for barriers.
    use_lora : bool
        If True, only export trainable (LoRA) parameters.
    get_model_name_parameters : callable, optional
        Custom function to iterate model named parameters.
        Falls back to model.named_parameters() if None.
    """
    if dist.get_rank() == 0:
        rollout_engine.pause_generation()

    dist.barrier(group=cpu_group)

    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
    main_rank = dist.get_rank() == 0

    buffer_size = 0
    named_tensors: list[tuple[str, torch.Tensor]] = []

    if get_model_name_parameters is not None:
        param_iterator = get_model_name_parameters()
    else:
        param_iterator = model.named_parameters()

    if use_lora:
        param_iterator = (
            (name, param)
            for name, param in param_iterator
            if param.requires_grad
        )

    for name, param in param_iterator:
        tensor = _get_full_tensor(param)

        # Non-main ranks only help to get full tensor (for FSDP gather)
        if not main_rank:
            continue

        tensor_size = tensor.numel() * tensor.element_size()

        if tensor_size + buffer_size > weight_chunked_mem_size:
            _update_tensor_bucket(rollout_engine, named_tensors)
            buffer_size = 0

        named_tensors.append((name, tensor))
        buffer_size += tensor_size

    # Flush remaining
    if named_tensors:
        _update_tensor_bucket(rollout_engine, named_tensors)

    dist.barrier(group=cpu_group)

    if dist.get_rank() == 0:
        rollout_engine.continue_generation()

    current_platform.synchronize()
    dist.barrier(group=cpu_group)


def _update_tensor_bucket(
    rollout_engine: InferenceEngine,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    """Send a bucket of tensors directly to the inference engine."""
    if not named_tensors:
        return

    fut = rollout_engine.update_weights_from_tensor(list(named_tensors))
    fut.result()
    named_tensors.clear()
