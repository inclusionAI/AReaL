# SPDX-License-Identifier: Apache-2.0
"""
LM-head hidden-state capture for the fused linear-cross-entropy fast path.

The fused LCE kernel needs ``(hidden, weight)`` instead of materialised
``[seq, vocab]`` logits.  This module temporarily monkey-patches
``output_layer.forward`` to capture those tensors for one microbatch.

Compatibility: incompatible with MuP (``use_mup``), MTP
(``mtp_num_layers > 0``), and critic heads.  The engine falls back to
the materialised path automatically when any of these conditions hold.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)

from areal.utils import logging

logger = logging.getLogger("FusedLCECapture")

FUSED_LCE_HIDDEN_KEY = "_fused_lce_hidden"
FUSED_LCE_WEIGHT_KEY = "_fused_lce_weight"


@dataclass
class _CaptureSlot:
    hidden: torch.Tensor | None = None
    weight: torch.Tensor | None = None


def _unwrap_to_post_process_module(model: torch.nn.Module) -> torch.nn.Module | None:
    inner = model
    for _ in range(8):
        if hasattr(inner, "output_layer") and inner.output_layer is not None:
            return inner
        if not hasattr(inner, "module"):
            return None
        inner = inner.module
    return None


def _is_compatible(post_process_module: torch.nn.Module) -> bool:
    config = getattr(post_process_module, "config", None)
    if config is None:
        return False

    if getattr(config, "use_mup", False):
        logger.warning(
            "Fused LCE disabled: MuP scaling is enabled (config.use_mup=True)."
        )
        return False
    if getattr(config, "mtp_num_layers", 0):
        logger.warning(
            "Fused LCE disabled: MTP is enabled (config.mtp_num_layers>0)."
        )
        return False

    output_layer = getattr(post_process_module, "output_layer", None)
    if output_layer is None:
        return False

    parallel_output = getattr(post_process_module, "parallel_output", True)
    if not parallel_output:
        logger.warning(
            "Fused LCE disabled: model has parallel_output=False; "
            "would require an extra TP gather."
        )
        return False

    return True


@contextmanager
def capture_lm_head_hidden(
    model: torch.nn.Module, *, enabled: bool
) -> Iterator[_CaptureSlot | None]:
    if not enabled:
        yield None
        return

    post_process = _unwrap_to_post_process_module(model)
    if post_process is None or not _is_compatible(post_process):
        yield None
        return

    output_layer = post_process.output_layer
    slot = _CaptureSlot()
    original_forward = output_layer.forward

    config = getattr(post_process, "config", None)
    sequence_parallel = bool(getattr(config, "sequence_parallel", False))
    tp_world_size = mpu.get_tensor_model_parallel_world_size()
    needs_sp_gather = sequence_parallel and tp_world_size > 1

    def _patched_forward(input_, weight=None, runtime_gather_output=None):
        actual_weight = weight if weight is not None else output_layer.weight

        hidden = input_
        if needs_sp_gather:
            hidden = gather_from_sequence_parallel_region(hidden)

        if hidden.dtype != actual_weight.dtype:
            hidden = hidden.to(actual_weight.dtype)

        slot.hidden = hidden
        slot.weight = actual_weight
        return hidden, None

    output_layer.forward = _patched_forward  # type: ignore[assignment]
    try:
        yield slot
    finally:
        try:
            del output_layer.forward
        except AttributeError:
            output_layer.forward = original_forward  # type: ignore[assignment]


__all__ = [
    "FUSED_LCE_HIDDEN_KEY",
    "FUSED_LCE_WEIGHT_KEY",
    "capture_lm_head_hidden",
]
