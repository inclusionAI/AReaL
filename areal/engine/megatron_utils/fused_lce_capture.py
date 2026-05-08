# SPDX-License-Identifier: Apache-2.0
"""
LM-head hidden-state capture for the fused linear-cross-entropy fast path.

The fused :func:`areal.utils.kernel.linear_cross_entropy` kernel needs the
pre-projection hidden state (``[seq, hidden]``) and the LM-head weight
(``[vocab, hidden]``, possibly vocab-sharded along the TP group) instead of
the materialised ``[seq, vocab]`` logits tensor. The Megatron-Core
:class:`GPTModel` does not expose either of these to AReaL's
``_compute_logprobs_and_loss`` call site by default, so we install a
temporary monkey-patch on ``output_layer.forward`` for the duration of one
microbatch forward pass:

1. Stashes the input tensor (``hidden``) and the actual weight (either the
   ``output_layer``'s own weight, or the embedding-tied weight passed in via
   ``weight=``).
2. Returns ``(hidden, None)`` instead of ``(logits, bias)``. Because
   :func:`areal.utils.data.unpad_logits` and
   :func:`postprocess_packed_seqs_context_parallel` are shape-agnostic on
   the leading sequence dim and propagate ``shape[1:]`` verbatim, the
   returned hidden tensor flows through the rest of the engine pipeline
   without modification — the engine's downstream code on the fused path
   never inspects the trailing dim except to take a min/max for diagnostic
   purposes, which we override with proxies in
   ``MegatronEngine._compute_logprobs_and_loss``.

The patch is installed only when ``enabled=True`` and uninstalled on
context exit (including on exception), so error-path leaks of the patched
method are impossible.

Compatibility notes:

* The patch is incompatible with Megatron-Core's MuP logit scaling
  (``config.use_mup``), MTP (``config.mtp_num_layers > 0``) and inference
  paths that materialise ``last_token_logits``. We assert against these
  configurations at install time and refuse to engage; the engine then
  falls back to the materialised path automatically.
* The patch is also incompatible with the critic value head, since that
  head is a 1-output-dim ``ColumnParallelLinear`` and the fused kernel
  requires the LM-head weight; the engine guards on ``is_critic`` before
  calling this helper.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from megatron.core import parallel_state as mpu

from areal.utils import logging

logger = logging.getLogger("FusedLCECapture")

# Keys used to pass captured tensors from forward_step → process_output.
# Centralised here to keep the engine and helper in lockstep.
FUSED_LCE_HIDDEN_KEY = "_fused_lce_hidden"
FUSED_LCE_WEIGHT_KEY = "_fused_lce_weight"


@dataclass
class _CaptureSlot:
    """Mutable single-shot stash populated by the patched ``forward``."""

    hidden: Optional[torch.Tensor] = None
    weight: Optional[torch.Tensor] = None


def _unwrap_to_post_process_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Strip DDP/Float16Module wrappers and return the inner module that
    owns ``output_layer`` (i.e. an mcore ``GPTModel`` on the last PP stage),
    or ``None`` if no such module is reachable on this rank.

    Returning ``None`` (instead of raising) lets the caller skip the patch
    transparently on intermediate pipeline stages.
    """
    inner = model
    # Loop bound: at most ~4 wrapper layers in practice (DDP, Float16Module,
    # vp wrapper). 8 is a generous upper bound that protects against
    # accidental cycles.
    for _ in range(8):
        if hasattr(inner, "output_layer") and inner.output_layer is not None:
            return inner
        if not hasattr(inner, "module"):
            return None
        inner = inner.module
    return None


def _is_compatible(post_process_module: torch.nn.Module) -> bool:
    """Refuse to engage when the model uses features incompatible with the
    fused kernel. Falling back is preferred over silently producing wrong
    numbers."""
    config = getattr(post_process_module, "config", None)
    if config is None:
        # Conservative default: don't patch unknown modules.
        return False

    if getattr(config, "use_mup", False):
        logger.warning(
            "Fused LCE: MuP scaling is enabled (config.use_mup=True); "
            "fused path is disabled for this microbatch."
        )
        return False
    if getattr(config, "mtp_num_layers", 0):
        logger.warning(
            "Fused LCE: MTP is enabled (config.mtp_num_layers>0); "
            "fused path is disabled for this microbatch."
        )
        return False

    output_layer = getattr(post_process_module, "output_layer", None)
    if output_layer is None:
        return False

    # Sequence parallel + TP gather inside output_layer is what we *want*
    # to bypass; AReaL runs with parallel_output=True which keeps logits
    # vocab-sharded — exactly what the fused kernel expects via tp_group.
    parallel_output = getattr(post_process_module, "parallel_output", True)
    if not parallel_output:
        # If gather_output=True, the engine has already requested the
        # full-vocab logits to be all-gathered; capturing hidden here would
        # mean the downstream kernel needs to gather instead, doubling
        # comms. Prefer the existing materialised path in that case.
        logger.warning(
            "Fused LCE: model has parallel_output=False; fused path is "
            "disabled to avoid an extra TP gather."
        )
        return False

    return True


@contextmanager
def capture_lm_head_hidden(
    model: torch.nn.Module, *, enabled: bool
) -> Iterator[Optional[_CaptureSlot]]:
    """Context manager that captures the input + weight handed to the
    ``output_layer`` of the wrapped Megatron GPT model.

    Yields:
        ``_CaptureSlot`` on the pipeline-last stage when ``enabled`` is
        True and the model is compatible; ``None`` otherwise. The caller is
        expected to inspect ``slot.hidden`` for ``None`` to decide whether
        the fused path is usable for this microbatch.
    """
    if not enabled:
        yield None
        return

    post_process = _unwrap_to_post_process_module(model)
    if post_process is None or not _is_compatible(post_process):
        # Either an intermediate PP stage or an incompatible config; the
        # engine will transparently fall back to the materialised path.
        yield None
        return

    output_layer = post_process.output_layer
    slot = _CaptureSlot()
    original_forward = output_layer.forward

    def _patched_forward(input_, weight=None, runtime_gather_output=None):
        # Resolve the actual weight: either passed in (weight tying) or the
        # output_layer's own parameter. We intentionally store a *reference*
        # (not detach) so autograd flows through both the kernel forward
        # and backward.
        actual_weight = weight if weight is not None else output_layer.weight
        slot.hidden = input_
        slot.weight = actual_weight
        # Return ``(input_, None)``: callers expect ``(logits, bias)`` and
        # only ever destructure with ``logits, _ = output_layer(...)``. The
        # downstream pipeline (``unpad_logits`` etc.) is shape-agnostic on
        # the trailing dim, so passing ``hidden`` through is safe; the
        # fused kernel will then consume the stashed tensors and produce
        # the real per-token logprobs.
        return input_, None

    # ``output_layer.forward = _patched_forward`` replaces the bound method
    # at instance level (via ``__dict__`` lookup), shadowing the class
    # method without mutating the class. Restoration in ``finally`` is
    # therefore exception-safe.
    output_layer.forward = _patched_forward  # type: ignore[assignment]
    try:
        yield slot
    finally:
        # Best-effort restoration. ``del`` removes the instance-level
        # binding and re-exposes the class-level method, which is what
        # callers will execute on subsequent forwards.
        try:
            del output_layer.forward
        except AttributeError:
            # If __dict__ assignment is not supported (rare for nn.Module
            # subclasses), fall back to direct restoration.
            output_layer.forward = original_forward  # type: ignore[assignment]


__all__ = [
    "FUSED_LCE_HIDDEN_KEY",
    "FUSED_LCE_WEIGHT_KEY",
    "capture_lm_head_hidden",
]
