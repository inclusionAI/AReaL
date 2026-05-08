# SPDX-License-Identifier: Apache-2.0
"""
High-level fused linear cross-entropy entry points for AReaL.

These wrappers bridge the :class:`LinearCrossEntropy` Triton kernel into
AReaL's existing :func:`gather_logprobs_entropy` interface so that the
Megatron path can opt in via a single configuration flag without
restructuring the model forward.

The wrappers:

* accept already-flat ``hidden`` of shape ``(num_tokens, hidden_size)`` and
  ``labels`` of shape ``(num_tokens,)`` (or higher-dimensional tensors with
  an explicit last hidden dim) so the call site looks identical to the
  existing materialised path;
* support optional tensor-parallel via ``tp_group`` for vocab-sharded
  ``weight`` matrices;
* fall back gracefully to the materialised reference path when Triton is
  unavailable or inputs are not on CUDA, so unit tests can still run on CPU.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist

from areal.utils import logging

logger = logging.getLogger("LinearCrossEntropy")


def _force_fallback() -> bool:
    """Allow ops/CI to disable the fused kernel via env var without code change."""
    return os.environ.get("AREAL_DISABLE_FUSED_LCE", "0") == "1"


def _kernel_available() -> bool:
    """Whether the Triton fused kernel can run on this host."""
    if _force_fallback():
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _reference_logprobs_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    tp_group: Optional[dist.ProcessGroup],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference (materialised-logits) implementation.

    Used when Triton is unavailable. Mathematically equivalent to the fused
    kernel up to floating-point reordering, which is why the test suite
    asserts with explicit rtol/atol rather than bitwise equality.
    """
    # Shape normalisation matches the fused kernel.
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_labels = labels.reshape(-1)

    logits = torch.matmul(flat_hidden.float(), weight.float().t())
    if temperature != 1.0:
        logits = logits / temperature

    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        # Vocab-parallel: gather full vocab logits across TP group.
        # Used only as a slow correctness fallback.
        world_size = dist.get_world_size(tp_group)
        gathered = [torch.empty_like(logits) for _ in range(world_size)]
        dist.all_gather(gathered, logits, group=tp_group)
        logits = torch.cat(gathered, dim=-1)

    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_softmax.gather(
        dim=-1, index=flat_labels.unsqueeze(-1)
    ).squeeze(-1)
    probs = log_softmax.exp()
    entropy = -(probs * log_softmax).sum(dim=-1)
    return log_probs_labels, entropy


def linear_cross_entropy_logprobs_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log-prob and entropy from hidden states + lm-head weight.

    This is the fused counterpart to
    :func:`areal.utils.functional.vocab_parallel.gather_logprobs_entropy`,
    but consumes ``hidden`` (last layer states) and ``weight`` (lm-head
    weight) directly instead of a materialised ``[num_tokens, vocab_size]``
    logits tensor. Memory savings scale with ``vocab_size``.

    Args:
        hidden: ``(..., hidden_size)`` last-layer hidden states.
        weight: ``(vocab_size, hidden_size)`` lm-head weight; may be sharded
            on the vocab dimension when ``tp_group`` is set.
        labels: ``(...,)`` integer label ids matching the leading dims of
            ``hidden``. With TP, labels MUST hold *global* vocab ids.
        temperature: softmax temperature.
        tp_group: optional tensor-parallel group when ``weight`` is sharded.

    Returns:
        ``(logprobs, entropy)`` both shaped like ``labels``.
    """
    leading_shape = labels.shape

    if _kernel_available():
        # Lazy import: keeps a hard Triton import out of the module path so
        # CPU-only environments can still load areal.utils.functional.
        from areal.utils.kernel.linear_cross_entropy import linear_cross_entropy

        if hidden.device.type != "cuda":
            logger.warning(
                "Fused LCE requested but hidden is on %s; falling back to reference path.",
                hidden.device,
            )
        else:
            try:
                logprobs, entropy = linear_cross_entropy(
                    hidden,
                    weight,
                    labels,
                    temperature,
                    "none",
                    tp_group,
                )
                return logprobs.reshape(leading_shape), entropy.reshape(leading_shape)
            except Exception as exc:  # pragma: no cover - fall back path
                logger.warning(
                    "Fused LCE kernel raised %s; falling back to reference path.",
                    exc,
                )

    logprobs, entropy = _reference_logprobs_entropy(
        hidden, weight, labels, temperature, tp_group
    )
    return logprobs.reshape(leading_shape), entropy.reshape(leading_shape)


def linear_cross_entropy_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Logprobs-only counterpart of :func:`linear_cross_entropy_logprobs_entropy`.

    Returns a tensor shaped like ``labels``.
    """
    logprobs, _ = linear_cross_entropy_logprobs_entropy(
        hidden, weight, labels, temperature, tp_group
    )
    return logprobs
