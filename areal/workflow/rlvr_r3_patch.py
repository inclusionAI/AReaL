"""
R3 helpers for the RLVR workflow.

These functions bridge the inference-time ``ModelResponse.routed_experts``
(a numpy array of shape ``(num_sgl_tokens, num_moe_layers * topk)``) into the
training-side tensor dict so that the Megatron engine can replay routing
decisions.

The conversion pipeline:
    1. ``extract_routed_experts`` -- called in ``arun_episode`` right after
       ``_collect_samples``.  Converts the numpy array to a left-padded
       torch tensor of shape ``(1, seq_len, num_moe_layers, topk)``.
    2. The tensor is added to the result dict under key ``"routed_experts"``.
    3. During training, the ``MegatronEngine`` R3 patch picks it up from
       the batch data and feeds it to ``setup_per_microbatch_replay_forward``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def extract_routed_experts(
    routed_experts_np: Optional[np.ndarray],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_moe_layers: int,
    topk: int,
    compress_dtype: bool = True,
) -> Optional[torch.Tensor]:
    """Convert ``ModelResponse.routed_experts`` into a training tensor.

    Args:
        routed_experts_np: ``np.ndarray`` of shape ``(num_sgl_tokens, num_moe_layers * topk)``
            as returned by the SGLang inference backend, or ``None``.
        input_ids: ``(1, seq_len)`` token ids (prompt + response).
        attention_mask: ``(1, seq_len)`` with 1 for real tokens, 0 for padding.
        num_moe_layers: Number of MoE layers in the model. **Required**.
        topk: Router top-k. **Required**.
        compress_dtype: Downcast to ``uint8`` / ``int16`` when possible.

    Returns:
        ``torch.Tensor`` of shape ``(1, seq_len, num_moe_layers, topk)`` or ``None``.
    """
    if routed_experts_np is None:
        return None

    try:
        from areal.engine.router_replay_utils import (
            preprocess_routed_experts_batch,
        )

        return preprocess_routed_experts_batch(
            routed_experts_np,
            input_ids,
            attention_mask,
            num_moe_layers=num_moe_layers,
            topk=topk,
            compress_dtype=compress_dtype,
        )
    except Exception:
        logger.warning(
            "[R3] Failed to preprocess routed_experts (shape=%s); skipping.",
            getattr(routed_experts_np, "shape", "unknown"),
            exc_info=True,
        )
        return None


def inject_routed_experts_into_result(
    result: dict[str, torch.Tensor],
    routed_experts: Optional[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Add ``routed_experts`` to the result dict if available.

    This is a trivial helper kept separate for clarity and to centralise
    the key name (``"routed_experts"``).
    """
    if routed_experts is not None:
        result["routed_experts"] = routed_experts
    return result
