# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

Note on num_moe_layers and topk:
    At the workflow level, we may not know the exact model config values.
    We store the raw numpy array shape info and let the engine layer
    (which has access to tf_config) do the final reshape.  As a
    practical compromise, we accept optional num_moe_layers and topk
    parameters and fall back to shape-based inference when not provided.
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
    num_moe_layers: Optional[int] = None,
    topk: Optional[int] = None,
    compress_dtype: bool = True,
) -> Optional[torch.Tensor]:
    """Convert ``ModelResponse.routed_experts`` into a training tensor.

    Args:
        routed_experts_np: ``np.ndarray`` of shape ``(num_sgl_tokens, num_moe_layers * topk)``
            as returned by the SGLang inference backend, or ``None``.
        input_ids: ``(1, seq_len)`` token ids (prompt + response).
        attention_mask: ``(1, seq_len)`` with 1 for real tokens, 0 for padding.
        num_moe_layers: Number of MoE layers. If None, inferred from shape.
        topk: Router top-k. If None, inferred from shape.
        compress_dtype: Downcast to ``uint8`` / ``int16`` when possible.

    Returns:
        ``torch.Tensor`` of shape ``(1, seq_len, num_moe_layers, topk)`` or ``None``.
    """
    if routed_experts_np is None:
        return None

    try:
        if num_moe_layers is not None and topk is not None:
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
        else:
            # Fallback: infer num_moe_layers and topk from shape
            return _infer_and_preprocess(
                routed_experts_np,
                input_ids,
                attention_mask,
                compress_dtype=compress_dtype,
            )
    except Exception:
        logger.warning(
            "[R3] Failed to preprocess routed_experts (shape=%s); skipping.",
            getattr(routed_experts_np, "shape", "unknown"),
            exc_info=True,
        )
        return None


_INFER_LOGGED: set[tuple[int, int]] = set()


def _infer_and_preprocess(
    routed_experts_np: np.ndarray,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    compress_dtype: bool = True,
) -> torch.Tensor:
    """Infer num_moe_layers and topk from shape, then preprocess.

    We try common topk values (6, 8, 4, 2, 1) that divide the flat
    dimension evenly.  This is a fallback when model config is not available.
    """
    flat_dim = routed_experts_np.shape[1]

    topk = None
    for candidate_topk in [6, 8, 4, 2, 1]:
        if flat_dim % candidate_topk == 0:
            topk = candidate_topk
            break
    if topk is None:
        topk = 1
        logger.warning(
            "[R3] Cannot infer topk from flat_dim=%d; falling back to topk=1.",
            flat_dim,
        )
    num_moe_layers = flat_dim // topk

    _key = (num_moe_layers, topk)
    if _key not in _INFER_LOGGED:
        _INFER_LOGGED.add(_key)
        logger.info(
            "[R3] rlvr workflow inferred num_moe_layers=%d, topk=%d from "
            "flat_dim=%d (this count includes any dense-FFN layers; the "
            "engine-side router_replay_utils.set_router_replay_data handles "
            "the dense-vs-MoE split).  For deterministic behaviour, pass "
            "num_moe_layers and topk explicitly from the model config.",
            num_moe_layers,
            topk,
            flat_dim,
        )

    from areal.engine.router_replay_utils import preprocess_routed_experts_batch

    return preprocess_routed_experts_batch(
        routed_experts_np,
        input_ids,
        attention_mask,
        num_moe_layers=num_moe_layers,
        topk=topk,
        compress_dtype=compress_dtype,
    )


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
