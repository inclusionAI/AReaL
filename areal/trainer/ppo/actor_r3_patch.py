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
R3 metrics and logging helpers for the PPO actor.

When Router Replay (R3) is enabled, these functions compute and log
statistics about the replayed routing decisions, such as:

- The fraction of micro-batches that carried replay data.
- Per-step summary of routing shapes and data types.

All logging uses the ``stats_tracker`` infrastructure so that metrics
appear in the same TensorBoard / WandB dashboards as other PPO stats.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from areal.utils import stats_tracker

logger = logging.getLogger(__name__)


def log_r3_data_stats(
    data: dict[str, Any],
    scope: str = "r3",
) -> None:
    """Log summary statistics about the ``routed_experts`` tensor in a
    training data dict.

    Called once per PPO update step (not per micro-batch) to avoid
    log spam.

    Args:
        data: The training data dict that may contain ``"routed_experts"``.
        scope: Stats-tracker scope prefix.
    """
    re = data.get("routed_experts")
    if re is None:
        return

    with stats_tracker.scope(scope):
        if isinstance(re, torch.Tensor):
            stats_tracker.scalar(
                r3_batch_size=re.shape[0],
                r3_seq_len=re.shape[1],
                r3_num_layers=re.shape[2] if re.dim() >= 3 else 0,
                r3_topk=re.shape[3] if re.dim() >= 4 else 0,
                r3_dtype_bytes=re.element_size(),
                r3_max_expert_id=re.max().item() if re.numel() > 0 else 0,
            )
        else:
            stats_tracker.scalar(r3_present=0)


def strip_routed_experts_before_loss(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Remove ``routed_experts`` from the data dict before the loss function.

    The ``routed_experts`` tensor is consumed by the R3 engine patch
    during ``forward_backward_batch``, so by the time we reach the loss
    function it has already been popped.  This function is a safety net.

    Returns the data dict (modified in-place).
    """
    data.pop("routed_experts", None)
    return data
