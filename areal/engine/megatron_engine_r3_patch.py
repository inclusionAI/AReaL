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
R3 Integration Patch for MegatronEngine.

This module wraps ``MegatronEngine.forward_backward_batch`` so that, when
the micro-batch data contains ``routed_experts`` tensors, each micro-batch's
forward step is preceded by a call to ``setup_per_microbatch_replay_forward``
and followed (after the full forward pass) by a switch to backward-replay
mode.

The patch handles the critical issue that ``routed_experts`` is a 4D tensor
``(bs, seq_len, num_moe_layers, topk)`` which will NOT be correctly split by
``split_padded_tensor_dict_into_mb_list`` (which only splits tensors with
``numel() == bs * max_seqlen``).  Instead, we extract ``routed_experts``
from ``mb_list.data`` before micro-batch splitting, and manually distribute
it to each micro-batch using the ``forward_indices`` and ``group_lens``
from ``MicroBatchList``.

Usage::

    from areal.engine.megatron_engine_r3_patch import patch_megatron_engine_for_r3
    patch_megatron_engine_for_r3(engine, enable_router_replay=True)
"""

from __future__ import annotations

import logging
import types
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ===================================================================
# Public API
# ===================================================================


def patch_megatron_engine_for_r3(
    engine,
    enable_router_replay: bool = False,
) -> None:
    """Patch a ``MegatronEngine`` instance to support Router Replay (R3).

    1. Applies Megatron-Core monkey-patches (TransformerConfig, TopKRouter,
       Dispatcher).
    2. Tags the engine with ``_r3_enabled = True``.
    3. Wraps ``forward_backward_batch`` to inject per-microbatch replay
       setup / teardown around the Megatron pipeline schedule.

    Args:
        engine: A ``MegatronEngine`` instance (already initialized).
        enable_router_replay: Master switch.
    """
    if not enable_router_replay:
        engine._r3_enabled = False
        logger.debug("[R3] Router replay not enabled; skipping engine patch.")
        return

    logger.info("[R3] Patching MegatronEngine for Router Replay (R3).")

    # Mark and save original
    engine._r3_enabled = True
    engine._r3_original_forward_backward_batch = engine.forward_backward_batch

    # Bind the wrapped method
    engine.forward_backward_batch = types.MethodType(
        _r3_forward_backward_batch, engine
    )

    logger.info("[R3] MegatronEngine patched successfully.")


# ===================================================================
# attention_mask reconstruction from cu_seqlens (Problem 5 fix)
# ===================================================================


def _reconstruct_attention_mask_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Reconstruct a 2D ``attention_mask`` from packed ``cu_seqlens``.

    After ``pack_tensor_dict``, the original ``attention_mask`` is replaced
    by ``cu_seqlens`` (shape ``(B+1,)``) and ``max_seqlen``.  For R3's
    ``set_router_replay_data`` we need an ``attention_mask`` of shape
    ``(B, padded_seq_len)`` where padded_seq_len = max_seqlen.

    Args:
        cu_seqlens: ``(B+1,)`` cumulative sequence lengths.
        max_seqlen: Maximum sequence length (the padded dimension).

    Returns:
        ``torch.Tensor`` of shape ``(B, max_seqlen)`` with dtype ``torch.bool``.
    """
    bs = cu_seqlens.shape[0] - 1
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # (B,)
    # Build mask: position j < seq_lens[i] -> True
    positions = torch.arange(max_seqlen, device=cu_seqlens.device).unsqueeze(0)  # (1, S)
    mask = positions < seq_lens.unsqueeze(1)  # (B, S)
    logger.debug(
        "[R3] Reconstructed attention_mask from cu_seqlens: "
        "bs=%d, max_seqlen=%d, seq_lens=%s.",
        bs,
        max_seqlen,
        seq_lens.tolist()[:8],  # log first 8 for brevity
    )
    return mask


# ===================================================================
# routed_experts splitting (Problem 7 fix: robust sample-count inference)
# ===================================================================


def _infer_mb_sample_count(
    mb_dict: dict,
    total_bs: int,
    n_mbs: int,
) -> int:
    """Infer the number of samples in a micro-batch dict.

    Tries multiple strategies in order of reliability:
    1. ``cu_seqlens`` -> ``len(cu_seqlens) - 1`` (packed format, most reliable)
    2. ``attention_mask.shape[0]`` (padded format)
    3. ``input_ids.shape[0]`` (fallback)
    4. Even division (last resort)
    """
    if isinstance(mb_dict, dict):
        # Strategy 1: cu_seqlens (packed format -- most common after pack_tensor_dict)
        cu = mb_dict.get("cu_seqlens")
        if cu is not None:
            return len(cu) - 1

        # Strategy 2: attention_mask (padded format)
        attn = mb_dict.get("attention_mask")
        if attn is not None and hasattr(attn, "shape"):
            return attn.shape[0]

        # Strategy 3: input_ids
        ids = mb_dict.get("input_ids")
        if ids is not None and hasattr(ids, "shape"):
            return ids.shape[0]

    # Strategy 4: last resort
    logger.warning(
        "[R3] _infer_mb_sample_count: no reliable key found, "
        "falling back to even division (%d / %d).",
        total_bs,
        n_mbs,
    )
    return total_bs // n_mbs


def _split_routed_experts_for_mbs(
    routed_experts: torch.Tensor,
    mb_list,
) -> list[torch.Tensor | None]:
    """Split the batch-level ``routed_experts`` tensor into per-micro-batch tensors.

    Uses ``mb_list.forward_indices`` and per-MB sample counts to correctly
    reorder and slice samples, mirroring how ``split_padded_tensor_dict_into_mb_list``
    splits other tensors.

    Args:
        routed_experts: ``(bs, max_seqlen, num_moe_layers, topk)``
        mb_list: ``MicroBatchList`` with ``forward_indices`` and ``group_lens``.

    Returns:
        List of tensors, one per micro-batch, each of shape
        ``(mb_bs, max_seqlen, num_moe_layers, topk)``.
    """
    if routed_experts is None:
        return [None] * len(mb_list)

    forward_indices = mb_list.forward_indices
    n_mbs = len(mb_list)

    if forward_indices is None:
        # No reordering -- just split evenly
        bs = routed_experts.shape[0]
        chunk = bs // n_mbs
        result = [routed_experts[i * chunk : (i + 1) * chunk] for i in range(n_mbs)]
        logger.debug(
            "[R3] _split_routed_experts_for_mbs: no forward_indices, "
            "split %d samples evenly into %d chunks of %d.",
            bs, n_mbs, chunk,
        )
        return result

    # Reorder by forward_indices (sample-level reordering)
    reordered = routed_experts[forward_indices]

    # Determine number of samples per micro-batch from mbs dicts.
    result = []
    offset = 0
    for i, mb_dict in enumerate(mb_list.mbs):
        n_samples = _infer_mb_sample_count(mb_dict, routed_experts.shape[0], n_mbs)
        result.append(reordered[offset : offset + n_samples])
        offset += n_samples

    logger.debug(
        "[R3] _split_routed_experts_for_mbs: split %d samples into %d mbs "
        "with sizes %s (forward_indices len=%d).",
        routed_experts.shape[0],
        n_mbs,
        [r.shape[0] for r in result],
        len(forward_indices),
    )
    return result


# ===================================================================
# Per-MB attention_mask extraction (Problem 5 fix)
# ===================================================================


def _get_attention_mask_for_mb(mb_item) -> torch.Tensor | None:
    """Extract or reconstruct ``attention_mask`` from a ``MicroBatchItem``.

    After ``pack_tensor_dict``, both ``orig_mb`` and ``padded_mb`` have
    ``cu_seqlens`` instead of ``attention_mask``.  We reconstruct the mask
    from ``cu_seqlens`` + ``max_seqlen`` in the padded_mb (which reflects
    the actual padded sequence length used by the model).

    Falls back to ``attention_mask`` if still present (e.g. tree training).
    """
    # Try padded_mb first (has the actual padded dimensions)
    if hasattr(mb_item, "padded_mb") and isinstance(mb_item.padded_mb, dict):
        # Direct attention_mask
        attn = mb_item.padded_mb.get("attention_mask")
        if attn is not None:
            return attn
        # Reconstruct from cu_seqlens
        cu = mb_item.padded_mb.get("cu_seqlens")
        max_sl = mb_item.padded_mb.get("max_seqlen")
        if cu is not None and max_sl is not None:
            return _reconstruct_attention_mask_from_cu_seqlens(cu, int(max_sl))

    # Try orig_mb as fallback
    if hasattr(mb_item, "orig_mb") and isinstance(mb_item.orig_mb, dict):
        attn = mb_item.orig_mb.get("attention_mask")
        if attn is not None:
            return attn
        cu = mb_item.orig_mb.get("cu_seqlens")
        max_sl = mb_item.orig_mb.get("max_seqlen")
        if cu is not None and max_sl is not None:
            return _reconstruct_attention_mask_from_cu_seqlens(cu, int(max_sl))

    return None


# ===================================================================
# Wrapped forward_backward_batch
# ===================================================================


def _r3_forward_backward_batch(
    self,
    mb_list,
    process_output_fn: Callable[
        [torch.Tensor, dict[str, Any]], torch.Tensor | None
    ],
    forward_only: bool = False,
) -> None:
    """Drop-in replacement for ``MegatronEngine.forward_backward_batch``
    that injects R3 replay setup around each micro-batch.

    If the data does not contain ``routed_experts``, delegates directly
    to the original method with zero overhead.
    """
    from areal.engine.router_replay_patch import RouterReplay
    from areal.engine.router_replay_utils import (
        clear_router_replay,
        setup_per_microbatch_replay_forward,
    )

    # ------------------------------------------------------------------
    # 1. Extract routed_experts from the batch data and split per-MB.
    #    routed_experts is (bs, max_seqlen, num_moe_layers, topk) and
    #    does NOT get split by split_padded_tensor_dict_into_mb_list.
    # ------------------------------------------------------------------
    routed_experts_batch = None
    if hasattr(mb_list, "data") and isinstance(mb_list.data, dict):
        routed_experts_batch = mb_list.data.pop("routed_experts", None)

    # Also clean from mbs and padded_mbs to avoid confusing downstream code.
    # Problem 1: these would contain the un-split full tensor via not_to_split broadcast.
    for mb_dict in mb_list.mbs:
        if isinstance(mb_dict, dict):
            mb_dict.pop("routed_experts", None)
    if mb_list.padded_mbs is not None:
        for mb_dict in mb_list.padded_mbs:
            if isinstance(mb_dict, dict):
                mb_dict.pop("routed_experts", None)

    if routed_experts_batch is None:
        logger.debug(
            "[R3] No routed_experts in batch data; using original "
            "forward_backward_batch."
        )
        return self._r3_original_forward_backward_batch(
            mb_list, process_output_fn, forward_only=forward_only
        )

    logger.info(
        "[R3] R3 forward_backward: %d micro-batches, routed_experts shape=%s, "
        "forward_only=%s",
        len(mb_list),
        routed_experts_batch.shape,
        forward_only,
    )

    # Split routed_experts per micro-batch
    per_mb_routed_experts = _split_routed_experts_for_mbs(
        routed_experts_batch, mb_list
    )

    # ------------------------------------------------------------------
    # 2. Store R3 data on the engine for the wrapped iterator.
    # ------------------------------------------------------------------
    self._r3_per_mb_experts = per_mb_routed_experts
    self._r3_mb_counter = 0
    model_config = self.tf_config

    # ------------------------------------------------------------------
    # 2b. Set PP size on RouterReplay for backward recompute remapping.
    #     Reset call counters so each forward_backward_batch starts clean.
    # ------------------------------------------------------------------
    try:
        from megatron.core import parallel_state as mpu
        pp_size = mpu.get_pipeline_model_parallel_world_size()
    except Exception:
        pp_size = getattr(model_config, "pipeline_model_parallel_size", 1)
    RouterReplay.pp_size = pp_size
    RouterReplay.reset_all_call_counters()
    logger.debug(
        "[R3] Set RouterReplay.pp_size=%d, reset all call counters "
        "(%d instances).",
        pp_size,
        len(RouterReplay.router_instances),
    )

    # ------------------------------------------------------------------
    # 3. Wrap the MicroBatchList iterator on the INSTANCE level
    #
    #    The iterator injects R3 setup before each micro-batch's forward.
    #    The call-counter mechanism in RouterReplay handles backward
    #    recompute correctly for both PP=1 and PP>1 scenarios:
    #
    #    - PP=1 (forward_backward_no_pipelining): all forwards first,
    #      then all backwards in REVERSE order. The call counter tracks
    #      which MB's data to use: forward calls [0..N-1], backward
    #      recompute calls [N..2N-1] remapped as [N-1..0].
    #
    #    - PP>1 (1F1B): interleaved forward-backward. Each MB's backward
    #      follows its forward closely. The call counter handles this
    #      with same-order remapping.
    # ------------------------------------------------------------------
    engine_ref = self

    class _R3MicroBatchIterator:
        """Wraps the micro-batch iterator to inject R3 setup."""

        def __init__(self, base_iter):
            self._base = base_iter

        def __iter__(self):
            return self

        def __next__(self):
            mb_item = next(self._base)

            idx = engine_ref._r3_mb_counter
            engine_ref._r3_mb_counter += 1
            re = (
                engine_ref._r3_per_mb_experts[idx]
                if idx < len(engine_ref._r3_per_mb_experts)
                else None
            )

            if re is not None:
                # Problem 5 fix: reconstruct attention_mask from cu_seqlens
                # when pack_tensor_dict has replaced it.
                attn_mask = _get_attention_mask_for_mb(mb_item)

                if attn_mask is not None:
                    # Truncate `re` sequence length to match `attn_mask`
                    # Since data is right-padded (real tokens are on the left),
                    # slicing the first max_seqlen elements safely removes extra padding.
                    # This prevents a shape mismatch when the micro-batch has a smaller max_seqlen
                    # than the full batch (which occurs when sequences are packed).
                    re_matched = re[:, :attn_mask.shape[1], ...]

                    try:
                        setup_per_microbatch_replay_forward(
                            re_matched.to(attn_mask.device),
                            attn_mask,
                            model_config,
                        )
                        logger.debug(
                            "[R3] Replay setup OK for micro-batch %d: "
                            "routed_experts=%s, attn_mask=%s.",
                            idx, re_matched.shape, attn_mask.shape,
                        )
                    except Exception:
                        logger.warning(
                            "[R3] Failed to setup replay for micro-batch %d.",
                            idx,
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        "[R3] Cannot find or reconstruct attention_mask for "
                        "micro-batch %d; skipping replay setup. "
                        "Keys in orig_mb: %s, keys in padded_mb: %s.",
                        idx,
                        list(mb_item.orig_mb.keys()) if hasattr(mb_item, "orig_mb") and isinstance(mb_item.orig_mb, dict) else "N/A",
                        list(mb_item.padded_mb.keys()) if hasattr(mb_item, "padded_mb") and isinstance(mb_item.padded_mb, dict) else "N/A",
                    )
            return mb_item

    original_class_iter = mb_list.__class__.__iter__

    def _r3_iter(mb_list_self):
        return _R3MicroBatchIterator(original_class_iter(mb_list_self))

    mb_list.__class__.__iter__ = _r3_iter

    try:
        # Megatron's forward_backward_func (e.g. 1F1B schedule) internally
        # interleaves forward and backward for each micro-batch.
        #
        # The call-counter mechanism in RouterReplay handles backward
        # recompute (activation checkpointing) correctly:
        # - Each RouterReplay instance tracks how many times its routing
        #   function is called via routing_call_count.
        # - Forward calls (call_idx < N) use routing_data_list[call_idx].
        # - Backward recompute calls (call_idx >= N) are remapped based
        #   on pp_size:
        #   * PP=1: reverse order (N-1-recompute_idx)
        #   * PP>1: same order (recompute_idx)
        #
        # This replaces the previous design that relied on target_topk_idx
        # (which got overwritten by subsequent MBs in PP=1 multi-MB case).
        self._r3_original_forward_backward_batch(
            mb_list, process_output_fn, forward_only=forward_only
        )
    finally:
        # Restore original class __iter__ and clean up R3 state
        mb_list.__class__.__iter__ = original_class_iter
        clear_router_replay()
        self._r3_per_mb_experts = None
        self._r3_mb_counter = 0
        logger.debug("[R3] forward_backward_batch cleanup complete.")
