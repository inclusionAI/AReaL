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

Ref some code from megatron or verl, adapted for AReaL.
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
    engine._r3_pending_routed_experts = None

    # Bind the wrapped method
    engine.forward_backward_batch = types.MethodType(
        _r3_forward_backward_batch, engine
    )

    logger.info("[R3] MegatronEngine patched successfully.")


# ===================================================================
# routed_experts alignment (right-padded rollout → left-aligned training)
# ===================================================================


def _align_routed_experts_to_mask(
    routed_experts: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Align ``routed_experts`` from right-padded rollout format to left-aligned
    training format, matching the token layout implied by ``cu_seqlens``.

    **Rollout format**: ``routed_experts`` is ``(bs, batch_max_seqlen, L, K)``
    with RIGHT padding (real tokens at the BEGINNING of each row, after
    batch-level concatenation; zeros are appended at the end).

    **Training format**: After ``pack_tensor_dict``, tokens are LEFT-aligned
    (real tokens first).  The ``cu_seqlens`` tells us each sample's actual
    length.

    This function extracts the first ``actual_len`` tokens from each
    sample in ``routed_experts`` and produces a ``(bs_aligned, max_seqlen, L, K)``
    tensor with real tokens at the LEFT (matching training convention).

    If ``cu_seqlens`` has more entries than ``routed_experts`` has rows
    (because ``pad_packed_tensor_dict`` appended a dummy padding sequence),
    the output is zero-padded along the batch dimension.

    Args:
        routed_experts: ``(bs, batch_max_seqlen, num_moe_layers, topk)``
        cu_seqlens: ``(n_seqs+1,)`` cumulative sequence lengths.
        max_seqlen: Maximum sequence length (from ``padded_mb["max_seqlen"]``).

    Returns:
        ``(n_seqs, max_seqlen, num_moe_layers, topk)`` aligned tensor.
    """
    re_bs, re_seqlen = routed_experts.shape[:2]
    extra_dims = routed_experts.shape[2:]  # (num_moe_layers, topk)
    n_seqs = cu_seqlens.shape[0] - 1
    seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()

    # Output: (n_seqs, max_seqlen, L, K) with real tokens left-aligned
    aligned = torch.zeros(
        n_seqs, max_seqlen, *extra_dims,
        dtype=routed_experts.dtype,
        device=routed_experts.device,
    )

    for i in range(min(n_seqs, re_bs)):
        actual_len = seq_lens[i]
        if actual_len <= 0:
            continue
        # Source: first actual_len tokens from right-padded routed_experts
        n = min(actual_len, re_seqlen, max_seqlen)
        aligned[i, :n] = routed_experts[i, :n]

    logger.debug(
        "[R3] _align_routed_experts_to_mask: re_shape=%s -> aligned_shape=%s, "
        "n_seqs=%d (re_bs=%d), seq_lens=%s.",
        routed_experts.shape, aligned.shape, n_seqs, re_bs, seq_lens[:8],
    )
    return aligned


# ===================================================================
# routed_experts splitting (robust sample-count inference)
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
# Per-MB cu_seqlens extraction
# ===================================================================


def _get_cu_seqlens_for_mb(mb_item) -> tuple[torch.Tensor, int] | None:
    """Extract ``cu_seqlens`` and ``max_seqlen`` from a ``MicroBatchItem``.

    Prefers ``padded_mb`` (which has the actual TP-aligned dimensions used
    by the model) over ``orig_mb``.

    Returns:
        ``(cu_seqlens, max_seqlen)`` or ``None`` if not available.
    """
    # Try padded_mb first (has TP-aligned cu_seqlens -- this is what the model sees)
    if hasattr(mb_item, "padded_mb") and isinstance(mb_item.padded_mb, dict):
        cu = mb_item.padded_mb.get("cu_seqlens")
        max_sl = mb_item.padded_mb.get("max_seqlen")
        if cu is not None and max_sl is not None:
            return cu, int(max_sl)

    # Try orig_mb as fallback (pre-padding cu_seqlens)
    if hasattr(mb_item, "orig_mb") and isinstance(mb_item.orig_mb, dict):
        cu = mb_item.orig_mb.get("cu_seqlens")
        max_sl = mb_item.orig_mb.get("max_seqlen")
        if cu is not None and max_sl is not None:
            return cu, int(max_sl)

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

    **CRITICAL FIX**: Uses ``cu_seqlens`` from the padded micro-batch
    (with per-sequence TP alignment) for packing replay data, ensuring
    token ordering matches exactly what Megatron's transformer layers see.
    """
    from areal.engine.router_replay_patch import RouterReplay, RouterReplayAction
    from areal.engine.router_replay_utils import (
        RouterReplayHelper,
        clear_router_replay,
        setup_per_microbatch_replay_forward,
    )

    # ------------------------------------------------------------------
    # 1. Retrieve routed_experts.
    # ------------------------------------------------------------------
    routed_experts_batch = None
    _from_side_channel = False

    # Strategy A: Side-channel (preferred path)
    if hasattr(self, '_r3_pending_routed_experts') and self._r3_pending_routed_experts is not None:
        routed_experts_batch = self._r3_pending_routed_experts
        self._r3_pending_routed_experts = None  # Consume it
        _from_side_channel = True
        logger.info(
            "[R3] Retrieved routed_experts from engine side-channel: shape=%s.",
            routed_experts_batch.shape,
        )

    # Strategy B: Legacy path from mb_list.data (backward compatibility)
    if routed_experts_batch is None and not forward_only:
        if hasattr(mb_list, "data") and isinstance(mb_list.data, dict):
            routed_experts_batch = mb_list.data.pop("routed_experts", None)
            if routed_experts_batch is not None:
                logger.info(
                    "[R3] Retrieved routed_experts from mb_list.data (legacy path): "
                    "shape=%s.",
                    routed_experts_batch.shape,
                )

    # Clean from mbs and padded_mbs to avoid confusing downstream code.
    for mb_dict in mb_list.mbs:
        if isinstance(mb_dict, dict):
            mb_dict.pop("routed_experts", None)
    if mb_list.padded_mbs is not None:
        for mb_dict in mb_list.padded_mbs:
            if isinstance(mb_dict, dict):
                mb_dict.pop("routed_experts", None)
    if hasattr(mb_list, "data") and isinstance(mb_list.data, dict):
        mb_list.data.pop("routed_experts", None)

    if routed_experts_batch is None:
        if forward_only:
            logger.debug(
                "[R3] forward_only=True and no side-channel routed_experts; "
                "skipping R3 replay (compute_logp/eval path)."
            )
        else:
            logger.debug(
                "[R3] No routed_experts found (neither side-channel nor mb_list.data); "
                "using original forward_backward_batch."
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

    # Compute seq_align_to (same as what _prepare_mb_list uses)
    from megatron.core import parallel_state as mpu
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = getattr(mpu, "get_context_parallel_world_size", lambda: 1)()
    seq_align_to = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    # ------------------------------------------------------------------
    # 2b. Set initial replay action to REPLAY_FORWARD.
    # ------------------------------------------------------------------
    RouterReplay.reset_agreement_stats()
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    logger.debug(
        "[R3] Set initial REPLAY_FORWARD action on %d router instances.",
        len(RouterReplay.router_instances),
    )

    # ------------------------------------------------------------------
    # 3. Wrap the MicroBatchList iterator
    # ------------------------------------------------------------------
    engine_ref = self
    _seq_align_to = seq_align_to

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

            # When backward recompute finishes and next forward starts,
            # switch back to REPLAY_FORWARD.
            if RouterReplayHelper.is_replay_backward_action(model_config):
                router_list = RouterReplayHelper.get_micro_batch_router_list(
                    model_config
                )
                for router in router_list:
                    router.set_router_replay_action(
                        RouterReplayAction.REPLAY_FORWARD
                    )

            if re is not None:
                # Extract cu_seqlens from padded_mb (TP-aligned, what the model sees)
                cu_info = _get_cu_seqlens_for_mb(mb_item)

                if cu_info is not None:
                    cu_seqlens, max_seqlen = cu_info
                    try:
                        # Use cu_seqlens for alignment instead of
                        # attention_mask. This ensures the packed token order
                        # matches Megatron's actual forward pass.

                        # First, get the ORIGINAL (pre-TP-alignment) cu_seqlens
                        # to know each sample's actual token count for
                        # extracting from routed_experts.
                        orig_cu = None
                        if hasattr(mb_item, "old_cu_seqlens") and mb_item.old_cu_seqlens is not None:
                            orig_cu = mb_item.old_cu_seqlens
                        elif hasattr(mb_item, "orig_mb") and isinstance(mb_item.orig_mb, dict):
                            orig_cu = mb_item.orig_mb.get("cu_seqlens")

                        if orig_cu is None:
                            # Fallback: use padded cu_seqlens directly
                            orig_cu = cu_seqlens

                        # Align routed_experts from left-padded to left-aligned
                        # using the ORIGINAL cu_seqlens (actual token counts).
                        aligned_re = _align_routed_experts_to_mask(
                            re, orig_cu, max_seqlen,
                        )

                        # Pass the PADDED cu_seqlens (with TP alignment)
                        # to set_router_replay_data so packing matches Megatron.
                        setup_per_microbatch_replay_forward(
                            aligned_re.to(cu_seqlens.device),
                            cu_seqlens,
                            model_config,
                            seq_align_to=_seq_align_to,
                        )
                        logger.debug(
                            "[R3] Replay setup OK for micro-batch %d: "
                            "original_re=%s, aligned_re=%s, cu_seqlens=%s "
                            "(seq_align_to=%d).",
                            idx, re.shape, aligned_re.shape, cu_seqlens.shape,
                            _seq_align_to,
                        )
                    except Exception:
                        logger.warning(
                            "[R3] Failed to setup replay for micro-batch %d.",
                            idx,
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        "[R3] Cannot find cu_seqlens for "
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

    # ------------------------------------------------------------------
    # 4. Register a forward hook for REPLAY_FORWARD -> REPLAY_BACKWARD toggle.
    # ------------------------------------------------------------------
    hook_handles = []

    def _r3_post_forward_hook(module, input, output):
        """Switch from REPLAY_FORWARD to REPLAY_BACKWARD after model forward."""
        if RouterReplayHelper.is_replay_forward_action(model_config):
            router_list = RouterReplayHelper.get_micro_batch_router_list(
                model_config
            )
            for router in router_list:
                router.set_router_replay_action(
                    RouterReplayAction.REPLAY_BACKWARD
                )

    for model_chunk in self.model:
        handle = model_chunk.register_forward_hook(_r3_post_forward_hook)
        hook_handles.append(handle)

    logger.debug(
        "[R3] Registered forward hooks on %d model chunks for "
        "FORWARD->BACKWARD toggle.",
        len(hook_handles),
    )

    try:
        self._r3_original_forward_backward_batch(
            mb_list, process_output_fn, forward_only=forward_only
        )
    finally:
        # Remove forward hooks
        for handle in hook_handles:
            handle.remove()
        # Restore original class __iter__ and clean up R3 state
        mb_list.__class__.__iter__ = original_class_iter

        # Harvest agreement stats BEFORE clearing replay state.
        _agreement = RouterReplay.harvest_agreement_stats()
        self._r3_last_agreement_stats = _agreement
        if _agreement.get("n_samples", 0) > 0:
            from areal.utils import stats_tracker
            with stats_tracker.scope("r3"):
                stats_tracker.scalar(
                    router_agreement_rate=_agreement["avg"],
                    router_agreement_rate_min=_agreement["min"],
                    router_agreement_rate_max=_agreement["max"],
                    router_agreement_n_samples=_agreement["n_samples"],
                    router_agreement_n_calls=_agreement["n_calls"],
                )

        clear_router_replay()
        self._r3_per_mb_experts = None
        self._r3_mb_counter = 0
        logger.debug("[R3] forward_backward_batch cleanup complete.")
