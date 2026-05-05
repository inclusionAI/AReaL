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

import types
from collections.abc import Callable
from typing import Any

import torch

from areal.utils import logging

# NOTE: use areal.utils.logging.getLogger with a stable registered
# name so the logger survives the dictConfig(disable_existing_loggers=True) re-init path.
logger = logging.getLogger("R3/megatron")


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

    logger.info("[R3] Patching MegatronEngine for Router Replay.")

    # Mark and save original
    engine._r3_enabled = True
    engine._r3_original_forward_backward_batch = engine.forward_backward_batch
    engine._r3_pending_routed_experts = None

    # ---------- R3 diagnostics: one-shot config snapshot so the NEXT
    # run's log unambiguously records the PP layout Megatron-Core
    # actually saw (num_layers, vp_size, pp_size, local offset/end,
    # router_instance count per PP rank).  This answers D2/D3/D5 in a
    # single early-startup line without polluting hot paths.
    try:
        from areal.engine.router_replay_patch import RouterReplay as _RR
        from areal.engine.router_replay_utils import (
            _r3_pp_tp_info as _ppi,
            _r3_verbose as _v,
            get_current_rank_layer_info as _info,
            is_moe_layer as _ism,
        )
        if _v():
            _tf = engine.tf_config
            _li = _info(_tf)
            _moe_list = [i for i in range(_li["start"], _li["end"]) if _ism(_tf, i)]
            _dense_list = [
                i for i in range(_li["start"], _li["end"]) if not _ism(_tf, i)
            ]
            logger.info(
                "[R3-STAGE0/patch_megatron_engine_for_r3] ENGINE_SNAPSHOT %s "
                "tf_config.num_layers=%d pp_size=%d vp_size=%s "
                "moe_layer_freq=%s first_k_dense_replace=%s "
                "local={start:%d end:%d count:%d} "
                "moe_layers_in_range=%s non_moe_layers_in_range=%s "
                "total_router_instances=%d "
                "inst_creator_ranks=%s",
                _ppi(_tf),
                _tf.num_layers,
                getattr(_tf, "pipeline_model_parallel_size", 1),
                getattr(_tf, "virtual_pipeline_model_parallel_size", None),
                getattr(_tf, "moe_layer_freq", None),
                getattr(_tf, "first_k_dense_replace", None),
                _li["start"], _li["end"], _li["count"],
                _moe_list, _dense_list,
                len(_RR.router_instances),
                [getattr(r, "creator_rank", -1) for r in _RR.router_instances],
            )
    except Exception:
        logger.exception(
            "[R3-STAGE0/patch_megatron_engine_for_r3] snapshot log failed"
        )
    # --------------------------------------------------------------

    # Bind the wrapped method
    engine.forward_backward_batch = types.MethodType(
        _r3_forward_backward_batch, engine
    )

    logger.debug("[R3] MegatronEngine patched successfully.")


# ===================================================================
# routed_experts alignment (right-padded rollout → left-aligned training)
# ===================================================================


def _align_routed_experts_to_mask(
    routed_experts: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    _r3_mb_idx: int | None = None,
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
    # Detailed alignment log: smoking-gun check for the "last generated token
    # has no routing" edge case (SGLang convention: num_sgl_tokens =
    # prompt_len + gen_len - 1). If cu_seqlens claims k real tokens but the
    # source only has k-1 non-zero rows, the k-th row here is a ZERO ROW that
    # will route to expert 0 unconditionally.
    try:
        from areal.engine.router_replay_utils import (
            _r3_pp_tp_info,
            _r3_should_log,
            _r3_tensor_sig,
            _r3_verbose,
        )

        if _r3_verbose() and _r3_should_log("_align_routed_experts_to_mask"):
            with torch.no_grad():
                per_row_zero_src = (
                    (routed_experts == 0).reshape(re_bs, re_seqlen, -1).all(dim=-1)
                )
                src_zero_rows_per_sample = per_row_zero_src.sum(dim=-1).tolist()
                per_row_zero_dst = (
                    (aligned == 0).reshape(n_seqs, max_seqlen, -1).all(dim=-1)
                )
                dst_zero_rows_per_sample = per_row_zero_dst.sum(dim=-1).tolist()
                # For each sample, locate first zero-row idx within the real-token window.
                first_zero_in_real = []
                for i in range(min(n_seqs, re_bs)):
                    L = int(seq_lens[i])
                    if L <= 0:
                        first_zero_in_real.append(-1)
                        continue
                    row = per_row_zero_src[i, :L]
                    idx = torch.nonzero(row, as_tuple=False)
                    first_zero_in_real.append(
                        int(idx[0].item()) if idx.numel() > 0 else -1
                    )
            logger.info(
                "[R3-STAGE3/_align_routed_experts_to_mask] mb=%s %s "
                "re_shape=%s aligned_shape=%s n_seqs=%d re_bs=%d "
                "seq_lens[:8]=%s src_zero_rows_per_sample[:8]=%s "
                "first_zero_in_real_window[:8]=%s "
                "dst_zero_rows_per_sample[:8]=%s | %s | %s",
                _r3_mb_idx,
                _r3_pp_tp_info(),
                tuple(routed_experts.shape),
                tuple(aligned.shape),
                n_seqs,
                re_bs,
                seq_lens[:8],
                src_zero_rows_per_sample[:8],
                first_zero_in_real[:8],
                dst_zero_rows_per_sample[:8],
                _r3_tensor_sig("src_re", routed_experts, max_sample=4),
                _r3_tensor_sig("aligned", aligned, max_sample=4),
            )
    except Exception:
        # diagnostic helper must never break the main flow
        pass
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
        reordered = routed_experts
    else:
        reordered = routed_experts[forward_indices]

    # Always derive per-micro-batch sample counts from ``mb_list.mbs`` rather
    # than assuming an even ``bs // n_mbs`` split -- the latter silently drops
    # samples when ``bs`` is not divisible by ``n_mbs``.
    result = []
    offset = 0
    for i, mb_dict in enumerate(mb_list.mbs):
        n_samples = _infer_mb_sample_count(mb_dict, routed_experts.shape[0], n_mbs)
        result.append(reordered[offset : offset + n_samples])
        offset += n_samples

    logger.debug(
        "[R3] _split_routed_experts_for_mbs: split %d samples into %d mbs "
        "with sizes %s (forward_indices=%s).",
        routed_experts.shape[0],
        n_mbs,
        [r.shape[0] for r in result],
        "None" if forward_indices is None else f"len={len(forward_indices)}",
    )
    try:
        from areal.engine.router_replay_utils import (
            _r3_hash64,
            _r3_per_sample_hashes,
            _r3_per_sample_nnz,
            _r3_per_sample_seq_real_len,
            _r3_pp_tp_info,
            _r3_should_log,
            _r3_tensor_sig,
            _r3_verbose,
        )

        if _r3_verbose() and _r3_should_log("_split_routed_experts_for_mbs"):
            pre_hash = _r3_per_sample_hashes(routed_experts, max_rows=32)
            post_hash = _r3_per_sample_hashes(reordered, max_rows=32)
            per_mb_hashes = [
                [hex(h) for h in _r3_per_sample_hashes(r, max_rows=16)]
                for r in result
            ]
            per_mb_nnz = [_r3_per_sample_nnz(r, max_rows=16) for r in result]
            per_mb_real = [_r3_per_sample_seq_real_len(r, max_rows=16) for r in result]
            logger.info(
                "[R3-STAGE3/_split_routed_experts_for_mbs] %s "
                "input_shape=%s input_hash=%s n_mbs=%d "
                "forward_indices=%s per_mb_shapes=%s per_mb_hashes=%s "
                "pre_reorder_per_sample_hash[:16]=%s "
                "post_reorder_per_sample_hash[:16]=%s "
                "per_mb_per_sample_hash=%s per_mb_per_sample_nnz=%s "
                "per_mb_per_sample_real_len=%s | %s",
                _r3_pp_tp_info(),
                tuple(routed_experts.shape),
                hex(_r3_hash64(routed_experts)),
                n_mbs,
                "None" if forward_indices is None
                else f"len={len(forward_indices)} first32={forward_indices[:32].tolist() if hasattr(forward_indices,'tolist') else list(forward_indices)[:32]}",
                [tuple(r.shape) for r in result],
                [hex(_r3_hash64(r)) for r in result],
                [hex(h) for h in pre_hash[:16]],
                [hex(h) for h in post_hash[:16]],
                per_mb_hashes,
                per_mb_nnz,
                per_mb_real,
                _r3_tensor_sig("routed_experts", routed_experts, max_sample=4),
            )
    except Exception:
        logger.exception(
            "[R3-STAGE3/_split_routed_experts_for_mbs] trace log failed"
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
    source = None
    if hasattr(mb_item, "padded_mb") and isinstance(mb_item.padded_mb, dict):
        cu = mb_item.padded_mb.get("cu_seqlens")
        max_sl = mb_item.padded_mb.get("max_seqlen")
        if cu is not None and max_sl is not None:
            source = ("padded_mb", cu, int(max_sl))

    # Try orig_mb as fallback (pre-padding cu_seqlens)
    if source is None and hasattr(mb_item, "orig_mb") and isinstance(mb_item.orig_mb, dict):
        cu = mb_item.orig_mb.get("cu_seqlens")
        max_sl = mb_item.orig_mb.get("max_seqlen")
        if cu is not None and max_sl is not None:
            source = ("orig_mb", cu, int(max_sl))

    if source is None:
        return None

    src_name, cu_out, max_sl_out = source
    try:
        from areal.engine.router_replay_utils import (
            _r3_pp_tp_info,
            _r3_should_log,
            _r3_tensor_sig,
            _r3_verbose,
        )

        if _r3_verbose() and _r3_should_log("_get_cu_seqlens_for_mb"):
            logger.info(
                "[R3-STAGE3/_get_cu_seqlens_for_mb] %s source=%s "
                "max_seqlen=%d | %s",
                _r3_pp_tp_info(),
                src_name,
                max_sl_out,
                _r3_tensor_sig("cu_seqlens", cu_out),
            )
    except Exception:
        pass
    return cu_out, max_sl_out


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
    from areal.engine.router_replay_patch import RouterReplay, RouterReplayAction
    from areal.engine.router_replay_utils import (
        RouterReplayHelper,
        _r3_pp_tp_info,
        _r3_should_log,
        _r3_tensor_sig,
        _r3_verbose,
        clear_router_replay,
        setup_per_microbatch_replay_forward,
    )

    # ------------------------------------------------------------------
    # 1. Retrieve routed_experts.
    # ------------------------------------------------------------------
    routed_experts_batch = None
    _from_side_channel = False
    _consumed_trace_id = getattr(self, "_r3_active_trace_id", None)

    # Strategy A: Side-channel (preferred path)
    if hasattr(self, '_r3_pending_routed_experts') and self._r3_pending_routed_experts is not None:
        routed_experts_batch = self._r3_pending_routed_experts
        self._r3_pending_routed_experts = None  # Consume it
        _from_side_channel = True
        try:
            from areal.engine.router_replay_utils import (
                _r3_hash64,
                _r3_per_sample_hashes,
                _r3_per_sample_nnz,
                _r3_per_sample_seq_real_len,
                _r3_pp_tp_info,
                _r3_verbose,
            )
            if _r3_verbose():
                # ---------- R3 diagnostics: D6 -- cross-PP batch-hash
                # consistency. Print FULL global hash + per-sample hashes
                # so PP rank 0 and PP rank 1 at the same trace_id can be
                # diff'd offline. If hashes differ, the side-channel
                # broadcast/scatter is wrong (root cause); if identical,
                # PP-input parity is proved and we can rule out D6.
                _full_hash = hex(_r3_hash64(routed_experts_batch))
                _all_per_sample = _r3_per_sample_hashes(
                    routed_experts_batch, max_rows=4096,
                )
                _all_per_sample_hex = [hex(h) for h in _all_per_sample]
                logger.info(
                    "[R3-STAGE3/_r3_forward_backward_batch] "
                    "SIDE_CHANNEL_CONSUME trace_id=%s %s forward_only=%s "
                    "shape=%s hash=%s per_sample_hash[:16]=%s "
                    "per_sample_nnz[:16]=%s per_sample_real_len[:16]=%s "
                    "n_samples_total=%d full_per_sample_hash=%s",
                    _consumed_trace_id,
                    _r3_pp_tp_info(),
                    forward_only,
                    routed_experts_batch.shape,
                    _full_hash,
                    _all_per_sample_hex[:16],
                    _r3_per_sample_nnz(routed_experts_batch, max_rows=16),
                    _r3_per_sample_seq_real_len(routed_experts_batch, max_rows=16),
                    len(_all_per_sample),
                    _all_per_sample_hex,
                )
        except Exception:
            logger.exception(
                "[R3-STAGE3/_r3_forward_backward_batch] "
                "SIDE_CHANNEL_CONSUME trace log failed"
            )
        logger.debug(
            "[R3] Retrieved routed_experts from engine side-channel: shape=%s.",
            routed_experts_batch.shape,
        )

    # Strategy B: Legacy path from mb_list.data (backward compatibility)
    if routed_experts_batch is None and not forward_only:
        if hasattr(mb_list, "data") and isinstance(mb_list.data, dict):
            routed_experts_batch = mb_list.data.pop("routed_experts", None)
            if routed_experts_batch is not None:
                logger.debug(
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
        "[R3] forward_backward_batch: %d micro-batches, routed_experts shape=%s, "
        "forward_only=%s",
        len(mb_list),
        routed_experts_batch.shape,
        forward_only,
    )
    if _r3_verbose() and _r3_should_log("_r3_forward_backward_batch/ENTER"):
        logger.info(
            "[R3-STAGE3/_r3_forward_backward_batch] ENTER %s "
            "n_mbs=%d forward_only=%s from_side_channel=%s "
            "has_padded_mbs=%s | %s",
            _r3_pp_tp_info(),
            len(mb_list),
            forward_only,
            _from_side_channel,
            mb_list.padded_mbs is not None,
            _r3_tensor_sig("routed_experts_batch", routed_experts_batch),
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
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

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

            if _r3_verbose() and _r3_should_log("_R3MicroBatchIterator.__next__"):
                logger.info(
                    "[R3-STAGE3/_R3MicroBatchIterator] ENTER mb_idx=%d %s "
                    "re_shape=%s has_orig_mb=%s has_padded_mb=%s "
                    "has_old_cu_seqlens=%s",
                    idx,
                    _r3_pp_tp_info(),
                    None if re is None else tuple(re.shape),
                    hasattr(mb_item, "orig_mb")
                    and isinstance(mb_item.orig_mb, dict),
                    hasattr(mb_item, "padded_mb")
                    and isinstance(mb_item.padded_mb, dict),
                    hasattr(mb_item, "old_cu_seqlens")
                    and mb_item.old_cu_seqlens is not None,
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
                if _r3_verbose() and _r3_should_log(
                    "_R3MicroBatchIterator.toggle_to_forward"
                ):
                    logger.info(
                        "[R3-STAGE3/_R3MicroBatchIterator] TOGGLE backward->forward "
                        "mb_idx=%d %s n_routers=%d",
                        idx,
                        _r3_pp_tp_info(),
                        len(router_list),
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
                        orig_cu_src = None
                        if hasattr(mb_item, "old_cu_seqlens") and mb_item.old_cu_seqlens is not None:
                            orig_cu = mb_item.old_cu_seqlens
                            orig_cu_src = "old_cu_seqlens"
                        elif hasattr(mb_item, "orig_mb") and isinstance(mb_item.orig_mb, dict):
                            orig_cu = mb_item.orig_mb.get("cu_seqlens")
                            orig_cu_src = "orig_mb.cu_seqlens"

                        if orig_cu is None:
                            # Fallback: use padded cu_seqlens directly
                            orig_cu = cu_seqlens
                            orig_cu_src = "padded_cu_seqlens (fallback)"

                        if _r3_verbose() and _r3_should_log(
                            "_R3MicroBatchIterator.pre_align"
                        ):
                            from areal.engine.router_replay_utils import (
                                _r3_hash64,
                                _r3_per_sample_hashes,
                                _r3_per_sample_nnz,
                                _r3_per_sample_seq_real_len,
                                _r3_current_trace_id,
                            )
                            logger.info(
                                "[R3-STAGE3/_R3MicroBatchIterator] PRE-ALIGN "
                                "mb_idx=%d trace_id=%d %s orig_cu_src=%s "
                                "max_seqlen=%d re_shape=%s re_hash=%s "
                                "per_sample_hash[:16]=%s per_sample_nnz[:16]=%s "
                                "per_sample_real_len[:16]=%s "
                                "orig_cu_diff[:16]=%s padded_cu_diff[:16]=%s "
                                "| %s | %s | %s",
                                idx,
                                _r3_current_trace_id(),
                                _r3_pp_tp_info(),
                                orig_cu_src,
                                max_seqlen,
                                tuple(re.shape),
                                hex(_r3_hash64(re)),
                                [hex(h) for h in _r3_per_sample_hashes(re, max_rows=16)],
                                _r3_per_sample_nnz(re, max_rows=16),
                                _r3_per_sample_seq_real_len(re, max_rows=16),
                                (orig_cu[1:] - orig_cu[:-1]).long().cpu().tolist()[:16]
                                if hasattr(orig_cu, "cpu") else "N/A",
                                (cu_seqlens[1:] - cu_seqlens[:-1]).long().cpu().tolist()[:16]
                                if hasattr(cu_seqlens, "cpu") else "N/A",
                                _r3_tensor_sig("re", re, max_sample=4),
                                _r3_tensor_sig("orig_cu", orig_cu),
                                _r3_tensor_sig("padded_cu", cu_seqlens),
                            )

                        # Align routed_experts from left-padded to left-aligned
                        # using the ORIGINAL cu_seqlens (actual token counts).
                        aligned_re = _align_routed_experts_to_mask(
                            re, orig_cu, max_seqlen, _r3_mb_idx=idx,
                        )

                        # Pass the PADDED cu_seqlens (with TP alignment)
                        # to set_router_replay_data so packing matches Megatron.
                        setup_per_microbatch_replay_forward(
                            aligned_re.to(cu_seqlens.device),
                            cu_seqlens,
                            model_config,
                            seq_align_to=_seq_align_to,
                        )
                        # ---------- R3 diagnostics: per-mb queue-depth
                        # snapshot RIGHT AFTER the dispatch finishes.
                        # Under PP=1 every router has fwd_q==1 here; under
                        # PP=2 1F1B the depth oscillates 1..PP_size.
                        try:
                            from areal.engine.router_replay_patch import (
                                RouterReplay as _RR,
                            )
                            from areal.engine.router_replay_utils import (
                                _r3_should_log as _sl2,
                                _r3_verbose as _v2,
                            )
                            if _v2() and _sl2(
                                "_R3MicroBatchIterator/post_dispatch_queue_audit"
                            ):
                                router_list = (
                                    RouterReplayHelper.get_micro_batch_router_list(
                                        model_config
                                    )
                                )
                                fwd_qs = [
                                    len(getattr(r, "replay_backward_list", []) or [])
                                    for r in router_list
                                ]
                                push_qs = [
                                    len(
                                        getattr(r, "replay_push_meta_list", []) or []
                                    )
                                    for r in router_list
                                ]
                                logger.info(
                                    "[R3-STAGE3/_R3MicroBatchIterator] "
                                    "POST_DISPATCH_QUEUE_AUDIT mb_idx=%d %s "
                                    "n_routers=%d fwd_q_lens=%s push_meta_q_lens=%s "
                                    "max_fwd_q=%d min_fwd_q=%d "
                                    "lens_locked=%s",
                                    idx,
                                    _r3_pp_tp_info(),
                                    len(router_list),
                                    fwd_qs,
                                    push_qs,
                                    max(fwd_qs) if fwd_qs else -1,
                                    min(fwd_qs) if fwd_qs else -1,
                                    fwd_qs == push_qs,
                                )
                        except Exception:
                            logger.exception(
                                "[R3-STAGE3/_R3MicroBatchIterator] "
                                "POST_DISPATCH_QUEUE_AUDIT diag log failed"
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

    # Use a per-instance class swap instead of rebinding the shared
    # ``mb_list.__class__.__iter__``. The latter is a global side effect
    # that also affects any other ``MicroBatchList`` objects alive in the
    # process (e.g. a concurrent engine). Here we create a dynamic
    # subclass whose ``__iter__`` injects the R3 setup logic, and assign
    # it only to *this* instance via ``__class__``. The original class
    # remains untouched. Restoration in the ``finally`` block merely
    # flips ``__class__`` back.
    _r3_original_mb_list_class = mb_list.__class__

    class _R3MicroBatchListProxy(_r3_original_mb_list_class):
        """Per-instance proxy that wraps __iter__ with R3 setup logic."""

        def __iter__(self_inner):
            return _R3MicroBatchIterator(
                _r3_original_mb_list_class.__iter__(self_inner)
            )

    mb_list.__class__ = _R3MicroBatchListProxy

    # ------------------------------------------------------------------
    # 4. Register a forward hook for REPLAY_FORWARD -> REPLAY_BACKWARD toggle.
    # ------------------------------------------------------------------
    hook_handles = []
    # ---------- R3 diagnostics: D8 -- track which model chunks fire
    # the post-forward hook. Under PP=2 + VP, multiple model chunks
    # share the fbfunc; if any chunk's hook misses, the action toggle
    # is skipped and backward pops see REPLAY_FORWARD, silently
    # returning live routing. A mismatch between len(self.model) and
    # hook_fire_counts[chunk_id] is a smoking gun.
    _r3_hook_fire_counts: dict[int, int] = {}
    _r3_toggle_count = {"n": 0}

    def _r3_post_forward_hook(module, input, output):
        """Switch from REPLAY_FORWARD to REPLAY_BACKWARD after model forward."""
        _chunk_id = id(module)
        _r3_hook_fire_counts[_chunk_id] = _r3_hook_fire_counts.get(_chunk_id, 0) + 1
        if RouterReplayHelper.is_replay_forward_action(model_config):
            router_list = RouterReplayHelper.get_micro_batch_router_list(
                model_config
            )
            for router in router_list:
                router.set_router_replay_action(
                    RouterReplayAction.REPLAY_BACKWARD
                )
            _r3_toggle_count["n"] += 1
            if _r3_verbose() and _r3_should_log("_r3_post_forward_hook"):
                logger.info(
                    "[R3-STAGE3/_r3_post_forward_hook] TOGGLE forward->backward "
                    "%s n_routers=%d mb_counter=%d chunk_id=%d "
                    "fire_count_this_chunk=%d total_toggles=%d "
                    "n_chunks_seen=%d",
                    _r3_pp_tp_info(),
                    len(router_list),
                    getattr(self, "_r3_mb_counter", -1),
                    _chunk_id,
                    _r3_hook_fire_counts[_chunk_id],
                    _r3_toggle_count["n"],
                    len(_r3_hook_fire_counts),
                )
        else:
            # Hook fired but action was not REPLAY_FORWARD -- this is
            # expected after the first toggle under 1F1B (subsequent mbs
            # see REPLAY_BACKWARD until the iterator flips them back).
            # We still log rarely to confirm behavior.
            if _r3_verbose() and _r3_should_log(
                "_r3_post_forward_hook/no_toggle"
            ):
                logger.info(
                    "[R3-STAGE3/_r3_post_forward_hook] NO_TOGGLE "
                    "(already backward or cleared) %s mb_counter=%d "
                    "chunk_id=%d fire_count_this_chunk=%d "
                    "n_chunks_seen=%d",
                    _r3_pp_tp_info(),
                    getattr(self, "_r3_mb_counter", -1),
                    _chunk_id,
                    _r3_hook_fire_counts[_chunk_id],
                    len(_r3_hook_fire_counts),
                )

    for model_chunk in self.model:
        handle = model_chunk.register_forward_hook(_r3_post_forward_hook)
        hook_handles.append(handle)

    # ---------- R3 diagnostics: reset FB-level aggregate counters so
    # the end-of-FB summary reflects only this call. Safe to reset
    # unconditionally: consumers read the dict inside the FB span.
    try:
        RouterReplay._r3_fb_stats = {}
    except Exception:
        pass

    try:
        self._r3_original_forward_backward_batch(
            mb_list, process_output_fn, forward_only=forward_only
        )
    finally:
        # Remove forward hooks
        for handle in hook_handles:
            handle.remove()
        # Restore the original class on this instance (undo the per-instance
        # class swap done above). The original class was never modified.
        mb_list.__class__ = _r3_original_mb_list_class

        # ---------- R3 diagnostics: END_OF_FB summary. One line per
        # forward_backward_batch call aggregating:
        #   * divergence_v1 (MATCH vs popped-vs-latest-target; under
        #     PP=2 1F1B, DIVERGE is EXPECTED — logging artifact).
        #   * divergence_v2 (MATCH vs popped-vs-own-push; under any
        #     PP layout MATCH is REQUIRED — REAL_MISMATCH is a bug).
        #   * hook fire counts per model-chunk: if unequal, one chunk
        #     missed its toggle (D8 smoking gun).
        #   * final queue residue across all routers (D9 smoking gun).
        try:
            if _r3_verbose() and _r3_should_log("_r3_forward_backward_batch/EXIT_SUMMARY"):
                from areal.engine.router_replay_patch import RouterReplay as _RR
                _fwd_q = [
                    len(getattr(r, "replay_backward_list", []) or [])
                    for r in _RR.router_instances
                ]
                _push_q = [
                    len(getattr(r, "replay_push_meta_list", []) or [])
                    for r in _RR.router_instances
                ]
                logger.info(
                    "[R3-STAGE3/_r3_forward_backward_batch] EXIT_SUMMARY %s "
                    "n_mbs=%d forward_only=%s trace_id=%s "
                    "fb_stats=%s n_model_chunks=%d hook_fire_counts=%s "
                    "total_toggles=%d residual_fwd_q=%s residual_push_q=%s "
                    "residual_max=%d",
                    _r3_pp_tp_info(),
                    len(mb_list),
                    forward_only,
                    _consumed_trace_id,
                    dict(_RR._r3_fb_stats),
                    len(self.model),
                    dict(_r3_hook_fire_counts),
                    _r3_toggle_count["n"],
                    _fwd_q,
                    _push_q,
                    max(_fwd_q) if _fwd_q else -1,
                )
        except Exception:
            logger.exception(
                "[R3-STAGE3/_r3_forward_backward_batch] "
                "EXIT_SUMMARY diag log failed"
            )

        clear_router_replay()
        self._r3_per_mb_experts = None
        self._r3_mb_counter = 0
