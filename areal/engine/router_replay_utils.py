"""
Router Replay Utilities for AReaL.

Handles the complete shape-transformation pipeline that converts rollout
routing indices into the layout expected by Megatron-Core's RouterReplay:

1. **Right-padding to left-alignment** -- rollout batch is right-padded;
   training uses left-aligned packed format.
2. **TP/SP splitting** -- sequence parallelism across tensor-model-parallel ranks.
3. **PP layer slicing** -- pipeline parallelism assigns different layers to ranks.
4. **Dense/MoE layer mapping** -- architectures with dense FFN layers before MoE.
"""

from __future__ import annotations

import inspect
import os
from typing import Optional

import torch

from areal.engine.router_replay_patch import RouterReplay, RouterReplayAction

from areal.utils import logging

# NOTE: use areal.utils.logging.getLogger with a stable registered
# name so the logger survives the dictConfig(disable_existing_loggers=True) re-init path.
logger = logging.getLogger("R3/utils")


# ===================================================================
# R3 detailed-logging helpers
# ===================================================================
# These helpers are used by EVERY R3 file (this module, router_replay_patch,
# megatron_engine_r3_patch, actor_r3_patch, actor.py, rlvr_r3_patch) so that
# all stages of the pipeline produce fingerprints in a consistent format.
#
# Controls (all opt-in via env vars so prod perf is not affected unless you
# deliberately enable):
#
#   AREAL_R3_VERBOSE=1              -- master switch; enables everything below.
#                                      Default: 1 (ON) so that if someone cares
#                                      to run with R3 and grep logs, they do
#                                      not need to set anything extra.
#   AREAL_R3_LOG_FIRST_N=30         -- for rate-limited hot paths, always log
#                                      the first N calls per key.
#   AREAL_R3_LOG_EVERY=100          -- after the first N, log every Nth call.
#   AREAL_R3_TENSOR_SAMPLE=8        -- how many leading elements to include in
#                                      a tensor signature.
#   AREAL_R3_ROUTER_LAYER_LIMIT=3   -- in patched_topk_routing, only print
#                                      per-layer details for up to the first
#                                      K routing calls of each type per step
#                                      (layer idx is approximated via a
#                                      per-action counter).
# ===================================================================


def _r3_verbose() -> bool:
    return os.environ.get("AREAL_R3_VERBOSE", "1") != "0"


_R3_LOG_CALL_COUNTS: dict[str, int] = {}
_R3_LOG_FIRST_N = int(os.environ.get("AREAL_R3_LOG_FIRST_N", "30"))
_R3_LOG_EVERY = int(os.environ.get("AREAL_R3_LOG_EVERY", "100"))
_R3_TENSOR_SAMPLE = int(os.environ.get("AREAL_R3_TENSOR_SAMPLE", "8"))
_R3_ROUTER_LAYER_LIMIT = int(os.environ.get("AREAL_R3_ROUTER_LAYER_LIMIT", "3"))


def _r3_should_log(key: str) -> bool:
    """Rate-limited logging gate. Returns True for the first
    ``AREAL_R3_LOG_FIRST_N`` calls against ``key``, then True once every
    ``AREAL_R3_LOG_EVERY`` calls thereafter. Monotonic per-process counter.
    """
    if not _r3_verbose():
        return False
    n = _R3_LOG_CALL_COUNTS.get(key, 0) + 1
    _R3_LOG_CALL_COUNTS[key] = n
    if n <= _R3_LOG_FIRST_N:
        return True
    return (n % max(_R3_LOG_EVERY, 1)) == 0


def _r3_call_count(key: str) -> int:
    return _R3_LOG_CALL_COUNTS.get(key, 0)


def _r3_tensor_sig(name: str, t, *, max_sample: int | None = None) -> str:
    """Compact human-readable fingerprint for a tensor or numpy array.

    Intentionally cheap: performs ONE ``.detach().cpu()`` copy and one
    reduction so it is safe to call from hot paths (still, prefer to gate
    via ``_r3_should_log``).
    """
    if t is None:
        return f"{name}=None"
    sample_n = _R3_TENSOR_SAMPLE if max_sample is None else max_sample
    try:
        if isinstance(t, torch.Tensor):
            tc = t.detach()
            if tc.device.type != "cpu":
                tc = tc.to("cpu", non_blocking=False)
            flat = tc.reshape(-1)
            total = int(flat.numel())
            if total == 0:
                return f"{name}(shape={tuple(t.shape)}, dtype={t.dtype}, empty)"
            nnz = int((flat != 0).sum().item())
            if tc.dtype in (
                torch.float16,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ):
                checksum = float(flat.float().double().sum().item())
                maxv = float(flat.float().abs().max().item())
                sample = [round(v, 6) for v in flat[:sample_n].float().tolist()]
            else:
                checksum = int(flat.long().sum().item())
                maxv = int(flat.long().abs().max().item())
                sample = flat[:sample_n].tolist()
            return (
                f"{name}(shape={tuple(t.shape)}, dtype={t.dtype}, "
                f"device={t.device}, nnz={nnz}/{total}, "
                f"sum={checksum}, |max|={maxv}, first{len(sample)}={sample})"
            )
        # numpy or generic array-like
        if hasattr(t, "shape") and hasattr(t, "dtype"):
            import numpy as np

            arr = t if isinstance(t, np.ndarray) else np.asarray(t)
            flat = arr.reshape(-1)
            total = int(flat.size)
            if total == 0:
                return f"{name}(shape={arr.shape}, dtype={arr.dtype}, empty, numpy)"
            nnz = int((flat != 0).sum())
            checksum = int(flat.astype("int64").sum()) if np.issubdtype(
                arr.dtype, np.integer
            ) else float(flat.astype("float64").sum())
            maxv = (
                int(np.abs(flat).max())
                if np.issubdtype(arr.dtype, np.integer)
                else float(np.abs(flat).max())
            )
            sample = flat[:sample_n].tolist()
            return (
                f"{name}(shape={tuple(arr.shape)}, dtype={arr.dtype}, numpy, "
                f"nnz={nnz}/{total}, sum={checksum}, |max|={maxv}, "
                f"first{len(sample)}={sample})"
            )
    except Exception as e:  # pragma: no cover - diagnostic helper must not raise
        return f"{name}=<sig-err:{type(e).__name__}:{e}>"
    return f"{name}={type(t).__name__}"


def _r3_zero_row_stats(top_indices: torch.Tensor) -> str:
    """Returns a string describing the count of all-zero rows in a
    ``(num_tokens, topk)`` target_topk_idx tensor. All-zero rows are the
    smoking gun for zero-fill hazard.
    """
    if top_indices is None or top_indices.ndim < 2:
        return "zero_row_stats=N/A"
    try:
        with torch.no_grad():
            zero_rows = (top_indices == 0).all(dim=-1)
            total = int(zero_rows.numel())
            z = int(zero_rows.sum().item())
        return f"zero_rows={z}/{total} ({100.0*z/max(total,1):.2f}%)"
    except Exception as e:
        return f"zero_row_stats=<err:{e}>"


def _r3_pp_tp_info(tf_config=None, vp_rank=None) -> str:
    """Short PP/TP/DP/SP/EP context string for log lines."""
    try:
        from megatron.core import parallel_state as mpu

        pp = mpu.get_pipeline_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        tp = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        cp = getattr(mpu, "get_context_parallel_world_size", lambda: 1)()
        dp = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        return (
            f"pp={pp_rank}/{pp} tp={tp_rank}/{tp} cp={cp} dp={dp_rank}/{dp}"
            + (f" vp={vp_rank}" if vp_rank is not None else "")
        )
    except Exception:
        return "pp=?/? tp=?/?"


# ===================================================================
# Root-cause hunting helpers (v2 — per-sample & per-mb fingerprints)
# ===================================================================
# We need to pinpoint whether the R3 replay indices that reach the router
# are the SAME bytes as those the rollout engine produced for the SAME
# sample.  The cheapest reliable way to do this is a per-sample 64-bit
# fold-hash of the int32 tensor bytes.  The hash is stable across device
# (we move to CPU once), preserves per-sample order, and survives
# reordering (each sample is hashed independently — we can then check
# any permutation via the multiset of hashes).
#
# We also expose a monotonically increasing trace-id that the actor
# increments every time it sets ``engine._r3_pending_routed_experts``
# so each end-to-end replay can be correlated across STAGE2 → STAGE3 →
# STAGE4 log lines.
# ===================================================================


# Global monotonically increasing trace-id.  Incremented at the side-channel
# SET site (actor._compute_logp / actor._ppo_update).  Read back at the
# CONSUMPTION site in ``_r3_forward_backward_batch``.  Exported via an
# env-independent module-level function so *all* stages print the same id.
_R3_TRACE_ID: int = 0


def _r3_next_trace_id() -> int:
    """Reserve & return a new trace-id.

    A trace-id identifies one SIDE_CHANNEL-SET -> CONSUME -> REPLAY cycle.
    """
    global _R3_TRACE_ID
    _R3_TRACE_ID += 1
    return _R3_TRACE_ID


def _r3_current_trace_id() -> int:
    return _R3_TRACE_ID


def _r3_hash64(t) -> int:
    """Return a stable 64-bit hash of a tensor/ndarray's int32 bytes.

    For a ``(bs, seqlen, L, K)`` routed_experts tensor this is cheap
    (one CPU copy) and deterministic regardless of dtype conversion.
    Returns 0 for ``None``.
    """
    if t is None:
        return 0
    try:
        if isinstance(t, torch.Tensor):
            tc = t.detach()
            if tc.device.type != "cpu":
                tc = tc.to("cpu", non_blocking=False)
            arr = tc.to(torch.int32).contiguous().numpy()
        else:
            import numpy as np

            arr = (t if isinstance(t, np.ndarray) else np.asarray(t)).astype("int32", copy=False)
        import hashlib

        return int.from_bytes(
            hashlib.blake2b(arr.tobytes(), digest_size=8).digest(),
            "big",
            signed=False,
        )
    except Exception:
        return -1


def _r3_per_sample_hashes(t, max_rows: int = 64) -> list[int]:
    """Return per-sample 64-bit hashes.

    For a 4D ``(bs, seqlen, L, K)`` tensor, returns one hash per sample
    (dim-0).  For 3D packed ``(total_aligned, L, K)`` returns one hash
    per row (capped at ``max_rows`` to keep log size sane).
    """
    if t is None:
        return []
    try:
        if isinstance(t, torch.Tensor):
            tc = t.detach()
            if tc.device.type != "cpu":
                tc = tc.to("cpu", non_blocking=False)
            arr = tc.to(torch.int32).contiguous().numpy()
        else:
            import numpy as np

            arr = (t if isinstance(t, np.ndarray) else np.asarray(t)).astype("int32", copy=False)
        import hashlib

        out = []
        for i in range(min(arr.shape[0], max_rows)):
            b = arr[i].tobytes()
            out.append(
                int.from_bytes(
                    hashlib.blake2b(b, digest_size=8).digest()[:4],
                    "big",
                    signed=False,
                )
            )
        return out
    except Exception:
        return [-1]


def _r3_per_sample_nnz(t, max_rows: int = 64) -> list[int]:
    """Return per-sample non-zero counts (rows where any expert id != 0)."""
    if t is None:
        return []
    try:
        if isinstance(t, torch.Tensor):
            tc = t.detach()
            if tc.device.type != "cpu":
                tc = tc.to("cpu", non_blocking=False)
            arr = tc.to(torch.int32).contiguous().numpy()
        else:
            import numpy as np

            arr = (t if isinstance(t, np.ndarray) else np.asarray(t)).astype("int32", copy=False)
        out = []
        for i in range(min(arr.shape[0], max_rows)):
            out.append(int((arr[i] != 0).any(axis=-1).sum()))
        return out
    except Exception:
        return [-1]


def _r3_per_sample_seq_real_len(t, max_rows: int = 64) -> list[int]:
    """Return per-sample "real-looking" length = index of last non-all-zero row + 1.

    Useful for verifying that the routed_experts tensor is right-padded
    as expected: the real length should equal the sample's attention
    mask sum (= cu_seqlens diff).  If it doesn't, alignment is off.
    """
    if t is None:
        return []
    try:
        if isinstance(t, torch.Tensor):
            tc = t.detach()
            if tc.device.type != "cpu":
                tc = tc.to("cpu", non_blocking=False)
            arr = tc.to(torch.int32).contiguous().numpy()
        else:
            import numpy as np

            arr = (t if isinstance(t, np.ndarray) else np.asarray(t)).astype("int32", copy=False)
        out = []
        for i in range(min(arr.shape[0], max_rows)):
            row_any = (arr[i] != 0).any(axis=(-1, -2)) if arr[i].ndim >= 2 else (arr[i] != 0)
            nz = row_any.nonzero()[0]
            out.append(int(nz[-1]) + 1 if len(nz) else 0)
        return out
    except Exception:
        return [-1]


# ===================================================================
# Layer computation helpers
# ===================================================================


def get_num_layers_to_build(config, vp_stage=None, pp_rank=None) -> int:
    """Determine the number of transformer layers to build for the current PP stage.

    Self-contained reimplementation that does not depend on
    ``megatron.core.transformer.transformer_block.get_num_layers_to_build``
    which may not exist in all megatron-core versions.
    """
    from megatron.core import parallel_state as mpu

    if pp_rank is None:
        pp_rank = mpu.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    # Custom pipeline layout
    if (
        hasattr(config, "pipeline_model_parallel_layout")
        and config.pipeline_model_parallel_layout is not None
    ):
        try:
            from megatron.core.transformer.enums import LayerType

            return config.pipeline_model_parallel_layout.get_num_layers_to_build(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        except ImportError:
            pass

    first_stage_layers = getattr(config, "num_layers_in_first_pipeline_stage", None)
    last_stage_layers = getattr(config, "num_layers_in_last_pipeline_stage", None)

    if first_stage_layers is not None or last_stage_layers is not None:
        layers_to_distribute = config.num_layers
        pipeline_stages_left = config.pipeline_model_parallel_size

        if first_stage_layers is not None:
            layers_to_distribute -= first_stage_layers
            pipeline_stages_left -= 1
        if last_stage_layers is not None:
            layers_to_distribute -= last_stage_layers
            pipeline_stages_left -= 1

        if pipeline_stages_left > 0:
            assert layers_to_distribute % pipeline_stages_left == 0
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        if is_first_pp_stage and first_stage_layers is not None:
            num_layers_per_pipeline_rank = first_stage_layers
        if is_last_pp_stage and last_stage_layers is not None:
            num_layers_per_pipeline_rank = last_stage_layers
    else:
        num_layers = config.num_layers
        if getattr(config, "account_for_embedding_in_pipeline_split", False):
            num_layers += 1
        if getattr(config, "account_for_loss_in_pipeline_split", False):
            num_layers += 1
        assert num_layers % config.pipeline_model_parallel_size == 0
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    vp_size = config.virtual_pipeline_model_parallel_size
    if vp_size is not None and config.pipeline_model_parallel_size > 1:
        assert num_layers_per_pipeline_rank % vp_size == 0
        num_layers_to_build = num_layers_per_pipeline_rank // vp_size
    else:
        num_layers_to_build = num_layers_per_pipeline_rank

    # Account for embedding/loss layers
    if getattr(config, "account_for_embedding_in_pipeline_split", False):
        if is_first_pp_stage and (
            vp_stage is None or vp_stage == 0
        ):
            num_layers_to_build -= 1

    if getattr(config, "account_for_loss_in_pipeline_split", False):
        vp_last = (vp_size is None) or (vp_stage == vp_size - 1)
        if is_last_pp_stage and vp_last:
            num_layers_to_build -= 1

    return num_layers_to_build


def is_moe_layer(tf_config, layer_idx: int) -> bool:
    """Check whether a given global layer index is an MoE layer."""
    moe_layer_freq = getattr(tf_config, "moe_layer_freq", None)
    if moe_layer_freq is None:
        # If not set, assume all layers are MoE
        return True
    if isinstance(moe_layer_freq, int):
        return layer_idx % moe_layer_freq == 0
    elif isinstance(moe_layer_freq, list):
        return moe_layer_freq[layer_idx] == 1
    else:
        raise ValueError(f"[R3] Unsupported moe_layer_freq type: {type(moe_layer_freq)}")


def get_moe_num_layers_to_build(config, vp_stage=None, pp_rank=None) -> int:
    """Count the number of MoE layers assigned to the current rank."""
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    total_layers = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)

    sig = inspect.signature(get_transformer_layer_offset)
    kwargs = {}
    if "vp_stage" in sig.parameters and "pp_rank" in sig.parameters:
        kwargs = {"vp_stage": vp_stage, "pp_rank": pp_rank}
    elif "pp_rank" in sig.parameters:
        kwargs = {"pp_rank": pp_rank}

    layer_offset = get_transformer_layer_offset(config, **kwargs)
    return sum(
        1
        for idx in range(layer_offset, layer_offset + total_layers)
        if is_moe_layer(config, idx)
    )


def get_current_rank_layer_info(tf_config, vp_rank=None) -> dict:
    """Return ``{"start", "end", "count"}`` for the current PP rank's layer range."""
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    if vp_rank is None:
        vp_rank = 0

    num_layers = get_num_layers_to_build(tf_config, vp_stage=vp_rank)

    sig = inspect.signature(get_transformer_layer_offset)
    kwargs = {}
    if "vp_stage" in sig.parameters:
        kwargs["vp_stage"] = vp_rank

    offset = get_transformer_layer_offset(tf_config, **kwargs)
    return {"start": offset, "end": offset + num_layers, "count": num_layers}


# ===================================================================
# RouterReplayHelper
# ===================================================================


class RouterReplayHelper:
    """Helper to query router replay state and locate local RouterReplay instances."""

    @staticmethod
    def get_micro_batch_router_list(tf_config, vp_rank=None) -> list:
        """Return the RouterReplay instances for the current (pp_rank, vp_stage)."""
        vp_size = tf_config.virtual_pipeline_model_parallel_size
        if vp_size is not None:
            vp_rank = 0 if vp_rank is None else vp_rank
            offset = 0
            for pre_vp_stage in range(vp_size):
                if pre_vp_stage == vp_rank:
                    break
                offset += get_moe_num_layers_to_build(tf_config, pre_vp_stage)
        else:
            offset = 0

        num_layers = get_moe_num_layers_to_build(tf_config, vp_rank)
        return RouterReplay.router_instances[offset : offset + num_layers]

    @staticmethod
    def is_replay_forward_action(tf_config, vp_rank=None) -> bool:
        instances = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return bool(
            instances
            and instances[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD
        )

    @staticmethod
    def is_replay_backward_action(tf_config, vp_rank=None) -> bool:
        instances = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return bool(
            instances
            and instances[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD
        )




def set_router_replay_data(
    layers_topk_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    tf_config,
    vp_rank: Optional[int] = None,
    seq_align_to: Optional[int] = None,
) -> None:
    """Scatter packed router top-k indices to SP ranks and update RouterReplay instances.

    The packing steps mirror ``pad_packed_tensor_dict`` in ``areal/utils/data.py``:

    1. Use ``cu_seqlens`` to extract each sample's real tokens from the
       left-aligned ``layers_topk_idx``.
    2. Pack tokens contiguously with per-sequence TP/CP alignment padding
       (each sequence padded to a multiple of ``seq_align_to``).
    3. When ``cp_size > 1``, CP-interleave-split dim-0 to match
       ``preprocess_packed_seqs_context_parallel``.
    4. ``scatter_to_sequence_parallel_region`` to split across TP/SP ranks.
    5. Permute to ``(num_layers, local_tokens, topk)`` and distribute to
       RouterReplay instances.

    Args:
        layers_topk_idx: ``(bs, max_seq_len, num_moe_layers, topk)`` -- the
            replay data (left-aligned, real tokens first).  After
            ``_align_routed_experts_to_mask``, this matches the attention_mask
            convention where real tokens occupy the leftmost positions.
        cu_seqlens: ``(bs+1,)`` or ``(bs+1+1,)`` -- cumulative sequence
            lengths from the PADDED micro-batch (after ``pad_packed_tensor_dict``).
            These define the actual token ordering that Megatron uses.
            If the last entry is a batch-level padding sequence, it will be
            handled by including a zero-filled routing segment.
        tf_config: Megatron TransformerConfig.
        vp_rank: Virtual pipeline stage rank override.
        seq_align_to: Per-sequence TP alignment factor (typically ``tp_size``
            or ``tp_size * cp_size * 2``).  If None, defaults to TP world size.
    """
    from megatron.core import parallel_state as mpu

    with torch.no_grad():
        device = torch.cuda.current_device()
        bs_re = layers_topk_idx.shape[0]
        num_layers = layers_topk_idx.shape[2]
        topk = layers_topk_idx.shape[3]

        # Determine the number of real sequences from cu_seqlens.
        # pad_packed_tensor_dict may add one extra entry for batch-level padding.
        n_cu_entries = cu_seqlens.shape[0]
        # Number of sequences in cu_seqlens (including potential batch padding seq)
        n_seqs_in_cu = n_cu_entries - 1

        # Extract per-sequence lengths from cu_seqlens
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()

        # Determine seq_align_to if not provided
        if seq_align_to is None:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            cp_size = getattr(mpu, "get_context_parallel_world_size", lambda: 1)()
            seq_align_to = tp_size * cp_size * 2 if cp_size > 1 else tp_size

        # Compute TP-aligned lengths (matching pad_packed_tensor_dict)
        aligned_lens = []
        for slen in seq_lens:
            pad = (-slen) % seq_align_to
            aligned_lens.append(slen + pad)

        total_aligned = sum(aligned_lens)

        # Pack routed_experts using cu_seqlens-aligned layout.
        # layers_topk_idx is left-ALIGNED: real tokens at positions [0, seq_len).
        # For each sequence i, we take the first seq_lens[i] tokens and place
        # them at aligned positions, with zero-padding for TP alignment gaps.
        packed = torch.zeros(
            total_aligned, num_layers, topk,
            dtype=layers_topk_idx.dtype,
            device=layers_topk_idx.device,
        )
        # ----------------------------------------------------------------
        # build a 1-D validity mask in lock-step with ``packed``.
        # True  = real token position (safe to replay).
        # False = padding (per-seq TP alignment slack OR batch-level padding
        #         sequence OR tail rows beyond the real payload).
        #
        # After ``scatter_to_sequence_parallel_region``, this mask is sliced
        # the same way as ``packed`` so each TP rank knows which of its local
        # rows are real.  Rows with mask==False MUST NOT be forced to the
        # recorded top-k (which is [0,0,...,0] for padding)
        # ----------------------------------------------------------------
        valid_mask = torch.zeros(
            total_aligned,
            dtype=torch.bool,
            device=layers_topk_idx.device,
        )

        aligned_offset = 0
        for i in range(min(n_seqs_in_cu, bs_re)):
            slen = seq_lens[i]
            if slen <= 0:
                aligned_offset += aligned_lens[i]
                continue
            # Take first slen tokens from this sample's routed_experts
            actual_len = min(slen, layers_topk_idx.shape[1])
            packed[aligned_offset : aligned_offset + actual_len] = (
                layers_topk_idx[i, :actual_len]
            )
            # Only the real-token span is marked valid; the per-seq
            # TP-alignment slack (aligned_lens[i] - actual_len) stays False.
            valid_mask[aligned_offset : aligned_offset + actual_len] = True
            aligned_offset += aligned_lens[i]

        # For any extra sequences in cu_seqlens beyond bs_re (batch padding),
        # the packed tensor already has zeros at those positions and
        # ``valid_mask`` stays False for that entire span.
        for i in range(bs_re, n_seqs_in_cu):
            aligned_offset += aligned_lens[i]

        # ----------------------------------------------------------------
        # strike "structurally all-zero" rows from the
        # validity mask even when they fall inside ``[0, seq_len)``.
        #
        # Motivation: SGLang's async rollout does NOT record routing for the
        # last generated token of each sequence (the EOS / boundary token)
        # because its routing metadata is finalised AFTER the forward pass
        # that produces it.  That leaves exactly one all-zero tail row per
        # sequence inside the "valid" span (evidence: every rollout EXIT
        # log shows ``tail_real_row_all_zero=True zero_rows_total=1/N``).
        # Plan A already masks the per-seq TP alignment slack and batch
        # padding sequences, but those tail rows slip through: they are
        # recorded positions whose routing happens to be [0,...,0] for
        # every one of the ``L * K`` slots.
        #
        # Because Moonlight-16B has 27 MoE layers with top-6 routing to 64
        # experts and ``torch.topk`` returns distinct indices, a real token
        # producing [0,0,0,0,0,0] across all 27 layers has probability
        # essentially 0.  It is safe -- and correct -- to treat every such
        # row as a recording gap and fall back to the LIVE router top-k
        # during replay, exactly like padding rows.  This keeps the
        # forward/backward spliced indices consistent (both branches see
        # the same ``target_valid_mask``) and restores the low
        # ``mean_abs_diff`` profile on every micro-batch.
        # ----------------------------------------------------------------
        with torch.no_grad():
            row_all_zero = (
                (packed == 0).reshape(packed.shape[0], -1).all(dim=-1)
            )
            n_strike = int((valid_mask & row_all_zero).sum().item())
            valid_mask = valid_mask & (~row_all_zero)

        # Step 2: CP split (before TP scatter).
        #
        # When ``cp_size > 1``, megatron-core's
        # ``preprocess_packed_seqs_context_parallel`` has already
        # interleaved-split the model's token axis so each CP rank only sees
        # ``total_aligned / cp_size`` tokens.  Router replay indices MUST
        # match that layout before the TP scatter; otherwise each TP rank
        # would end up with ``cp_size``x too many rows and overwrite the
        # wrong positions.  We reuse ``split_packed_seqs_for_context_parallel``
        # with the PADDED ``cu_seqlens`` the caller provided (see caller
        # comment "Pass the PADDED cu_seqlens (with TP alignment) ...").
        #
        # Contract: ``cu_seqlens`` here describes the SAME packed layout as
        # ``packed`` (dim-0 aligned), which is what the engine passes in.
        packed = packed.to(device)
        valid_mask = valid_mask.to(device)
        cp_size = getattr(mpu, "get_context_parallel_world_size", lambda: 1)()
        if cp_size > 1:
            from areal.engine.megatron_utils.packed_context_parallel import (
                split_packed_seqs_for_context_parallel,
            )
            cu_seqlens_dev = cu_seqlens.to(device)
            packed = split_packed_seqs_for_context_parallel(packed, cu_seqlens_dev)
            # Preserve bool semantics: split as int32 then recast.
            valid_mask = split_packed_seqs_for_context_parallel(
                valid_mask.to(torch.int32), cu_seqlens_dev
            ).bool()

        # Step 3: Scatter to SP ranks (TP)
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
            local_tokens = scatter_to_sequence_parallel_region(packed)
            # Scatter the mask on dim-0 as well.  ``scatter_to_sequence_parallel_region``
            # expects a tensor with a sequence dimension on dim 0; promote the
            # bool mask to the same dtype as ``packed`` to keep the op's
            # collective-compat contract intact, then cast back.
            mask_buf = valid_mask.to(packed.dtype).unsqueeze(-1).unsqueeze(-1)
            local_mask = scatter_to_sequence_parallel_region(mask_buf)[..., 0, 0].bool()
        else:
            local_tokens = packed
            local_mask = valid_mask
        # local_tokens: (local_tokens_count, num_layers, topk)
        # local_mask:   (local_tokens_count,)

        # Step 4: Permute to (num_layers, local_tokens_count, topk)
        layers_topk = local_tokens.permute(1, 0, 2)

        # Step 5: Distribute to RouterReplay instances for local PP layers
        local_info = get_current_rank_layer_info(tf_config, vp_rank)
        offset, end = local_info["start"], local_info["end"]
        router_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)

        if len(router_list) == 0:
            logger.warning(
                "[R3] set_router_replay_data: no RouterReplay instances found "
                "for PP offset=%d..%d, vp_rank=%s. "
                "Total router_instances=%d.",
                offset, end, vp_rank,
                len(RouterReplay.router_instances),
            )
            return

        # Determine indexing: if dim-0 covers all layers, use absolute index;
        # otherwise (only MoE layers), use MoE-layer ordinal.
        index_by_layer = len(layers_topk) == tf_config.num_layers

        moe_idx = sum(1 for i in range(offset) if is_moe_layer(tf_config, i))

        router_offset = 0
        for layer_idx in range(offset, end):
            if not is_moe_layer(tf_config, layer_idx):
                continue
            if router_offset >= len(router_list):
                logger.warning(
                    "[R3] set_router_replay_data: router_offset=%d >= "
                    "len(router_list)=%d. Layer assignment mismatch at "
                    "layer_idx=%d.",
                    router_offset, len(router_list), layer_idx,
                )
                break
            router = router_list[router_offset]
            idx = layer_idx if index_by_layer else moe_idx
            if idx >= len(layers_topk):
                logger.warning(
                    "[R3] set_router_replay_data: layer index %d >= "
                    "layers_topk dim-0 (%d). Skipping.",
                    idx, len(layers_topk),
                )
                moe_idx += 1
                router_offset += 1
                continue
            slab = layers_topk[idx].to(torch.int64)
            router.set_target_indices(slab, valid_mask=local_mask)
            router_offset += 1
            moe_idx += 1

        logger.debug(
            "[R3] set_router_replay_data: distributed %d layers of replay data "
            "to %d/%d router instances (PP layers %d..%d, tp_size=%d).",
            router_offset,
            len(router_list),
            len(RouterReplay.router_instances),
            offset,
            end,
            tp_size,
        )


# ===================================================================
# Per-microbatch replay control
# ===================================================================


def setup_per_microbatch_replay_forward(
    routed_experts: torch.Tensor,
    cu_seqlens: torch.Tensor,
    tf_config,
    vp_rank: Optional[int] = None,
    seq_align_to: Optional[int] = None,
) -> None:
    """Set up RouterReplay for a single micro-batch's forward pass.

    Args:
        routed_experts: ``(batch, padded_seq, num_moe_layers, topk)``
            Left-aligned routing indices (real tokens first).
        cu_seqlens: ``(batch+1,)`` or ``(batch+1+1,)`` cumulative sequence
            lengths from the padded micro-batch.
        tf_config: Megatron TransformerConfig.
        vp_rank: Virtual pipeline stage rank override.
        seq_align_to: Per-sequence TP alignment factor.
    """
    routed_experts = routed_experts.to(torch.int32)
    set_router_replay_data(
        routed_experts, cu_seqlens, tf_config, vp_rank,
        seq_align_to=seq_align_to,
    )


def setup_per_microbatch_replay_backward() -> None:
    """Switch to backward replay mode for activation-checkpoint recomputation."""
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    logger.debug("[R3] Switched to backward replay mode.")


def clear_router_replay() -> None:
    """Clear all RouterReplay state after a full forward-backward pass."""
    n_instances = len(RouterReplay.router_instances)
    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_router_replay_action()
    logger.debug("[R3] Router replay state cleared (%d instances).", n_instances)


# ===================================================================
# preprocess_routed_experts_batch
# ===================================================================


def preprocess_routed_experts_batch(
    routed_experts_np,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_moe_layers: int,
    topk: int,
    compress_dtype: bool = True,
) -> torch.Tensor:
    """Convert a numpy ``routed_experts`` array from the inference engine
    into a left-padded torch tensor.

    The inference engine returns shape ``(num_tokens, num_moe_layers * topk)``
    where ``num_tokens = prompt_len + gen_len - 1`` (SGLang convention).
    We reshape to ``(1, seq_len, num_moe_layers, topk)`` with left-padding.

    Args:
        routed_experts_np: ``np.ndarray`` of shape ``(num_tokens, num_moe_layers*topk)``.
        input_ids: ``(1, seq_len)``.
        attention_mask: ``(1, seq_len)``.
        num_moe_layers: Number of MoE layers in the model. No guessing!
        topk: Router top-k value. No guessing!
        compress_dtype: Downcast to ``uint8``/``int16`` if possible.

    Returns:
        ``torch.Tensor`` of shape ``(1, seq_len, num_moe_layers, topk)``.
    """
    import numpy as np

    if routed_experts_np is None:
        return None

    seq_len = input_ids.shape[1]
    num_sgl_tokens = routed_experts_np.shape[0]
    flat_dim = routed_experts_np.shape[1]

    expected_flat = num_moe_layers * topk
    if flat_dim != expected_flat:
        logger.warning(
            "[R3] preprocess_routed_experts_batch: flat_dim=%d != "
            "num_moe_layers(%d) * topk(%d) = %d. "
            "Attempting to infer from flat_dim.",
            flat_dim,
            num_moe_layers,
            topk,
            expected_flat,
        )
        # Fallback: try to use flat_dim directly
        if flat_dim % topk == 0:
            num_moe_layers = flat_dim // topk
        elif flat_dim % num_moe_layers == 0:
            topk = flat_dim // num_moe_layers
        else:
            raise ValueError(
                f"[R3] Cannot reshape routed_experts: flat_dim={flat_dim} "
                f"is not divisible by topk={topk} or num_moe_layers={num_moe_layers}."
            )

    reshaped = routed_experts_np.reshape(num_sgl_tokens, num_moe_layers, topk)
    tensor = torch.from_numpy(reshaped.astype(np.int32))

    # Build (1, seq_len, num_moe_layers, topk) with RIGHT padding.
    real_tokens = int(attention_mask.sum().item())
    padded = torch.zeros(1, seq_len, num_moe_layers, topk, dtype=torch.int32)
    if num_sgl_tokens > real_tokens:
        # Pathological case (~2.4% of samples in observed runs):
        # SGLang returned routing for MORE tokens than the final request
        # actually carries (e.g. KV-preempt + retry, multi-turn rollout,
        # or an abandoned prefill prefix whose routing was not dropped).
        # Taking the HEAD ``tensor[:real_tokens]`` here would bind this
        # sample to UNRELATED tokens' expert decisions and cause
        # catastrophic router-replay misalignment: per-sample k3_kl jumps
        # from ~1e-4 (normal) to ~1.0, producing the "~40% normal + ~60%
        # broken" bimodal rollout-vs-train logp divergence.
        #
        # Safe behavior: disable R3 for THIS sample by leaving ``padded``
        # as all-zeros. The training-side replay path treats all-zero
        # rows as "no recorded routing" and falls back to the live router
        # (see ``valid_mask`` splicing in ``router_replay_patch.py``),
        # which makes this sample behave like an R3-off sample instead
        # of a corrupted one.
        logger.warning(
            "[R3] preprocess_routed_experts_batch: num_sgl_tokens=%d > "
            "real_tokens=%d (ratio=%.2f, seq_len=%d). This is the "
            "'double-rollout' / preempt-retry path; taking tensor[:real] "
            "here would MIS-ALIGN routing to unrelated tokens. Disabling "
            "R3 for this sample (returning all-zero routed_experts so "
            "replay falls back to live routing).",
            num_sgl_tokens,
            real_tokens,
            num_sgl_tokens / max(real_tokens, 1),
            seq_len,
        )
    else:
        n = min(num_sgl_tokens, real_tokens)
        padded[0, :n] = tensor[:n]

    if compress_dtype:
        max_val = padded.max().item()
        if max_val < 256:
            padded = padded.to(torch.uint8)
        elif max_val < 32768:
            padded = padded.to(torch.int16)

    right_pad = seq_len - real_tokens
    logger.debug(
        "[R3] preprocess_routed_experts_batch: shape=%s dtype=%s "
        "(num_moe_layers=%d, topk=%d, sgl_tokens=%d, real_tokens=%d, "
        "right_pad=%d).",
        padded.shape,
        padded.dtype,
        num_moe_layers,
        topk,
        num_sgl_tokens,
        real_tokens,
        right_pad,
    )

    return padded
