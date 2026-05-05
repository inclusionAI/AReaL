"""
Monkey-patches for Megatron-Core MoE components to support Router Replay (R3).

Router Replay forces the TopKRouter to use pre-recorded expert assignments
(from rollout inference) instead of computing new ones during training.
This eliminates the train/inference routing mismatch caused by weight
staleness in asynchronous RL training.

Ref some code from megatron or verl, adapted for AReaL.
"""

from __future__ import annotations

import inspect
import types
import warnings
from enum import Enum
from functools import wraps

import torch

from areal.utils import logging

# NOTE: use areal.utils.logging.getLogger with a stable registered
# name so the logger survives the dictConfig(disable_existing_loggers=True) re-init path.
logger = logging.getLogger("R3/patch")

# ---------------------------------------------------------------------------
# Optional megatron-core imports with fallback
# ---------------------------------------------------------------------------
try:
    from megatron.core.transformer.moe.moe_utils import (
        apply_router_token_dropping,
        compute_routing_scores_for_aux_loss,
    )
except ImportError:
    apply_router_token_dropping = None
    compute_routing_scores_for_aux_loss = None
    warnings.warn(
        "[R3] Could not import apply_router_token_dropping / "
        "compute_routing_scores_for_aux_loss from megatron.core; "
        "some MoE features may be unavailable.",
        stacklevel=2,
    )

try:
    from megatron.core.transformer.moe.moe_utils import group_limited_topk
except ImportError:
    group_limited_topk = None

try:
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )
except ImportError:
    MoEAlltoAllTokenDispatcher = None

from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


# ===================================================================
# RouterReplayAction enum and RouterReplay class
# ===================================================================


class RouterReplayAction(Enum):
    """Actions controlling the MoE routing replay behaviour."""

    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class RouterReplay:
    """Manages recording and replaying of MoE routing decisions.

    Each MoE layer gets one ``RouterReplay`` instance.  The class-level
    list ``router_instances`` holds all of them so that global operations
    (set action, distribute data, clear state) are straightforward.
    """

    # Class-level list of all router instances (one per MoE layer).
    router_instances: list["RouterReplay"] = []

    # Class-level pipeline parallelism size for backward remapping.
    # Set by the engine patch before forward_backward_func.
    pp_size: int = 1

    # ---------- R3 diagnostics (PP=2 root-cause hunt) ----------
    # Per forward_backward_batch aggregate counters.  Keys set by the
    # REPLAY_BACKWARD/consume path and the forward-hook; values are reset
    # at FB entry by _r3_forward_backward_batch and dumped at FB exit.
    # This gives one END-OF-FB summary line to diff PP=1 vs PP=2 runs.
    # Always class-level dict (no cross-rank comm — each rank maintains
    # its own, which is the correctness unit).
    _r3_fb_stats: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Class-level (static) helpers
    # ------------------------------------------------------------------

    @staticmethod
    def set_replay_data(all_layers_topk_indices: list) -> None:
        """Distribute per-layer topk indices to ``RouterReplay`` instances.

        Args:
            all_layers_topk_indices: List of tensors, one per MoE layer,
                each of shape ``(num_tokens, topk)``.  Order must match
                instantiation order.
        """
        if len(all_layers_topk_indices) != len(RouterReplay.router_instances):
            raise ValueError(
                f"[R3] Number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match number of router instances "
                f"({len(RouterReplay.router_instances)})."
            )
        for i, inst in enumerate(RouterReplay.router_instances):
            inst.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def get_recorded_data() -> list:
        """Collect recorded topk indices from all instances."""
        return [r.get_recorded_indices() for r in RouterReplay.router_instances]

    @staticmethod
    def clear_global_indices() -> None:
        """Clear recorded and target indices on all instances."""
        for r in RouterReplay.router_instances:
            r.clear_indices()

    @staticmethod
    def set_global_router_replay_action(action: RouterReplayAction) -> None:
        """Set the replay action for all router instances."""
        for r in RouterReplay.router_instances:
            r.set_router_replay_action(action)
        try:
            from areal.engine.router_replay_utils import (
                _r3_pp_tp_info,
                _r3_should_log,
            )

            if _r3_should_log(f"set_global_router_replay_action/{action.value}"):
                logger.info(
                    "[R3-STAGE4/set_global_router_replay_action] %s action=%s "
                    "applied_to=%d router_instances",
                    _r3_pp_tp_info(),
                    action.value,
                    len(RouterReplay.router_instances),
                )
        except Exception:
            pass

    @staticmethod
    def clear_global_router_replay_action() -> None:
        """Clear the replay action on all router instances."""
        for r in RouterReplay.router_instances:
            r.clear_router_replay_action()

    def __init__(self) -> None:
        self.target_topk_idx: torch.Tensor | None = None
        self.recorded_topk_idx: torch.Tensor | None = None
        self.router_replay_action: RouterReplayAction | None = None
        self.replay_backward_list: list[torch.Tensor] = []
        # 1-D bool mask (shape=(num_tokens,)) marking which
        # rows of ``target_topk_idx`` correspond to real tokens.  Padded
        # rows (seq-alignment slack + batch padding) must not be forced to
        # the recorded top-k (which is [0,...,0]); instead we let them fall
        # back to the live router output so they produce no replay signal.
        self.target_valid_mask: torch.Tensor | None = None
        self.replay_backward_mask_list: list[torch.Tensor | None] = []
        # ---------- R3 diagnostics (PP=2 root-cause hunt) ----------
        # Per-push metadata ring -- parallel to ``replay_backward_list`` so
        # the BACKWARD consumer can compare the popped slab against the
        # slab that was REGISTERED AT THAT PUSH (not the most recent
        # target, which under 1F1B scheduling has been overwritten by a
        # later mb). This cleanly separates "logging artifact" from "real
        # backward queue corruption". See code-rules/distributed.md hang
        # section -- the metadata ring is local state, no cross-rank comm.
        self.replay_push_meta_list: list[dict] = []
        self.creation_order: int = len(RouterReplay.router_instances)
        try:
            import torch.distributed as _dist
            self.creator_rank: int = _dist.get_rank() if _dist.is_initialized() else -1
        except Exception:
            self.creator_rank = -1
        # ------------------------------------------------------------
        RouterReplay.router_instances.append(self)

    def set_target_indices(
        self,
        topk_indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        """Sets the target topk indices (and optional row-validity mask) for replay.

        Args:
            topk_indices: ``(num_tokens, topk)`` replay indices.
            valid_mask: Optional ``(num_tokens,)`` bool tensor. ``True`` means
                the row is a real token and replay should override live routing;
                ``False`` means the row is padding (batch or TP-alignment slack)
                and replay MUST fall back to live routing to avoid forcing those
                rows to expert 0. When ``None``, all rows are treated as real
                (legacy behaviour).
        """
        self.target_topk_idx = topk_indices
        self.target_valid_mask = valid_mask
        self.replay_backward_list.append(topk_indices)
        self.replay_backward_mask_list.append(valid_mask)
        # ---------- R3 diagnostics: capture push metadata at the SAME
        # call site so REPLAY_BACKWARD/consume can later prove that the
        # popped slab equals the slab that was originally pushed (the
        # only correctness criterion). Hashing here is gated by
        # _r3_should_log so steady-state cost is one int + one None.
        # ---------------------------------------------------------------
        try:
            from areal.engine.router_replay_utils import (
                _r3_current_trace_id as _tid,
                _r3_hash64 as _h64,
                _r3_should_log as _sl,
                _r3_verbose as _v,
            )
            if _v() and _sl("RouterReplay.set_target_indices/push_meta"):
                _slab_h = hex(_h64(topk_indices))
                _mask_h = (
                    hex(_h64(valid_mask.to(torch.int32)))
                    if valid_mask is not None else "None"
                )
                self.replay_push_meta_list.append({
                    "push_id": getattr(self, "_r3_push_counter", 0),
                    "trace_id": _tid(),
                    "slab_hash": _slab_h,
                    "mask_hash": _mask_h,
                    "slab_shape": tuple(topk_indices.shape),
                })
                self._r3_push_counter = getattr(self, "_r3_push_counter", 0) + 1
            else:
                # Always append a placeholder so list lengths stay locked
                # to ``replay_backward_list``; pop side will skip None.
                self.replay_push_meta_list.append(None)
        except Exception:
            try:
                self.replay_push_meta_list.append(None)
            except Exception:
                pass
        # Cheap diagnostic: record every set in first few layers/mb. Gated
        # via _r3_should_log so steady-state overhead is ~one integer
        # increment.
        try:
            from areal.engine.router_replay_utils import (
                _r3_current_trace_id,
                _r3_hash64,
                _r3_pp_tp_info,
                _r3_should_log,
                _r3_tensor_sig,
                _r3_verbose,
                _r3_zero_row_stats,
            )

            if _r3_verbose() and _r3_should_log("RouterReplay.set_target_indices"):
                # instance index in the class-level list tells us which
                # MoE layer this replay slot refers to
                try:
                    inst_idx = RouterReplay.router_instances.index(self)
                except ValueError:
                    inst_idx = -1
                _slab_hash = hex(_r3_hash64(topk_indices))
                _mask_hash = (
                    hex(_r3_hash64(valid_mask.to(torch.int32)))
                    if valid_mask is not None else "None"
                )
                logger.info(
                    "[R3-STAGE3b/set_target_indices] trace_id=%d inst#%d %s %s "
                    "slab_shape=%s slab_hash=%s mask_hash=%s "
                    "| %s | backward_queue_len=%d (post-push)",
                    _r3_current_trace_id(),
                    inst_idx,
                    _r3_pp_tp_info(),
                    _r3_zero_row_stats(topk_indices),
                    tuple(topk_indices.shape),
                    _slab_hash,
                    _mask_hash,
                    _r3_tensor_sig("topk_indices", topk_indices),
                    len(self.replay_backward_list),
                )
        except Exception:
            pass

    def get_recorded_indices(self) -> torch.Tensor | None:
        return self.recorded_topk_idx

    def record_indices(self, topk_indices: torch.Tensor) -> None:
        self.recorded_topk_idx = topk_indices

    def clear_indices(self) -> None:
        # ---------- R3 diagnostics: dump tail-state queue sizes BEFORE
        # clearing so residual queues (a smoking gun for lost backward
        # pops under PP=2 1F1B) are always visible in logs.
        try:
            from areal.engine.router_replay_utils import (
                _r3_pp_tp_info,
                _r3_should_log,
                _r3_verbose,
            )
            if _r3_verbose() and _r3_should_log("RouterReplay.clear_indices/tail_state"):
                logger.info(
                    "[R3-STAGE3c/clear_indices] %s inst#%d fwd_q=%d "
                    "mask_q=%d push_meta_q=%d",
                    _r3_pp_tp_info(),
                    self.creation_order,
                    len(self.replay_backward_list),
                    len(self.replay_backward_mask_list),
                    len(getattr(self, "replay_push_meta_list", []) or []),
                )
        except Exception:
            pass
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.target_valid_mask = None
        self.replay_backward_list = []
        self.replay_backward_mask_list = []
        self.replay_push_meta_list = []

    def set_router_replay_action(self, action: RouterReplayAction) -> None:
        self.router_replay_action = action

    def clear_router_replay_action(self) -> None:
        self.router_replay_action = None


# ===================================================================
# Patched routing implementation
# ===================================================================


def _R3_routing_log(
    action_name: str,
    *,
    scores: torch.Tensor,
    top_indices: torch.Tensor,
    topk: int,
    compute_topk_fn,
    num_groups=None,
    group_topk=None,
) -> None:
    """Rate-limited diagnostic for the replay branches.

    Key quantities:
      * ``shape_match``   -- does target_topk_idx align with this layer's
        token count? If NOT, replay is being fed the wrong slab.
      * ``zero_rows``     -- fraction of all-zero rows in the replay
        indices; zero rows collapse routing to expert 0.
      * ``live_vs_replay`` -- overlap between replay top-k and the live
        top-k the router would have picked right now. 100% = no staleness
        (rollout weights == train weights). 0% = total mismatch.
    """
    from areal.engine.router_replay_utils import (
        _r3_call_count,
        _r3_pp_tp_info,
        _r3_should_log,
        _r3_tensor_sig,
        _r3_verbose,
        _r3_zero_row_stats,
        _R3_ROUTER_LAYER_LIMIT,
    )

    if not _r3_verbose():
        return
    key = f"patched_routing/{action_name}"
    call_n = _r3_call_count(key)
    # We always want an early, concentrated burst of per-layer details at
    # startup (helps catch first-step config problems) and then a sparse
    # steady-state sample.
    if not _r3_should_log(key):
        return
    with torch.no_grad():
        shape_match = top_indices.shape[0] == scores.shape[0]
        if shape_match:
            try:
                _, live_top = compute_topk_fn(
                    scores, topk, num_groups=num_groups, group_topk=group_topk
                )
                # per-token overlap ratio
                set_live = live_top.sort(dim=-1).values
                set_rep = top_indices.sort(dim=-1).values
                # equality per (token, slot)
                overlap = (set_live == set_rep).float().mean().item()
            except Exception as e:
                overlap = f"err:{e}"
        else:
            overlap = None
    logger.info(
        "[R3-STAGE4/patched_routing] %s call#%d %s "
        "scores_shape=%s topk=%d target_shape=%s shape_match=%s "
        "live_vs_replay_topk_overlap=%s %s | %s | %s",
        action_name,
        call_n,
        _r3_pp_tp_info(),
        tuple(scores.shape),
        topk,
        tuple(top_indices.shape),
        shape_match,
        overlap,
        _r3_zero_row_stats(top_indices),
        _r3_tensor_sig("scores", scores, max_sample=4),
        _r3_tensor_sig("top_indices", top_indices, max_sample=8),
    )


def _patched_topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    score_function: str,
    expert_bias: torch.Tensor,
    fused: bool,
    router_replay: RouterReplay | None,
    scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patched ``topk_routing_with_score_function`` supporting router replay."""
    num_tokens, num_experts = logits.shape

    def _compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk and group_limited_topk is not None:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        routing_action = (
            router_replay.router_replay_action if router_replay is not None else None
        )

        if routing_action is None:
            return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

        if routing_action == RouterReplayAction.RECORD:
            probs, top_indices = _compute_topk(
                scores, topk, num_groups=num_groups, group_topk=group_topk
            )
            if router_replay is not None:
                router_replay.record_indices(top_indices)
            return probs, top_indices

        elif routing_action == RouterReplayAction.REPLAY_FORWARD:
            if router_replay is None or router_replay.target_topk_idx is None:
                # Fallback if replay data is not available
                logger.warning(
                    "[R3] REPLAY_FORWARD: no replay indices available, "
                    "falling back to normal routing."
                )
                return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

            # Use the provided indices for replay
            top_indices = router_replay.target_topk_idx
            top_indices = top_indices.to(scores.device)
            # splice padded rows with the LIVE router top-k so
            # that TP-alignment / batch padding slack (which was recorded as
            # all-zeros) does not force those rows to expert 0.
            valid_mask = getattr(router_replay, "target_valid_mask", None)
            try:
                from areal.engine.router_replay_utils import (
                    _r3_current_trace_id as _tid,
                    _r3_hash64 as _h64,
                    _r3_should_log as _sl,
                    _r3_verbose as _v,
                )
                if _v() and _sl("REPLAY_FORWARD/consume"):
                    try:
                        _inst_idx = RouterReplay.router_instances.index(router_replay)
                    except ValueError:
                        _inst_idx = -1
                    logger.info(
                        "[R3-STAGE4/REPLAY_FORWARD/consume] trace_id=%d inst#%d "
                        "scores_shape=%s target_shape=%s shape_match=%s "
                        "target_hash=%s mask_hash=%s backward_queue_len=%d",
                        _tid(),
                        _inst_idx,
                        tuple(scores.shape),
                        tuple(top_indices.shape),
                        top_indices.shape[0] == scores.shape[0],
                        hex(_h64(top_indices)),
                        "None" if valid_mask is None
                        else hex(_h64(valid_mask.to(torch.int32))),
                        len(router_replay.replay_backward_list),
                    )
            except Exception:
                pass
            if valid_mask is not None and valid_mask.shape[0] == top_indices.shape[0]:
                _, live_top = _compute_topk(
                    scores, topk, num_groups=num_groups, group_topk=group_topk
                )
                top_indices = torch.where(
                    valid_mask.to(scores.device).unsqueeze(-1),
                    top_indices,
                    live_top,
                )
            _R3_routing_log(
                "REPLAY_FORWARD",
                scores=scores,
                top_indices=top_indices,
                topk=topk,
                compute_topk_fn=_compute_topk,
                num_groups=num_groups,
                group_topk=group_topk,
            )
            probs = scores.gather(1, top_indices)

            return probs, top_indices

        elif routing_action == RouterReplayAction.REPLAY_BACKWARD:
            if router_replay is None or not router_replay.replay_backward_list:
                # Fallback if replay data is not available
                logger.warning(
                    "[R3] REPLAY_BACKWARD: no backward indices available, "
                    "falling back to normal routing."
                )
                return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            # Use the last recorded indices for backward replay
            _bw_queue_len_before = len(router_replay.replay_backward_list)
            _bw_mask_queue_len_before = len(
                getattr(router_replay, "replay_backward_mask_list", []) or []
            )
            top_indices = router_replay.replay_backward_list.pop(0)
            top_indices = top_indices.to(scores.device)
            # pop the matching per-row validity mask (if any)
            # so the backward recompute sees the same spliced indices as the
            # original forward pass.  Without this, activation-checkpoint
            # recomputation re-introduces the all-zero padding rows and the
            # gradient path contradicts the forward pass.
            bw_mask_list = getattr(router_replay, "replay_backward_mask_list", None)
            if bw_mask_list:
                bw_valid_mask = bw_mask_list.pop(0)
            else:
                bw_valid_mask = None
            # ---------- R3 diagnostics: pop the matching push-meta entry
            # so the divergence verdict below compares popped-slab against
            # the slab that was REGISTERED AT PUSH TIME (the real
            # correctness criterion under 1F1B PP scheduling).
            _push_meta_list = getattr(router_replay, "replay_push_meta_list", None)
            _bw_push_meta = None
            if _push_meta_list:
                try:
                    _bw_push_meta = _push_meta_list.pop(0)
                except IndexError:
                    _bw_push_meta = None
            # ---- R3 deep-trace: log backward pop order + hashes ----
            try:
                from areal.engine.router_replay_utils import (
                    _r3_current_trace_id as _tid,
                    _r3_hash64 as _h64,
                    _r3_should_log as _sl,
                    _r3_verbose as _v,
                )

                if _v() and _sl("REPLAY_BACKWARD/consume"):
                    try:
                        _inst_idx = RouterReplay.router_instances.index(router_replay)
                    except ValueError:
                        _inst_idx = -1
                    _popped_slab_hash = hex(_h64(top_indices))
                    _popped_mask_hash = (
                        "None"
                        if bw_valid_mask is None
                        else hex(_h64(bw_valid_mask.to(torch.int32)))
                    )
                    _target_hash = (
                        "None"
                        if router_replay.target_topk_idx is None
                        else hex(_h64(router_replay.target_topk_idx))
                    )
                    _divergence = (
                        "None"
                        if router_replay.target_topk_idx is None
                        else (
                            "MATCH"
                            if (
                                router_replay.target_topk_idx.shape
                                == top_indices.shape
                                and hex(
                                    _h64(
                                        router_replay.target_topk_idx.to(
                                            top_indices.device
                                        )
                                    )
                                )
                                == _popped_slab_hash
                            )
                            else "DIVERGE_vs_FWD_TARGET"
                        )
                    )
                    logger.info(
                        "[R3-STAGE4/REPLAY_BACKWARD/consume] trace_id=%d inst#%d "
                        "scores_shape=%s popped_shape=%s shape_match_scores=%s "
                        "popped_slab_hash=%s popped_mask_hash=%s "
                        "current_target_hash=%s divergence=%s "
                        "queue_len_before=%d queue_len_after=%d "
                        "mask_queue_len_before=%d mask_queue_len_after=%d "
                        "push_meta=%s divergence_v2=%s",
                        _tid(),
                        _inst_idx,
                        tuple(scores.shape),
                        tuple(top_indices.shape),
                        top_indices.shape[0] == scores.shape[0],
                        _popped_slab_hash,
                        _popped_mask_hash,
                        _target_hash,
                        _divergence,
                        _bw_queue_len_before,
                        len(router_replay.replay_backward_list),
                        _bw_mask_queue_len_before,
                        len(
                            getattr(router_replay, "replay_backward_mask_list", [])
                            or []
                        ),
                        _bw_push_meta,
                        # divergence_v2 is the DEFINITIVE verdict: it
                        # compares popped slab against the slab recorded
                        # at the matching push site, not against the
                        # most recent (potentially overwritten) target.
                        # MATCH here = backward queue is correct under PP.
                        (
                            "NO_PUSH_META"
                            if _bw_push_meta is None
                            else (
                                "MATCH"
                                if _bw_push_meta.get("slab_hash") == _popped_slab_hash
                                else "REAL_MISMATCH"
                            )
                        ),
                    )
            except Exception:
                logger.exception(
                    "[R3-STAGE4/REPLAY_BACKWARD/consume] trace log failed"
                )
            # ---------- R3 diagnostics: FB-level aggregate counters
            # (gated by _r3_verbose so prod path is untouched). Counters
            # are reset at _r3_forward_backward_batch entry and dumped at
            # exit, giving one summary line per FB call.
            try:
                from areal.engine.router_replay_utils import (
                    _r3_hash64 as _h64x,
                    _r3_verbose as _vx,
                )
                if _vx():
                    _stats = RouterReplay._r3_fb_stats
                    if router_replay.target_topk_idx is None:
                        _stats["divergence_v1_none"] = (
                            _stats.get("divergence_v1_none", 0) + 1
                        )
                    else:
                        _v1_match = (
                            router_replay.target_topk_idx.shape == top_indices.shape
                            and _h64x(
                                router_replay.target_topk_idx.to(top_indices.device)
                            ) == _h64x(top_indices)
                        )
                        _stats["divergence_v1_match" if _v1_match
                               else "divergence_v1_diverge"] = (
                            _stats.get(
                                "divergence_v1_match" if _v1_match
                                else "divergence_v1_diverge", 0,
                            ) + 1
                        )
                    if _bw_push_meta is None:
                        _stats["divergence_v2_no_meta"] = (
                            _stats.get("divergence_v2_no_meta", 0) + 1
                        )
                    else:
                        _v2_match = (
                            _bw_push_meta.get("slab_hash")
                            == hex(_h64x(top_indices))
                        )
                        _stats["divergence_v2_match" if _v2_match
                               else "divergence_v2_real_mismatch"] = (
                            _stats.get(
                                "divergence_v2_match" if _v2_match
                                else "divergence_v2_real_mismatch", 0,
                            ) + 1
                        )
                    _stats["bw_pop_total"] = _stats.get("bw_pop_total", 0) + 1
            except Exception:
                pass
            if (
                bw_valid_mask is not None
                and bw_valid_mask.shape[0] == top_indices.shape[0]
            ):
                _, live_top = _compute_topk(
                    scores, topk, num_groups=num_groups, group_topk=group_topk
                )
                top_indices = torch.where(
                    bw_valid_mask.to(scores.device).unsqueeze(-1),
                    top_indices,
                    live_top,
                )
            _R3_routing_log(
                "REPLAY_BACKWARD",
                scores=scores,
                top_indices=top_indices,
                topk=topk,
                compute_topk_fn=_compute_topk,
                num_groups=num_groups,
                group_topk=group_topk,
            )
            probs = scores.gather(1, top_indices)
            return probs, top_indices

        else:
            return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    # --- Score function dispatch ---
    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"[R3] Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    if torch.are_deterministic_algorithms_enabled():
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)
        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_(
            (rows, top_indices),
            torch.ones_like(probs, dtype=routing_map.dtype),
            accumulate=False,
        )
        routing_map = routing_map.bool()
    else:
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map


# ===================================================================
# Aux-loss helpers
# ===================================================================


def _get_aux_loss_coeff(_self, aux_loss_type: str) -> float:
    """Return the aux loss coeff for the given auxiliary loss type."""
    if isinstance(_self.routing_type, str):
        if _self.routing_type == aux_loss_type:
            return _self.config.moe_aux_loss_coeff
    if isinstance(_self.routing_type, list):
        try:
            idx = _self.routing_type.index(aux_loss_type)
            return _self.config.moe_aux_loss_coeff[idx]
        except (ValueError, IndexError):
            return 0.0
    return 0.0


def _is_aux_loss_enabled(_self) -> bool:
    """Check if any auxiliary loss is enabled."""
    for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
        if _get_aux_loss_coeff(_self, aux_loss_type) > 0:
            return True
    return False


# ===================================================================
# patched_routing -- replaces TopKRouter.routing
# ===================================================================


def patched_routing(self, logits: torch.Tensor, *args, **kwargs):
    """Patched ``TopKRouter.routing`` that supports router replay.

    Drop-in replacement for ``TopKRouter.routing`` that delegates to
    ``_patched_topk_routing_with_score_function`` which honours the
    ``RouterReplayAction`` set on the per-layer ``RouterReplay`` instance.
    """
    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    moe_router_fusion = getattr(self.config, "moe_router_fusion", False)

    # Calculate probs and routing_map for token dispatching
    if self.routing_type == "sinkhorn":
        probs, routing_map = self.sinkhorn_load_balancing(logits)
    else:
        probs, routing_map = _patched_topk_routing_with_score_function(
            logits=logits,
            topk=self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            fused=moe_router_fusion,
            router_replay=getattr(self, "router_replay", None),
        )

    # Apply token dropping to probs and routing_map.
    if (
        self.config.moe_expert_capacity_factor is not None
        and apply_router_token_dropping is not None
    ):
        probs, routing_map = apply_router_token_dropping(
            probs,
            routing_map,
            router_topk=self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            drop_policy=self.config.moe_token_drop_policy,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        )

    if not hasattr(self, "is_aux_loss_enabled"):
        self.is_aux_loss_enabled = types.MethodType(_is_aux_loss_enabled, self)

    # Apply aux loss
    if (
        self.training
        and torch.is_grad_enabled()
        and self.is_aux_loss_enabled()
        and compute_routing_scores_for_aux_loss is not None
    ):
        routing_map_for_aux_loss, scores_for_aux_loss = (
            compute_routing_scores_for_aux_loss(
                logits,
                self.topk,
                self.score_function,
                fused=self.config.moe_router_fusion,
            )
        )
        probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
        probs = self._apply_seq_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
        )
        probs = self._apply_global_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss
        )

    # Update expert bias and tokens_per_expert
    if self.enable_expert_bias and torch.is_grad_enabled():
        with torch.no_grad():
            self.local_tokens_per_expert += routing_map.sum(dim=0)

    return probs, routing_map


# ===================================================================
# Sentinel to prevent double-patching
# ===================================================================
_PATCHES_APPLIED = False

# Store original methods for undo
_ORIGINAL_TF_CONFIG_INIT = None
_ORIGINAL_TOPK_ROUTER_INIT = None
_ORIGINAL_TOPK_ROUTER_ROUTING = None
_ORIGINAL_DISPATCHER_PREPROCESS = None


# ===================================================================
# apply_router_replay_patch
# ===================================================================


def apply_router_replay_patch() -> None:
    """Apply all Megatron-Core monkey-patches required for Router Replay.

    Safe to call multiple times -- subsequent calls are no-ops.
    Must be called **before** model creation.
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        logger.debug("[R3] Router replay patches already applied; skipping.")
        return

    logger.info("[R3] Applying Router Replay patches...")

    RouterReplay.router_instances.clear()

    _patch_transformer_config_init()
    _patch_topk_router_init()
    _patch_topk_router_routing()
    _patch_alltoall_dispatcher_preprocess()

    _PATCHES_APPLIED = True
    logger.debug("[R3] All Router Replay patches applied successfully.")


def remove_router_replay_patch() -> None:
    """Undo all patches (primarily for test cleanup)."""
    global _PATCHES_APPLIED
    _undo_transformer_config_patch()
    _undo_topk_router_init_patch()
    _undo_topk_router_routing_patch()
    _undo_dispatcher_patch()
    RouterReplay.router_instances.clear()
    _PATCHES_APPLIED = False
    logger.debug("[R3] All Router Replay patches removed.")


# ===================================================================
# Patch 1: TransformerConfig.__init__
# ===================================================================


def _patch_transformer_config_init() -> None:
    """Patch ``TransformerConfig.__init__`` to accept ``enable_routing_replay``."""
    global _ORIGINAL_TF_CONFIG_INIT

    if getattr(TransformerConfig, "_r3_config_patched", False):
        return

    # Inspect the current signature to add enable_routing_replay
    try:
        sig = inspect.signature(TransformerConfig.__init__)
        native_params = sig.parameters
        params = list(sig.parameters.values())
    except Exception:
        sig = None
        native_params = {}
        params = []

    ext_attr = "enable_routing_replay"

    if ext_attr not in native_params and sig is not None:
        new_param = inspect.Parameter(
            ext_attr, inspect.Parameter.KEYWORD_ONLY, default=False
        )
        if params and params[-1].kind == inspect.Parameter.VAR_KEYWORD:
            params.insert(-1, new_param)
        else:
            params.append(new_param)
        try:
            TransformerConfig.__init__.__signature__ = sig.replace(parameters=params)
        except Exception as e:
            logger.warning("[R3] Failed to update TransformerConfig signature: %s", e)

    _ORIGINAL_TF_CONFIG_INIT = TransformerConfig.__init__

    @wraps(_ORIGINAL_TF_CONFIG_INIT)
    def patched_tf_config_init(self, *args, **kwargs):
        enable_routing_replay = kwargs.get("enable_routing_replay", False)
        if "enable_routing_replay" not in native_params:
            enable_routing_replay = kwargs.pop("enable_routing_replay", False)
        _ORIGINAL_TF_CONFIG_INIT(self, *args, **kwargs)
        self.enable_routing_replay = enable_routing_replay

    TransformerConfig.__init__ = patched_tf_config_init
    TransformerConfig._r3_config_patched = True
    logger.debug("[R3] TransformerConfig.__init__ patched to accept enable_routing_replay.")


def _undo_transformer_config_patch() -> None:
    global _ORIGINAL_TF_CONFIG_INIT
    if _ORIGINAL_TF_CONFIG_INIT is not None:
        TransformerConfig.__init__ = _ORIGINAL_TF_CONFIG_INIT
        if hasattr(TransformerConfig, "_r3_config_patched"):
            del TransformerConfig._r3_config_patched
        _ORIGINAL_TF_CONFIG_INIT = None


# ===================================================================
# Patch 2: TopKRouter.__init__
# ===================================================================


def _patch_topk_router_init() -> None:
    """Patch ``TopKRouter.__init__`` to create a ``RouterReplay`` instance."""
    global _ORIGINAL_TOPK_ROUTER_INIT

    if getattr(TopKRouter, "_r3_init_patched", False):
        return

    _ORIGINAL_TOPK_ROUTER_INIT = TopKRouter.__init__

    def patched_init(self, *args, **kwargs):
        _ORIGINAL_TOPK_ROUTER_INIT(self, *args, **kwargs)
        self.router_replay = None
        if getattr(self.config, "enable_routing_replay", False):
            self.router_replay = RouterReplay()
            logger.debug(
                "[R3] TopKRouter: created RouterReplay instance "
                "(total instances: %d).",
                len(RouterReplay.router_instances),
            )

    TopKRouter.__init__ = patched_init
    TopKRouter._r3_init_patched = True
    logger.debug("[R3] TopKRouter.__init__ patched.")


def _undo_topk_router_init_patch() -> None:
    global _ORIGINAL_TOPK_ROUTER_INIT
    if _ORIGINAL_TOPK_ROUTER_INIT is not None:
        TopKRouter.__init__ = _ORIGINAL_TOPK_ROUTER_INIT
        if hasattr(TopKRouter, "_r3_init_patched"):
            del TopKRouter._r3_init_patched
        _ORIGINAL_TOPK_ROUTER_INIT = None


# ===================================================================
# Patch 3: TopKRouter.routing
# ===================================================================


def _patch_topk_router_routing() -> None:
    """Patch ``TopKRouter.routing`` with the replay-aware version."""
    global _ORIGINAL_TOPK_ROUTER_ROUTING

    if getattr(TopKRouter, "_r3_routing_patched", False):
        return

    _ORIGINAL_TOPK_ROUTER_ROUTING = TopKRouter.routing
    TopKRouter.routing = patched_routing
    TopKRouter._r3_routing_patched = True
    logger.debug("[R3] TopKRouter.routing patched.")


def _undo_topk_router_routing_patch() -> None:
    global _ORIGINAL_TOPK_ROUTER_ROUTING
    if _ORIGINAL_TOPK_ROUTER_ROUTING is not None:
        TopKRouter.routing = _ORIGINAL_TOPK_ROUTER_ROUTING
        if hasattr(TopKRouter, "_r3_routing_patched"):
            del TopKRouter._r3_routing_patched
        _ORIGINAL_TOPK_ROUTER_ROUTING = None


# ===================================================================
# Patch 4: MoEAlltoAllTokenDispatcher.preprocess
# ===================================================================


def _patch_alltoall_dispatcher_preprocess() -> None:
    """Patch dispatcher preprocess to handle duplicate indices from replay."""
    global _ORIGINAL_DISPATCHER_PREPROCESS

    if MoEAlltoAllTokenDispatcher is None:
        logger.warning(
            "[R3] Cannot import MoEAlltoAllTokenDispatcher -- "
            "skipping preprocess patch."
        )
        return

    if getattr(MoEAlltoAllTokenDispatcher, "_r3_preprocess_patched", False):
        return

    _ORIGINAL_DISPATCHER_PREPROCESS = MoEAlltoAllTokenDispatcher.preprocess

    def patched_preprocess(self, routing_map):
        result = _ORIGINAL_DISPATCHER_PREPROCESS(self, routing_map)
        if (
            getattr(self.config, "enable_routing_replay", False)
            and not self.drop_and_pad
            and self.config.moe_expert_capacity_factor is None
            and not (
                getattr(self.config, "moe_router_padding_for_quantization", None)
                or getattr(self.config, "moe_router_padding_for_fp8", None)
            )
        ):
            self.num_out_tokens = int(routing_map.sum().item())
        return result

    MoEAlltoAllTokenDispatcher.preprocess = patched_preprocess
    MoEAlltoAllTokenDispatcher._r3_preprocess_patched = True
    logger.debug("[R3] MoEAlltoAllTokenDispatcher.preprocess patched.")


def _undo_dispatcher_patch() -> None:
    global _ORIGINAL_DISPATCHER_PREPROCESS
    if MoEAlltoAllTokenDispatcher is None:
        return
    if _ORIGINAL_DISPATCHER_PREPROCESS is not None:
        MoEAlltoAllTokenDispatcher.preprocess = _ORIGINAL_DISPATCHER_PREPROCESS
        if hasattr(MoEAlltoAllTokenDispatcher, "_r3_preprocess_patched"):
            del MoEAlltoAllTokenDispatcher._r3_preprocess_patched
        _ORIGINAL_DISPATCHER_PREPROCESS = None
