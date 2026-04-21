"""
Runtime monkey-patch for sglang to support per-PP-rank NCCL weight update groups.

When sglang runs with pipeline parallelism (pp_size > 1), weight updates from
the training side come per-PP-stage.  Each training PP rank creates a separate
NCCL group that only includes the sglang workers at the matching PP rank.

Without this patch, ``init_weights_update_group`` forces *every* TP worker
(across all PP ranks) to join the same NCCL group, which is incorrect when
different PP ranks hold different model shards.

The patch intercepts three code-paths inside sglang:

1. **``InitWeightsUpdateGroupReqInput``** -- the HTTP request dataclass gains
   an optional ``pp_rank`` field so the caller can specify which PP stage
   the new group targets.

2. **``ModelRunner.init_weights_update_group``** -- when ``pp_rank`` is
   supplied, only workers whose ``self.pp_rank`` matches actually call
   ``init_custom_process_group``.  Workers at other PP ranks record the
   group name with a ``None`` sentinel so subsequent operations can
   short-circuit.

3. **``ModelRunner.update_weights_from_distributed``** and
   **``ModelRunner.destroy_weights_update_group``** -- if the local worker
   did not join the group (sentinel is ``None``), the operation is skipped
   gracefully instead of crashing.

Backward compatibility
----------------------
When ``pp_rank`` is *not* present in the HTTP payload (or is ``None``), every
worker joins the group unconditionally.  For PP=1 the behaviour is identical
to the original sglang code.

Usage
-----
::

    from areal.patches.sglang_pp_weight_update import apply_sglang_pp_patch
    apply_sglang_pp_patch()

This must be called **before** the sglang server starts handling weight update
requests -- typically right after importing sglang but before
``launch_server`` or ``Engine()`` is invoked.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time as _time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel used to mark a group that this worker intentionally did NOT join.
# Stored in ``ModelRunner._model_update_group[group_name]`` so that
# downstream code can distinguish "never initialised" from "skipped because
# of PP rank mismatch".
# ---------------------------------------------------------------------------
_PP_SKIP_SENTINEL = "__pp_rank_skip__"

_PATCHED = False


def apply_sglang_pp_patch() -> None:
    """Apply monkey-patches to sglang for per-PP-rank NCCL group support.

    This patches:

    1. ``InitWeightsUpdateGroupReqInput``  -- adds optional ``pp_rank`` field.
    2. ``ModelRunner.init_weights_update_group`` -- routes group creation to
       the correct PP-rank workers.
    3. ``ModelRunner.update_weights_from_distributed`` -- skips broadcast
       receive for workers that did not join the group.
    4. ``ModelRunner.destroy_weights_update_group`` -- skips destruction for
       workers that did not join the group.

    The function is idempotent: calling it more than once is a no-op.
    """
    global _PATCHED
    if _PATCHED:
        logger.debug("sglang PP weight-update patch already applied; skipping.")
        return

    try:
        _patch_io_struct()
        _patch_model_runner()
        _PATCHED = True
        logger.info("Successfully applied sglang PP weight-update patch.")
    except Exception as e:
        logger.warning(
            "Failed to apply sglang PP weight-update patch: %s. "
            "Per-PP-rank weight updates may not work correctly.",
            e,
            exc_info=True,
        )


# ===================================================================== #
#  1.  Patch InitWeightsUpdateGroupReqInput to accept ``pp_rank``       #
# ===================================================================== #


def _patch_io_struct() -> None:
    """Add an optional ``pp_rank`` field to ``InitWeightsUpdateGroupReqInput``.

    sglang's dataclass does not know about ``pp_rank``.  We dynamically
    rebuild the dataclass with the extra field so that FastAPI / pydantic
    will accept the field from the JSON payload and propagate it through
    the internal IPC path.
    """
    from sglang.srt.managers import io_struct

    OrigCls = io_struct.InitWeightsUpdateGroupReqInput

    # If the field already exists (e.g. future sglang version), nothing to do.
    existing_fields = {f.name for f in dataclasses.fields(OrigCls)}
    if "pp_rank" in existing_fields:
        logger.info(
            "InitWeightsUpdateGroupReqInput already has 'pp_rank' field; "
            "skipping io_struct patch."
        )
        return

    # Build a new dataclass that inherits from the original and adds pp_rank.
    # We use ``dataclasses.make_dataclass`` so that the result is a proper
    # dataclass with all the original fields preserved.
    new_fields = [
        ("pp_rank", int | None, dataclasses.field(default=None)),
    ]

    PatchedCls = dataclasses.make_dataclass(
        "InitWeightsUpdateGroupReqInput",
        fields=new_fields,
        bases=(OrigCls,),
    )

    PatchedCls.__module__ = OrigCls.__module__
    PatchedCls.__qualname__ = OrigCls.__qualname__

    # Overwrite in the module so that FastAPI picks up the new schema.
    io_struct.InitWeightsUpdateGroupReqInput = PatchedCls

    # Also patch the http_server module's local reference if it has already
    # been imported, since ``from io_struct import ...`` binds a local name.
    try:
        from sglang.srt.entrypoints import http_server

        http_server.InitWeightsUpdateGroupReqInput = PatchedCls
    except (ImportError, AttributeError):
        pass

    logger.debug("Patched InitWeightsUpdateGroupReqInput with pp_rank field.")


# ===================================================================== #
#  2.  Patch ModelRunner weight-update methods                          #
# ===================================================================== #


def _patch_model_runner() -> None:
    """Monkey-patch ``ModelRunner`` methods for per-PP-rank group support."""

    from sglang.srt.model_executor.model_runner import ModelRunner

    # -- init_weights_update_group ----------------------------------------
    _orig_init_group = ModelRunner.init_weights_update_group

    def _patched_init_weights_update_group(
        self: "ModelRunner",
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        pp_rank: Optional[int] = None,
    ):
        """Per-PP-rank aware group initialisation.

        Parameters
        ----------
        pp_rank : int or None
            If provided, only workers whose ``self.pp_rank == pp_rank``
            actually join the NCCL group.  Workers at other PP ranks
            store a ``None`` sentinel under *group_name* so that
            ``update_weights_from_distributed`` and
            ``destroy_weights_update_group`` can short-circuit.

            If ``None``, *all* workers join (original behaviour for PP=1).
        """
        if pp_rank is not None and self.pp_rank != pp_rank:
            # This worker is at a different PP rank -- skip group creation.
            self._model_update_group[group_name] = None  # sentinel
            logger.info(
                "Skipping group '%s': requested pp_rank=%d but worker is pp_rank=%d.",
                group_name, pp_rank, self.pp_rank,
            )
            return True, (
                f"Skipped group creation (pp_rank mismatch: "
                f"requested={pp_rank}, local={self.pp_rank})."
            )

        # Either pp_rank is None (all-join, PP=1) or matches this worker.
        if pp_rank is not None:
            logger.info(
                "Worker pp_rank=%d tp_rank=%d joining per-PP-rank group '%s'.",
                self.pp_rank, self.tp_rank, group_name,
            )

        _final_nccl_rank = rank_offset + getattr(self, 'tp_rank', 0)

        _t0 = _time.monotonic()
        _stop = threading.Event()

        def _wd():
            for s in [30, 60, 120]:
                if _stop.wait(s):
                    return
                elapsed = _time.monotonic() - _t0
                logger.warning(
                    "init_weights_update_group BLOCKED %.0fs: "
                    "pp=%s tp=%s nccl_rank=%d ws=%d group=%s",
                    elapsed, getattr(self, 'pp_rank', '?'),
                    getattr(self, 'tp_rank', '?'),
                    _final_nccl_rank, world_size, group_name,
                )

        _wd_t = threading.Thread(target=_wd, daemon=True)
        _wd_t.start()

        try:
            result = _orig_init_group(
                self,
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )
        except Exception as e:
            _stop.set()
            logger.error(
                "init_weights_update_group EXCEPTION after %.2fs: "
                "pp=%s tp=%s nccl_rank=%d group=%s: %s",
                _time.monotonic() - _t0,
                self.pp_rank, self.tp_rank,
                _final_nccl_rank, group_name, e,
                exc_info=True,
            )
            raise

        _stop.set()
        _elapsed = _time.monotonic() - _t0
        logger.info(
            "init_weights_update_group completed in %.2fs: "
            "pp=%s tp=%s group=%s",
            _elapsed, self.pp_rank, self.tp_rank, group_name,
        )
        return result

    ModelRunner.init_weights_update_group = _patched_init_weights_update_group

    # -- update_weights_from_distributed ----------------------------------
    _orig_update_distributed = ModelRunner.update_weights_from_distributed

    def _patched_update_weights_from_distributed(
        self: "ModelRunner",
        names,
        dtypes,
        shapes,
        group_name: str,
        load_format: Optional[str] = None,
    ):
        """Skip broadcast receive if this worker did not join *group_name*."""
        pg = self._model_update_group.get(group_name, _PP_SKIP_SENTINEL)

        if pg is None:
            # Sentinel: this worker was deliberately excluded from the group.
            logger.debug(
                "Skipping update_weights_from_distributed for group '%s': "
                "worker pp_rank=%d did not join.",
                group_name, self.pp_rank,
            )
            return True, (
                f"Skipped weight update (worker pp_rank={self.pp_rank} "
                f"did not join group '{group_name}')."
            )

        if pg is _PP_SKIP_SENTINEL:
            # The group was never registered at all -- fall through to the
            # original method which will raise a clear assertion error.
            pass

        return _orig_update_distributed(
            self, names, dtypes, shapes, group_name, load_format
        )

    ModelRunner.update_weights_from_distributed = (
        _patched_update_weights_from_distributed
    )

    # -- destroy_weights_update_group -------------------------------------
    _orig_destroy_group = ModelRunner.destroy_weights_update_group

    def _patched_destroy_weights_update_group(
        self: "ModelRunner",
        group_name: str,
    ):
        """Skip destruction if this worker holds a ``None`` sentinel."""
        pg = self._model_update_group.get(group_name, _PP_SKIP_SENTINEL)

        if pg is None:
            # Remove the sentinel and report success.
            self._model_update_group.pop(group_name, None)
            logger.debug(
                "Skipping destroy for group '%s': worker pp_rank=%d did not join.",
                group_name, self.pp_rank,
            )
            return True, (
                f"Skipped group destruction (worker pp_rank={self.pp_rank} "
                f"did not join group '{group_name}')."
            )

        return _orig_destroy_group(self, group_name)

    ModelRunner.destroy_weights_update_group = _patched_destroy_weights_update_group

    # -- Patch BaseTpWorker.init_weights_update_group to forward pp_rank --
    _patch_tp_worker()

    logger.debug("Patched ModelRunner weight-update methods for PP support.")


# ===================================================================== #
#  3.  Patch BaseTpWorker to forward pp_rank from the request object    #
# ===================================================================== #


def _patch_tp_worker() -> None:
    """Patch ``BaseTpWorker.init_weights_update_group`` to forward ``pp_rank``.

    The original implementation in ``BaseTpWorker`` calls::

        self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )

    We patch it to additionally pass ``recv_req.pp_rank`` (which may be
    ``None`` if the caller did not set it).
    """
    from sglang.srt.managers.tp_worker import BaseTpWorker

    def _patched_tp_init_weights_update_group(self, recv_req):
        """Forward pp_rank from the request object to model_runner."""
        pp_rank = getattr(recv_req, "pp_rank", None)
        logger.info(
            "BaseTpWorker forwarding init_weights_update_group: "
            "pp_rank=%s group_name=%s",
            pp_rank, recv_req.group_name,
        )

        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
            pp_rank=pp_rank,
        )
        return success, message

    BaseTpWorker.init_weights_update_group = _patched_tp_init_weights_update_group

    logger.debug("Patched BaseTpWorker.init_weights_update_group for PP support.")
