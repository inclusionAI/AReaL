"""
Monkey-patch for SGLang's ModelRunner.init_weights_update_group to fix
NCCL rank collision when Pipeline Parallelism (PP) > 1.

Root Cause
----------
SGLang's ``ModelRunner.init_weights_update_group`` computes the NCCL rank as:

    rank = rank_offset + self.tp_rank

This ignores ``pp_rank``.  With PP=2, TP=2, rank_offset=1 for one SGLang
server instance, the four workers get:

    PP0-TP0 → 1+0 = 1
    PP0-TP1 → 1+1 = 2
    PP1-TP0 → 1+0 = 1   ← COLLISION with PP0-TP0
    PP1-TP1 → 1+1 = 2   ← COLLISION with PP0-TP1

Two processes trying to join the same NCCL group with identical ranks causes
a hang, which surfaces as a 504 Gateway Timeout on ``/callback/update_weights_xccl``.

Fix
---
Patch the rank computation to:

    rank = rank_offset + pp_rank * tp_size + tp_rank

Correct mapping (PP=2, TP=2, rank_offset=1):

    PP0-TP0 → 1 + 0*2 + 0 = 1
    PP0-TP1 → 1 + 0*2 + 1 = 2
    PP1-TP0 → 1 + 1*2 + 0 = 3
    PP1-TP1 → 1 + 1*2 + 1 = 4

This ensures every worker gets a unique rank in the NCCL weight-update group.
"""

import logging

logger = logging.getLogger("areal.patches.pp_rank_fix")


def apply_sglang_pp_rank_fix():
    """Apply the monkey-patch to ModelRunner.init_weights_update_group.

    Safe to call multiple times – subsequent calls are no-ops.
    """
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner, "_areal_pp_rank_fixed", False):
        logger.debug("PP rank fix already applied, skipping.")
        return

    _orig_init_weights_update_group = ModelRunner.init_weights_update_group

    def _patched_init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        pp_rank = getattr(self, "pp_rank", 0)
        tp_rank = getattr(self, "tp_rank", 0)
        tp_size = getattr(self, "tp_size", 1)

        correct_local_index = pp_rank * tp_size + tp_rank

        logger.info(
            "[AReaL PP Rank Fix] init_weights_update_group: "
            "pp_rank=%d, tp_rank=%d, tp_size=%d, rank_offset=%d, "
            "original_rank=%d, corrected_rank=%d, world_size=%d, "
            "group_name=%s, backend=%s",
            pp_rank,
            tp_rank,
            tp_size,
            rank_offset,
            rank_offset + tp_rank,
            rank_offset + correct_local_index,
            world_size,
            group_name,
            backend,
        )

        # Temporarily override self.tp_rank so that the original method's
        #     rank = rank_offset + self.tp_rank
        # produces the correct value:
        #     rank = rank_offset + (pp_rank * tp_size + tp_rank)
        saved_tp_rank = self.tp_rank
        self.tp_rank = correct_local_index
        try:
            return _orig_init_weights_update_group(
                self,
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )
        finally:
            self.tp_rank = saved_tp_rank

    ModelRunner.init_weights_update_group = _patched_init_weights_update_group
    ModelRunner._areal_pp_rank_fixed = True
    logger.info(
        "[AReaL PP Rank Fix] Successfully patched ModelRunner.init_weights_update_group"
    )
