"""Monkey-patch for SGLang GroupCoordinator.send/recv PP weight-tying bug.

Bug: SGLang's model code (e.g. glm4.py, qwen2.py with PP patches) passes
global ranks to GroupCoordinator.send(dst=) and recv(src=), but these methods
expect LOCAL group indices (0..world_size-1). This causes IndexError when
TP > 1 because PP group ranks become non-contiguous (e.g. [0, 2] for PP=2 TP=2).

Fix: Intercept send/recv calls and auto-convert global ranks to local indices
when the passed value exceeds the group's world_size.
"""

import logging

logger = logging.getLogger("areal.patches.sglang_pp_fix")

_PATCHED = False


def _resolve_group_local_rank(ranks, world_size, rank_value, param_name):
    """Convert a potentially global rank to a group-local index.

    If rank_value < world_size, it's already a valid local index (or happens
    to coincide with the global rank, which is the TP=1 case). Return as-is.

    If rank_value >= world_size, it must be a global rank. Look it up in the
    group's ranks list and return the local index.
    """
    if rank_value < world_size:
        return rank_value

    # rank_value >= world_size => must be a global rank, convert to local index
    try:
        local_idx = ranks.index(rank_value)
    except ValueError:
        raise IndexError(
            f"[AReaL PP fix] {param_name}={rank_value} is not a member of "
            f"this group (ranks={ranks}, world_size={world_size}). "
            f"Cannot convert global rank to local index."
        )

    logger.debug(
        f"[AReaL PP fix] Converted {param_name} global_rank={rank_value} "
        f"-> local_index={local_idx} (group ranks={ranks})"
    )
    return local_idx


def apply_sglang_pp_fix():
    """Apply monkey-patch to SGLang's GroupCoordinator.send and .recv.

    Safe to call multiple times; only patches once.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from sglang.srt.distributed.parallel_state import GroupCoordinator
    except ImportError:
        logger.warning(
            "Cannot import sglang.srt.distributed.parallel_state.GroupCoordinator. "
            "SGLang PP fix not applied."
        )
        return

    _orig_send = GroupCoordinator.send
    _orig_recv = GroupCoordinator.recv

    def _patched_send(self, tensor, dst=None):
        """Patched send: auto-converts global rank dst to local index."""
        if dst is not None:
            dst = _resolve_group_local_rank(
                self.ranks, self.world_size, dst, "dst"
            )
        return _orig_send(self, tensor, dst)

    def _patched_recv(self, size, dtype, src=None):
        """Patched recv: auto-converts global rank src to local index."""
        if src is not None:
            src = _resolve_group_local_rank(
                self.ranks, self.world_size, src, "src"
            )
        return _orig_recv(self, size, dtype, src)

    GroupCoordinator.send = _patched_send
    GroupCoordinator.recv = _patched_recv
    _PATCHED = True

    logger.info(
        "Applied SGLang PP fix: GroupCoordinator.send/recv now auto-convert "
        "global ranks to group-local indices."
    )
