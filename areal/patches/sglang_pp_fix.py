"""
Monkey-patch for SGLang Pipeline Parallelism bug.

SGLang's GroupCoordinator.send/recv expect LOCAL group indices (0..world_size-1)
as dst/src, but some model files (e.g., qwen2.py weight-tying) pass GLOBAL ranks
via first_rank/last_rank properties. This patch transparently converts global ranks
to local group indices.
"""

_PATCHED = False


def _resolve_group_local_rank(coordinator, rank_value, param_name):
    ranks = coordinator.ranks
    world_size = coordinator.world_size
    rank_in_group = coordinator.rank_in_group

    # Case 1: value >= world_size → definitely a global rank, convert it
    if rank_value >= world_size:
        if rank_value in ranks:
            return ranks.index(rank_value)
        return rank_value

    # Case 2: value < world_size — could be correct local index OR global rank
    if rank_value in ranks and ranks.index(rank_value) != rank_value:
        # Ambiguous: targeted fix for PP weight-tying pattern
        if param_name == "src" and rank_in_group == world_size - 1 and rank_value == ranks[0]:
            return 0
        if param_name == "dst" and rank_in_group == 0 and rank_value == ranks[-1]:
            return world_size - 1

    return rank_value


def apply_sglang_pp_fix():
    global _PATCHED
    if _PATCHED:
        return
    try:
        from sglang.srt.distributed.parallel_state import GroupCoordinator
    except ImportError:
        return

    _orig_send = GroupCoordinator.send
    _orig_recv = GroupCoordinator.recv

    def _patched_send(self, tensor, dst=None):
        if dst is not None:
            dst = _resolve_group_local_rank(self, dst, "dst")
        return _orig_send(self, tensor, dst)

    def _patched_recv(self, size, dtype, src=None):
        if src is not None:
            src = _resolve_group_local_rank(self, src, "src")
        return _orig_recv(self, size, dtype, src)

    GroupCoordinator.send = _patched_send
    GroupCoordinator.recv = _patched_recv
    _PATCHED = True
