"""
tests/test_per_pp_rank_nccl_group.py

Unit tests for the per-PP-rank NCCL group design:
  - NCCL group naming conventions
  - world_size computation
  - rank_offset computation for various topologies
  - Compatibility with PP=1 (no regression)

These tests are pure-Python (no GPU, no torch.distributed) and validate
the *logic* of group construction rather than actual NCCL communication.

Run with:
    pytest tests/test_per_pp_rank_nccl_group.py -v
"""

from __future__ import annotations

import dataclasses
import itertools
import unittest
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lightweight stand-ins for AReaL types (same as in test_megatron_pp_nccl.py)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ParallelConfig:
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1

    @property
    def world_size(self):
        # type: () -> int
        return self.dp_size * self.tp_size * self.pp_size


@dataclasses.dataclass
class AllocationConfig:
    parallel: ParallelConfig = dataclasses.field(default_factory=ParallelConfig)
    num_servers: int = 1


@dataclasses.dataclass
class WeightUpdateMeta:
    nccl_master_address: str = ""
    nccl_master_port: int = 0
    nccl_group_name: str = ""
    gen_allocation: AllocationConfig = dataclasses.field(
        default_factory=AllocationConfig
    )
    pp_rank: Optional[int] = None
    param_names: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Computation helpers -- these mirror the logic in the modified source files
# and are tested independently here to ensure correctness.
# ---------------------------------------------------------------------------


def compute_per_pp_world_size(num_servers, tp_size):
    # type: (int, int) -> int
    """World size for one per-PP-rank NCCL group.

    = 1 (training source) + num_servers * tp_size (inference workers).
    """
    return 1 + num_servers * tp_size


def compute_rank_offset(server_idx, tp_size):
    # type: (int, int) -> int
    """Rank offset for server ``server_idx`` inside a per-PP NCCL group.

    Training source is rank 0.  Server 0 starts at rank 1.
    """
    return 1 + server_idx * tp_size


def compute_group_name(pp_rank):
    # type: (int) -> str
    return "update_weight_group_{}".format(pp_rank)


def compute_pp1_world_size(parallel):
    # type: (ParallelConfig) -> int
    """Legacy single-group world_size: gen_world_size + 1."""
    return parallel.world_size + 1


def is_pp_source_rank(dp_rank, tp_rank):
    # type: (int, int) -> bool
    """Only dp_rank=0 and tp_rank=0 is the PP source rank for a given PP
    stage."""
    return dp_rank == 0 and tp_rank == 0


# =====================================================================
# Test cases
# =====================================================================


class TestGroupNaming(unittest.TestCase):
    """Verify NCCL group naming follows the pattern
    ``update_weight_group_{pp_rank}``."""

    def test_pp0(self):
        self.assertEqual(compute_group_name(0), "update_weight_group_0")

    def test_pp3(self):
        self.assertEqual(compute_group_name(3), "update_weight_group_3")

    def test_names_unique(self):
        names = {compute_group_name(i) for i in range(8)}
        self.assertEqual(len(names), 8)


class TestWorldSizeComputation(unittest.TestCase):
    """Validate per-PP world_size = 1 + num_servers * tp_size."""

    def test_basic(self):
        # 2 servers, TP=2 -> 1 + 2*2 = 5
        self.assertEqual(compute_per_pp_world_size(2, 2), 5)

    def test_single_server(self):
        self.assertEqual(compute_per_pp_world_size(1, 4), 5)

    def test_four_servers_tp8(self):
        self.assertEqual(compute_per_pp_world_size(4, 8), 33)

    def test_one_server_tp1(self):
        self.assertEqual(compute_per_pp_world_size(1, 1), 2)


class TestRankOffsetComputation(unittest.TestCase):
    """Validate rank_offset = 1 + server_idx * tp_size."""

    def test_server0_tp2(self):
        self.assertEqual(compute_rank_offset(0, 2), 1)

    def test_server1_tp2(self):
        self.assertEqual(compute_rank_offset(1, 2), 3)

    def test_server3_tp4(self):
        self.assertEqual(compute_rank_offset(3, 4), 13)

    def test_offsets_are_contiguous(self):
        """Offsets for consecutive servers should be contiguous blocks of
        tp_size ranks."""
        tp = 4
        for s in range(5):
            off = compute_rank_offset(s, tp)
            next_off = compute_rank_offset(s + 1, tp)
            self.assertEqual(next_off - off, tp)

    def test_last_rank_equals_world_size_minus_1(self):
        """The highest rank in the group (last server, last TP worker)
        should be world_size - 1."""
        num_servers, tp_size = 3, 4
        ws = compute_per_pp_world_size(num_servers, tp_size)
        last_server_offset = compute_rank_offset(num_servers - 1, tp_size)
        max_rank = last_server_offset + tp_size - 1
        self.assertEqual(max_rank, ws - 1)


class TestPP1Compatibility(unittest.TestCase):
    """When pp_size == 1 the legacy formula must produce the same result as
    before: world_size = dp * tp * pp + 1 = gen_world_size + 1."""

    def test_dp2_tp4_pp1(self):
        p = ParallelConfig(dp_size=2, tp_size=4, pp_size=1)
        self.assertEqual(compute_pp1_world_size(p), 9)

    def test_dp1_tp1_pp1(self):
        p = ParallelConfig(dp_size=1, tp_size=1, pp_size=1)
        self.assertEqual(compute_pp1_world_size(p), 2)


class TestPPSourceRank(unittest.TestCase):
    """Only dp_rank=0 && tp_rank=0 should be PP source."""

    def test_source(self):
        self.assertTrue(is_pp_source_rank(0, 0))

    def test_non_source_dp(self):
        self.assertFalse(is_pp_source_rank(1, 0))

    def test_non_source_tp(self):
        self.assertFalse(is_pp_source_rank(0, 1))

    def test_non_source_both(self):
        self.assertFalse(is_pp_source_rank(1, 1))

    def test_exactly_one_source_per_pp_stage(self):
        """In a DP=2, TP=2 setup each PP stage has 4 ranks.
        Exactly 1 should be the source."""
        dp_size, tp_size = 2, 2
        sources = [
            (d, t)
            for d, t in itertools.product(range(dp_size), range(tp_size))
            if is_pp_source_rank(d, t)
        ]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0], (0, 0))


class TestParametricTopologies(unittest.TestCase):
    """Sweep over a range of topologies and check invariants."""

    TOPOLOGIES = [
        # (num_servers, tp_size, pp_size)
        (1, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
        (2, 4, 4),
        (4, 8, 2),
        (8, 8, 4),
    ]

    def test_world_size_positive(self):
        for ns, tp, pp in self.TOPOLOGIES:
            ws = compute_per_pp_world_size(ns, tp)
            self.assertGreater(ws, 1, msg="ns={}, tp={}, pp={}".format(ns, tp, pp))

    def test_rank_offsets_no_overlap(self):
        """Rank ranges for different servers must not overlap within a
        single per-PP group."""
        for ns, tp, pp in self.TOPOLOGIES:
            for pp_rank in range(pp):
                ranges = []
                for s in range(ns):
                    off = compute_rank_offset(s, tp)
                    ranges.append(set(range(off, off + tp)))
                # All pairwise disjoint.
                for i in range(len(ranges)):
                    for j in range(i + 1, len(ranges)):
                        self.assertTrue(
                            ranges[i].isdisjoint(ranges[j]),
                            msg="Overlap for ns={}, tp={}, pp_rank={}".format(
                                ns, tp, pp_rank
                            ),
                        )

    def test_rank_0_is_training_only(self):
        """Rank 0 must never appear in any server's rank range (it is
        reserved for the training source)."""
        for ns, tp, pp in self.TOPOLOGIES:
            for s in range(ns):
                off = compute_rank_offset(s, tp)
                self.assertGreaterEqual(off, 1)

    def test_total_groups_equals_pp_size(self):
        """The number of independent NCCL groups created should equal
        pp_size."""
        for ns, tp, pp in self.TOPOLOGIES:
            names = {compute_group_name(r) for r in range(pp)}
            self.assertEqual(len(names), pp)


if __name__ == "__main__":
    unittest.main()
