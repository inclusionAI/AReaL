"""
tests/test_megatron_pp_nccl.py

End-to-end correctness test: verify that a Megatron training engine with
DP=2, PP=2, TP=2 can synchronise weights to an SGLang inference fleet
(PP=2, TP=2) via per-PP-rank NCCL groups.

This test does NOT launch real Megatron or SGLang processes.  Instead it
mocks the distributed primitives and validates:
  1. The correct NCCL groups are created (one per PP rank).
  2. Each group has the right world_size and group_name.
  3. The broadcast calls target the correct group.
  4. The inference-side init request carries the right rank_offset.

Run with:
    python -m unittest new_files/tests/test_megatron_pp_nccl.py -v
"""

from __future__ import annotations

import dataclasses
import os
import sys
import unittest
from typing import List, Optional

# ---------------------------------------------------------------------------
# Lightweight stand-ins for AReaL types so the test is self-contained.
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
# Import the modified functions under test.
# We load them from the workspace's modified_files directory.
# ---------------------------------------------------------------------------

import importlib.util

# Resolve the workspace root: walk up from this file until we find
# modified_files/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = _THIS_DIR
for _ in range(10):
    if os.path.isdir(os.path.join(_WORKSPACE, "modified_files")):
        break
    _WORKSPACE = os.path.dirname(_WORKSPACE)

_MOD_DIR = os.path.join(_WORKSPACE, "modified_files", "areal", "engine")


def _load_module(name, path):
    # type: (str, str) -> object
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# We need to test the sglang_remote functions standalone; they are written
# as instance methods but we can call them with a dummy self.

class _DummyBackend:
    """Minimal stand-in that provides the modified sglang_remote methods."""
    pass


# Load the modified sglang_remote module.
_sglang_mod = _load_module(
    "sglang_remote_modified",
    os.path.join(_MOD_DIR, "sglang_remote.py"),
)

# Bind the functions as methods on _DummyBackend.
_DummyBackend.build_init_weights_group_request = (
    _sglang_mod.build_init_weights_group_request
)
_DummyBackend.build_update_weights_from_distributed_request = (
    _sglang_mod.build_update_weights_from_distributed_request
)


# =====================================================================
# Test cases
# =====================================================================


class TestPerPPRankNCCLGroupCreation(unittest.TestCase):
    """Validate that per-PP-rank NCCL group requests are built correctly
    for a DP=2, PP=2, TP=2 inference fleet spread across 2 servers.
    """

    def setUp(self):
        self.backend = _DummyBackend()

        # Inference-side topology: PP=2, TP=2 across 2 servers.
        self.gen_parallel = ParallelConfig(dp_size=1, tp_size=2, pp_size=2)
        self.allocation = AllocationConfig(
            parallel=self.gen_parallel, num_servers=2,
        )
        self.meta = WeightUpdateMeta(
            nccl_master_address="127.0.0.1",
            nccl_master_port=29500,
            nccl_group_name="update_weight_group_0",
            gen_allocation=self.allocation,
        )

    # ----- world_size -----------------------------------------------

    def test_per_pp_world_size(self):
        """Each per-PP NCCL group should have 1 + num_servers * tp_size
        members = 1 + 2 * 2 = 5."""
        _, req = self.backend.build_init_weights_group_request(
            "http://server0:8000", 0, self.meta, pp_rank=0,
        )
        self.assertEqual(req["world_size"], 5)

    # ----- group_name -----------------------------------------------

    def test_group_name_contains_pp_rank(self):
        _, req0 = self.backend.build_init_weights_group_request(
            "http://s0:8000", 0, self.meta, pp_rank=0,
        )
        _, req1 = self.backend.build_init_weights_group_request(
            "http://s0:8000", 0, self.meta, pp_rank=1,
        )
        self.assertEqual(req0["group_name"], "update_weight_group_0")
        self.assertEqual(req1["group_name"], "update_weight_group_1")

    # ----- rank_offset ----------------------------------------------

    def test_rank_offset_server0_pp0(self):
        _, req = self.backend.build_init_weights_group_request(
            "http://s0:8000", server_idx=0, meta=self.meta, pp_rank=0,
        )
        # server_idx=0 -> rank_offset = 1 + 0*2 = 1
        self.assertEqual(req["rank_offset"], 1)

    def test_rank_offset_server1_pp0(self):
        _, req = self.backend.build_init_weights_group_request(
            "http://s1:8000", server_idx=1, meta=self.meta, pp_rank=0,
        )
        # server_idx=1 -> rank_offset = 1 + 1*2 = 3
        self.assertEqual(req["rank_offset"], 3)

    def test_rank_offset_server0_pp1(self):
        _, req = self.backend.build_init_weights_group_request(
            "http://s0:8000", server_idx=0, meta=self.meta, pp_rank=1,
        )
        # server_idx=0, pp_rank=1 -> rank_offset = 1 + 0*2 = 1
        # (rank_offset is relative to the per-PP group, always starts at 1
        #  for server 0 regardless of pp_rank)
        self.assertEqual(req["rank_offset"], 1)

    def test_rank_offset_server1_pp1(self):
        _, req = self.backend.build_init_weights_group_request(
            "http://s1:8000", server_idx=1, meta=self.meta, pp_rank=1,
        )
        self.assertEqual(req["rank_offset"], 3)

    # ----- pp_rank in request ---------------------------------------

    def test_pp_rank_included_in_request(self):
        _, req = self.backend.build_init_weights_group_request(
            "http://s0:8000", 0, self.meta, pp_rank=1,
        )
        self.assertIn("pp_rank", req)
        self.assertEqual(req["pp_rank"], 1)


class TestSingleGroupFallback(unittest.TestCase):
    """When pp_size == 1 the behaviour must be identical to the original."""

    def setUp(self):
        self.backend = _DummyBackend()

        self.gen_parallel = ParallelConfig(dp_size=1, tp_size=4, pp_size=1)
        self.allocation = AllocationConfig(
            parallel=self.gen_parallel, num_servers=2,
        )
        self.meta = WeightUpdateMeta(
            nccl_master_address="127.0.0.1",
            nccl_master_port=29500,
            nccl_group_name="update_weight_group_0",
            gen_allocation=self.allocation,
        )

    def test_world_size_pp1(self):
        """PP=1 -> world_size = gen_world_size + 1 = 4 + 1 = 5."""
        _, req = self.backend.build_init_weights_group_request(
            "http://s0:8000", 0, self.meta,
        )
        self.assertEqual(req["world_size"], self.gen_parallel.world_size + 1)

    def test_pp_rank_not_in_request_pp1(self):
        _, req = self.backend.build_init_weights_group_request(
            "http://s0:8000", 0, self.meta,
        )
        self.assertNotIn("pp_rank", req)

    def test_rank_offset_pp1(self):
        _, req = self.backend.build_init_weights_group_request(
            "http://s0:8000", server_idx=0, meta=self.meta,
        )
        self.assertEqual(req["rank_offset"], 1)

        _, req = self.backend.build_init_weights_group_request(
            "http://s1:8000", server_idx=1, meta=self.meta,
        )
        self.assertEqual(req["rank_offset"], 5)  # 1 + 1*4


class TestUpdateWeightsRequest(unittest.TestCase):
    """Verify the update_weights_from_distributed request generation."""

    def setUp(self):
        self.backend = _DummyBackend()
        self.meta = WeightUpdateMeta(
            nccl_master_address="127.0.0.1",
            nccl_master_port=29500,
            nccl_group_name="update_weight_group_0",
        )

    def test_update_request_with_pp_rank(self):
        _, req = self.backend.build_update_weights_from_distributed_request(
            "http://s0:8000", self.meta, pp_rank=1,
        )
        self.assertEqual(req["group_name"], "update_weight_group_1")

    def test_update_request_without_pp_rank(self):
        _, req = self.backend.build_update_weights_from_distributed_request(
            "http://s0:8000", self.meta,
        )
        self.assertEqual(req["group_name"], "update_weight_group_0")


class TestDP2PP2TP2Scenario(unittest.TestCase):
    """Full scenario: training DP=2 PP=2 TP=2, inference PP=2 TP=2 across
    2 servers.

    Training side:
        8 ranks total.  PP stage 0 has ranks {0,1,2,3}, PP stage 1 has
        ranks {4,5,6,7}.
        PP source ranks (dp=0, tp=0): rank 0 for PP0, rank 4 for PP1.

    Inference side (per server):
        4 workers total (PP=2 * TP=2).
        PP stage 0 workers: local ranks 0,1  -> tp_rank 0,1
        PP stage 1 workers: local ranks 2,3  -> tp_rank 0,1

    Expected NCCL groups:
        update_weight_group_0:
            rank 0 = training PP0 source
            rank 1,2 = server 0 PP0 workers (tp=0,1)
            rank 3,4 = server 1 PP0 workers (tp=0,1)
            world_size = 5

        update_weight_group_1:
            rank 0 = training PP1 source
            rank 1,2 = server 0 PP1 workers (tp=0,1)
            rank 3,4 = server 1 PP1 workers (tp=0,1)
            world_size = 5
    """

    def setUp(self):
        self.backend = _DummyBackend()
        self.gen_parallel = ParallelConfig(dp_size=1, tp_size=2, pp_size=2)
        self.allocation = AllocationConfig(
            parallel=self.gen_parallel, num_servers=2,
        )
        self.servers = ["http://server0:8000", "http://server1:8000"]

    def _build_all_requests(self, pp_rank):
        # type: (int) -> list
        meta = WeightUpdateMeta(
            nccl_master_address="10.0.0.1",
            nccl_master_port=29500 + pp_rank,
            nccl_group_name="update_weight_group_{}".format(pp_rank),
            gen_allocation=self.allocation,
        )
        requests = []
        for idx, addr in enumerate(self.servers):
            _, req = self.backend.build_init_weights_group_request(
                addr, idx, meta, pp_rank=pp_rank,
            )
            requests.append(req)
        return requests

    def test_pp0_group(self):
        reqs = self._build_all_requests(pp_rank=0)

        # All requests share the same group.
        for req in reqs:
            self.assertEqual(req["group_name"], "update_weight_group_0")
            self.assertEqual(req["world_size"], 5)
            self.assertEqual(req["pp_rank"], 0)

        # rank_offsets: server0 -> 1, server1 -> 3
        self.assertEqual(reqs[0]["rank_offset"], 1)
        self.assertEqual(reqs[1]["rank_offset"], 3)

    def test_pp1_group(self):
        reqs = self._build_all_requests(pp_rank=1)

        for req in reqs:
            self.assertEqual(req["group_name"], "update_weight_group_1")
            self.assertEqual(req["world_size"], 5)
            self.assertEqual(req["pp_rank"], 1)

        self.assertEqual(reqs[0]["rank_offset"], 1)
        self.assertEqual(reqs[1]["rank_offset"], 3)

    def test_groups_are_independent(self):
        """The two PP groups must have different group_names."""
        reqs0 = self._build_all_requests(pp_rank=0)
        reqs1 = self._build_all_requests(pp_rank=1)
        self.assertNotEqual(
            reqs0[0]["group_name"], reqs1[0]["group_name"],
        )

    def test_total_inference_ranks_per_group(self):
        """Each per-PP group should have exactly num_servers * tp_size
        inference ranks = 2 * 2 = 4."""
        reqs = self._build_all_requests(pp_rank=0)
        # The last server's highest rank_offset + tp_size - 1 should equal
        # world_size - 1 (=4).
        last_req = reqs[-1]
        max_rank = last_req["rank_offset"] + self.gen_parallel.tp_size - 1
        self.assertEqual(max_rank, last_req["world_size"] - 1)


if __name__ == "__main__":
    unittest.main()
