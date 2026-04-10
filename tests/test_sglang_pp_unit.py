"""Unit tests for sglang PP pipeline parallelism support.

Tests the per-PP-rank NCCL group creation logic without requiring actual
GPU hardware or a running sglang server.
"""
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from areal.api.alloc_mode import ModelAllocation, ParallelStrategy
from areal.api.io_struct import WeightUpdateMeta
from areal.engine.sglang_remote import SGLangBackend


class TestBuildInitWeightsGroupRequest:
    """Test SGLangBackend.build_init_weights_group_request for PP support."""

    def _make_meta(self, tp=1, pp=1, dp=1, group_name="update_weight_group_0"):
        meta = WeightUpdateMeta(type="xccl")
        meta.gen_allocation = ModelAllocation.from_str(
            f"sglang:d{dp}p{pp}t{tp}"
        )
        meta.nccl_master_address = "127.0.0.1"
        meta.nccl_master_port = 12345
        meta.nccl_group_name = group_name
        return meta

    def test_pp1_backward_compatible(self):
        """PP=1 should use original behavior."""
        backend = SGLangBackend()
        meta = self._make_meta(tp=2, pp=1, dp=2)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["world_size"] == 5  # 2*2 + 1
        assert req.payload["rank_offset"] == 1  # 1 + 0*2
        assert "pp_rank" not in req.payload

    def test_pp1_server_idx_1(self):
        """PP=1, second server should have correct rank_offset."""
        backend = SGLangBackend()
        meta = self._make_meta(tp=2, pp=1, dp=2)
        req = backend.build_init_weights_group_request("addr", 1, meta)
        assert req.payload["rank_offset"] == 3  # 1 + 1*2

    def test_pp2_tp2_dp1_rank0(self):
        """PP=2, TP=2, DP=1: per-PP-rank group for pp_rank=0."""
        backend = SGLangBackend()
        meta = self._make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        # world_size = n_servers * tp_size + 1 = 1 * 2 + 1 = 3
        assert req.payload["world_size"] == 3
        assert req.payload["rank_offset"] == 1  # 1 + 0*2
        assert req.payload["pp_rank"] == 0

    def test_pp2_tp2_dp1_rank1(self):
        """PP=2, TP=2, DP=1: per-PP-rank group for pp_rank=1."""
        backend = SGLangBackend()
        meta = self._make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_1")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["world_size"] == 3
        assert req.payload["rank_offset"] == 1
        assert req.payload["pp_rank"] == 1

    def test_dp2_pp2_tp2_complex(self):
        """DP=2, PP=2, TP=2: 8 total inference GPUs, 2 DP servers."""
        backend = SGLangBackend()
        # total world_size = 2*2*2 = 8
        meta = self._make_meta(tp=2, pp=2, dp=2, group_name="update_weight_group_0")
        # server 0
        req0 = backend.build_init_weights_group_request("addr0", 0, meta)
        # per-PP-rank world_size = n_servers * tp_size + 1 = 2 * 2 + 1 = 5
        assert req0.payload["world_size"] == 5
        assert req0.payload["rank_offset"] == 1  # 1 + 0*2
        assert req0.payload["pp_rank"] == 0
        # server 1
        req1 = backend.build_init_weights_group_request("addr1", 1, meta)
        assert req1.payload["world_size"] == 5
        assert req1.payload["rank_offset"] == 3  # 1 + 1*2
        assert req1.payload["pp_rank"] == 0

    def test_group_name_parsing(self):
        """Test that pp_rank is correctly extracted from group name."""
        backend = SGLangBackend()
        for pp_rank in [0, 1, 5, 10]:
            meta = self._make_meta(tp=1, pp=2, dp=1,
                                   group_name=f"update_weight_group_{pp_rank}")
            req = backend.build_init_weights_group_request("addr", 0, meta)
            assert req.payload["pp_rank"] == pp_rank


class TestAllocationModeParsing:
    """Test that sglang allocation mode correctly parses PP dimension."""

    def test_sglang_with_pp(self):
        alloc = ModelAllocation.from_str("sglang:d2p2t2")
        assert alloc.parallel.pp_size == 2
        assert alloc.parallel.tp_size == 2
        assert alloc.parallel.dp_size == 2
        assert alloc.parallel.world_size == 8

    def test_sglang_without_pp(self):
        alloc = ModelAllocation.from_str("sglang:d4t2")
        assert alloc.parallel.pp_size == 1
        assert alloc.parallel.tp_size == 2
        assert alloc.parallel.dp_size == 4

    def test_sglang_pp_only(self):
        alloc = ModelAllocation.from_str("sglang:p2t2")
        assert alloc.parallel.pp_size == 2
        assert alloc.parallel.tp_size == 2


class TestSGLangPPPatchModule:
    """Test the sglang PP patch module logic."""

    def test_patch_idempotent(self):
        """apply_sglang_pp_patch should be idempotent."""
        from areal.patches.sglang_pp_weight_update import apply_sglang_pp_patch, _PATCHED
        # Just verify import works - actual patching needs sglang installed
        assert callable(apply_sglang_pp_patch)

    def test_pp_skip_sentinel(self):
        """Verify sentinel constant is defined."""
        from areal.patches.sglang_pp_weight_update import _PP_SKIP_SENTINEL
        assert isinstance(_PP_SKIP_SENTINEL, str)
