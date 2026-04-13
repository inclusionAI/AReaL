"""Unit tests for sglang PP pipeline parallelism support.

Tests the per-PP-rank NCCL group creation logic in SGLangBackend and related
allocation mode parsing without requiring actual GPU hardware or a running
sglang server.

Covers three scenarios:
  1. PP=1 (original / backward compatible)
  2. PP>1 with per-PP-rank groups (Megatron-style, group name ends with _{digit})
  3. PP>1 with single group (FSDP-style, group name without PP rank suffix)

Also tests allocation mode parsing with PP dimension and patch module imports.
"""
import pytest
from unittest.mock import MagicMock, patch

from areal.api.alloc_mode import ModelAllocation, ParallelStrategy
from areal.api.io_struct import WeightUpdateMeta
from areal.engine.sglang_remote import SGLangBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(tp=1, pp=1, dp=1, group_name="update_weight_group_0"):
    """Build a WeightUpdateMeta with the given parallel dimensions."""
    meta = WeightUpdateMeta(type="xccl")
    meta.gen_allocation = ModelAllocation.from_str(f"sglang:d{dp}p{pp}t{tp}")
    meta.nccl_master_address = "127.0.0.1"
    meta.nccl_master_port = 12345
    meta.nccl_group_name = group_name
    return meta


# ===================================================================== #
#  Scenario 1: PP=1 (backward compatible, single group)                 #
# ===================================================================== #

class TestPP1BackwardCompatible:
    """PP=1 should use original behavior: single NCCL group, no pp_rank."""

    def test_pp1_tp2_dp2_server0(self):
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=2)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        # world_size = total_gen_workers + 1 = 2*1*2 + 1 = 5
        assert req.payload["world_size"] == 5
        # rank_offset = 1 + server_idx * tp_size = 1 + 0*2 = 1
        assert req.payload["rank_offset"] == 1
        assert "pp_rank" not in req.payload

    def test_pp1_tp2_dp2_server1(self):
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=2)
        req = backend.build_init_weights_group_request("addr", 1, meta)
        # rank_offset = 1 + 1*2 = 3
        assert req.payload["rank_offset"] == 3
        assert req.payload["world_size"] == 5
        assert "pp_rank" not in req.payload

    def test_pp1_tp1_dp1(self):
        """Simplest case: single GPU inference."""
        backend = SGLangBackend()
        meta = _make_meta(tp=1, pp=1, dp=1)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["world_size"] == 2  # 1 + 1
        assert req.payload["rank_offset"] == 1
        assert "pp_rank" not in req.payload

    def test_pp1_tp4_dp1(self):
        backend = SGLangBackend()
        meta = _make_meta(tp=4, pp=1, dp=1)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["world_size"] == 5  # 4 + 1
        assert req.payload["rank_offset"] == 1
        assert "pp_rank" not in req.payload

    def test_pp1_group_name_preserved(self):
        """Group name from meta should be passed through unchanged."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=1, group_name="my_custom_group")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["group_name"] == "my_custom_group"

    def test_pp1_endpoint(self):
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=1)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.endpoint == "/init_weights_update_group"

    def test_pp1_master_address_and_port(self):
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=1)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["master_port"] == str(12345)


# ===================================================================== #
#  Scenario 2: PP>1 with per-PP-rank groups (Megatron engine style)     #
# ===================================================================== #

class TestPerPPRankGroups:
    """PP>1, group name ends with _{digit} -> per-PP-rank groups."""

    def test_pp2_tp2_dp1_rank0(self):
        """PP=2, TP=2, DP=1: per-PP-rank group for pp_rank=0."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        # n_servers = world_size / (tp * pp) = 4 / (2*2) = 1
        # per-PP world_size = n_servers * tp + 1 = 1*2 + 1 = 3
        assert req.payload["world_size"] == 3
        # rank_offset = 1 + server_idx * tp = 1 + 0*2 = 1
        assert req.payload["rank_offset"] == 1
        assert req.payload["pp_rank"] == 0

    def test_pp2_tp2_dp1_rank1(self):
        """PP=2, TP=2, DP=1: per-PP-rank group for pp_rank=1."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_1")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["world_size"] == 3
        assert req.payload["rank_offset"] == 1
        assert req.payload["pp_rank"] == 1

    def test_dp2_pp2_tp2_server0(self):
        """DP=2, PP=2, TP=2: 8 total inference GPUs, server 0."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=2, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr0", 0, meta)
        # n_servers = 8 / (2*2) = 2
        # per-PP world_size = 2*2 + 1 = 5
        assert req.payload["world_size"] == 5
        assert req.payload["rank_offset"] == 1  # 1 + 0*2
        assert req.payload["pp_rank"] == 0

    def test_dp2_pp2_tp2_server1(self):
        """DP=2, PP=2, TP=2: 8 total inference GPUs, server 1."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=2, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr1", 1, meta)
        assert req.payload["world_size"] == 5
        assert req.payload["rank_offset"] == 3  # 1 + 1*2
        assert req.payload["pp_rank"] == 0

    def test_dp2_pp2_tp2_rank1_server0(self):
        """DP=2, PP=2, TP=2: pp_rank=1, server 0."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=2, group_name="update_weight_group_1")
        req = backend.build_init_weights_group_request("addr0", 0, meta)
        assert req.payload["world_size"] == 5
        assert req.payload["rank_offset"] == 1
        assert req.payload["pp_rank"] == 1

    def test_pp4_tp2_dp1_rank3(self):
        """PP=4 with higher pp_rank to verify general extraction."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=4, dp=1, group_name="update_weight_group_3")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        # n_servers = 8 / (2*4) = 1
        # per-PP world = 1*2 + 1 = 3
        assert req.payload["world_size"] == 3
        assert req.payload["pp_rank"] == 3

    def test_group_name_with_pp_rank_preserved(self):
        """The full group name should be preserved in payload."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["group_name"] == "update_weight_group_0"


# ===================================================================== #
#  Scenario 3: PP>1 with single group (FSDP engine style)              #
# ===================================================================== #

class TestSingleGroupWithPP:
    """PP>1, group name without digit suffix -> single group for all workers."""

    def test_pp2_tp2_dp1_single_group(self):
        """PP=2, TP=2, DP=1: single group, all workers join."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        # instance_size = tp * pp = 2*2 = 4
        # world_size = total_gen_workers + 1 = 4 + 1 = 5
        assert req.payload["world_size"] == 5
        # rank_offset = 1 + 0 * instance_size = 1
        assert req.payload["rank_offset"] == 1
        assert "pp_rank" not in req.payload

    def test_pp2_tp2_dp2_single_group_server0(self):
        """PP=2, TP=2, DP=2: single group, server 0."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=2, group_name="update_weight_group")
        req = backend.build_init_weights_group_request("addr0", 0, meta)
        # instance_size = 2*2 = 4
        # world_size = 8 + 1 = 9
        assert req.payload["world_size"] == 9
        assert req.payload["rank_offset"] == 1  # 1 + 0*4
        assert "pp_rank" not in req.payload

    def test_pp2_tp2_dp2_single_group_server1(self):
        """PP=2, TP=2, DP=2: single group, server 1."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=2, group_name="update_weight_group")
        req = backend.build_init_weights_group_request("addr1", 1, meta)
        # rank_offset = 1 + 1*4 = 5
        assert req.payload["rank_offset"] == 5
        assert req.payload["world_size"] == 9
        assert "pp_rank" not in req.payload

    def test_pp2_tp1_dp4_single_group_server2(self):
        """PP=2, TP=1, DP=4: single group, server 2."""
        backend = SGLangBackend()
        meta = _make_meta(tp=1, pp=2, dp=4, group_name="update_weight_group")
        req = backend.build_init_weights_group_request("addr2", 2, meta)
        # instance_size = 1*2 = 2
        # rank_offset = 1 + 2*2 = 5
        assert req.payload["rank_offset"] == 5
        # world_size = 8 + 1 = 9
        assert req.payload["world_size"] == 9
        assert "pp_rank" not in req.payload


# ===================================================================== #
#  Group name parsing edge cases                                        #
# ===================================================================== #

class TestGroupNameParsing:
    """Test that pp_rank extraction from group name handles edge cases."""

    def test_sequential_pp_ranks(self):
        """All pp_rank values from 0..N should be correctly extracted."""
        backend = SGLangBackend()
        for pp_rank in [0, 1, 5, 10]:
            meta = _make_meta(
                tp=1, pp=2, dp=1, group_name=f"update_weight_group_{pp_rank}"
            )
            req = backend.build_init_weights_group_request("addr", 0, meta)
            assert req.payload["pp_rank"] == pp_rank

    def test_group_name_no_suffix_is_single_group(self):
        """Group name without trailing digit -> Scenario 3 (single group)."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert "pp_rank" not in req.payload

    def test_group_name_with_text_suffix_is_single_group(self):
        """Group name ending with non-digit -> Scenario 3."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_abc")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert "pp_rank" not in req.payload

    def test_group_name_digit_suffix_only_triggers_when_pp_gt_1(self):
        """Even with digit suffix, PP=1 should use Scenario 1 (no pp_rank)."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        # PP=1 -> Scenario 1, no pp_rank regardless of group name
        assert "pp_rank" not in req.payload


# ===================================================================== #
#  Allocation mode parsing with PP dimension                            #
# ===================================================================== #

class TestAllocationModeParsing:
    """Test that sglang allocation mode correctly parses the PP dimension."""

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

    def test_megatron_with_pp(self):
        alloc = ModelAllocation.from_str("megatron:d2p2t2")
        assert alloc.parallel.pp_size == 2
        assert alloc.parallel.tp_size == 2
        assert alloc.parallel.dp_size == 2
        assert alloc.parallel.world_size == 8

    def test_fsdp_with_pp(self):
        alloc = ModelAllocation.from_str("fsdp:d2p2t2")
        assert alloc.parallel.pp_size == 2
        assert alloc.parallel.tp_size == 2

    def test_world_size_computation(self):
        """world_size = dp * pp * tp."""
        alloc = ModelAllocation.from_str("sglang:d3p2t4")
        assert alloc.parallel.world_size == 3 * 2 * 4


# ===================================================================== #
#  Backward compatibility per engine type                               #
# ===================================================================== #

class TestBackwardCompatibilityPerEngine:
    """Verify that each engine type's group naming convention maps to the
    correct scenario in build_init_weights_group_request."""

    def test_megatron_pp1_uses_scenario1(self):
        """Megatron with PP=1: group_name='update_weight_group_0' but PP=1
        means Scenario 1 (no pp_rank in payload)."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=2, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert "pp_rank" not in req.payload
        assert req.payload["world_size"] == 5  # 4 + 1

    def test_megatron_pp2_uses_scenario2(self):
        """Megatron with PP=2: group_name='update_weight_group_0' and PP>1
        triggers Scenario 2 (per-PP-rank)."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["pp_rank"] == 0
        assert req.payload["world_size"] == 3  # 1*2 + 1

    def test_fsdp_pp1_uses_scenario1(self):
        """FSDP with PP=1: group_name='update_weight_group', Scenario 1."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=2, group_name="update_weight_group")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert "pp_rank" not in req.payload
        assert req.payload["world_size"] == 5  # 4 + 1

    def test_fsdp_pp2_per_pp_rank_groups(self):
        """FSDP with PP=2 but per-PP-rank group names: group_name still
        ends with _{digit}, so it uses Scenario 2 (per-PP-rank).
        Note: FSDP's _init_per_pp_weight_update_groups creates per-PP-rank
        group names like 'update_weight_group_0', same as Megatron."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["pp_rank"] == 0
        assert req.payload["world_size"] == 3


# ===================================================================== #
#  Patch module importability and constants                             #
# ===================================================================== #

class TestSGLangPPPatchModule:
    """Test the sglang PP patch module can be imported and has expected symbols."""

    def test_apply_patch_is_callable(self):
        from areal.patches.sglang_pp_weight_update import apply_sglang_pp_patch
        assert callable(apply_sglang_pp_patch)

    def test_patched_flag_exists(self):
        from areal.patches.sglang_pp_weight_update import _PATCHED
        assert isinstance(_PATCHED, bool)

    def test_pp_skip_sentinel_defined(self):
        from areal.patches.sglang_pp_weight_update import _PP_SKIP_SENTINEL
        assert isinstance(_PP_SKIP_SENTINEL, str)
        assert len(_PP_SKIP_SENTINEL) > 0

    def test_patch_idempotent_flag(self):
        """apply_sglang_pp_patch sets _PATCHED; second call is a no-op."""
        import areal.patches.sglang_pp_weight_update as mod
        # Just verify the flag mechanism exists; actual patching needs sglang.
        assert hasattr(mod, "_PATCHED")
        assert hasattr(mod, "apply_sglang_pp_patch")
