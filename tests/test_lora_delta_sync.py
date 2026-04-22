"""Unit tests for LoRA delta weight synchronization logic.

Tests cover:
- LoRA weight extraction (filtering lora_A/lora_B parameters)
- Base weight filtering (excluding lora_ prefixed parameters)
- base_sync_done state management on WeightUpdateMeta
- WeightUpdateMeta serialization / round-trip
- TrainEngineConfig.lora_delta_sync default value and validation
- SGLang backend load_lora_adapter and disk update call paths (mocked)

Note: When lora_delta_sync is enabled, both base-model weights and adapter
weights are synced via disk (no NCCL process group required).  Base-model
weights use ``/update_weights_from_disk`` and adapter weights use
``/load_lora_adapter``.
"""

import copy
import dataclasses
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn as nn

from areal.api import ModelAllocation, ParamSpec, WeightUpdateMeta
from areal.api.cli_args import TrainEngineConfig
from areal.api.io_struct import get_versioned_lora_name

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_model_params():
    """Return a dict simulating named_parameters of a LoRA-wrapped model.

    Keys follow the PEFT naming convention produced by
    ``peft.get_peft_model`` on a HuggingFace transformer.
    """
    params = {
        # Base model weights (non-LoRA)
        "base_model.model.model.embed_tokens.weight": torch.randn(1000, 64),
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.randn(64, 64),
        "base_model.model.model.layers.0.self_attn.k_proj.base_layer.weight": torch.randn(64, 64),
        "base_model.model.model.layers.0.self_attn.v_proj.base_layer.weight": torch.randn(64, 64),
        "base_model.model.model.layers.0.self_attn.o_proj.base_layer.weight": torch.randn(64, 64),
        "base_model.model.lm_head.weight": torch.randn(1000, 64),
        # LoRA adapter weights
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(8, 64),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(64, 8),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight": torch.randn(8, 64),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight": torch.randn(64, 8),
    }
    return params


def _is_lora_param(name: str) -> bool:
    """Check if a parameter name corresponds to a LoRA adapter weight."""
    return "lora_A" in name or "lora_B" in name


def _is_base_param(name: str) -> bool:
    """Check if a parameter name corresponds to a base model weight (non-LoRA)."""
    return "lora_" not in name


def _filter_lora_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract only LoRA adapter parameters from a full state dict.

    This mirrors the logic used in FSDPEngine._update_weights_delta_sync_disk
    when base_sync_done=True: only parameters whose names contain 'lora_A'
    or 'lora_B' are saved to disk and loaded via ``/load_lora_adapter``.
    """
    return {k: v for k, v in params.items() if _is_lora_param(k)}


def _filter_base_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract only base model parameters (exclude lora_ prefixed params).

    This mirrors the logic for the first sync (base_sync_done=False):
    only non-LoRA parameters are saved to disk via
    ``/update_weights_from_disk`` so the inference engine can load
    the base model.
    """
    return {k: v for k, v in params.items() if _is_base_param(k)}


# ===========================================================================
# Test: LoRA weight extraction
# ===========================================================================

class TestLoRAWeightExtraction:
    """Test that LoRA adapter weights can be correctly separated from base weights."""

    def test_filter_lora_params_returns_only_lora(self):
        params = _make_dummy_model_params()
        lora_params = _filter_lora_params(params)

        assert len(lora_params) == 4
        for name in lora_params:
            assert "lora_A" in name or "lora_B" in name

    def test_filter_lora_params_excludes_base(self):
        params = _make_dummy_model_params()
        lora_params = _filter_lora_params(params)

        for name in lora_params:
            assert "base_layer" not in name or "lora_" in name

    def test_filter_lora_params_empty_when_no_lora(self):
        params = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        }
        lora_params = _filter_lora_params(params)
        assert len(lora_params) == 0

    def test_lora_param_shapes_match_rank(self):
        """LoRA A has shape (rank, in_features), B has (out_features, rank)."""
        params = _make_dummy_model_params()
        lora_params = _filter_lora_params(params)

        lora_rank = 8
        for name, tensor in lora_params.items():
            if "lora_A" in name:
                assert tensor.shape[0] == lora_rank
            elif "lora_B" in name:
                assert tensor.shape[1] == lora_rank


# ===========================================================================
# Test: base weight filtering
# ===========================================================================

class TestBaseWeightFiltering:
    """Test that base model weights are correctly filtered (lora_ excluded)."""

    def test_filter_base_params_excludes_lora(self):
        params = _make_dummy_model_params()
        base_params = _filter_base_params(params)

        for name in base_params:
            assert "lora_A" not in name
            assert "lora_B" not in name

    def test_filter_base_params_count(self):
        params = _make_dummy_model_params()
        base_params = _filter_base_params(params)
        # 6 base params in the dummy model
        assert len(base_params) == 6

    def test_base_and_lora_cover_all(self):
        params = _make_dummy_model_params()
        base_params = _filter_base_params(params)
        lora_params = _filter_lora_params(params)
        assert len(base_params) + len(lora_params) == len(params)

    def test_no_overlap_between_base_and_lora(self):
        params = _make_dummy_model_params()
        base_names = set(_filter_base_params(params).keys())
        lora_names = set(_filter_lora_params(params).keys())
        assert base_names & lora_names == set()


# ===========================================================================
# Test: base_sync_done state management
# ===========================================================================

class TestBaseSyncDoneState:
    """Test base_sync_done flag on WeightUpdateMeta."""

    def test_default_base_sync_done_is_false(self):
        meta = WeightUpdateMeta(type="xccl")
        assert meta.base_sync_done is False

    def test_set_base_sync_done(self):
        meta = WeightUpdateMeta(type="xccl")
        meta.base_sync_done = True
        assert meta.base_sync_done is True

    def test_lora_delta_sync_default_is_false(self):
        meta = WeightUpdateMeta(type="xccl")
        assert meta.lora_delta_sync is False

    def test_lora_delta_sync_enable(self):
        meta = WeightUpdateMeta(type="xccl", lora_delta_sync=True)
        assert meta.lora_delta_sync is True

    def test_first_sync_sends_base_then_subsequent_sends_lora(self):
        """Simulate the two-phase sync lifecycle."""
        meta = WeightUpdateMeta(
            type="xccl",
            use_lora=True,
            lora_delta_sync=True,
            lora_name="test-lora",
        )
        params = _make_dummy_model_params()

        # Phase 1: first sync -- base_sync_done=False => send base weights
        assert meta.base_sync_done is False
        base_params = _filter_base_params(params)
        assert len(base_params) > 0

        # Mark base sync as complete
        meta.base_sync_done = True

        # Phase 2: subsequent sync -- base_sync_done=True => send only LoRA
        assert meta.base_sync_done is True
        lora_params = _filter_lora_params(params)
        assert len(lora_params) > 0
        assert len(lora_params) < len(params)

    def test_copy_preserves_base_sync_done(self):
        meta = WeightUpdateMeta(type="xccl", base_sync_done=True)
        meta_copy = copy.copy(meta)
        assert meta_copy.base_sync_done is True


# ===========================================================================
# Test: WeightUpdateMeta serialization / round-trip
# ===========================================================================

class TestWeightUpdateMetaSerialization:
    """Test that WeightUpdateMeta can be serialized and deserialized."""

    def test_asdict_round_trip(self):
        meta = WeightUpdateMeta(
            type="xccl",
            use_lora=True,
            lora_name="my-lora",
            lora_int_id=1,
            lora_delta_sync=True,
            base_sync_done=False,
            peft_config={"r": 16, "lora_alpha": 16},
            weight_chunked_mem_mb=512,
        )
        d = asdict(meta)
        assert d["type"] == "xccl"
        assert d["use_lora"] is True
        assert d["lora_delta_sync"] is True
        assert d["base_sync_done"] is False
        assert d["peft_config"]["r"] == 16

    def test_asdict_contains_delta_sync_fields(self):
        meta = WeightUpdateMeta(type="disk")
        d = asdict(meta)
        assert "lora_delta_sync" in d
        assert "base_sync_done" in d

    def test_with_version_preserves_delta_sync(self):
        meta = WeightUpdateMeta(
            type="xccl",
            lora_delta_sync=True,
            base_sync_done=True,
            path="/tmp/weight_update",
        )
        versioned = meta.with_version(3)
        assert versioned.lora_delta_sync is True
        assert versioned.base_sync_done is True
        assert versioned.version == 3
        assert "v3" in versioned.path

    def test_from_fsdp_xccl_with_defaults(self):
        alloc = ModelAllocation.from_str("sglang:d1")
        meta = WeightUpdateMeta.from_fsdp_xccl(gen_allocation=alloc)
        assert meta.lora_delta_sync is False
        assert meta.base_sync_done is False


# ===========================================================================
# Test: TrainEngineConfig.lora_delta_sync
# ===========================================================================

class TestTrainEngineConfigLoraDeltaSync:
    """Test that TrainEngineConfig.lora_delta_sync defaults and propagates."""

    def test_default_is_false(self):
        config = TrainEngineConfig(
            experiment_name="test",
            trial_name="t",
            backend="fsdp:d1",
        )
        assert config.lora_delta_sync is False

    def test_can_enable(self):
        config = TrainEngineConfig(
            experiment_name="test",
            trial_name="t",
            backend="fsdp:d1",
            use_lora=True,
            lora_delta_sync=True,
        )
        assert config.lora_delta_sync is True
        assert config.use_lora is True


# ===========================================================================
# Test: get_versioned_lora_name utility
# ===========================================================================

class TestGetVersionedLoraName:
    def test_basic(self):
        assert get_versioned_lora_name("lora-gsm8k", 1) == "lora-gsm8k-v1"

    def test_version_zero(self):
        assert get_versioned_lora_name("my-lora", 0) == "my-lora-v0"

    def test_version_large(self):
        assert get_versioned_lora_name("adapter", 999) == "adapter-v999"


# ===========================================================================
# Test: ParamSpec construction for LoRA params
# ===========================================================================

class TestParamSpecForLoRA:
    """Test ParamSpec creation from LoRA tensor metadata."""

    def test_param_spec_from_lora_tensor(self):
        tensor = torch.randn(8, 64, dtype=torch.bfloat16)
        spec = ParamSpec(
            name="model.layers.0.self_attn.q_proj.lora_A.weight",
            shape=tuple(tensor.shape),
            dtype="bfloat16",
        )
        assert spec.name == "model.layers.0.self_attn.q_proj.lora_A.weight"
        assert spec.shape == (8, 64)
        assert spec.dtype == "bfloat16"

    def test_param_spec_size(self):
        spec = ParamSpec(name="test", shape=(8, 64), dtype="bfloat16")
        # bfloat16 = 2 bytes, 8*64 = 512 elements => 1024 bytes
        assert spec.size == 1024

    def test_param_spec_size_float32(self):
        spec = ParamSpec(name="test", shape=(16, 32), dtype="float32")
        # float32 = 4 bytes, 16*32 = 512 => 2048
        assert spec.size == 2048


# ===========================================================================
# Test: SGLang load_lora_adapter mock
# ===========================================================================

class TestSGLangLoadLoRAAdapterMock:
    """Test that the SGLang backend's load_lora_adapter is called correctly
    during LoRA delta sync via mocking."""

    def test_build_disk_lora_update_request(self):
        """Verify that build_disk_weight_update_requests produces the correct
        /load_lora_adapter endpoint when use_lora=True."""
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        meta = WeightUpdateMeta(
            type="disk",
            use_lora=True,
            lora_name="test-lora",
            version=2,
            path="/tmp/lora_weights",
        )
        requests = backend.build_disk_weight_update_requests(meta)
        assert len(requests.requests) == 1
        req = requests.requests[0]
        assert req.endpoint == "/load_lora_adapter"
        assert req.payload["lora_name"] == "test-lora-v2"
        assert req.payload["lora_path"] == "/tmp/lora_weights"

    def test_build_distributed_rejects_lora_without_delta_sync(self):
        """SGLang distributed update currently raises for use_lora=True
        (unless lora_delta_sync changes this in the future)."""
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        meta = WeightUpdateMeta(
            type="xccl",
            use_lora=True,
        )
        with pytest.raises(ValueError, match="does not support LoRA"):
            backend.build_distributed_weight_update_requests(meta, [])

    def test_build_distributed_base_sync_no_lora(self):
        """When use_lora=False, standard distributed update works."""
        from areal.engine.sglang_remote import SGLangBackend

        backend = SGLangBackend()
        meta = WeightUpdateMeta(
            type="xccl",
            use_lora=False,
            nccl_group_name="test_group",
        )
        specs = [
            ParamSpec(name="model.embed_tokens.weight", shape=(1000, 64), dtype="bfloat16"),
        ]
        requests = backend.build_distributed_weight_update_requests(meta, specs)
        assert len(requests.requests) == 1
        req = requests.requests[0]
        assert req.endpoint == "/update_weights_from_distributed"
        assert "model.embed_tokens.weight" in req.payload["names"]


# ===========================================================================
# Test: Delta sync parameter selection logic
# ===========================================================================

class TestDeltaSyncParameterSelection:
    """Test the parameter selection logic that would be used in
    FSDPEngine._update_weights_from_distributed when lora_delta_sync is enabled."""

    def _select_params_for_sync(
        self,
        all_params: dict[str, torch.Tensor],
        lora_delta_sync: bool,
        base_sync_done: bool,
        use_lora: bool,
    ) -> dict[str, torch.Tensor]:
        """Simulate the parameter selection logic.

        When lora_delta_sync=True (disk-based delta sync):
          - First sync (base_sync_done=False): send ALL parameters
            (base via /update_weights_from_disk + lora via /load_lora_adapter).
          - Subsequent syncs (base_sync_done=True): send ONLY lora parameters
            (via /load_lora_adapter).

        When lora_delta_sync=False and use_lora=True:
          - Always send only trainable (lora) parameters (existing behavior).
        """
        if not use_lora:
            return all_params

        if lora_delta_sync:
            if not base_sync_done:
                # First sync: send everything
                return all_params
            else:
                # Subsequent: only LoRA adapter params
                return _filter_lora_params(all_params)
        else:
            # Non-delta-sync LoRA: only trainable (LoRA) params
            return _filter_lora_params(all_params)

    def test_first_sync_sends_all(self):
        params = _make_dummy_model_params()
        selected = self._select_params_for_sync(
            params, lora_delta_sync=True, base_sync_done=False, use_lora=True
        )
        assert len(selected) == len(params)

    def test_subsequent_sync_sends_only_lora(self):
        params = _make_dummy_model_params()
        selected = self._select_params_for_sync(
            params, lora_delta_sync=True, base_sync_done=True, use_lora=True
        )
        assert len(selected) == 4
        for name in selected:
            assert _is_lora_param(name)

    def test_non_delta_sync_always_sends_lora_only(self):
        params = _make_dummy_model_params()
        # base_sync_done should not matter when lora_delta_sync=False
        for bsd in [False, True]:
            selected = self._select_params_for_sync(
                params, lora_delta_sync=False, base_sync_done=bsd, use_lora=True
            )
            assert len(selected) == 4

    def test_no_lora_sends_all(self):
        params = _make_dummy_model_params()
        selected = self._select_params_for_sync(
            params, lora_delta_sync=False, base_sync_done=False, use_lora=False
        )
        assert len(selected) == len(params)

    def test_subsequent_sync_much_smaller(self):
        """Delta sync should transmit significantly fewer bytes on subsequent syncs."""
        params = _make_dummy_model_params()
        first = self._select_params_for_sync(
            params, lora_delta_sync=True, base_sync_done=False, use_lora=True
        )
        subsequent = self._select_params_for_sync(
            params, lora_delta_sync=True, base_sync_done=True, use_lora=True
        )
        first_bytes = sum(t.numel() * t.element_size() for t in first.values())
        subsequent_bytes = sum(t.numel() * t.element_size() for t in subsequent.values())
        assert subsequent_bytes < first_bytes
        # In practice, LoRA params are a small fraction of total
        assert subsequent_bytes < first_bytes * 0.5
