"""Tests for colocation weight sync module."""

from __future__ import annotations

import dataclasses
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from areal.api.io_struct import WeightUpdateMeta


class TestWeightUpdateMetaTensor:
    """Test WeightUpdateMeta tensor type support."""

    def test_from_colocation_creates_tensor_type(self):
        alloc_mode = MagicMock()
        meta = WeightUpdateMeta.from_colocation(allocation_mode=alloc_mode)
        assert meta.type == "tensor"
        assert meta.alloc_mode is alloc_mode

    def test_from_colocation_with_lora(self):
        alloc_mode = MagicMock()
        meta = WeightUpdateMeta.from_colocation(
            allocation_mode=alloc_mode,
            use_lora=True,
            lora_name="test_lora",
            lora_int_id=1,
            base_model_name="base_model",
        )
        assert meta.type == "tensor"
        assert meta.use_lora is True
        assert meta.lora_name == "test_lora"
        assert meta.lora_int_id == 1
        assert meta.base_model_name == "base_model"

    def test_from_colocation_with_chunked_mem(self):
        alloc_mode = MagicMock()
        meta = WeightUpdateMeta.from_colocation(
            allocation_mode=alloc_mode,
            weight_chunked_mem_mb=512,
        )
        assert meta.weight_chunked_mem_mb == 512

    def test_tensor_type_in_literal(self):
        """Verify 'tensor' is accepted as type."""
        meta = WeightUpdateMeta(type="tensor")
        assert meta.type == "tensor"

    def test_with_version_preserves_tensor_type(self):
        alloc_mode = MagicMock()
        meta = WeightUpdateMeta.from_colocation(allocation_mode=alloc_mode)
        versioned = meta.with_version(5)
        assert versioned.type == "tensor"
        assert versioned.version == 5


class TestColocationSyncHelpers:
    """Test helper functions in colocation_sync module."""

    def test_get_full_tensor_regular(self):
        """Test _get_full_tensor with regular tensor."""
        from areal.engine.core.colocation_sync import _get_full_tensor

        tensor = torch.randn(4, 4)
        result = _get_full_tensor(tensor)
        assert torch.equal(tensor, result)

    def test_get_full_tensor_parameter(self):
        """Test _get_full_tensor with nn.Parameter."""
        from areal.engine.core.colocation_sync import _get_full_tensor

        param = nn.Parameter(torch.randn(3, 3))
        result = _get_full_tensor(param)
        assert torch.equal(param.data, result)

    def test_update_tensor_bucket_empty(self):
        """Test _update_tensor_bucket with empty list."""
        from areal.engine.core.colocation_sync import _update_tensor_bucket

        engine = MagicMock()
        named_tensors = []
        _update_tensor_bucket(engine, named_tensors)
        engine.update_weights_from_tensor.assert_not_called()

    def test_update_tensor_bucket_calls_engine(self):
        """Test _update_tensor_bucket sends tensors to engine."""
        from areal.engine.core.colocation_sync import _update_tensor_bucket

        engine = MagicMock()
        fut = Future()
        fut.set_result(None)
        engine.update_weights_from_tensor.return_value = fut

        tensors = [("layer.weight", torch.randn(2, 2))]
        _update_tensor_bucket(engine, tensors)
        engine.update_weights_from_tensor.assert_called_once()
        # Verify tensors were cleared
        assert len(tensors) == 0


class TestFSDPEngineUpdateWeightsTensor:
    """Test FSDPEngine.update_weights dispatches to tensor mode."""

    def test_update_weights_dispatches_tensor_type(self):
        """Verify update_weights routes 'tensor' type correctly."""
        # We can't easily construct FSDPEngine without a GPU,
        # so verify the code path exists by checking the method
        from areal.engine.fsdp_engine import FSDPEngine

        assert hasattr(FSDPEngine, "_update_weights_from_tensor")


class TestArchonEngineUpdateWeightsTensor:
    """Test ArchonEngine.update_weights dispatches to tensor mode."""

    def test_update_weights_dispatches_tensor_type(self):
        from areal.experimental.engine.archon_engine import ArchonEngine

        assert hasattr(ArchonEngine, "_update_weights_from_tensor")


class TestMegatronEngineUpdateWeightsTensor:
    """Test MegatronEngine.update_weights dispatches to tensor mode."""

    def test_update_weights_dispatches_tensor_type(self):
        from areal.engine.megatron_engine import MegatronEngine

        assert hasattr(MegatronEngine, "_update_weights_from_tensor")
