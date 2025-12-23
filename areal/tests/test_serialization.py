"""Tests for RPC serialization utilities."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
from areal.tests.utils import get_model_path


@dataclass
class SampleData:
    name: str
    value: int
    tensor: torch.Tensor


class TestSerializationRoundTrip:
    """Test serialization/deserialization for all supported types."""

    def test_primitives_and_collections(self):
        """Test primitives and basic collections."""
        payload = {
            "str": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": [2, 3]},
        }
        serialized = serialize_value(payload)
        deserialized = deserialize_value(serialized)
        assert deserialized == payload

    def test_tensors(self):
        """Test torch tensors including special dtypes."""
        tensors = {
            "int": torch.tensor([1, 2, 3], dtype=torch.int64),
            "float32": torch.randn(3, 4, dtype=torch.float32),
            "bfloat16": torch.randn(2, 3, dtype=torch.bfloat16),
            "meta": torch.empty(5, 10, dtype=torch.float32, device="meta"),
        }

        for name, original in tensors.items():
            serialized = serialize_value(original)
            assert serialized["type"] == "tensor"
            deserialized = deserialize_value(serialized)

            if original.is_meta:
                assert deserialized.is_meta
            else:
                assert torch.equal(deserialized, original)
            assert deserialized.dtype == original.dtype
            assert deserialized.shape == original.shape

    def test_numpy_arrays(self):
        """Test NumPy array serialization."""
        arrays = {
            "int": np.array([1, 2, 3], dtype=np.int32),
            "float": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "bool": np.array([True, False], dtype=np.bool_),
        }

        for name, original in arrays.items():
            serialized = serialize_value(original)
            assert serialized["type"] == "ndarray"
            deserialized = deserialize_value(serialized)
            np.testing.assert_array_equal(deserialized, original, strict=True)
            assert deserialized.dtype == original.dtype

    def test_numpy_object_array_rejected(self):
        """Object arrays should be rejected."""
        array = np.array([{"a": 1}], dtype=object)
        with pytest.raises(ValueError, match="Object or void dtype"):
            serialize_value(array)

    def test_dataclass(self):
        """Test dataclass serialization with nested tensors."""
        original = SampleData(
            name="test",
            value=42,
            tensor=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )

        serialized = serialize_value(original)
        assert serialized["type"] == "dataclass"
        assert serialized["class_path"].endswith("SampleData")

        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, SampleData)
        assert deserialized.name == original.name
        assert deserialized.value == original.value
        assert torch.equal(deserialized.tensor, original.tensor)

    def test_tokenizer(self):
        """Test Hugging Face tokenizer serialization."""
        original = AutoTokenizer.from_pretrained(
            get_model_path(
                "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
            )
        )

        serialized = serialize_value(original)
        assert serialized["type"] == "tokenizer"

        deserialized = deserialize_value(serialized)
        assert deserialized.vocab_size == original.vocab_size
        assert deserialized.encode("test") == original.encode("test")

    def test_nested_structure(self):
        """Test complex nested structure with multiple types."""
        payload = {
            "tensor": torch.tensor([1.0, 2.0, 3.0]),
            "array": np.array([4, 5, 6]),
            "dataclass": SampleData(
                name="nested",
                value=7,
                tensor=torch.zeros(2, 2),
            ),
            "list": [torch.ones(3), np.zeros(2)],
            "meta": {"text": "value"},
        }

        serialized = serialize_value(payload)
        deserialized = deserialize_value(serialized)

        assert torch.equal(deserialized["tensor"], payload["tensor"])
        np.testing.assert_array_equal(deserialized["array"], payload["array"])
        assert isinstance(deserialized["dataclass"], SampleData)
        assert deserialized["dataclass"].name == "nested"
        assert torch.equal(deserialized["list"][0], payload["list"][0])
        assert deserialized["meta"]["text"] == "value"

    @pytest.mark.skipif(
        not hasattr(torch, "cuda") or not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_cuda_tensors_moved_to_cpu(self):
        """CUDA tensors should be serialized as CPU tensors."""
        original = torch.randn(3, 3).cuda()
        serialized = serialize_value(original)
        deserialized = deserialize_value(serialized)

        assert deserialized.device.type == "cpu"
        assert torch.equal(deserialized, original.cpu())
