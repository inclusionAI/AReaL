"""Pytest test suite for tensor serialization utilities."""

import pytest
import torch

from areal.scheduler.rpc.serialization import (
    SerializedTensor,
    deserialize_value,
    serialize_value,
)


class TestSerializedTensor:
    """Test suite for SerializedTensor Pydantic model."""

    def test_from_tensor_float32(self):
        """Test serialization of float32 tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        serialized = SerializedTensor.from_tensor(tensor)

        assert serialized.type == "tensor"
        assert serialized.shape == [3]
        assert serialized.dtype == "torch.float32"

    def test_from_tensor_various_dtypes(self):
        """Test serialization of tensors with various dtypes."""
        dtypes = [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.bool,
            torch.uint8,
        ]

        for dtype in dtypes:
            tensor = torch.tensor([1, 2, 3], dtype=dtype)
            serialized = SerializedTensor.from_tensor(tensor)
            assert serialized.dtype == str(dtype)

    def test_from_tensor_various_shapes(self):
        """Test serialization of tensors with various shapes."""
        shapes = [
            (),  # scalar
            (5,),  # 1D
            (3, 4),  # 2D
            (2, 3, 4),  # 3D
            (2, 3, 4, 5),  # 4D
        ]

        for shape in shapes:
            tensor = torch.randn(shape)
            serialized = SerializedTensor.from_tensor(tensor)
            assert serialized.shape == list(shape)

    def test_from_tensor_with_requires_grad(self):
        """Test serialization ignores requires_grad flag."""
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        serialized = SerializedTensor.from_tensor(tensor)
        # Serialization should work but requires_grad is not preserved
        assert serialized.type == "tensor"

    def test_roundtrip_float32(self):
        """Test serialize-deserialize roundtrip for float32 tensor."""
        original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        serialized = SerializedTensor.from_tensor(original)
        reconstructed = serialized.to_tensor()

        assert torch.allclose(original, reconstructed)
        assert reconstructed.dtype == original.dtype
        assert reconstructed.shape == original.shape

    def test_roundtrip_various_dtypes(self):
        """Test roundtrip for various dtypes."""
        test_cases = [
            (torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32), torch.float32),
            (torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), torch.float64),
            (torch.tensor([1, 2, 3], dtype=torch.int32), torch.int32),
            (torch.tensor([1, 2, 3], dtype=torch.int64), torch.int64),
            (torch.tensor([True, False, True], dtype=torch.bool), torch.bool),
        ]

        for original, expected_dtype in test_cases:
            serialized = SerializedTensor.from_tensor(original)
            reconstructed = serialized.to_tensor()

            assert reconstructed.dtype == expected_dtype
            if expected_dtype == torch.bool:
                assert torch.equal(original, reconstructed)
            else:
                assert torch.allclose(original.float(), reconstructed.float())

    def test_roundtrip_ignores_requires_grad(self):
        """Test roundtrip does not preserve requires_grad."""
        original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        serialized = SerializedTensor.from_tensor(original)
        reconstructed = serialized.to_tensor()

        # requires_grad is not preserved
        assert reconstructed.requires_grad is False
        assert torch.allclose(original.detach(), reconstructed)

    def test_empty_tensor(self):
        """Test serialization of empty tensor."""
        tensor = torch.tensor([])
        serialized = SerializedTensor.from_tensor(tensor)
        reconstructed = serialized.to_tensor()

        assert reconstructed.shape == torch.Size([0])
        assert torch.equal(tensor, reconstructed)

    def test_large_tensor(self):
        """Test serialization of large tensor."""
        tensor = torch.randn(100, 100)
        serialized = SerializedTensor.from_tensor(tensor)
        reconstructed = serialized.to_tensor()

        assert torch.allclose(tensor, reconstructed)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor(self):
        """Test serialization of CUDA tensor (moves to CPU)."""
        tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        serialized = SerializedTensor.from_tensor(tensor)

        # Serialization works with CUDA tensors
        assert serialized.type == "tensor"

        # Reconstructed tensor is always on CPU
        reconstructed = serialized.to_tensor()
        assert reconstructed.device.type == "cpu"
        assert torch.allclose(tensor.cpu(), reconstructed)


class TestSerializeValue:
    """Test suite for serialize_value function."""

    def test_serialize_none(self):
        """Test serialization of None."""
        assert serialize_value(None) is None

    def test_serialize_primitives(self):
        """Test serialization of primitive types."""
        assert serialize_value(42) == 42
        assert serialize_value(3.14) == 3.14
        assert serialize_value("hello") == "hello"
        assert serialize_value(True) is True

    def test_serialize_tensor(self):
        """Test serialization of torch tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = serialize_value(tensor)

        assert isinstance(result, dict)
        assert result["type"] == "tensor"
        assert "data" in result
        assert "shape" in result
        assert "dtype" in result

    def test_serialize_list_of_primitives(self):
        """Test serialization of list of primitives."""
        original = [1, 2, 3, "hello", True]
        result = serialize_value(original)

        assert result == original

    def test_serialize_list_of_tensors(self):
        """Test serialization of list of tensors."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = serialize_value(tensors)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(item["type"] == "tensor" for item in result)

    def test_serialize_dict_of_primitives(self):
        """Test serialization of dict with primitives."""
        original = {"a": 1, "b": "hello", "c": 3.14}
        result = serialize_value(original)

        assert result == original

    def test_serialize_dict_of_tensors(self):
        """Test serialization of dict with tensors."""
        tensors = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
        }
        result = serialize_value(tensors)

        assert isinstance(result, dict)
        assert all(result[key]["type"] == "tensor" for key in result)

    def test_serialize_nested_dict(self):
        """Test serialization of nested dictionary."""
        nested = {
            "level1": {"level2": {"tensor": torch.tensor([1.0, 2.0]), "value": 42}}
        }
        result = serialize_value(nested)

        assert isinstance(result, dict)
        assert isinstance(result["level1"], dict)
        assert isinstance(result["level1"]["level2"], dict)
        assert result["level1"]["level2"]["tensor"]["type"] == "tensor"
        assert result["level1"]["level2"]["value"] == 42

    def test_serialize_mixed_structure(self):
        """Test serialization of complex mixed structure."""
        mixed = {
            "tensors": [torch.tensor([1.0]), torch.tensor([2.0])],
            "metadata": {"batch_size": 32, "device": "cpu"},
            "mask": torch.tensor([True, False, True]),
        }
        result = serialize_value(mixed)

        assert isinstance(result["tensors"], list)
        assert result["tensors"][0]["type"] == "tensor"
        assert result["metadata"]["batch_size"] == 32
        assert result["mask"]["type"] == "tensor"

    def test_serialize_tuple(self):
        """Test serialization of tuple (converts to list)."""
        original = (1, 2, torch.tensor([3.0]))
        result = serialize_value(original)

        assert isinstance(result, list)
        assert result[0] == 1
        assert result[1] == 2
        assert result[2]["type"] == "tensor"


class TestDeserializeValue:
    """Test suite for deserialize_value function."""

    def test_deserialize_none(self):
        """Test deserialization of None."""
        assert deserialize_value(None) is None

    def test_deserialize_primitives(self):
        """Test deserialization of primitive types."""
        assert deserialize_value(42) == 42
        assert deserialize_value(3.14) == 3.14
        assert deserialize_value("hello") == "hello"
        assert deserialize_value(True) is True

    def test_deserialize_tensor(self):
        """Test deserialization of serialized tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        serialized = serialize_value(tensor)
        result = deserialize_value(serialized)

        assert isinstance(result, torch.Tensor)
        assert torch.allclose(tensor, result)

    def test_deserialize_list_of_primitives(self):
        """Test deserialization of list of primitives."""
        original = [1, 2, 3, "hello", True]
        result = deserialize_value(original)

        assert result == original

    def test_deserialize_list_of_tensors(self):
        """Test deserialization of list of tensors."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        serialized = serialize_value(tensors)
        result = deserialize_value(serialized)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, torch.Tensor) for item in result)
        assert torch.allclose(tensors[0], result[0])
        assert torch.allclose(tensors[1], result[1])

    def test_deserialize_dict_of_tensors(self):
        """Test deserialization of dict with tensors."""
        original = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
        }
        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        assert isinstance(result, dict)
        assert all(isinstance(result[key], torch.Tensor) for key in result)
        assert torch.equal(original["input_ids"], result["input_ids"])
        assert torch.equal(original["attention_mask"], result["attention_mask"])

    def test_deserialize_nested_structure(self):
        """Test deserialization of nested structure."""
        original = {
            "level1": {
                "tensor": torch.tensor([1.0, 2.0]),
                "value": 42,
            }
        }
        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        assert isinstance(result["level1"]["tensor"], torch.Tensor)
        assert result["level1"]["value"] == 42
        assert torch.allclose(original["level1"]["tensor"], result["level1"]["tensor"])

    def test_deserialize_invalid_tensor_dict(self):
        """Test deserialization handles invalid tensor dict gracefully."""
        # Dict with type="tensor" but missing required fields
        invalid = {"type": "tensor", "invalid_field": "value"}
        result = deserialize_value(invalid)

        # Should treat as regular dict if parsing fails
        assert isinstance(result, dict)
        assert result["type"] == "tensor"


class TestRoundtrip:
    """Test suite for full serialize-deserialize roundtrips."""

    def test_roundtrip_simple_tensor(self):
        """Test roundtrip for simple tensor."""
        original = torch.tensor([1.0, 2.0, 3.0])
        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        assert torch.allclose(original, result)

    def test_roundtrip_trajectory_dict(self):
        """Test roundtrip for typical trajectory dictionary."""
        trajectory = {
            "input_ids": torch.tensor([101, 102, 103, 104]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "rewards": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "logprobs": torch.tensor([-1.0, -2.0, -3.0, -4.0]),
        }

        serialized = serialize_value(trajectory)
        result = deserialize_value(serialized)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(trajectory.keys())
        for key in trajectory:
            assert torch.allclose(trajectory[key].float(), result[key].float())

    def test_roundtrip_mixed_types(self):
        """Test roundtrip for mixed type structure."""
        original = {
            "tensors": [torch.tensor([1.0]), torch.tensor([2.0])],
            "metadata": {"count": 2, "name": "test"},
            "value": 42,
            "flag": True,
        }

        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        assert len(result["tensors"]) == 2
        assert torch.allclose(original["tensors"][0], result["tensors"][0])
        assert result["metadata"] == original["metadata"]
        assert result["value"] == original["value"]
        assert result["flag"] == original["flag"]

    def test_roundtrip_with_none_values(self):
        """Test roundtrip with None values in structure."""
        original = {
            "tensor": torch.tensor([1.0, 2.0]),
            "optional": None,
            "nested": {"value": 42, "empty": None},
        }

        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        assert torch.allclose(original["tensor"], result["tensor"])
        assert result["optional"] is None
        assert result["nested"]["value"] == 42
        assert result["nested"]["empty"] is None

    def test_roundtrip_empty_structures(self):
        """Test roundtrip for empty structures."""
        test_cases = [
            {},  # Empty dict
            [],  # Empty list
            {"empty_list": [], "empty_dict": {}},  # Nested empty
        ]

        for original in test_cases:
            serialized = serialize_value(original)
            result = deserialize_value(serialized)
            assert result == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
