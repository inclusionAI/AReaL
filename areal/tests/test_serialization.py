from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from areal.scheduler.rpc.serializer import Serializer


class TestSerializer:
    """Test the Serializer class functionality."""

    serializer = Serializer()

    def test_basic_types(self):
        """Test serialization of basic Python types."""

        # Test primitives
        test_cases = [
            ("hello", "string"),
            (42, "int"),
            (3.14, "float"),
            (True, "bool"),
            (False, "bool"),
            (None, "None"),
        ]

        for value, desc in test_cases:
            serialized = self.serializer.serialize(value)
            deserialized = self.serializer.deserialize(serialized)
            assert deserialized == value, f"Failed for {desc}: {value}"

    def test_collections(self):
        """Test serialization of lists, tuples, and dicts."""

        # Test list
        original_list = [1, 2, 3, "hello", 4.5]
        serialized = self.serializer.serialize(original_list)
        deserialized = self.serializer.deserialize(serialized)
        assert deserialized == original_list

        # Test dict
        original_dict = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        serialized = self.serializer.serialize(original_dict)
        deserialized = self.serializer.deserialize(serialized)
        assert deserialized == original_dict

    def test_numpy_arrays(self):
        """Test serialization of numpy arrays."""

        # Test various numpy arrays
        test_arrays = [
            np.array([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            np.array([1.0, 2.0, 3.0]),
            np.array([True, False, True]),
            np.zeros((10, 10)),  # Larger array to test buffer management
            np.ones((5,)),  # Small array to test inline encoding
        ]

        for original in test_arrays:
            serialized = self.serializer.serialize(original)
            deserialized = self.serializer.deserialize(serialized, np.ndarray)
            np.testing.assert_array_equal(
                deserialized,
                original,
                err_msg=f"Arrays are not equal: {deserialized} != {original}",
                strict=True,
            )
            assert deserialized.dtype == original.dtype

    def test_torch_tensors(self):
        """Test serialization of PyTorch tensors."""

        # Test various torch tensors
        test_tensors = [
            torch.tensor([1, 2, 3]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.randn(10, 10),  # Larger tensor
            torch.zeros(5),  # Small tensor
            torch.tensor([1, 2, 3], dtype=torch.int64),
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        ]

        for original in test_tensors:
            serialized = self.serializer.serialize(original)
            deserialized = self.serializer.deserialize(serialized, torch.Tensor)
            assert torch.equal(deserialized, original)
            assert deserialized.dtype == original.dtype
            assert deserialized.shape == original.shape

    def test_slice_objects(self):
        """Test serialization of slice objects."""

        test_slices = [
            slice(None),
            slice(5),
            slice(1, 10),
            slice(1, 10, 2),
            slice(None, None, -1),
        ]

        for original in test_slices:
            serialized = self.serializer.serialize(original)
            deserialized = self.serializer.deserialize(serialized, slice)
            assert deserialized == original

    def test_functions(self):
        """Test serialization of function objects."""

        def test_function(x, y=10):
            return x + y

        with pytest.raises(NotImplementedError):
            self.serializer.serialize(test_function)

    def test_dataclass(self):
        """Test serialization of dataclasses."""

        @dataclass
        class TestDataClass:
            name: str
            value: int
            tensor: torch.Tensor

        original = TestDataClass(name="test", value=42, tensor=torch.randn(2, 3))
        serialized = self.serializer.serialize(original)
        deserialized = self.serializer.deserialize(serialized, TestDataClass)

        assert isinstance(deserialized, TestDataClass)
        assert deserialized.name == "test"
        assert deserialized.value == 42
        assert torch.equal(deserialized.tensor, original.tensor)

    def test_tokenizers(self):
        """Test serialization of Hugging Face tokenizers."""
        from transformers import PreTrainedTokenizerFast

        # Create a simple tokenizer (using a small model for testing)
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-bert"
        )

        serialized = self.serializer.serialize(tokenizer)
        deserialized = self.serializer.deserialize(serialized, PreTrainedTokenizerFast)

        # Test that the deserialized tokenizer has the same basic properties
        assert deserialized.vocab_size == tokenizer.vocab_size
        assert deserialized.model_max_length == tokenizer.model_max_length
        assert deserialized.name_or_path == tokenizer.name_or_path

        test_text = "Hello world"
        original_tokens = tokenizer.encode(test_text)
        deserialized_tokens = deserialized.encode(test_text)
        assert original_tokens == deserialized_tokens

    @pytest.mark.skipif(
        not hasattr(torch, "cuda") or not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_cuda_tensors(self):
        """Test serialization of CUDA tensors."""

        # Create a CUDA tensor
        original = torch.randn(5, 5).cuda()
        serialized = self.serializer.serialize(original)
        deserialized = self.serializer.deserialize(serialized, torch.Tensor)

        # Should be moved back to CPU
        assert deserialized.device.type == "cpu"
        assert torch.equal(deserialized, original.cpu())


if __name__ == "__main__":
    pytest.main([__file__])
