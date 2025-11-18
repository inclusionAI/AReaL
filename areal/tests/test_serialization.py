from dataclasses import dataclass

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from areal.scheduler.rpc.serialization import deserialize_value, serialize_value


@dataclass
class SampleData:
    name: str
    value: int
    tensor: torch.Tensor


class TestSerializationUtils:
    def test_primitives_round_trip(self):
        values = ["hello", 42, 3.14, True, False, None]
        for value in values:
            serialized = serialize_value(value)
            assert serialized == value
            assert deserialize_value(serialized) == value

    def test_collections_round_trip(self):
        payload = {
            "list": [1, 2, 3, "hello", 4.5],
            "dict": {"a": 1, "b": "hello", "c": [1, 2, 3]},
        }
        serialized = serialize_value(payload)
        assert serialized == payload
        assert deserialize_value(serialized) == payload

    def test_numpy_arrays(self):
        arrays = [
            np.array([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            np.array([1.0, 2.0, 3.0]),
            np.array([True, False, True]),
            np.zeros((10, 10)),
            np.ones((5,)),
        ]
        for original in arrays:
            serialized = serialize_value(original)
            assert serialized["type"] == "ndarray"
            deserialized = deserialize_value(serialized)
            np.testing.assert_array_equal(
                deserialized,
                original,
                err_msg=f"Arrays are not equal: {deserialized} != {original}",
                strict=True,
            )
            assert deserialized.dtype == original.dtype

    def test_numpy_object_array_rejected(self):
        array = np.array([{"a": 1}], dtype=object)
        with pytest.raises(ValueError):
            serialize_value(array)

    def test_torch_tensors(self):
        tensors = [
            torch.tensor([1, 2, 3]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.randn(10, 10),
            torch.zeros(5),
            torch.tensor([1, 2, 3], dtype=torch.int64),
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
        ]
        for original in tensors:
            serialized = serialize_value(original)
            assert serialized["type"] == "tensor"
            deserialized = deserialize_value(serialized)
            assert torch.equal(deserialized, original)
            assert deserialized.dtype == original.dtype
            assert tuple(deserialized.shape) == tuple(original.shape)

    def test_dataclass_round_trip(self):
        original = SampleData(
            name="test",
            value=42,
            tensor=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )
        serialized = serialize_value(original)
        assert serialized["type"] == "dataclass"
        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, SampleData)
        assert deserialized.name == original.name
        assert deserialized.value == original.value
        assert torch.equal(deserialized.tensor, original.tensor)

    def test_nested_structure_round_trip(self):
        payload = {
            "tensor": torch.tensor([1, 2, 3], dtype=torch.int64),
            "array": np.arange(4).reshape(2, 2),
            "dataclass": SampleData(
                name="inner",
                value=7,
                tensor=torch.tensor([0.5, 0.25]),
            ),
            "items": [torch.zeros(1), np.ones(3)],
            "meta": {"text": "hello"},
            "tokenizer": AutoTokenizer.from_pretrained(
                "hf-internal-testing/tiny-random-bert"
            ),
        }
        deserialized = deserialize_value(serialize_value(payload))
        assert torch.equal(deserialized["tensor"], payload["tensor"])
        np.testing.assert_array_equal(
            deserialized["array"], payload["array"], strict=True
        )
        assert isinstance(deserialized["dataclass"], SampleData)
        assert deserialized["dataclass"].name == payload["dataclass"].name
        assert deserialized["dataclass"].value == payload["dataclass"].value
        assert torch.equal(
            deserialized["dataclass"].tensor, payload["dataclass"].tensor
        )
        assert torch.equal(deserialized["items"][0], payload["items"][0])
        np.testing.assert_array_equal(
            deserialized["items"][1], payload["items"][1], strict=True
        )
        assert deserialized["meta"] == payload["meta"]
        assert isinstance(deserialized["tokenizer"], PreTrainedTokenizerFast)

    def test_tokenizers(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-bert"
        )
        serialized = serialize_value(tokenizer)
        assert serialized["type"] == "tokenizer"
        deserialized = deserialize_value(serialized)
        assert isinstance(deserialized, PreTrainedTokenizerFast)
        assert deserialized.vocab_size == tokenizer.vocab_size
        assert deserialized.model_max_length == tokenizer.model_max_length
        assert deserialized.name_or_path == tokenizer.name_or_path
        test_text = "Hello world"
        assert deserialized.encode(test_text) == tokenizer.encode(test_text)

    @pytest.mark.skipif(
        not hasattr(torch, "cuda") or not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_cuda_tensors(self):
        original = torch.randn(5, 5).cuda()
        serialized = serialize_value(original)
        assert serialized["type"] == "tensor"
        deserialized = deserialize_value(serialized)
        assert deserialized.device.type == "cpu"
        assert torch.equal(deserialized, original.cpu())


if __name__ == "__main__":
    pytest.main([__file__])
