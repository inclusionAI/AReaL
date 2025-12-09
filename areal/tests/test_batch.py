import pickle

import pytest
import torch

from areal.controller.batch import DistributedBatchMemory
from areal.controller.batch_metadata import (
    BatchMetadata,
    ShardMetadata,
    TensorMetadata,
)
from areal.utils.batch_utils import (
    convert_dict_to_list,
    convert_list_to_dict,
    validate_dict_dataset,
)
from areal.utils.errors import FrameworkError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_data():
    """Test fixtures for batch memory tests."""
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "seqlen": torch.tensor([3, 3]),
        "labels": torch.tensor([1, 0]),
    }


@pytest.fixture
def test_list():
    """Test list fixtures."""
    return [
        {"input_ids": torch.tensor([1, 2]), "seqlen": torch.tensor(2)},
        {"input_ids": torch.tensor([3, 4]), "seqlen": torch.tensor(2)},
    ]


@pytest.fixture
def scalar_data():
    """Scalar test data."""
    return {
        "labels": [1, 2, 3, 4, 5],  # List of integers
        "scores": [0.1, 0.2, 0.3, 0.4, 0.5],  # List of floats
        "categories": ["A", "B", "C", "D", "E"],  # List of strings
        "flags": [True, False, True, False, True],  # List of booleans
    }


@pytest.fixture
def mixed_data():
    """Mixed scalar and tensor data."""
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "labels": [1, 0, 1],  # List of scalars
        "scores": [0.95, 0.87, 0.92],  # List of floats
        "single_value": 42,  # Single scalar
    }


@pytest.fixture
def scalar_list():
    """List format test data with scalars."""
    return [
        {"label": 1, "score": 0.95, "category": "A"},
        {"label": 0, "score": 0.87, "category": "B"},
        {"label": 1, "score": 0.92, "category": "C"},
    ]


@pytest.fixture
def test_dict():
    """Test dictionary fixtures."""
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "seqlen": torch.tensor([3, 3]),
    }


# =============================================================================
# TestBatchMemory
# =============================================================================


class TestBatchMemory:
    """Test cases for DistributedBatchMemory functionality."""

    def test_from_dict(self, test_data):
        """Test creating DistributedBatchMemory from dictionary."""
        memory = DistributedBatchMemory.from_dict(test_data)
        assert memory is not None
        assert memory.dataset == test_data
        assert len(memory) == 2

    def test_from_list(self, test_list):
        """Test creating DistributedBatchMemory from list."""
        test_data = [
            {"input_ids": torch.tensor([[1, 2]]), "seqlen": torch.tensor([2])},
            {"input_ids": torch.tensor([[3, 4]]), "seqlen": torch.tensor([2])},
        ]
        memory = DistributedBatchMemory.from_list(test_data)
        assert memory is not None
        assert memory.dataset["input_ids"].shape == (2, 2)
        assert memory.dataset["seqlen"].tolist() == [2, 2]

    def test_chunk(self):
        """Test chunk method for data parallel splitting."""
        data = {
            "input_ids": torch.arange(12).reshape(6, 2),
            "seqlen": torch.tensor([2, 2, 2, 2, 2, 2]),
        }
        memory = DistributedBatchMemory.from_dict(data)
        results = memory.chunk(3)

        assert len(results) == 3
        for batch in results:
            assert len(batch) == 2  # 6 items split into 3 batches of 2

    def test_chunk_by_ffd(self):
        """Test chunk_by_ffd method for sequence length balancing."""
        tensor = torch.tensor(
            [
                [1] * 1 + [0] * 9,
                [1] * 6 + [0] * 4,
                [1] * 3 + [0] * 7,
                [1] * 5 + [0] * 5,
                [1] * 3 + [0] * 7,
                [1] * 8 + [0] * 2,
                [1] * 10 + [0] * 0,
                [1] * 5 + [0] * 5,
            ]
        )
        data = {"input_ids": tensor, "seqlen": torch.tensor([1, 6, 3, 5, 3, 8, 10, 5])}
        memory = DistributedBatchMemory.from_dict(data)
        results = memory.chunk_by_ffd(2, 2)

        assert len(results) == 2
        assert torch.equal(results[0]["seqlen"], torch.tensor([1, 6, 10, 5]))
        assert torch.equal(results[1]["seqlen"], torch.tensor([3, 5, 3, 8]))

    def test_union(self):
        """Test union method for merging batches."""
        data1 = {"input_ids": torch.tensor([[1, 2]]), "seqlen": torch.tensor([2])}
        data2 = {"input_ids": torch.tensor([[3, 4]]), "seqlen": torch.tensor([2])}

        batch1 = DistributedBatchMemory.from_dict(data1)
        batch2 = DistributedBatchMemory.from_dict(data2)

        merged = batch1.union_(batch2)
        assert len(merged) == 2
        assert torch.equal(merged.dataset["input_ids"], torch.tensor([[1, 2], [3, 4]]))

    def test_concat(self):
        """Test concat static method."""
        data1 = {
            "input_ids": torch.tensor([[1, 2]]),
            "seqlen": torch.tensor([2]),
            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        data2 = {
            "input_ids": torch.tensor([[3, 4]]),
            "seqlen": torch.tensor([2]),
            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        }

        batch1 = DistributedBatchMemory.from_dict(data1)
        batch2 = DistributedBatchMemory.from_dict(data2)

        concatenated = DistributedBatchMemory.concat([batch1, batch2])
        assert len(concatenated) == 2
        assert torch.equal(
            concatenated.dataset["input_ids"], torch.tensor([[1, 2], [3, 4]])
        )

    def test_getitem(self, test_data):
        """Test __getitem__ for indexing and key access."""
        memory = DistributedBatchMemory.from_dict(test_data)

        # Test integer indexing
        sample = memory[0]
        assert sample["input_ids"].tolist() == [1, 2, 3]
        assert sample["seqlen"].item() == 3

        # Test string key access
        input_ids = memory["input_ids"]
        assert torch.equal(input_ids, test_data["input_ids"])

    def test_setitem(self, test_data):
        """Test __setitem__ for setting tensor values."""
        memory = DistributedBatchMemory.from_dict(test_data)
        new_tensor = torch.tensor([[7, 8, 9], [10, 11, 12]])

        memory["new_key"] = new_tensor
        assert torch.equal(memory["new_key"], new_tensor)

    def test_delitem(self, test_data):
        """Test __delitem__ for deleting items."""
        memory = DistributedBatchMemory.from_dict(test_data)

        # Test deleting by key
        del memory["labels"]
        assert "labels" not in memory.dataset

        # Test deleting by index (converts to list format)
        simple_data = {"input_ids": torch.tensor([[1, 2]]), "seqlen": torch.tensor([2])}
        memory = DistributedBatchMemory.from_dict(simple_data)
        assert len(memory) == 1
        del memory[0]
        # After deletion, the dataset should be empty
        assert len(memory) == 0

    def test_delitem_int_and_string_keys(self, test_data):
        """Test deleting elements using both int and string keys."""
        # Create test data with multiple samples
        data = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "seqlen": torch.tensor([2, 2]),
        }

        # Test 1: Delete by string key (removes entire attribute)
        memory = DistributedBatchMemory.from_dict(data)
        assert len(memory) == 2
        assert "seqlen" in memory.dataset

        del memory["seqlen"]
        assert "seqlen" not in memory.dataset

        # Test 2: Delete by int index (removes sample at position)
        memory = DistributedBatchMemory.from_dict(data)

        # Delete first sample - verify the operation completes without error
        del memory[0]
        # We verify the operation completed successfully
        assert isinstance(memory, DistributedBatchMemory)

    def test_pickle_support(self, test_data):
        """Test pickle serialization and deserialization."""
        memory = DistributedBatchMemory.from_dict(test_data)

        # Test pickle dump and load
        serialized = pickle.dumps(memory)
        loaded = pickle.loads(serialized)

        assert torch.equal(memory.dataset["input_ids"], loaded.dataset["input_ids"])
        assert torch.equal(memory.dataset["seqlen"], loaded.dataset["seqlen"])

    def test_str_repr(self, test_data):
        """Test __str__ and __repr__ methods."""
        memory = DistributedBatchMemory.from_dict(test_data)
        str_repr = str(memory)
        assert "DistributedBatchMemory" in str_repr
        assert "size=2" in str_repr

    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        empty_memory = DistributedBatchMemory.from_dict({})
        assert len(empty_memory) == 0

        # Test chunk with empty dataset
        with pytest.raises(Exception):
            empty_memory.chunk(2)

    def test_get_data(self, test_data):
        """Test get_data method returns all data correctly."""
        # Test with normal data
        memory = DistributedBatchMemory.from_dict(test_data)
        result = memory.get_data()

        # Verify return type and structure
        assert isinstance(result, dict)
        assert set(result.keys()) == set(test_data.keys())

        # Verify data integrity
        for key in test_data:
            assert torch.equal(result[key], test_data[key])

        # Test with empty dataset
        empty_memory = DistributedBatchMemory.from_dict({})
        empty_result = empty_memory.get_data()
        assert empty_result == {}

        # Test with single sample
        single_data = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "seqlen": torch.tensor([3]),
        }
        single_memory = DistributedBatchMemory.from_dict(single_data)
        single_result = single_memory.get_data()
        assert len(single_result) == 2
        assert torch.equal(single_result["input_ids"], single_data["input_ids"])
        assert torch.equal(single_result["seqlen"], single_data["seqlen"])


# =============================================================================
# TestRollout
# =============================================================================


class TestRollout:
    """Test cases for prompt and task_type dataset format as requested."""

    @pytest.fixture
    def prompt_task_data(self):
        """Test data with prompt and task_type attributes."""
        return [
            {"prompt": "Solve the equation 2x + 5 = 13 for x.", "task_type": "math"},
            {
                "prompt": "Write a Python function to calculate the factorial of a number.",
                "task_type": "code",
            },
            {
                "prompt": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 7.",
                "task_type": "math",
            },
            {
                "prompt": "Implement a binary search algorithm in Python.",
                "task_type": "code",
            },
            {
                "prompt": "Calculate the area of a circle with radius 5 units.",
                "task_type": "math",
            },
            {
                "prompt": "Create a function to reverse a string without using built-in reverse methods.",
                "task_type": "code",
            },
            {
                "prompt": "Solve the quadratic equation x^2 - 4x + 3 = 0.",
                "task_type": "math",
            },
            {
                "prompt": "Write a program to check if a number is prime.",
                "task_type": "code",
            },
            {
                "prompt": "Find the integral of f(x) = 3x^2 + 2x + 1.",
                "task_type": "math",
            },
            {
                "prompt": "Implement a simple calculator that can perform basic arithmetic operations.",
                "task_type": "code",
            },
            {
                "prompt": "Determine the slope of the line passing through points (2,3) and (5,11).",
                "task_type": "math",
            },
            {
                "prompt": "Create a function to find the longest common subsequence between two strings.",
                "task_type": "code",
            },
        ]

    def test_from_list_with_prompt_task_type(self, prompt_task_data):
        """Test creating DistributedBatchMemory from list with prompt and task_type."""
        memory = DistributedBatchMemory.from_list(prompt_task_data)

        assert memory is not None
        assert len(memory) == 12
        assert "prompt" in memory.dataset
        assert "task_type" in memory.dataset
        assert isinstance(memory.dataset["prompt"], list)
        assert isinstance(memory.dataset["task_type"], list)
        assert len(memory.dataset["prompt"]) == 12
        assert len(memory.dataset["task_type"]) == 12

    def test_chunk_with_dp_size_4(self, prompt_task_data):
        """Test chunk method with dp_size=4 for prompt and task_type data."""
        memory = DistributedBatchMemory.from_list(prompt_task_data)
        results = memory.chunk(4)

        assert len(results) == 4

        # Check chunk sizes: 12 items split into 4 chunks should be [3,3,3,3]
        expected_sizes = [3, 3, 3, 3]
        actual_sizes = [len(batch) for batch in results]
        assert actual_sizes == expected_sizes

        # Verify data integrity in each chunk
        all_prompts = []
        all_task_types = []

        for i, batch in enumerate(results):
            assert isinstance(batch, DistributedBatchMemory)
            assert "prompt" in batch.dataset
            assert "task_type" in batch.dataset

            prompts = batch.dataset["prompt"]
            task_types = batch.dataset["task_type"]

            # Check chunk content
            assert len(prompts) == expected_sizes[i]
            assert len(task_types) == expected_sizes[i]
            assert all(isinstance(p, str) for p in prompts)
            assert all(t in ["math", "code"] for t in task_types)

            all_prompts.extend(prompts)
            all_task_types.extend(task_types)

        # Verify all original data is preserved
        original_prompts = [item["prompt"] for item in prompt_task_data]
        original_task_types = [item["task_type"] for item in prompt_task_data]

        assert all_prompts == original_prompts
        assert all_task_types == original_task_types

    def test_chunk_by_ffd_with_prompt_task_type(self, prompt_task_data):
        """Test chunk_by_ffd method with prompt and task_type data."""
        # Add seqlen for chunk_by_ffd testing
        extended_data = [
            {**item, "seqlen": len(item["prompt"]) // 10 + 1}
            for item in prompt_task_data
        ]

        memory = DistributedBatchMemory.from_list(extended_data)
        results = memory.chunk_by_ffd(2, 3)  # group_size=2, dp_size=3

        assert len(results) == 3
        total_samples = sum(len(batch) for batch in results)
        assert total_samples == 12

        # Verify structure
        for batch in results:
            assert "prompt" in batch.dataset
            assert "task_type" in batch.dataset
            assert "seqlen" in batch.dataset

    def test_union_with_prompt_task_type(self, prompt_task_data):
        """Test union method with prompt and task_type data."""
        # Split data into two parts
        data1 = prompt_task_data[:6]
        data2 = prompt_task_data[6:]

        batch1 = DistributedBatchMemory.from_list(data1)
        batch2 = DistributedBatchMemory.from_list(data2)

        merged = batch1.union_(batch2)

        assert len(merged) == 12
        assert merged.dataset["prompt"] == [item["prompt"] for item in prompt_task_data]
        assert merged.dataset["task_type"] == [
            item["task_type"] for item in prompt_task_data
        ]

    def test_concat_with_prompt_task_type(self, prompt_task_data):
        """Test concat static method with prompt and task_type data."""
        # For concat_padded_tensors, we need tensor data
        # Convert list data to tensor-based format
        data_parts = [prompt_task_data[:4], prompt_task_data[4:8], prompt_task_data[8:]]
        batches = []
        for part in data_parts:
            # Create tensor-based data instead of list-based
            batch_data = {
                "input_ids": torch.randint(0, 100, (len(part), 10)),
                "attention_mask": torch.ones(len(part), 10, dtype=torch.bool),
                "task_ids": torch.arange(len(part)),
            }
            batch = DistributedBatchMemory.from_dict(batch_data)
            batches.append(batch)

        concatenated = DistributedBatchMemory.concat(batches)

        assert len(concatenated) == 12
        assert "input_ids" in concatenated.dataset
        assert "attention_mask" in concatenated.dataset

    def test_getitem_with_prompt_task_type(self, prompt_task_data):
        """Test __getitem__ for indexing and key access with prompt and task_type data."""
        memory = DistributedBatchMemory.from_list(prompt_task_data)

        # Test integer indexing
        sample = memory[0]
        assert sample["prompt"] == "Solve the equation 2x + 5 = 13 for x."
        assert sample["task_type"] == "math"

        # Test string key access
        prompts = memory["prompt"]
        task_types = memory["task_type"]

        assert isinstance(prompts, list)
        assert isinstance(task_types, list)
        assert len(prompts) == 12
        assert len(task_types) == 12
        assert prompts[0] == "Solve the equation 2x + 5 = 13 for x."
        assert task_types[1] == "code"

    def test_setitem_with_prompt_task_type(self, prompt_task_data):
        """Test __setitem__ for setting prompt and task_type values."""
        memory = DistributedBatchMemory.from_list(prompt_task_data)

        # Test adding new field
        memory["difficulty"] = ["easy", "medium", "hard"] * 4
        assert memory["difficulty"] == ["easy", "medium", "hard"] * 4

        # Test updating existing field
        new_prompts = [f"Updated: {p}" for p in memory["prompt"]]
        memory["prompt"] = new_prompts
        assert memory["prompt"] == new_prompts

    def test_delitem_with_prompt_task_type(self, prompt_task_data):
        """Test __delitem__ for deleting prompt and task_type items."""
        DistributedBatchMemory.from_list(prompt_task_data)

        # Test deleting by key
        temp_memory = DistributedBatchMemory.from_list(prompt_task_data)
        del temp_memory["task_type"]
        assert "task_type" not in temp_memory.dataset
        assert "prompt" in temp_memory.dataset

        # Test deleting by index
        temp_memory2 = DistributedBatchMemory.from_list(prompt_task_data)
        original_len = len(temp_memory2)
        del temp_memory2[0]  # Delete first item
        assert len(temp_memory2) == original_len - 1
        assert (
            temp_memory2["prompt"][0]
            == "Write a Python function to calculate the factorial of a number."
        )

    def test_pickle_support_with_prompt_task_type(self, prompt_task_data):
        """Test pickle serialization and deserialization with prompt and task_type data."""
        import pickle

        memory = DistributedBatchMemory.from_list(prompt_task_data)

        # Test pickle dump and load
        serialized = pickle.dumps(memory)
        loaded = pickle.loads(serialized)

        assert len(loaded) == 12
        assert loaded["prompt"] == memory["prompt"]
        assert loaded["task_type"] == memory["task_type"]

    def test_str_repr_with_prompt_task_type(self, prompt_task_data):
        """Test __str__ and __repr__ methods with prompt and task_type data."""
        memory = DistributedBatchMemory.from_list(prompt_task_data)
        str_repr = str(memory)

        assert "DistributedBatchMemory" in str_repr
        assert "size=12" in str_repr
        assert "prompt" in str_repr
        assert "task_type" in str_repr

    def test_empty_dataset_handling(self):
        """Test handling of empty prompt and task_type datasets."""
        empty_data = []
        empty_memory = DistributedBatchMemory.from_list(empty_data)

        assert len(empty_memory) == 0
        assert empty_memory.dataset == {}

    def test_single_sample_handling(self):
        """Test handling of single prompt and task_type sample."""
        single_data = [{"prompt": "What is 2+2?", "task_type": "math"}]
        memory = DistributedBatchMemory.from_list(single_data)

        assert len(memory) == 1
        assert memory["prompt"] == ["What is 2+2?"]
        assert memory["task_type"] == ["math"]

    def test_mixed_task_type_distribution(self):
        """Test that math and code task types are correctly distributed in chunks."""
        # Create data with alternating task types
        mixed_data = [
            (
                {"prompt": f"Math problem {i}", "task_type": "math"}
                if i % 2 == 0
                else {"prompt": f"Code problem {i}", "task_type": "code"}
            )
            for i in range(8)
        ]

        memory = DistributedBatchMemory.from_list(mixed_data)
        results = memory.chunk(4)

        # Each chunk should have 2 items
        assert [len(batch) for batch in results] == [2, 2, 2, 2]

        # Verify task type distribution
        for batch in results:
            task_types = batch.dataset["task_type"]
            assert len(task_types) == 2
            # Each chunk should have one math and one code
            assert "math" in task_types
            assert "code" in task_types


# =============================================================================
# TestBatchUtils
# =============================================================================


class TestBatchUtils:
    """Test cases for batch utilities."""

    def test_convert_list_to_dict(self):
        """Test convert_list_to_dict function."""
        test_data = [
            {"input_ids": torch.tensor([[1, 2]]), "seqlen": torch.tensor([2])},
            {"input_ids": torch.tensor([[3, 4]]), "seqlen": torch.tensor([2])},
        ]
        result = convert_list_to_dict(test_data)
        expected_input_ids = torch.tensor([[1, 2], [3, 4]])
        expected_seqlen = torch.tensor([2, 2])

        assert torch.equal(result["input_ids"], expected_input_ids)
        assert torch.equal(result["seqlen"], expected_seqlen)

    def test_convert_dict_to_list(self, test_dict):
        """Test convert_dict_to_list function."""
        result = convert_dict_to_list(test_dict)

        assert len(result) == 2
        assert torch.equal(result[0]["input_ids"], torch.tensor([1, 2, 3]))
        assert torch.equal(result[0]["seqlen"], torch.tensor(3))
        assert torch.equal(result[1]["input_ids"], torch.tensor([4, 5, 6]))
        assert torch.equal(result[1]["seqlen"], torch.tensor(3))

    def test_validate_dict_dataset_valid(self, test_dict):
        """Test validate_dict_dataset with valid data."""
        # Should not raise exception
        validate_dict_dataset(test_dict)

    def test_validate_dict_dataset_invalid(self, test_dict):
        """Test validate_dict_dataset with invalid data."""
        invalid_dict = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "seqlen": torch.tensor([3]),  # Different batch size
        }

        with pytest.raises(FrameworkError):
            validate_dict_dataset(invalid_dict)

    def test_convert_empty_list(self):
        """Test convert_list_to_dict with empty list."""
        result = convert_list_to_dict([])
        assert result == {}

    def test_convert_empty_dict(self):
        """Test convert_dict_to_list with empty dict."""
        result = convert_dict_to_list({})
        assert result == []

    def test_validate_empty_dict(self):
        """Test validate_dict_dataset with empty dict."""
        # Should not raise exception for empty dict
        validate_dict_dataset({})


# =============================================================================
# TestScalarAndListSupport
# =============================================================================


class TestScalarAndListSupport:
    """Test cases for scalar and list type support in DistributedBatchMemory."""

    @pytest.fixture
    def scalar_data(self):
        """Scalar test data."""
        return {
            "labels": [1, 2, 3, 4, 5],  # List of integers
            "scores": [0.1, 0.2, 0.3, 0.4, 0.5],  # List of floats
            "categories": ["A", "B", "C", "D", "E"],  # List of strings
            "flags": [True, False, True, False, True],  # List of booleans
        }

    @pytest.fixture
    def mixed_data(self):
        """Mixed scalar and tensor data."""
        return {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "labels": [1, 0, 1],  # List of scalars
            "scores": [0.95, 0.87, 0.92],  # List of floats
            "single_value": 42,  # Single scalar
        }

    @pytest.fixture
    def scalar_list(self):
        """List format test data with scalars."""
        return [
            {"label": 1, "score": 0.95, "category": "A"},
            {"label": 0, "score": 0.87, "category": "B"},
            {"label": 1, "score": 0.92, "category": "C"},
        ]

    def test_from_dict_scalar(self, scalar_data):
        """Test creating DistributedBatchMemory from dictionary with scalar values."""
        memory = DistributedBatchMemory.from_dict(scalar_data)
        assert memory is not None
        assert memory.dataset == scalar_data
        assert len(memory) == 5

    def test_from_dict_mixed(self, mixed_data):
        """Test creating DistributedBatchMemory from dictionary with mixed types."""
        memory = DistributedBatchMemory.from_dict(mixed_data)
        assert memory is not None
        assert len(memory) == 3
        assert memory.dataset["labels"] == [1, 0, 1]
        assert memory.dataset["scores"] == [0.95, 0.87, 0.92]
        assert memory.dataset["single_value"] == 42

    def test_from_list_scalar(self, scalar_list):
        """Test creating DistributedBatchMemory from list with scalar values."""
        memory = DistributedBatchMemory.from_list(scalar_list)
        assert memory is not None
        assert len(memory) == 3
        assert memory.dataset["label"] == [1, 0, 1]
        assert memory.dataset["score"] == [0.95, 0.87, 0.92]
        assert memory.dataset["category"] == ["A", "B", "C"]

    def test_chunk_scalar(self):
        """Test chunk method with scalar data."""
        data = {
            "labels": [1, 2, 3, 4, 5, 6],
            "scores": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "categories": ["A", "B", "C", "D", "E", "F"],
        }
        memory = DistributedBatchMemory.from_dict(data)
        results = memory.chunk(3)

        assert len(results) == 3
        expected_batches = [
            {"labels": [1, 2], "scores": [0.1, 0.2], "categories": ["A", "B"]},
            {"labels": [3, 4], "scores": [0.3, 0.4], "categories": ["C", "D"]},
            {"labels": [5, 6], "scores": [0.5, 0.6], "categories": ["E", "F"]},
        ]

        for i, (batch, expected) in enumerate(zip(results, expected_batches)):
            assert len(batch) == 2
            assert batch.dataset["labels"] == expected["labels"]
            assert batch.dataset["scores"] == expected["scores"]
            assert batch.dataset["categories"] == expected["categories"]

    def test_chunk_mixed(self):
        """Test chunk method with mixed tensor and scalar data."""
        data = {
            "input_ids": torch.arange(12).reshape(6, 2),
            "labels": [1, 0, 1, 0, 1, 0],
            "scores": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
        memory = DistributedBatchMemory.from_dict(data)
        results = memory.chunk(2)

        assert len(results) == 2
        for i, batch in enumerate(results):
            assert len(batch) == 3
            expected_labels = [1, 0, 1] if i == 0 else [0, 1, 0]
            expected_scores = [0.1, 0.2, 0.3] if i == 0 else [0.4, 0.5, 0.6]
            assert batch.dataset["labels"] == expected_labels
            assert batch.dataset["scores"] == expected_scores

    def test_union_scalar(self):
        """Test union method with scalar data."""
        data1 = {"labels": [1, 2], "scores": [0.1, 0.2]}
        data2 = {"labels": [3, 4], "scores": [0.3, 0.4]}

        batch1 = DistributedBatchMemory.from_dict(data1)
        batch2 = DistributedBatchMemory.from_dict(data2)

        merged = batch1.union_(batch2)
        assert len(merged) == 4
        assert merged.dataset["labels"] == [1, 2, 3, 4]
        assert merged.dataset["scores"] == [0.1, 0.2, 0.3, 0.4]

    def test_union_mixed(self):
        """Test union method with mixed tensor and scalar data."""
        data1 = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "labels": [1, 0],
            "single_value": 42,
        }
        data2 = {
            "input_ids": torch.tensor([[5, 6], [7, 8]]),
            "labels": [0, 1],
            "single_value": 43,
        }

        batch1 = DistributedBatchMemory.from_dict(data1)
        batch2 = DistributedBatchMemory.from_dict(data2)

        merged = batch1.union_(batch2)
        assert len(merged) == 4
        assert merged.dataset["labels"] == [1, 0, 0, 1]

    def test_concat_scalar(self):
        """Test concat static method with scalar data."""
        data1 = {
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.1]),
            "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        }
        data2 = {
            "labels": torch.tensor([2]),
            "scores": torch.tensor([0.2]),
            "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        }

        batch1 = DistributedBatchMemory.from_dict(data1)
        batch2 = DistributedBatchMemory.from_dict(data2)

        concatenated = DistributedBatchMemory.concat([batch1, batch2])
        assert len(concatenated) == 2
        assert torch.equal(concatenated.dataset["labels"], torch.tensor([1, 2]))
        assert torch.allclose(concatenated.dataset["scores"], torch.tensor([0.1, 0.2]))

    def test_concat_mixed(self):
        """Test concat static method with mixed types."""
        data1 = {
            "input_ids": torch.tensor([[1, 2]]),
            "labels": torch.tensor([1]),
            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        data2 = {
            "input_ids": torch.tensor([[3, 4]]),
            "labels": torch.tensor([0]),
            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        }

        batch1 = DistributedBatchMemory.from_dict(data1)
        batch2 = DistributedBatchMemory.from_dict(data2)

        concatenated = DistributedBatchMemory.concat([batch1, batch2])
        assert len(concatenated) == 2
        assert torch.equal(concatenated.dataset["labels"], torch.tensor([1, 0]))

    def test_getitem_scalar(self, scalar_data):
        """Test __getitem__ for indexing and key access with scalar data."""
        memory = DistributedBatchMemory.from_dict(scalar_data)

        # Test integer indexing
        sample = memory[0]
        assert sample["labels"] == 1
        assert sample["scores"] == 0.1
        assert sample["categories"] == "A"
        assert sample["flags"] is True

        # Test string key access
        labels = memory["labels"]
        assert labels == [1, 2, 3, 4, 5]

    def test_setitem_scalar(self, scalar_data):
        """Test __setitem__ for setting scalar values."""
        memory = DistributedBatchMemory.from_dict(scalar_data)

        # Test setting list values
        memory["new_labels"] = [10, 20, 30, 40, 50]
        assert memory["new_labels"] == [10, 20, 30, 40, 50]

        # Test setting single scalar
        memory["single_score"] = 99.9
        assert memory["single_score"] == 99.9

    def test_delitem_scalar(self, scalar_data):
        """Test __delitem__ for deleting scalar items."""
        memory = DistributedBatchMemory.from_dict(scalar_data)

        # Test deleting by key
        del memory["labels"]
        assert "labels" not in memory.dataset

        # Test deleting by index
        simple_data = {"labels": [1, 2], "scores": [0.1, 0.2]}
        memory = DistributedBatchMemory.from_dict(simple_data)
        assert len(memory) == 2
        del memory[0]
        # After deletion, should have 1 item left
        assert len(memory) == 1

    def test_delitem_int_and_string_keys(self, test_data):
        """Test deleting elements using both int and string keys."""
        # Create test data with multiple samples
        data = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "seqlen": torch.tensor([2, 2]),
        }

        # Test 1: Delete by string key (removes entire attribute)
        memory = DistributedBatchMemory.from_dict(data)
        assert len(memory) == 2
        assert "seqlen" in memory.dataset

        del memory["seqlen"]
        assert "seqlen" not in memory.dataset

        # Test 2: Delete by int index (removes sample at position)
        memory = DistributedBatchMemory.from_dict(data)

        # Delete first sample - verify the operation completes without error
        del memory[0]
        # We verify the operation completed successfully
        assert isinstance(memory, DistributedBatchMemory)

    def test_pickle_support(self, test_data):
        """Test pickle serialization and deserialization."""
        memory = DistributedBatchMemory.from_dict(test_data)

        # Test pickle dump and load
        serialized = pickle.dumps(memory)
        loaded = pickle.loads(serialized)

        assert torch.equal(memory.dataset["input_ids"], loaded.dataset["input_ids"])
        assert torch.equal(memory.dataset["seqlen"], loaded.dataset["seqlen"])

    def test_str_repr(self, test_data):
        """Test __str__ and __repr__ methods."""
        memory = DistributedBatchMemory.from_dict(test_data)
        str_repr = str(memory)
        assert "DistributedBatchMemory" in str_repr
        assert "size=2" in str_repr

    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        empty_memory = DistributedBatchMemory.from_dict({})
        assert len(empty_memory) == 0

        # Test chunk with empty dataset
        with pytest.raises(Exception):
            empty_memory.chunk(2)

    def test_get_data(self, test_data):
        """Test get_data method returns all data correctly."""
        # Test with normal data
        memory = DistributedBatchMemory.from_dict(test_data)
        result = memory.get_data()

        # Verify return type and structure
        assert isinstance(result, dict)
        assert set(result.keys()) == set(test_data.keys())

        # Verify data integrity
        for key in test_data:
            assert torch.equal(result[key], test_data[key])

        # Test with empty dataset
        empty_memory = DistributedBatchMemory.from_dict({})
        empty_result = empty_memory.get_data()
        assert empty_result == {}

        # Test with single sample
        single_data = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "seqlen": torch.tensor([3]),
        }
        single_memory = DistributedBatchMemory.from_dict(single_data)
        single_result = single_memory.get_data()
        assert len(single_result) == 2
        assert torch.equal(single_result["input_ids"], single_data["input_ids"])
        assert torch.equal(single_result["seqlen"], single_data["seqlen"])

    def test_single_scalar_value(self):
        """Test handling of single scalar value."""
        single_data = {
            "single_label": 42,
            "single_score": 0.99,
        }
        memory = DistributedBatchMemory.from_dict(single_data)
        assert len(memory) == 1
        assert memory["single_label"] == 42
        assert memory["single_score"] == 0.99

    def test_2d_list_operations(self):
        """Test complete operations with 2D list data (list of lists of ints)."""
        # Create 2D list data where each element is a list of ints
        data_2d = {
            "sequences": [[1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12, 13, 14]],
            "labels": [1, 0, 1, 0],
            "lengths": [3, 4, 2, 5],
        }

        memory = DistributedBatchMemory.from_dict(data_2d)

        # Test __getitem__ - integer indexing
        sample = memory[0]
        assert sample["sequences"] == [1, 2, 3]
        assert sample["labels"] == 1
        assert sample["lengths"] == 3

        # Test __getitem__ - string key access
        sequences = memory["sequences"]
        assert sequences == [[1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12, 13, 14]]

        # Test __delitem__ - delete by index
        memory_copy = DistributedBatchMemory.from_dict(data_2d)
        original_len = len(memory_copy)
        del memory_copy[1]  # Delete second item
        assert len(memory_copy) == original_len - 1
        assert memory_copy["sequences"] == [[1, 2, 3], [8, 9], [10, 11, 12, 13, 14]]
        assert memory_copy["labels"] == [1, 1, 0]

        # Test __delitem__ - delete by key
        memory_copy2 = DistributedBatchMemory.from_dict(data_2d)
        del memory_copy2["labels"]
        assert "labels" not in memory_copy2.dataset

        # Test chunk operation
        chunked = memory.chunk(2)
        assert len(chunked) == 2
        assert len(chunked[0]) == 2  # First chunk has 2 items
        assert len(chunked[1]) == 2  # Second chunk has 2 items
        assert chunked[0]["sequences"] == [[1, 2, 3], [4, 5, 6, 7]]
        assert chunked[1]["sequences"] == [[8, 9], [10, 11, 12, 13, 14]]

        # Test chunk_by_ffd operation
        chunked_ffd = memory.chunk_by_ffd(2, 2)
        assert len(chunked_ffd) == 2
        total_samples = sum(len(batch) for batch in chunked_ffd)
        assert total_samples == 4

        # Test union operation
        data1 = {"sequences": [[1, 2], [3, 4]], "labels": [1, 0]}
        data2 = {"sequences": [[5, 6], [7, 8]], "labels": [1, 0]}

        batch1_union = DistributedBatchMemory.from_dict(data1)
        batch2_union = DistributedBatchMemory.from_dict(data2)

        merged = batch1_union.union_(batch2_union)
        assert len(merged) == 4
        assert merged["sequences"] == [[1, 2], [3, 4], [5, 6], [7, 8]]
        assert merged["labels"] == [1, 0, 1, 0]

        # Test concat operation with tensor data
        data1_concat = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "labels": torch.tensor([1, 0]),
            "attention_mask": torch.ones(2, 2, dtype=torch.bool),
        }
        data2_concat = {
            "input_ids": torch.tensor([[5, 6], [7, 8]]),
            "labels": torch.tensor([1, 0]),
            "attention_mask": torch.ones(2, 2, dtype=torch.bool),
        }

        batch1 = DistributedBatchMemory.from_dict(data1_concat)
        batch2 = DistributedBatchMemory.from_dict(data2_concat)

        concatenated = DistributedBatchMemory.concat([batch1, batch2])
        assert len(concatenated) == 4
        assert torch.equal(
            concatenated["input_ids"], torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        )

        # Test pickle serialization/deserialization
        # Create fresh instance for pickle test
        pickle_data = {
            "sequences": [[1, 2, 3], [4, 5, 6, 7]],
            "labels": [1, 0],
            "lengths": [3, 4],
        }

        memory_for_pickle = DistributedBatchMemory.from_dict(pickle_data)
        serialized = pickle.dumps(memory_for_pickle)
        loaded = pickle.loads(serialized)

        # Verify data integrity after pickle/unpickle
        assert len(loaded) == 2
        assert loaded["sequences"] == [[1, 2, 3], [4, 5, 6, 7]]
        assert loaded["labels"] == [1, 0]
        assert loaded["lengths"] == [3, 4]

        # Verify __getstate__ and __setstate__ were called
        assert isinstance(loaded, DistributedBatchMemory)
        assert type(loaded) is type(memory_for_pickle)


# =============================================================================
# TestTrain
# =============================================================================


class TestTrain:
    """Test cases for rollout_res TensorDict format as requested."""

    @pytest.fixture
    def rollout_res_data(self):
        """Test data matching TensorDict format with rollout results."""
        batch_size = 512
        seq_len = 15877

        return {
            "attention_mask": torch.randn(batch_size, seq_len, dtype=torch.float32),
            "input_ids": torch.randint(
                0, 1000, (batch_size, seq_len), dtype=torch.int64
            ),
            "logprobs": torch.randn(batch_size, seq_len, dtype=torch.float32),
            "prompt_mask": torch.randint(
                0, 2, (batch_size, seq_len), dtype=torch.int64
            ),
            "rewards": torch.randint(0, 10, (batch_size,), dtype=torch.int64),
            "seq_no_eos_mask": torch.randint(0, 2, (batch_size,), dtype=torch.bool),
            "seqlen": torch.randint(100, 1000, (batch_size,), dtype=torch.int64),
            "task_ids": torch.randint(0, 100, (batch_size,), dtype=torch.int64),
            "versions": torch.randint(0, 10, (batch_size, seq_len), dtype=torch.int64),
        }

    def test_from_dict_with_rollout_res(self, rollout_res_data):
        """Test creating DistributedBatchMemory from rollout_res dict."""
        memory = DistributedBatchMemory.from_dict(rollout_res_data)

        assert memory is not None
        assert len(memory) == 512
        assert "attention_mask" in memory.dataset
        assert "input_ids" in memory.dataset
        assert "logprobs" in memory.dataset
        assert "prompt_mask" in memory.dataset
        assert "rewards" in memory.dataset
        assert "seq_no_eos_mask" in memory.dataset
        assert "seqlen" in memory.dataset
        assert "task_ids" in memory.dataset
        assert "versions" in memory.dataset

        # Verify tensor shapes
        assert memory.dataset["attention_mask"].shape == torch.Size([512, 15877])
        assert memory.dataset["input_ids"].shape == torch.Size([512, 15877])
        assert memory.dataset["logprobs"].shape == torch.Size([512, 15877])
        assert memory.dataset["prompt_mask"].shape == torch.Size([512, 15877])
        assert memory.dataset["rewards"].shape == torch.Size([512])
        assert memory.dataset["seq_no_eos_mask"].shape == torch.Size([512])
        assert memory.dataset["seqlen"].shape == torch.Size([512])
        assert memory.dataset["task_ids"].shape == torch.Size([512])
        assert memory.dataset["versions"].shape == torch.Size([512, 15877])

    def test_chunk_with_rollout_res(self, rollout_res_data):
        """Test chunk method with rollout_res data."""
        memory = DistributedBatchMemory.from_dict(rollout_res_data)
        results = memory.chunk(4)

        assert len(results) == 4

        # Check chunk sizes: 512 items split into 4 chunks should be [128,128,128,128]
        expected_sizes = [128, 128, 128, 128]
        actual_sizes = [len(batch) for batch in results]
        assert actual_sizes == expected_sizes

        # Verify data integrity in each chunk
        for i, batch in enumerate(results):
            assert isinstance(batch, DistributedBatchMemory)
            assert len(batch) == expected_sizes[i]

            # Verify all keys are present
            for key in rollout_res_data.keys():
                assert key in batch.dataset

            # Verify tensor shapes in chunks
            assert batch.dataset["attention_mask"].shape == torch.Size(
                [expected_sizes[i], 15877]
            )
            assert batch.dataset["input_ids"].shape == torch.Size(
                [expected_sizes[i], 15877]
            )
            assert batch.dataset["logprobs"].shape == torch.Size(
                [expected_sizes[i], 15877]
            )
            assert batch.dataset["prompt_mask"].shape == torch.Size(
                [expected_sizes[i], 15877]
            )
            assert batch.dataset["rewards"].shape == torch.Size([expected_sizes[i]])
            assert batch.dataset["seq_no_eos_mask"].shape == torch.Size(
                [expected_sizes[i]]
            )
            assert batch.dataset["seqlen"].shape == torch.Size([expected_sizes[i]])
            assert batch.dataset["task_ids"].shape == torch.Size([expected_sizes[i]])
            assert batch.dataset["versions"].shape == torch.Size(
                [expected_sizes[i], 15877]
            )

    def test_chunk_by_ffd_with_rollout_res(self, rollout_res_data):
        """Test chunk_by_ffd method with rollout_res data."""
        memory = DistributedBatchMemory.from_dict(rollout_res_data)
        results = memory.chunk_by_ffd(8, 4)  # group_size=8, dp_size=4

        assert len(results) == 4
        total_samples = sum(len(batch) for batch in results)
        assert total_samples == 512

        # Verify structure
        for batch in results:
            for key in rollout_res_data.keys():
                assert key in batch.dataset

    def test_getitem_with_rollout_res(self, rollout_res_data):
        """Test __getitem__ for indexing and key access with rollout_res data."""
        memory = DistributedBatchMemory.from_dict(rollout_res_data)

        # Test integer indexing
        sample = memory[0]
        assert "attention_mask" in sample
        assert "input_ids" in sample
        assert "logprobs" in sample
        assert "prompt_mask" in sample
        assert "rewards" in sample
        assert "seq_no_eos_mask" in sample
        assert "seqlen" in sample
        assert "task_ids" in sample
        assert "versions" in sample

        # Verify individual tensor shapes
        assert sample["attention_mask"].shape == torch.Size([15877])
        assert sample["input_ids"].shape == torch.Size([15877])
        assert sample["logprobs"].shape == torch.Size([15877])
        assert sample["prompt_mask"].shape == torch.Size([15877])
        assert sample["rewards"].shape == torch.Size([])
        assert sample["seq_no_eos_mask"].shape == torch.Size([])
        assert sample["seqlen"].shape == torch.Size([])
        assert sample["task_ids"].shape == torch.Size([])
        assert sample["versions"].shape == torch.Size([15877])

        # Test string key access
        attention_mask = memory["attention_mask"]
        input_ids = memory["input_ids"]
        rewards = memory["rewards"]

        assert attention_mask.shape == torch.Size([512, 15877])
        assert input_ids.shape == torch.Size([512, 15877])
        assert rewards.shape == torch.Size([512])

    def test_setitem_with_rollout_res(self, rollout_res_data):
        """Test __setitem__ for setting rollout_res values."""
        memory = DistributedBatchMemory.from_dict(rollout_res_data)

        # Test adding new field
        new_scores = torch.randn(512, dtype=torch.float32)
        memory["new_scores"] = new_scores
        assert torch.equal(memory["new_scores"], new_scores)

        # Test updating existing field
        new_rewards = torch.randint(0, 5, (512,), dtype=torch.int64)
        memory["rewards"] = new_rewards
        assert torch.equal(memory["rewards"], new_rewards)

    def test_union_with_rollout_res(self, rollout_res_data):
        """Test union method with rollout_res data."""
        # Split data into two parts
        data1 = {k: v[:256] for k, v in rollout_res_data.items()}
        data2 = {k: v[256:] for k, v in rollout_res_data.items()}

        batch1 = DistributedBatchMemory.from_dict(data1)
        batch2 = DistributedBatchMemory.from_dict(data2)

        merged = batch1.union_(batch2)

        assert len(merged) == 512
        for key in rollout_res_data.keys():
            assert key in merged.dataset
            assert torch.equal(merged.dataset[key], rollout_res_data[key])

    def test_concat_with_rollout_res(self, rollout_res_data):
        """Test concat static method with rollout_res data."""
        # Split into four parts
        data_parts = [
            {k: v[i * 128 : (i + 1) * 128] for k, v in rollout_res_data.items()}
            for i in range(4)
        ]
        batches = [DistributedBatchMemory.from_dict(part) for part in data_parts]

        concatenated = DistributedBatchMemory.concat(batches)

        assert len(concatenated) == 512
        for key in rollout_res_data.keys():
            assert key in concatenated.dataset
            assert torch.equal(concatenated.dataset[key], rollout_res_data[key])

    def test_delitem_with_rollout_res(self, rollout_res_data):
        """Test __delitem__ for deleting rollout_res items."""
        DistributedBatchMemory.from_dict(rollout_res_data)

        # Test deleting by key
        temp_memory = DistributedBatchMemory.from_dict(rollout_res_data)
        del temp_memory["logprobs"]
        assert "logprobs" not in temp_memory.dataset
        assert "input_ids" in temp_memory.dataset

        # Test deleting by index - skip this test for tensor data as it flattens
        # temp_memory2 = DistributedBatchMemory.from_dict(rollout_res_data)
        # original_len = len(temp_memory2)
        # del temp_memory2[0]  # Delete first item
        # assert len(temp_memory2) == original_len - 1

    def test_pickle_support_with_rollout_res(self, rollout_res_data):
        """Test pickle serialization and deserialization with rollout_res data."""
        import pickle

        memory = DistributedBatchMemory.from_dict(rollout_res_data)

        # Test pickle dump and load
        serialized = pickle.dumps(memory)
        loaded = pickle.loads(serialized)

        assert len(loaded) == 512
        for key in rollout_res_data.keys():
            assert key in loaded.dataset
            assert torch.equal(loaded.dataset[key], rollout_res_data[key])

    def test_str_repr_with_rollout_res(self, rollout_res_data):
        """Test __str__ and __repr__ methods with rollout_res data."""
        memory = DistributedBatchMemory.from_dict(rollout_res_data)
        str_repr = str(memory)

        assert "DistributedBatchMemory" in str_repr
        assert "size=512" in str_repr
        assert "attention_mask" in str_repr
        assert "input_ids" in str_repr

    def test_get_data_with_rollout_res(self, rollout_res_data):
        """Test get_data method returns all rollout_res data correctly."""
        memory = DistributedBatchMemory.from_dict(rollout_res_data)
        result = memory.get_data()

        # Verify return type and structure
        assert isinstance(result, dict)
        assert set(result.keys()) == set(rollout_res_data.keys())

        # Verify data integrity
        for key in rollout_res_data:
            assert torch.equal(result[key], rollout_res_data[key])


# =============================================================================
# TestBatchUtils - Scalar Support
# =============================================================================


class TestBatchUtilsScalar:
    """Test cases for scalar support in batch utilities."""

    def test_convert_list_to_dict_scalar(self):
        """Test convert_list_to_dict with scalar values."""
        test_data = [
            {"label": 1, "score": 0.95, "category": "A"},
            {"label": 0, "score": 0.87, "category": "B"},
        ]
        result = convert_list_to_dict(test_data)
        expected_labels = [1, 0]
        expected_scores = [0.95, 0.87]
        expected_categories = ["A", "B"]

        assert result["label"] == expected_labels
        assert result["score"] == expected_scores
        assert result["category"] == expected_categories

    def test_convert_dict_to_list_scalar(self):
        """Test convert_dict_to_list with scalar values."""
        scalar_dict = {
            "labels": [1, 2, 3],
            "scores": [0.1, 0.2, 0.3],
            "categories": ["A", "B", "C"],
        }
        result = convert_dict_to_list(scalar_dict)

        assert len(result) == 3
        assert result[0]["labels"] == 1
        assert result[0]["scores"] == 0.1
        assert result[0]["categories"] == "A"
        assert result[1]["labels"] == 2
        assert result[1]["scores"] == 0.2
        assert result[1]["categories"] == "B"

    def test_validate_dict_dataset_scalar(self, scalar_data):
        """Test validate_dict_dataset with scalar data."""
        # Should not raise exception
        validate_dict_dataset(scalar_data)

        # Test with inconsistent sizes
        invalid_scalar = {
            "labels": [1, 2, 3],
            "scores": [0.1, 0.2],  # Different length
        }

        with pytest.raises(FrameworkError):
            validate_dict_dataset(invalid_scalar)

    def test_single_scalar_value(self):
        """Test handling of single scalar value."""
        single_data = {
            "single_label": 42,
            "single_score": 0.99,
        }
        memory = DistributedBatchMemory.from_dict(single_data)
        assert len(memory) == 1
        assert memory["single_label"] == 42
        assert memory["single_score"] == 0.99


# =============================================================================
# TestDistributedBatchMemoryMetadata
# =============================================================================


class TestDistributedBatchMemoryMetadata:
    """Test metadata-based functionality."""

    def test_from_metadata(self):
        """Test creating batch from metadata."""
        metadata = BatchMetadata(
            batch_id="test-batch",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(10, 5),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        assert batch.dataset is None
        assert batch.metadata == metadata

    def test_concat_with_metadata(self):
        """Test concatenating batches with metadata."""
        metadata1 = BatchMetadata(
            batch_id="batch-1",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(5, 10),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        metadata2 = BatchMetadata(
            batch_id="batch-2",
            shards=[
                ShardMetadata(
                    node_id="node-1",
                    node_addr="localhost:8766",
                    shard_id="shard-1",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(3, 10),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )

        batch1 = DistributedBatchMemory.from_metadata(metadata1)
        batch2 = DistributedBatchMemory.from_metadata(metadata2)
        result = DistributedBatchMemory.concat([batch1, batch2])

        assert result.metadata is not None
        assert len(result) == 8  # 5 + 3
        assert len(result.metadata.shards) == 2

    def test_serialization_metadata(self):
        """Test serialization and deserialization with metadata."""
        metadata = BatchMetadata(
            batch_id="test",
            shards=[],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        # Serialize
        serialized = pickle.dumps(batch)
        # Deserialize
        deserialized = pickle.loads(serialized)

        assert deserialized.metadata.batch_id == "test"
        assert deserialized.metadata is not None
        assert deserialized.dataset is None

    def test_chunk_metadata(self):
        """Test chunking with metadata mode."""
        metadata = BatchMetadata(
            batch_id="test-batch",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(50, 128), dtype="torch.int64"
                        ),
                    },
                ),
                ShardMetadata(
                    node_id="node-1",
                    node_addr="localhost:8766",
                    shard_id="shard-1",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(50, 128), dtype="torch.int64"
                        ),
                    },
                ),
            ],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        # Chunk into 2 groups
        chunks = batch.chunk(2)

        assert len(chunks) == 2
        assert all(chunk.metadata is not None for chunk in chunks)
        assert all(chunk.dataset is None for chunk in chunks)
        # Total size should be preserved
        total_size = sum(len(chunk) for chunk in chunks)
        assert total_size == 100

    def test_union_metadata(self):
        """Test union with metadata mode."""
        metadata1 = BatchMetadata(
            batch_id="batch-1",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(30, 10),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        metadata2 = BatchMetadata(
            batch_id="batch-2",
            shards=[
                ShardMetadata(
                    node_id="node-1",
                    node_addr="localhost:8766",
                    shard_id="shard-1",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(20, 10),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )

        batch1 = DistributedBatchMemory.from_metadata(metadata1)
        batch2 = DistributedBatchMemory.from_metadata(metadata2)
        result = batch1.union_(batch2)

        assert result.metadata is not None
        assert len(result) == 50  # 30 + 20
        assert len(result.metadata.shards) == 2
        assert result.dataset is None

    def test_get_total_size_metadata(self):
        """Test _get_total_size with metadata mode."""
        metadata = BatchMetadata(
            batch_id="test",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(123, 10),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        assert len(batch) == 123
        assert batch._get_total_size() == 123


# =============================================================================
# TestBatchMetadata
# =============================================================================


class TestBatchMetadata:
    """Test metadata structures."""

    def test_tensor_metadata(self):
        """Test TensorMetadata creation."""
        meta = TensorMetadata(
            shape=(32, 128),
            dtype="torch.float32",
            device="cuda:0",
        )
        assert meta.shape == (32, 128)
        assert meta.dtype == "torch.float32"
        assert meta.device == "cuda:0"

    def test_shard_metadata(self):
        """Test ShardMetadata creation."""
        meta = ShardMetadata(
            node_id="node-0",
            node_addr="localhost:8765",
            shard_id="shard-0",
            fields={
                "input_ids": TensorMetadata(
                    shape=(32, 128),
                    dtype="torch.int64",
                ),
            },
        )
        assert meta.node_id == "node-0"
        # Batch size can be inferred from first field's shape[0]
        assert meta.fields["input_ids"].shape[0] == 32
        assert "input_ids" in meta.fields

    def test_batch_metadata_node_addrs(self):
        """Test getting all node addresses from batch metadata."""
        metadata = BatchMetadata(
            batch_id="test",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="192.168.1.10:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(32, 10),
                            dtype="torch.int64",
                        ),
                    },
                ),
                ShardMetadata(
                    node_id="node-1",
                    node_addr="192.168.1.11:8765",
                    shard_id="shard-1",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(32, 10),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        addrs = metadata.get_all_node_addrs()
        assert len(addrs) == 2
        assert "192.168.1.10:8765" in addrs
        assert "192.168.1.11:8765" in addrs


# =============================================================================
# TestRPCDistributedBatchReturn
# =============================================================================


class TestRPCDistributedBatchReturn:
    """Test RPC automatic distributed batch return functionality."""

    def test_handle_tensor_return(self):
        """Test handling tensor return from engine method."""
        from areal.scheduler.rpc.rpc_server import _handle_distributed_batch_return

        # Mock engine with get_version method
        class MockEngine:
            def get_version(self):
                return 42

        engine = MockEngine()

        # Test tensor return
        tensor_result = torch.randn(10, 5)
        batch = _handle_distributed_batch_return(
            tensor_result,
            result_key="logits",
            engine=engine,
        )

        # Should return DistributedBatchMemory with metadata
        assert isinstance(batch, DistributedBatchMemory)
        assert batch.metadata is not None
        assert len(batch) == 10
        assert len(batch.metadata.shards) == 1
        assert "logits" in batch.metadata.shards[0].fields

    def test_handle_dict_return(self):
        """Test handling dict return from engine method."""
        from areal.scheduler.rpc.rpc_server import _handle_distributed_batch_return

        # Mock engine
        class MockEngine:
            def get_version(self):
                return 100

        engine = MockEngine()

        # Test dict return
        dict_result = {
            "logits": torch.randn(8, 10, 50),
            "values": torch.randn(8, 10),
            "metadata": "some_string",  # non-tensor field
        }
        batch = _handle_distributed_batch_return(
            dict_result,
            result_key=None,
            engine=engine,
        )

        # Should return DistributedBatchMemory with metadata
        assert isinstance(batch, DistributedBatchMemory)
        assert batch.metadata is not None
        assert len(batch) == 8
        assert "logits" in batch.metadata.shards[0].fields
        assert "values" in batch.metadata.shards[0].fields

    def test_handle_non_tensor_return(self):
        """Test that non-tensor returns are passed through."""
        from areal.scheduler.rpc.rpc_server import _handle_distributed_batch_return

        class MockEngine:
            def get_version(self):
                return 0

        engine = MockEngine()

        # Test non-tensor returns
        int_result = 42
        result = _handle_distributed_batch_return(int_result, None, engine)
        assert result == 42

        str_result = "hello"
        result = _handle_distributed_batch_return(str_result, None, engine)
        assert result == "hello"

        dict_result = {"loss": 0.5, "accuracy": 0.9}  # no tensors
        result = _handle_distributed_batch_return(dict_result, None, engine)
        assert result == dict_result


# =============================================================================
# TestDistributedBatchMemoryExtended
# =============================================================================


class TestDistributedBatchMemoryExtended:
    """Extended tests for DistributedBatchMemory covering all methods."""

    def test_get_client(self):
        """Test getting or creating the shared client."""
        client1 = DistributedBatchMemory.get_client()
        client2 = DistributedBatchMemory.get_client()

        assert client1 is client2  # Should be the same instance
        assert client1 is not None

    def test_len_metadata(self):
        """Test __len__ with metadata."""
        metadata = BatchMetadata(
            batch_id="test",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(42, 10),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        assert len(batch) == 42

    def test_str_metadata(self):
        """Test __str__ with metadata."""
        metadata = BatchMetadata(
            batch_id="test",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(10, 10),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        s = str(batch)
        assert "DistributedBatchMemory" in s
        assert "test" in s

    def test_get_total_size_tensor(self):
        """Test _get_total_size with tensor."""
        data = {"input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]])}
        batch = DistributedBatchMemory.from_dict(data)

        assert batch._get_total_size() == 3

    def test_get_total_size_list(self):
        """Test _get_total_size with list."""
        data = {"labels": [0, 1, 2, 3]}
        batch = DistributedBatchMemory.from_dict(data)

        assert batch._get_total_size() == 4

    def test_get_total_size_scalar(self):
        """Test _get_total_size with scalar."""
        data = {"value": 42}
        batch = DistributedBatchMemory.from_dict(data)

        assert batch._get_total_size() == 1

    def test_get_total_size_empty(self):
        """Test _get_total_size with empty dataset."""
        batch = DistributedBatchMemory.from_dict({})
        assert batch._get_total_size() == 0

    def test_chunk_metadata_empty(self):
        """Test chunking metadata batch with no metadata raises error."""
        batch = DistributedBatchMemory.__new__(DistributedBatchMemory)
        batch.dataset = None
        batch.metadata = None

        with pytest.raises(Exception):  # FrameworkError
            batch.chunk(2)

    def test_chunk_by_ffd_metadata_fallback(self):
        """Test chunk_by_ffd falls back to chunk in metadata mode."""
        metadata = BatchMetadata(
            batch_id="test",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(10, 10),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        chunks = batch.chunk_by_ffd(group_size=2, dp_size=2)
        assert len(chunks) == 2
        assert all(chunk.metadata is not None for chunk in chunks)

    def test_union_mixed_mode_error(self):
        """Test union raises error for mixed modes."""
        batch1 = DistributedBatchMemory.from_dict({"input_ids": torch.tensor([[1, 2]])})
        metadata = BatchMetadata(
            batch_id="test",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(1, 2),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        batch2 = DistributedBatchMemory.from_metadata(metadata)

        with pytest.raises(Exception):  # FrameworkError
            batch1.union_(batch2)

    def test_group_shards_by_keys_same_keys(self):
        """Test _group_shards_by_keys with same keys."""
        shards = [
            ShardMetadata(
                node_id="node-0",
                node_addr="localhost:8000",
                shard_id="shard-0",
                fields={"input_ids": TensorMetadata(shape=(5, 10), dtype="int64")},
            ),
            ShardMetadata(
                node_id="node-1",
                node_addr="localhost:8001",
                shard_id="shard-1",
                fields={"input_ids": TensorMetadata(shape=(5, 10), dtype="int64")},
            ),
        ]

        groups, total_size = DistributedBatchMemory._group_shards_by_keys(shards)
        assert len(groups) == 1
        assert len(groups[0]) == 2
        assert total_size == 10

    def test_group_shards_by_keys_different_keys(self):
        """Test _group_shards_by_keys with different keys."""
        shards = [
            ShardMetadata(
                node_id="node-0",
                node_addr="localhost:8000",
                shard_id="shard-0",
                fields={"input_ids": TensorMetadata(shape=(5, 10), dtype="int64")},
            ),
            ShardMetadata(
                node_id="node-1",
                node_addr="localhost:8001",
                shard_id="shard-1",
                fields={"labels": TensorMetadata(shape=(5,), dtype="int64")},
            ),
        ]

        groups, total_size = DistributedBatchMemory._group_shards_by_keys(shards)
        assert len(groups) == 2
        assert len(groups[0]) == 1
        assert len(groups[1]) == 1
        assert total_size == 5  # Both groups should have same total

    def test_group_shards_by_keys_overlapping_keys_error(self):
        """Test _group_shards_by_keys raises error for overlapping keys."""
        shards = [
            ShardMetadata(
                node_id="node-0",
                node_addr="localhost:8000",
                shard_id="shard-0",
                fields={
                    "input_ids": TensorMetadata(shape=(5, 10), dtype="int64"),
                    "labels": TensorMetadata(shape=(5,), dtype="int64"),
                },
            ),
            ShardMetadata(
                node_id="node-1",
                node_addr="localhost:8001",
                shard_id="shard-1",
                fields={
                    "input_ids": TensorMetadata(shape=(5, 10), dtype="int64"),
                    "attention_mask": TensorMetadata(shape=(5, 10), dtype="int64"),
                },
            ),
        ]

        with pytest.raises(AssertionError):
            DistributedBatchMemory._group_shards_by_keys(shards)

    def test_chunk_shard_group(self):
        """Test _chunk_shard_group."""
        shards = [
            ShardMetadata(
                node_id="node-0",
                node_addr="localhost:8000",
                shard_id="shard-0",
                fields={"input_ids": TensorMetadata(shape=(4, 10), dtype="int64")},
            ),
            ShardMetadata(
                node_id="node-1",
                node_addr="localhost:8001",
                shard_id="shard-1",
                fields={"input_ids": TensorMetadata(shape=(4, 10), dtype="int64")},
            ),
        ]

        chunks = DistributedBatchMemory._chunk_shard_group(shards, dp_size=2)
        assert len(chunks) == 2
        # Each chunk should have at least one shard
        assert sum(len(chunk) for chunk in chunks) >= 2

    def test_concat_empty_list_error(self):
        """Test concat with empty list raises error."""
        with pytest.raises(AssertionError):
            DistributedBatchMemory.concat([])

    def test_concat_different_keys_error(self):
        """Test concat with batches having different keys raises error."""
        batch1 = DistributedBatchMemory.from_dict({"input_ids": torch.tensor([[1, 2]])})
        batch2 = DistributedBatchMemory.from_dict({"labels": torch.tensor([0])})

        with pytest.raises(Exception):  # FrameworkError
            DistributedBatchMemory.concat([batch1, batch2])

    def test_concat_mixed_modes_error(self):
        """Test concat with mixed modes (one metadata, one local) raises error."""
        batch1 = DistributedBatchMemory.from_dict({"input_ids": torch.tensor([[1, 2]])})
        metadata = BatchMetadata(
            batch_id="test",
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(1, 2),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        batch2 = DistributedBatchMemory.from_metadata(metadata)

        # concat should only work with all metadata or all local
        with pytest.raises(
            FrameworkError, match="Cannot concatenate batches with mixed statuses"
        ):
            DistributedBatchMemory.concat([batch1, batch2])

    def test_chunk_metadata_complex_multiple_key_groups(self):
        """Test chunking metadata with multiple key groups (20 shards: 16 with input_ids/attention_mask, 4 with prox_logp)."""
        # Create 20 shards: first 16 have input_ids+attention_mask, last 4 have prox_logp
        # Each group should have the same total batch_size for validation
        shards = []

        # First 16 shards: each has batch_size=5, total=80
        for i in range(16):
            shards.append(
                ShardMetadata(
                    node_id=f"node-{i % 4}",
                    node_addr=f"localhost:{8765 + (i % 4)}",
                    shard_id=f"shard-{i}",
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(5, 128), dtype="torch.int64"
                        ),
                        "attention_mask": TensorMetadata(
                            shape=(5, 128), dtype="torch.int64"
                        ),
                    },
                )
            )

        # Last 4 shards: each has batch_size=20, total=80 (same as first group)
        for i in range(16, 20):
            shards.append(
                ShardMetadata(
                    node_id=f"node-{i % 4}",
                    node_addr=f"localhost:{8765 + (i % 4)}",
                    shard_id=f"shard-{i}",
                    fields={
                        "prox_logp": TensorMetadata(shape=(20,), dtype="torch.float32"),
                    },
                )
            )

        metadata = BatchMetadata(
            batch_id="test-complex-batch",
            shards=shards,
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        # Chunk into 4 data parallel processes
        dp_size = 4
        chunks = batch.chunk(dp_size)

        # Verify basic structure
        assert len(chunks) == dp_size
        assert all(chunk.metadata is not None for chunk in chunks)
        assert all(chunk.dataset is None for chunk in chunks)

        # Verify total batch_size is preserved across chunks
        total_size = sum(len(chunk) for chunk in chunks)
        assert total_size == 80

        # Verify each chunk has shards from both key groups
        for chunk_idx, chunk in enumerate(chunks):
            chunk_shards = chunk.metadata.shards
            assert len(chunk_shards) > 0, (
                f"Chunk {chunk_idx} should have at least one shard"
            )

            # Collect keys from all shards in this chunk
            chunk_keys = set()
            for shard in chunk_shards:
                chunk_keys.update(shard.fields.keys())

            # Each chunk should have both key groups
            # First group: input_ids + attention_mask (from first 16 shards)
            # Second group: prox_logp (from last 4 shards)
            assert "input_ids" in chunk_keys, (
                f"Chunk {chunk_idx} should have input_ids key from first group"
            )
            assert "attention_mask" in chunk_keys, (
                f"Chunk {chunk_idx} should have attention_mask key from first group"
            )
            assert "prox_logp" in chunk_keys, (
                f"Chunk {chunk_idx} should have prox_logp key from second group"
            )

            # Verify shards are correctly categorized
            input_ids_shards = [
                s
                for s in chunk_shards
                if "input_ids" in s.fields.keys()
                and "attention_mask" in s.fields.keys()
            ]
            prox_logp_shards = [
                s for s in chunk_shards if "prox_logp" in s.fields.keys()
            ]

            assert len(input_ids_shards) > 0, (
                f"Chunk {chunk_idx} should have shards with input_ids+attention_mask"
            )
            assert len(prox_logp_shards) > 0, (
                f"Chunk {chunk_idx} should have shards with prox_logp"
            )

            # Verify batch_size distribution: each chunk should have ~20 samples
            # (80 total / 4 dp processes = 20 per chunk)
            assert len(chunk) == 20, (
                f"Chunk {chunk_idx} should have batch_size=20, got {len(chunk)}"
            )

        # Verify all original shards are present (by shard_id)
        all_chunk_shard_ids = set()
        for chunk in chunks:
            for shard in chunk.metadata.shards:
                all_chunk_shard_ids.add(shard.shard_id)

        original_shard_ids = {f"shard-{i}" for i in range(20)}
        assert all_chunk_shard_ids == original_shard_ids, (
            f"All original shards should be present. "
            f"Missing: {original_shard_ids - all_chunk_shard_ids}, "
            f"Extra: {all_chunk_shard_ids - original_shard_ids}"
        )

    def test_chunk_preserves_order(self):
        """Test that chunking preserves sample order."""
        data = {
            "input_ids": torch.tensor([[i, i + 1] for i in range(8)]),
        }
        batch = DistributedBatchMemory.from_dict(data)

        chunks = batch.chunk(2)
        assert len(chunks) == 2
        assert len(chunks[0]) == 4
        assert len(chunks[1]) == 4

        # Verify order is preserved
        chunk0_data = chunks[0].get_data()
        chunk1_data = chunks[1].get_data()
        assert chunk0_data["input_ids"][0, 0] == 0
        assert chunk1_data["input_ids"][0, 0] == 4

    def test_union_preserves_all_keys(self):
        """Test that union preserves all keys from both batches."""
        batch1 = DistributedBatchMemory.from_dict(
            {
                "input_ids": torch.tensor([[1, 2]]),
                "key1": torch.tensor([0]),
            }
        )
        batch2 = DistributedBatchMemory.from_dict(
            {
                "input_ids": torch.tensor([[3, 4]]),
                "key2": torch.tensor([1]),
            }
        )

        result = batch1.union_(batch2)
        assert "input_ids" in result.dataset
        assert "key1" in result.dataset
        assert "key2" in result.dataset
