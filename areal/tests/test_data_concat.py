"""Tests for concat_list_of_dicts and concat_list_of_dicts_along_seq functions."""

import pytest
import torch

from areal.utils.data import (
    concat_list_of_dicts_along_seq,
    concat_padded_tensors,
)


class TestConcatListOfDictsAlongSeq:
    """Tests for concat_list_of_dicts_along_seq function."""

    def test_empty_list(self):
        """Test with empty list returns empty dict."""
        result = concat_list_of_dicts_along_seq([])
        assert result == {}

    def test_single_dict(self):
        """Test with single dict returns same tensors."""
        d = {
            "logprobs": torch.randn(1, 10),
            "attention_mask": torch.ones(1, 10),
        }
        result = concat_list_of_dicts_along_seq([d])
        assert result["logprobs"].shape == (1, 10)
        assert torch.equal(result["logprobs"], d["logprobs"])

    def test_concat_along_sequence_dim(self):
        """Test concatenating along sequence dimension."""
        d1 = {
            "logprobs": torch.randn(1, 5),
            "attention_mask": torch.ones(1, 5),
        }
        d2 = {
            "logprobs": torch.randn(1, 10),
            "attention_mask": torch.ones(1, 10),
        }
        d3 = {
            "logprobs": torch.randn(1, 7),
            "attention_mask": torch.ones(1, 7),
        }
        result = concat_list_of_dicts_along_seq([d1, d2, d3])
        # Should concat along sequence dimension (dim=1)
        # Total seq_len = 5 + 10 + 7 = 22
        assert result["logprobs"].shape == (1, 22)
        assert result["attention_mask"].shape == (1, 22)

    def test_1d_tensors(self):
        """Test 1D tensors are concatenated along dim=0."""
        d1 = {"rewards": torch.tensor([1.0, 2.0])}
        d2 = {"rewards": torch.tensor([3.0])}
        result = concat_list_of_dicts_along_seq([d1, d2])
        assert result["rewards"].shape == (1,)
        assert torch.equal(result["rewards"], torch.tensor([3.0]))

    def test_batch_size_must_match(self):
        """Test that batch sizes must match for 2D tensors."""
        d1 = {"logprobs": torch.randn(1, 5)}
        d2 = {"logprobs": torch.randn(2, 10)}  # Different batch size
        with pytest.raises(AssertionError, match="must have same batch size"):
            concat_list_of_dicts_along_seq([d1, d2])

    def test_common_keys_only(self):
        """Test only common keys are included in result."""
        d1 = {"a": torch.randn(1, 5), "b": torch.randn(1, 5)}
        d2 = {"a": torch.randn(1, 10), "c": torch.randn(1, 10)}
        result = concat_list_of_dicts_along_seq([d1, d2])
        assert "a" in result
        assert result["a"].shape == (1, 15)
        assert "b" not in result
        assert "c" not in result


class TestConcatPaddedTensors:
    """Tests for concat_padded_tensors function."""

    def test_empty_list(self):
        """Test with empty list returns empty dict."""
        result = concat_padded_tensors([])
        assert result == {}

    def test_same_length_tensors(self):
        """Test with same length tensors."""
        d1 = {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }
        d2 = {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }
        result = concat_padded_tensors([d1, d2])
        assert result["input_ids"].shape == (2, 10)
        assert result["attention_mask"].shape == (2, 10)

    def test_different_length_with_padding(self):
        """Test with different length tensors."""
        d1 = {
            "input_ids": torch.randint(0, 100, (1, 5)),
            "attention_mask": torch.ones(1, 5),
        }
        d2 = {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }
        result = concat_padded_tensors([d1, d2])
        assert result["input_ids"].shape == (2, 10)
        assert result["attention_mask"].shape == (2, 10)
        # First row attention_mask should be padded with 0s
        assert result["attention_mask"][0, :5].sum() == 5
        assert result["attention_mask"][0, 5:].sum() == 0


class TestWorkflowIntegration:
    """Integration tests simulating the workflow in proxy.py."""

    def test_n_samples_shape(self):
        """Test that result has shape[0] = n_samples after processing.

        This simulates the workflow:
        1. Each session produces multiple completions (e.g., multi-turn conversation)
        2. Completions from same session are concatenated along seq dim -> shape [1, total_seq]
        3. Results from different sessions are concatenated along batch dim -> shape [n_samples, max_seq]
        """
        n_samples = 3

        # Simulate completions from 3 sessions (n_samples=3)
        # Each session has multiple completions with varying lengths
        session_completions = {
            "session_0": [
                {"logprobs": torch.randn(1, 100), "attention_mask": torch.ones(1, 100)},
                {"logprobs": torch.randn(1, 150), "attention_mask": torch.ones(1, 150)},
            ],
            "session_1": [
                {"logprobs": torch.randn(1, 200), "attention_mask": torch.ones(1, 200)},
            ],
            "session_2": [
                {"logprobs": torch.randn(1, 80), "attention_mask": torch.ones(1, 80)},
                {"logprobs": torch.randn(1, 120), "attention_mask": torch.ones(1, 120)},
                {"logprobs": torch.randn(1, 50), "attention_mask": torch.ones(1, 50)},
            ],
        }

        trajs = []
        for session_id, completion_list in session_completions.items():
            # Concatenate completions from same session along sequence dimension
            # This produces shape [1, sum(seq_lens)]
            traj = concat_list_of_dicts_along_seq(completion_list)
            trajs.append(traj)

        # Expected seq lengths after concat along seq dim:
        # session_0: 100 + 150 = 250
        # session_1: 200
        # session_2: 80 + 120 + 50 = 250

        assert trajs[0]["logprobs"].shape == (1, 250)
        assert trajs[1]["logprobs"].shape == (1, 200)
        assert trajs[2]["logprobs"].shape == (1, 250)

        # Concatenate trajectories from different sessions along batch dimension
        results = concat_padded_tensors(trajs)

        # Final shape should be [n_samples, max_seq_len] = [3, 250]
        assert results["logprobs"].shape[0] == n_samples
        assert results["logprobs"].shape == (3, 250)
        assert results["attention_mask"].shape == (3, 250)

        # Check attention_mask padding for session_1 (shorter seq)
        # session_1 has seq_len=200, padded to 250
        assert results["attention_mask"][1, :200].sum() == 200
        assert results["attention_mask"][1, 200:].sum() == 0

    def test_variable_completion_counts(self):
        """Test with different number of completions per session."""
        n_samples = 4

        session_completions = {
            f"session_{i}": [
                {
                    "logprobs": torch.randn(1, 50 + j * 10),
                    "attention_mask": torch.ones(1, 50 + j * 10),
                }
                for j in range(i + 1)  # session_0 has 1 completion, session_3 has 4
            ]
            for i in range(n_samples)
        }

        trajs = []
        for session_id, completion_list in session_completions.items():
            traj = concat_list_of_dicts_along_seq(completion_list)
            trajs.append(traj)

        results = concat_padded_tensors(trajs)

        # Verify shape[0] = n_samples
        assert results["logprobs"].shape[0] == n_samples
        assert results["attention_mask"].shape[0] == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
