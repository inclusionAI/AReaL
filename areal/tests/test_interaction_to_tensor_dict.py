"""Tests for InteractionWithTokenLogpReward.to_tensor_dict method."""

from dataclasses import dataclass

import pytest
import torch


@dataclass
class MockModelResponse:
    """Mock model response for testing."""

    input_tokens: list[int]
    output_tokens: list[int]
    output_logprobs: list[float]
    output_version: int = 0

    @property
    def input_len(self) -> int:
        return len(self.input_tokens)

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)

    @property
    def output_versions(self) -> list[int]:
        return [self.output_version] * self.output_len


class TestInteractionToTensorDict:
    """Tests for to_tensor_dict method."""

    def test_basic_tensor_dict_keys(self):
        """Test that to_tensor_dict returns all required keys."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        resp = MockModelResponse(
            input_tokens=[1, 2, 3, 4, 5],
            output_tokens=[6, 7, 8],
            output_logprobs=[-0.5, -0.3, -0.2],
            output_version=1,
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
            task_id=2,
            eos_token_id=8,
            pad_token_id=0,
        )

        result = interaction.to_tensor_dict()

        # Check all required keys are present
        required_keys = [
            "input_ids",
            "loss_mask",
            "prompt_mask",
            "logprobs",
            "versions",
            "attention_mask",
            "rewards",
            "seqlen",
            "task_ids",
            "seq_no_eos_mask",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_tensor_shapes(self):
        """Test that tensors have correct shapes."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        resp = MockModelResponse(
            input_tokens=[1, 2, 3, 4, 5],  # 5 tokens
            output_tokens=[6, 7, 8],  # 3 tokens
            output_logprobs=[-0.5, -0.3, -0.2],
            output_version=1,
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
        )

        result = interaction.to_tensor_dict()
        seq_len = 5 + 3  # 8 total

        # 2D tensors should have shape [1, seq_len]
        assert result["input_ids"].shape == (1, seq_len)
        assert result["loss_mask"].shape == (1, seq_len)
        assert result["prompt_mask"].shape == (1, seq_len)
        assert result["logprobs"].shape == (1, seq_len)
        assert result["versions"].shape == (1, seq_len)
        assert result["attention_mask"].shape == (1, seq_len)

        # 1D tensors should have shape [1]
        assert result["rewards"].shape == (1,)
        assert result["seqlen"].shape == (1,)
        assert result["task_ids"].shape == (1,)
        assert result["seq_no_eos_mask"].shape == (1,)

    def test_prompt_mask_inverse_of_loss_mask(self):
        """Test that prompt_mask is the inverse of loss_mask."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        resp = MockModelResponse(
            input_tokens=[1, 2, 3],  # 3 prompt tokens
            output_tokens=[4, 5],  # 2 output tokens
            output_logprobs=[-0.5, -0.3],
            output_version=1,
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
        )

        result = interaction.to_tensor_dict()

        # loss_mask: [0, 0, 0, 1, 1] (1 for output tokens)
        # prompt_mask: [1, 1, 1, 0, 0] (1 for prompt tokens)
        expected_loss_mask = torch.tensor([[0, 0, 0, 1, 1]])
        expected_prompt_mask = torch.tensor([[1, 1, 1, 0, 0]])

        assert torch.equal(result["loss_mask"], expected_loss_mask)
        assert torch.equal(result["prompt_mask"], expected_prompt_mask)

        # Verify they are inverses
        assert torch.equal(
            result["prompt_mask"] + result["loss_mask"],
            torch.ones_like(result["loss_mask"]),
        )

    def test_seqlen_value(self):
        """Test that seqlen contains the correct sequence length."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        resp = MockModelResponse(
            input_tokens=[1, 2, 3, 4],  # 4 tokens
            output_tokens=[5, 6, 7],  # 3 tokens
            output_logprobs=[-0.5, -0.3, -0.2],
            output_version=1,
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
        )

        result = interaction.to_tensor_dict()

        # seqlen should be 4 + 3 = 7
        assert result["seqlen"].item() == 7

    def test_task_id(self):
        """Test that task_ids contains the correct task ID."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        resp = MockModelResponse(
            input_tokens=[1, 2],
            output_tokens=[3],
            output_logprobs=[-0.5],
            output_version=1,
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
            task_id=5,
        )

        result = interaction.to_tensor_dict()
        assert result["task_ids"].item() == 5

    def test_seq_no_eos_mask_with_eos(self):
        """Test seq_no_eos_mask when sequence ends with EOS."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        eos_token = 100
        resp = MockModelResponse(
            input_tokens=[1, 2],
            output_tokens=[3, eos_token],  # Ends with EOS
            output_logprobs=[-0.5, -0.3],
            output_version=1,
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
            eos_token_id=eos_token,
        )

        result = interaction.to_tensor_dict()
        # Should be False because sequence ends with EOS
        assert not result["seq_no_eos_mask"].item()

    def test_seq_no_eos_mask_without_eos(self):
        """Test seq_no_eos_mask when sequence doesn't end with EOS."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        eos_token = 100
        resp = MockModelResponse(
            input_tokens=[1, 2],
            output_tokens=[3, 4],  # Doesn't end with EOS
            output_logprobs=[-0.5, -0.3],
            output_version=1,
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
            eos_token_id=eos_token,
        )

        result = interaction.to_tensor_dict()
        # Should be True because sequence doesn't end with EOS
        assert result["seq_no_eos_mask"].item()

    def test_seq_no_eos_mask_with_pad(self):
        """Test seq_no_eos_mask when sequence ends with PAD."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        pad_token = 0
        resp = MockModelResponse(
            input_tokens=[1, 2],
            output_tokens=[3, pad_token],  # Ends with PAD
            output_logprobs=[-0.5, -0.3],
            output_version=1,
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
            pad_token_id=pad_token,
        )

        result = interaction.to_tensor_dict()
        # Should be False because sequence ends with PAD
        assert not result["seq_no_eos_mask"].item()

    def test_seq_no_eos_mask_default_without_token_ids(self):
        """Test seq_no_eos_mask defaults to True when no token IDs provided."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        resp = MockModelResponse(
            input_tokens=[1, 2],
            output_tokens=[3, 4],
            output_logprobs=[-0.5, -0.3],
            output_version=1,
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
            # No eos_token_id or pad_token_id provided
        )

        result = interaction.to_tensor_dict()
        # Should default to True when token IDs are not provided
        assert result["seq_no_eos_mask"].item()

    def test_caching(self):
        """Test that results are cached."""
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        resp = MockModelResponse(
            input_tokens=[1, 2],
            output_tokens=[3],
            output_logprobs=[-0.5],
            output_version=1,
        )

        interaction = InteractionWithTokenLogpReward(
            model_response=resp,
            reward=1.0,
        )

        result1 = interaction.to_tensor_dict()
        result2 = interaction.to_tensor_dict()

        # Should return the same cached object
        assert result1 is result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
