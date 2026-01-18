"""Tests for MoE utility functions.

Run tests:
    pytest areal/tests/experimental/archon/test_moe_utils.py -v
"""

import pytest
import torch

from areal.experimental.models.archon.moe.utils import (
    _permute,
    _unpermute,
    indices_padding_wrapper,
    merge_expert_outputs,
    permute_tokens,
    set_token_group_alignment_size,
    unpermute_tokens,
)


class TestPermuteTokens:
    """Tests for permute_tokens and unpermute_tokens."""

    def test_permute_tokens_basic(self):
        """Test basic token permutation."""
        num_tokens = 4
        dim = 64
        num_experts = 3
        top_k = 2

        tokens = torch.randn(num_tokens, dim)
        # Expert assignments: token 0 -> [0,1], token 1 -> [2,0], etc.
        indices = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 1]])

        permuted, sorted_idx, num_per_expert = permute_tokens(
            tokens, indices, num_experts
        )

        # Check shapes
        assert permuted.shape == (num_tokens * top_k, dim)
        assert sorted_idx.shape == (num_tokens * top_k,)
        assert num_per_expert.shape == (num_experts,)

        # Check token counts sum to total
        assert num_per_expert.sum().item() == num_tokens * top_k

    def test_permute_tokens_grouping(self):
        """Test that tokens are correctly grouped by expert."""
        num_tokens = 4
        dim = 8
        num_experts = 2

        tokens = torch.randn(num_tokens, dim)
        # Token 0,2 go to expert 0; token 1,3 go to expert 1
        indices = torch.tensor([[0], [1], [0], [1]])

        permuted, sorted_idx, num_per_expert = permute_tokens(
            tokens, indices, num_experts
        )

        # Check that expert 0 has 2 tokens and expert 1 has 2 tokens
        assert num_per_expert[0].item() == 2
        assert num_per_expert[1].item() == 2

        # First 2 permuted tokens should be for expert 0 (tokens 0, 2)
        # Last 2 permuted tokens should be for expert 1 (tokens 1, 3)
        assert sorted_idx[:2].tolist() == [0, 2]
        assert sorted_idx[2:].tolist() == [1, 3]

    def test_permute_unpermute_roundtrip(self):
        """Test that permute followed by unpermute recovers original order."""
        num_tokens = 8
        dim = 64
        num_experts = 4
        top_k = 2

        tokens = torch.randn(num_tokens, dim)
        indices = torch.randint(0, num_experts, (num_tokens, top_k))

        permuted, sorted_idx, _ = permute_tokens(tokens, indices, num_experts)

        # Simulate expert computation (identity for this test)
        expert_output = permuted.clone()

        unpermuted = unpermute_tokens(expert_output, sorted_idx, num_tokens, top_k)

        # Each token appears top_k times in expanded form
        expected = tokens.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, dim)
        assert torch.allclose(unpermuted, expected)

    def test_permute_empty_expert(self):
        """Test handling of experts that receive no tokens."""
        num_tokens = 4
        dim = 32
        num_experts = 4

        tokens = torch.randn(num_tokens, dim)
        # Only experts 0 and 1 receive tokens; experts 2,3 are empty
        indices = torch.tensor([[0], [1], [0], [1]])

        permuted, sorted_idx, num_per_expert = permute_tokens(
            tokens, indices, num_experts
        )

        # Check empty experts have 0 count
        assert num_per_expert[2].item() == 0
        assert num_per_expert[3].item() == 0


class TestMergeExpertOutputs:
    """Tests for merge_expert_outputs."""

    def test_merge_score_before_experts(self):
        """Test merging when scores were applied before expert computation."""
        num_tokens = 4
        top_k = 2
        dim = 64

        # Expert outputs already scaled by scores
        unpermuted = torch.randn(num_tokens * top_k, dim)
        scores = torch.randn(
            num_tokens, top_k
        )  # Not used when score_before_experts=True

        merged = merge_expert_outputs(
            unpermuted, scores, num_tokens, top_k, score_before_experts=True
        )

        assert merged.shape == (num_tokens, dim)
        # Should be sum over top_k dimension
        expected = unpermuted.view(num_tokens, top_k, dim).sum(dim=1)
        assert torch.allclose(merged, expected)

    def test_merge_score_after_experts(self):
        """Test merging with weighted sum when scores applied after."""
        num_tokens = 4
        top_k = 2
        dim = 64

        unpermuted = torch.randn(num_tokens * top_k, dim)
        scores = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

        merged = merge_expert_outputs(
            unpermuted, scores, num_tokens, top_k, score_before_experts=False
        )

        assert merged.shape == (num_tokens, dim)

        # Verify weighted sum manually
        reshaped = unpermuted.view(num_tokens, top_k, dim)
        expected = (
            torch.bmm(scores.unsqueeze(1).float(), reshaped.float())
            .squeeze(1)
            .to(unpermuted.dtype)
        )
        assert torch.allclose(merged, expected, atol=1e-5)


class TestPermuteUnpermute:
    """Tests for _permute and _unpermute functions."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_permute_unpermute_roundtrip_cuda(self):
        """Test that _permute followed by _unpermute recovers original tokens on CUDA."""
        num_tokens = 32
        dim = 64
        num_experts = 4
        ep_degree = 1  # No EP (single rank)

        tokens = torch.randn(num_tokens, dim, device="cuda")
        # Simulated token distribution across experts
        num_tokens_per_expert = torch.tensor(
            [8, 10, 7, 7], dtype=torch.int64, device="cuda"
        )

        input_shape, permuted, indices, aligned_counts = _permute(
            tokens, num_tokens_per_expert, ep_degree, num_experts
        )

        # Simulate expert computation (identity)
        output = permuted.clone()

        unpermuted = _unpermute(output, input_shape, indices)

        assert unpermuted.shape == tokens.shape
        assert torch.allclose(unpermuted, tokens)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_alignment_padding_cuda(self):
        """Test that aligned counts are multiples of alignment size on CUDA."""
        alignment = 8
        set_token_group_alignment_size(alignment)

        num_tokens = 23  # Odd number to test padding
        dim = 32
        num_experts = 3
        ep_degree = 1

        tokens = torch.randn(num_tokens, dim, device="cuda")
        num_tokens_per_expert = torch.tensor(
            [5, 11, 7], dtype=torch.int64, device="cuda"
        )

        input_shape, permuted, indices, aligned_counts = _permute(
            tokens, num_tokens_per_expert, ep_degree, num_experts
        )

        # Check that aligned counts are multiples of alignment
        for count in aligned_counts:
            assert count.item() % alignment == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_empty_expert_handling_cuda(self):
        """Test handling of empty experts on CUDA."""
        alignment = 8
        set_token_group_alignment_size(alignment)

        num_tokens = 16
        dim = 32
        num_experts = 4
        ep_degree = 1

        tokens = torch.randn(num_tokens, dim, device="cuda")
        # Expert 2 gets no tokens
        num_tokens_per_expert = torch.tensor(
            [6, 6, 0, 4], dtype=torch.int64, device="cuda"
        )

        input_shape, permuted, indices, aligned_counts = _permute(
            tokens, num_tokens_per_expert, ep_degree, num_experts
        )

        # Empty expert should still get aligned minimum size
        assert aligned_counts[2].item() >= alignment

        output = permuted.clone()
        unpermuted = _unpermute(output, input_shape, indices)
        assert unpermuted.shape == tokens.shape


class TestIndicesPaddingWrapper:
    """Tests for indices_padding_wrapper."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wrapper_basic_cuda(self):
        """Test that wrapper correctly wraps expert function on CUDA."""
        num_tokens = 24
        dim = 64
        hidden_dim = 128
        num_experts = 4

        # Create dummy expert weights
        w1 = torch.randn(num_experts, hidden_dim, dim, device="cuda")
        w2 = torch.randn(num_experts, dim, hidden_dim, device="cuda")
        w3 = torch.randn(num_experts, hidden_dim, dim, device="cuda")

        x = torch.randn(num_tokens, dim, device="cuda")
        num_tokens_per_expert = torch.tensor(
            [6, 6, 6, 6], dtype=torch.int64, device="cuda"
        )

        # Simple identity-like function for testing
        def dummy_experts(w1, w2, w3, x, num_tokens_per_expert):
            return x  # Just return input

        wrapped = indices_padding_wrapper(dummy_experts)
        result = wrapped(w1, w2, w3, x, num_tokens_per_expert)

        assert result.shape == x.shape
        assert torch.allclose(result, x)


class TestEndToEndPermutation:
    """End-to-end tests simulating MoE forward pass."""

    def test_full_moe_permutation_flow(self):
        """Test complete permutation flow as used in MoE."""
        batch_size = 2
        seq_len = 8
        dim = 64
        num_experts = 4
        top_k = 2

        # Input tokens
        tokens = torch.randn(batch_size * seq_len, dim)

        # Simulate router output
        indices = torch.randint(0, num_experts, (batch_size * seq_len, top_k))
        scores = torch.softmax(torch.randn(batch_size * seq_len, top_k), dim=-1)

        # Step 1: Permute tokens by expert
        permuted, sorted_idx, num_per_expert = permute_tokens(
            tokens, indices, num_experts
        )

        # Step 2: Apply scores before expert (optional)
        # In real MoE, this would be: permuted * scores_sorted
        scores_sorted = scores.view(-1)[sorted_idx]
        scaled_permuted = permuted * scores_sorted.unsqueeze(-1)

        # Step 3: Simulate expert computation (identity for test)
        expert_output = scaled_permuted.clone()

        # Step 4: Unpermute
        unpermuted = unpermute_tokens(
            expert_output, sorted_idx, batch_size * seq_len, top_k
        )

        # Step 5: Merge outputs
        merged = merge_expert_outputs(
            unpermuted, scores, batch_size * seq_len, top_k, score_before_experts=True
        )

        assert merged.shape == tokens.shape

    def test_gradient_flow(self):
        """Test that gradients flow correctly through permutation."""
        num_tokens = 8
        dim = 32
        num_experts = 2
        top_k = 1

        tokens = torch.randn(num_tokens, dim, requires_grad=True)
        indices = torch.randint(0, num_experts, (num_tokens, top_k))

        permuted, sorted_idx, _ = permute_tokens(tokens, indices, num_experts)

        # Compute loss and backprop
        loss = permuted.sum()
        loss.backward()

        assert tokens.grad is not None
        # All gradients should be 1 since it's just a permutation
        assert torch.allclose(tokens.grad, torch.ones_like(tokens.grad))
