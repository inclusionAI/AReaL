"""Tests for GroupedExperts module.

Run tests:
    pytest areal/tests/experimental/archon/test_grouped_experts.py -v
"""

import pytest
import torch

from areal.experimental.models.archon.moe.grouped_experts import (
    GroupedExperts,
    _check_grouped_mm_available,
    _run_experts_for_loop,
)


class TestGroupedExpertsBasic:
    """Basic tests for GroupedExperts."""

    def test_init(self):
        """Test GroupedExperts initialization."""
        dim = 64
        hidden_dim = 128
        num_experts = 4

        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=False)

        assert experts.dim == dim
        assert experts.hidden_dim == hidden_dim
        assert experts.num_experts == num_experts
        assert experts.w1.shape == (num_experts, hidden_dim, dim)
        assert experts.w2.shape == (num_experts, dim, hidden_dim)
        assert experts.w3.shape == (num_experts, hidden_dim, dim)

    def test_forward_for_loop(self):
        """Test forward pass using for-loop backend."""
        dim = 32
        hidden_dim = 64
        num_experts = 4
        num_tokens = 10

        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=False)
        experts.init_weights()

        # Input tokens already sorted by expert
        x = torch.randn(num_tokens, dim)
        # Token distribution: 3, 2, 4, 1
        num_tokens_per_expert = torch.tensor([3, 2, 4, 1], dtype=torch.int64)

        output = experts(x, num_tokens_per_expert)

        assert output.shape == x.shape

    def test_forward_empty_expert(self):
        """Test forward with an expert receiving no tokens."""
        dim = 32
        hidden_dim = 64
        num_experts = 4
        num_tokens = 8

        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=False)
        experts.init_weights()

        x = torch.randn(num_tokens, dim)
        # Expert 1 receives no tokens
        num_tokens_per_expert = torch.tensor([3, 0, 3, 2], dtype=torch.int64)

        output = experts(x, num_tokens_per_expert)

        assert output.shape == x.shape

    def test_forward_single_expert(self):
        """Test forward with all tokens going to one expert."""
        dim = 32
        hidden_dim = 64
        num_experts = 4
        num_tokens = 10

        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=False)
        experts.init_weights()

        x = torch.randn(num_tokens, dim)
        # All tokens go to expert 0
        num_tokens_per_expert = torch.tensor([10, 0, 0, 0], dtype=torch.int64)

        output = experts(x, num_tokens_per_expert)

        assert output.shape == x.shape


class TestGroupedExpertsInitWeights:
    """Tests for weight initialization."""

    def test_init_weights_default(self):
        """Test default weight initialization."""
        dim = 32
        hidden_dim = 64
        num_experts = 4

        experts = GroupedExperts(dim, hidden_dim, num_experts)
        experts.init_weights()

        # Check weights are initialized (not zeros)
        assert not torch.allclose(experts.w1, torch.zeros_like(experts.w1))
        assert not torch.allclose(experts.w2, torch.zeros_like(experts.w2))
        assert not torch.allclose(experts.w3, torch.zeros_like(experts.w3))

    def test_init_weights_custom_std(self):
        """Test weight initialization with custom std."""
        dim = 32
        hidden_dim = 64
        num_experts = 4

        experts = GroupedExperts(dim, hidden_dim, num_experts)
        experts.init_weights(init_std=0.01)

        # w1 should have std ~0.02, w2/w3 should have std ~0.01
        # Just verify they're not zeros and have reasonable values
        assert experts.w1.std() < 0.1
        assert experts.w2.std() < 0.05
        assert experts.w3.std() < 0.05


class TestGroupedExpertsGradients:
    """Tests for gradient computation."""

    def test_gradient_flow(self):
        """Test that gradients flow correctly through experts."""
        dim = 32
        hidden_dim = 64
        num_experts = 4
        num_tokens = 10

        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=False)
        experts.init_weights()

        x = torch.randn(num_tokens, dim, requires_grad=True)
        num_tokens_per_expert = torch.tensor([3, 2, 4, 1], dtype=torch.int64)

        output = experts(x, num_tokens_per_expert)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert experts.w1.grad is not None
        assert experts.w2.grad is not None
        assert experts.w3.grad is not None

    def test_gradient_per_expert(self):
        """Test that only relevant expert weights receive gradients."""
        dim = 16
        hidden_dim = 32
        num_experts = 4

        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=False)
        experts.init_weights()

        # Only expert 0 receives tokens
        x = torch.randn(5, dim, requires_grad=True)
        num_tokens_per_expert = torch.tensor([5, 0, 0, 0], dtype=torch.int64)

        output = experts(x, num_tokens_per_expert)
        loss = output.sum()
        loss.backward()

        # Only expert 0 should have non-zero gradients
        assert experts.w1.grad[0].abs().sum() > 0
        assert experts.w1.grad[1:].abs().sum() == 0
        assert experts.w2.grad[0].abs().sum() > 0
        assert experts.w2.grad[1:].abs().sum() == 0


class TestRunExpertsForLoop:
    """Tests for the _run_experts_for_loop function."""

    def test_swiglu_computation(self):
        """Verify SwiGLU computation: silu(x @ w1.T) * (x @ w3.T) @ w2.T."""
        dim = 8
        hidden_dim = 16
        num_experts = 2

        torch.manual_seed(42)
        w1 = torch.randn(num_experts, hidden_dim, dim)
        w2 = torch.randn(num_experts, dim, hidden_dim)
        w3 = torch.randn(num_experts, hidden_dim, dim)

        x = torch.randn(3, dim)  # 3 tokens, all to expert 0
        num_tokens_per_expert = torch.tensor([3, 0])

        output = _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

        # Manual computation for expert 0
        h = torch.nn.functional.silu(x @ w1[0].T)
        h = h * (x @ w3[0].T)
        expected = h @ w2[0].T

        assert torch.allclose(output[:3], expected, atol=1e-5)


class TestGroupedMMBackend:
    """Tests for grouped_mm backend."""

    def test_check_grouped_mm_available(self):
        """Test grouped_mm availability check."""
        available = _check_grouped_mm_available()
        # This should return True only if CUDA is available (grouped_mm requires CUDA)
        assert isinstance(available, bool)
        if not torch.cuda.is_available():
            assert not available, "grouped_mm should not be available without CUDA"

    @pytest.mark.skipif(
        not _check_grouped_mm_available(), reason="torch._grouped_mm requires CUDA"
    )
    def test_grouped_mm_vs_for_loop(self):
        """Compare grouped_mm and for-loop outputs."""
        dim = 32
        hidden_dim = 64
        num_experts = 4
        num_tokens = 20
        device = torch.device("cuda")

        torch.manual_seed(42)

        # Create two experts with same weights
        experts_grouped = GroupedExperts(
            dim, hidden_dim, num_experts, use_grouped_mm=True
        ).to(device)
        experts_loop = GroupedExperts(
            dim, hidden_dim, num_experts, use_grouped_mm=False
        ).to(device)

        # Copy weights
        with torch.no_grad():
            experts_loop.w1.copy_(experts_grouped.w1)
            experts_loop.w2.copy_(experts_grouped.w2)
            experts_loop.w3.copy_(experts_grouped.w3)

        experts_grouped.init_weights()
        with torch.no_grad():
            experts_loop.w1.copy_(experts_grouped.w1)
            experts_loop.w2.copy_(experts_grouped.w2)
            experts_loop.w3.copy_(experts_grouped.w3)

        x = torch.randn(num_tokens, dim, device=device)
        num_tokens_per_expert = torch.tensor(
            [5, 5, 5, 5], dtype=torch.int64, device=device
        )

        # Note: grouped_mm requires aligned inputs, so we test with evenly distributed tokens
        out_grouped = experts_grouped(x, num_tokens_per_expert)
        out_loop = experts_loop(x, num_tokens_per_expert)

        # Outputs should be close (may differ due to bf16 casting in grouped_mm)
        assert torch.allclose(out_grouped, out_loop, atol=1e-2, rtol=1e-2)


class TestGroupedExpertsIntegration:
    """Integration tests simulating MoE usage."""

    def test_with_permutation(self):
        """Test GroupedExperts with token permutation from routing."""
        from areal.experimental.models.archon.moe.utils import (
            permute_tokens,
            unpermute_tokens,
        )

        dim = 32
        hidden_dim = 64
        num_experts = 4
        num_tokens = 16
        top_k = 2

        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=False)
        experts.init_weights()

        # Original tokens
        tokens = torch.randn(num_tokens, dim)

        # Simulate router output
        indices = torch.randint(0, num_experts, (num_tokens, top_k))

        # Permute tokens by expert
        permuted, sorted_idx, num_per_expert = permute_tokens(
            tokens, indices, num_experts
        )

        # Expert computation
        expert_output = experts(permuted, num_per_expert)

        # Unpermute back
        unpermuted = unpermute_tokens(expert_output, sorted_idx, num_tokens, top_k)

        assert unpermuted.shape == (num_tokens * top_k, dim)

    def test_batch_sequence_input(self):
        """Test with batch x sequence shaped input."""
        dim = 64
        hidden_dim = 128
        num_experts = 8
        batch_size = 2
        seq_len = 8
        top_k = 2

        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=False)
        experts.init_weights()

        # Flatten batch x sequence
        tokens = torch.randn(batch_size * seq_len, dim)

        # Simulate router: each token goes to top_k experts
        from areal.experimental.models.archon.moe.utils import permute_tokens

        indices = torch.randint(0, num_experts, (batch_size * seq_len, top_k))
        permuted, _, num_per_expert = permute_tokens(tokens, indices, num_experts)

        output = experts(permuted, num_per_expert)

        assert output.shape[0] == batch_size * seq_len * top_k
        assert output.shape[1] == dim
