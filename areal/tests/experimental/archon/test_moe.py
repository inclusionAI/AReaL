"""Tests for MoE composite module.

Run tests:
    pytest areal/tests/experimental/archon/test_moe.py -v
"""

import pytest
import torch

from areal.experimental.models.archon.moe import MoE, MoEArgs

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
)


class TestMoEBasic:
    """Basic tests for MoE module."""

    def test_init(self):
        """Test MoE initialization."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(num_experts=4, top_k=2)

        moe = MoE(moe_args, dim, hidden_dim)

        assert moe.num_experts == 4
        assert moe.top_k == 2
        assert moe.dim == dim
        assert moe.hidden_dim == hidden_dim

    def test_forward_shape(self):
        """Test forward pass output shape."""
        dim = 64
        hidden_dim = 128
        batch_size = 2
        seq_len = 16
        moe_args = MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False)

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(batch_size, seq_len, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape

    def test_forward_values(self):
        """Test that forward produces reasonable values."""
        dim = 32
        hidden_dim = 64
        batch_size = 2
        seq_len = 8
        moe_args = MoEArgs(num_experts=4, top_k=1, use_grouped_mm=False)

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(batch_size, seq_len, dim, device="cuda")
        out = moe(x)

        # Output should not be all zeros or NaN
        assert not torch.all(out == 0)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))


class TestMoEConfigurations:
    """Tests for different MoE configurations."""

    def test_top_k_1(self):
        """Test with top_k=1 (each token to one expert)."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(num_experts=4, top_k=1, use_grouped_mm=False)

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(2, 8, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape

    def test_top_k_2(self):
        """Test with top_k=2 (each token to two experts)."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(num_experts=8, top_k=2, use_grouped_mm=False)

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(2, 8, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape

    def test_many_experts(self):
        """Test with many experts (like Qwen3-MoE)."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(num_experts=64, top_k=8, use_grouped_mm=False)

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(2, 8, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape


class TestMoESharedExperts:
    """Tests for shared experts."""

    def test_with_shared_experts(self):
        """Test MoE with shared experts."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(
            num_experts=4, top_k=2, num_shared_experts=1, use_grouped_mm=False
        )

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        assert moe.shared_experts is not None

        x = torch.randn(2, 8, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape

    def test_without_shared_experts(self):
        """Test MoE without shared experts."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(
            num_experts=4, top_k=2, num_shared_experts=0, use_grouped_mm=False
        )

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        assert moe.shared_experts is None

        x = torch.randn(2, 8, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape


class TestMoEScoreApplication:
    """Tests for score application modes."""

    def test_score_before_experts(self):
        """Test scoring before expert computation."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(
            num_experts=4,
            top_k=2,
            score_before_experts=True,
            use_grouped_mm=False,
        )

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(2, 8, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape

    def test_score_after_experts(self):
        """Test scoring after expert computation."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(
            num_experts=4,
            top_k=2,
            score_before_experts=False,
            use_grouped_mm=False,
        )

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(2, 8, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape


class TestMoELoadBalancing:
    """Tests for load balancing functionality."""

    def test_tokens_per_expert_tracking(self):
        """Test that tokens_per_expert is tracked."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(
            num_experts=4, top_k=2, load_balance_coeff=1e-3, use_grouped_mm=False
        )

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        # Initially zero
        assert torch.all(moe.tokens_per_expert == 0)

        x = torch.randn(2, 8, dim, device="cuda")
        moe(x)

        # After forward, should have counts
        assert moe.tokens_per_expert.sum() > 0
        # Total should be batch_size * seq_len * top_k
        expected_total = 2 * 8 * 2
        assert moe.tokens_per_expert.sum().item() == expected_total

    def test_expert_bias_buffer(self):
        """Test expert_bias buffer exists when load_balance_coeff is set."""
        dim = 64
        hidden_dim = 128

        # With load balancing
        moe_args = MoEArgs(
            num_experts=4, top_k=2, load_balance_coeff=1e-3, use_grouped_mm=False
        )
        moe = MoE(moe_args, dim, hidden_dim).cuda()
        assert moe.expert_bias is not None

        # Without load balancing
        moe_args = MoEArgs(
            num_experts=4, top_k=2, load_balance_coeff=None, use_grouped_mm=False
        )
        moe = MoE(moe_args, dim, hidden_dim).cuda()
        assert moe.expert_bias is None


class TestMoEGradients:
    """Tests for gradient flow."""

    def test_gradient_flow(self):
        """Test that gradients flow through MoE."""
        dim = 32
        hidden_dim = 64
        moe_args = MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False)

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(2, 8, dim, device="cuda", requires_grad=True)
        out = moe(x)

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert moe.router.gate.weight.grad is not None
        assert moe.experts.w1.grad is not None

    def test_gradient_all_experts(self):
        """Test that gradients reach all expert weights with enough tokens."""
        dim = 32
        hidden_dim = 64
        num_experts = 4
        moe_args = MoEArgs(
            num_experts=num_experts,
            top_k=1,
            _debug_force_load_balance=True,  # Ensure all experts get tokens
            use_grouped_mm=False,
        )

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        # Use enough tokens so each expert gets some
        x = torch.randn(1, num_experts * 4, dim, device="cuda", requires_grad=True)
        out = moe(x)

        loss = out.sum()
        loss.backward()

        # All experts should have non-zero gradients
        for expert_idx in range(num_experts):
            assert moe.experts.w1.grad[expert_idx].abs().sum() > 0


class TestMoEIntegration:
    """Integration tests for MoE."""

    def test_qwen3_like_config(self):
        """Test with Qwen3-MoE-like configuration."""
        dim = 128
        hidden_dim = 256
        moe_args = MoEArgs(
            num_experts=128,
            num_shared_experts=0,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            use_grouped_mm=False,
        )

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(1, 16, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape

    def test_multiple_forward_passes(self):
        """Test multiple forward passes accumulate tokens_per_expert."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(
            num_experts=4, top_k=2, load_balance_coeff=1e-3, use_grouped_mm=False
        )

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(2, 8, dim, device="cuda")

        moe(x)
        count_after_1 = moe.tokens_per_expert.sum().item()

        moe(x)
        count_after_2 = moe.tokens_per_expert.sum().item()

        # Counts should accumulate
        assert count_after_2 == count_after_1 * 2

    def test_batch_size_1(self):
        """Test with batch_size=1."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False)

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(1, 16, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape

    def test_single_token(self):
        """Test with single token."""
        dim = 64
        hidden_dim = 128
        moe_args = MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False)

        moe = MoE(moe_args, dim, hidden_dim).cuda()
        moe.init_weights(init_std=0.02)

        x = torch.randn(1, 1, dim, device="cuda")
        out = moe(x)

        assert out.shape == x.shape
