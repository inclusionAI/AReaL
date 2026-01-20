"""Tests for TokenChoiceTopKRouter.

Run tests:
    pytest areal/tests/experimental/archon/test_router.py -v
"""

import pytest
import torch

from areal.experimental.models.archon.moe.router import TokenChoiceTopKRouter

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for histc with int64 input"
)


class TestTokenChoiceTopKRouterBasic:
    """Basic tests for TokenChoiceTopKRouter."""

    def test_init(self):
        """Test router initialization."""
        dim = 64
        num_experts = 8
        top_k = 2

        router = TokenChoiceTopKRouter(dim, num_experts, top_k)

        assert router.num_experts == num_experts
        assert router.top_k == top_k
        assert router.gate.in_features == dim
        assert router.gate.out_features == num_experts

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        dim = 64
        num_experts = 8
        top_k = 2
        num_tokens = 16

        router = TokenChoiceTopKRouter(dim, num_experts, top_k).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")
        top_scores, selected_indices, num_per_expert = router(x)

        assert top_scores.shape == (num_tokens, top_k)
        assert selected_indices.shape == (num_tokens, top_k)
        assert num_per_expert.shape == (num_experts,)

    def test_token_counts_sum(self):
        """Test that token counts sum to total assignments."""
        dim = 64
        num_experts = 8
        top_k = 2
        num_tokens = 32

        router = TokenChoiceTopKRouter(dim, num_experts, top_k).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")
        _, _, num_per_expert = router(x)

        # Total should be num_tokens * top_k
        assert num_per_expert.sum().item() == num_tokens * top_k

    def test_expert_indices_valid(self):
        """Test that expert indices are in valid range."""
        dim = 64
        num_experts = 8
        top_k = 2
        num_tokens = 32

        router = TokenChoiceTopKRouter(dim, num_experts, top_k).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")
        _, selected_indices, _ = router(x)

        assert selected_indices.min() >= 0
        assert selected_indices.max() < num_experts


class TestScoreFunctions:
    """Tests for different scoring functions."""

    def test_sigmoid_scores(self):
        """Test sigmoid scoring function."""
        dim = 64
        num_experts = 8
        top_k = 2
        num_tokens = 16

        router = TokenChoiceTopKRouter(
            dim, num_experts, top_k, score_func="sigmoid"
        ).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")
        top_scores, _, _ = router(x)

        # Sigmoid scores should be in (0, 1) before scaling
        # After selection, still bounded
        assert top_scores.min() >= 0
        assert top_scores.max() <= 1

    def test_softmax_scores(self):
        """Test softmax scoring function."""
        dim = 64
        num_experts = 8
        top_k = 2
        num_tokens = 16

        router = TokenChoiceTopKRouter(
            dim, num_experts, top_k, score_func="softmax"
        ).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")
        top_scores, _, _ = router(x)

        # Softmax scores should be positive
        assert top_scores.min() >= 0

    def test_route_norm(self):
        """Test route normalization."""
        dim = 64
        num_experts = 8
        top_k = 2
        num_tokens = 16

        router = TokenChoiceTopKRouter(
            dim, num_experts, top_k, route_norm=True, route_scale=1.0
        ).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")
        top_scores, _, _ = router(x)

        # Normalized scores should sum to ~1 per token
        score_sums = top_scores.sum(dim=-1)
        assert torch.allclose(score_sums, torch.ones_like(score_sums), atol=1e-5)

    def test_route_scale(self):
        """Test route scaling."""
        dim = 64
        num_experts = 8
        top_k = 2
        num_tokens = 16
        scale = 2.0

        router_unscaled = TokenChoiceTopKRouter(
            dim, num_experts, top_k, route_scale=1.0
        ).cuda()
        router_scaled = TokenChoiceTopKRouter(
            dim, num_experts, top_k, route_scale=scale
        ).cuda()

        # Copy weights
        with torch.no_grad():
            router_scaled.gate.weight.copy_(router_unscaled.gate.weight)

        router_unscaled.init_weights()
        with torch.no_grad():
            router_scaled.gate.weight.copy_(router_unscaled.gate.weight)

        x = torch.randn(num_tokens, dim, device="cuda")
        scores_unscaled, _, _ = router_unscaled(x)
        scores_scaled, _, _ = router_scaled(x)

        # Scaled should be scale times unscaled
        assert torch.allclose(scores_scaled, scores_unscaled * scale, atol=1e-5)


class TestExpertBias:
    """Tests for expert bias / load balancing."""

    def test_expert_bias_affects_routing(self):
        """Test that expert bias affects routing decisions."""
        dim = 64
        num_experts = 4
        top_k = 1
        num_tokens = 100

        router = TokenChoiceTopKRouter(dim, num_experts, top_k).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")

        # Without bias
        _, indices_no_bias, counts_no_bias = router(x, expert_bias=None)

        # With strong bias towards expert 0
        bias = torch.tensor(
            [-10.0, -10.0, -10.0, 10.0], device="cuda"
        )  # Favor expert 3
        _, indices_with_bias, counts_with_bias = router(x, expert_bias=bias)

        # Expert 3 should get more tokens with bias
        assert counts_with_bias[3] > counts_no_bias[3]

    def test_expert_bias_not_in_scores(self):
        """Test that expert bias doesn't affect the returned scores."""
        dim = 64
        num_experts = 4
        top_k = 1
        num_tokens = 16

        router = TokenChoiceTopKRouter(dim, num_experts, top_k).cuda()
        router.init_weights()

        # Small bias shouldn't change scores much
        bias = torch.tensor([0.01, 0.0, 0.0, -0.01], device="cuda")

        torch.manual_seed(42)
        x1 = torch.randn(num_tokens, dim, device="cuda")
        scores_no_bias, indices_no_bias, _ = router(x1, expert_bias=None)

        torch.manual_seed(42)
        x2 = torch.randn(num_tokens, dim, device="cuda")
        scores_with_bias, indices_with_bias, _ = router(x2, expert_bias=bias)

        # Where indices match, scores should be identical
        mask = indices_no_bias == indices_with_bias
        if mask.any():
            assert torch.allclose(
                scores_no_bias[mask], scores_with_bias[mask], atol=1e-5
            )


class TestNodeLimitedRouting:
    """Tests for node-limited (group-limited) routing."""

    def test_node_limited_routing_shape(self):
        """Test node-limited routing output shapes."""
        dim = 64
        num_experts = 8
        num_expert_groups = 2  # 4 experts per group
        num_limited_groups = 1  # Select 1 group
        top_k = 2
        num_tokens = 16

        router = TokenChoiceTopKRouter(
            dim,
            num_experts,
            top_k,
            num_expert_groups=num_expert_groups,
            num_limited_groups=num_limited_groups,
        ).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")
        top_scores, selected_indices, num_per_expert = router(x)

        assert top_scores.shape == (num_tokens, top_k)
        assert selected_indices.shape == (num_tokens, top_k)

    def test_node_limited_constrains_selection(self):
        """Test that node-limited routing constrains expert selection to groups."""
        dim = 64
        num_experts = 8
        num_expert_groups = 2  # Groups: [0,1,2,3] and [4,5,6,7]
        num_limited_groups = 1
        top_k = 2
        num_tokens = 100

        router = TokenChoiceTopKRouter(
            dim,
            num_experts,
            top_k,
            num_expert_groups=num_expert_groups,
            num_limited_groups=num_limited_groups,
        ).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")
        _, selected_indices, _ = router(x)

        # For each token, experts should be from same group or at most 2 groups
        experts_per_group = num_experts // num_expert_groups
        for token_idx in range(num_tokens):
            experts = selected_indices[token_idx].tolist()
            groups = set(e // experts_per_group for e in experts)
            assert len(groups) <= num_limited_groups

    def test_node_limited_validation(self):
        """Test validation errors for node-limited routing config."""
        dim = 64
        num_experts = 8
        top_k = 2

        # num_limited_groups required when num_expert_groups is set
        router = TokenChoiceTopKRouter(
            dim, num_experts, top_k, num_expert_groups=2, num_limited_groups=None
        ).cuda()

        x = torch.randn(10, dim, device="cuda")
        with pytest.raises(ValueError, match="num_limited_groups must be set"):
            router(x)

        # num_experts must be divisible by num_expert_groups
        router = TokenChoiceTopKRouter(
            dim, num_experts, top_k, num_expert_groups=3, num_limited_groups=1
        ).cuda()
        with pytest.raises(ValueError, match="must be divisible by"):
            router(x)


class TestDebugForceLoadBalance:
    """Tests for debug load balancing mode."""

    def test_force_load_balance(self):
        """Test that debug mode forces balanced routing."""
        dim = 64
        num_experts = 4
        top_k = 1
        num_tokens = 100

        router = TokenChoiceTopKRouter(
            dim, num_experts, top_k, _debug_force_load_balance=True
        ).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")
        _, _, num_per_expert = router(x)

        # With round-robin, each expert should get ~25 tokens
        expected = num_tokens * top_k // num_experts
        for count in num_per_expert:
            assert count.item() == expected

    def test_force_load_balance_top_k_2(self):
        """Test balanced routing with top_k=2."""
        dim = 64
        num_experts = 4
        top_k = 2
        num_tokens = 100

        router = TokenChoiceTopKRouter(
            dim, num_experts, top_k, _debug_force_load_balance=True
        ).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda")
        _, _, num_per_expert = router(x)

        # With round-robin, each expert should get 50 assignments
        expected = num_tokens * top_k // num_experts
        for count in num_per_expert:
            assert count.item() == expected


class TestRouterGradients:
    """Tests for gradient flow through router."""

    # NOTE: Upgrading PyTorch will resolve this in the future.
    @pytest.mark.slow
    def test_gradient_flow(self):
        """Test that gradients flow through router."""
        dim = 64
        num_experts = 8
        top_k = 2
        num_tokens = 16

        router = TokenChoiceTopKRouter(dim, num_experts, top_k).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda", requires_grad=True)
        top_scores, _, _ = router(x)

        loss = top_scores.sum()
        loss.backward()

        assert x.grad is not None
        assert router.gate.weight.grad is not None

    def test_gradient_not_through_indices(self):
        """Test that gradients don't flow through discrete indices."""
        dim = 64
        num_experts = 8
        top_k = 2
        num_tokens = 16

        router = TokenChoiceTopKRouter(dim, num_experts, top_k).cuda()
        router.init_weights()

        x = torch.randn(num_tokens, dim, device="cuda", requires_grad=True)
        top_scores, selected_indices, _ = router(x)

        # Indices should not require gradients
        assert not selected_indices.requires_grad


class TestHistcBehavior:
    """Tests for histc token counting behavior.

    These tests verify the correct behavior of torch.histc for counting
    tokens per expert, particularly around boundary conditions.
    """

    def test_histc_counts_boundary_expert(self):
        """Test that tokens assigned to the last expert are counted correctly.

        This test verifies the fix for the histc max parameter bug where tokens
        assigned to expert (num_experts - 1) could be miscounted when using
        max=num_experts-1 instead of max=num_experts.
        """
        num_experts = 4
        # All tokens go to the last expert (index 3)
        indices = torch.tensor([3, 3, 3, 3], device="cuda")

        counts = torch.histc(indices, bins=num_experts, min=0, max=num_experts).to(
            torch.int64
        )

        expected = torch.tensor([0, 0, 0, 4], device="cuda")
        assert torch.equal(counts, expected), f"Expected {expected}, got {counts}"

    def test_histc_counts_all_experts(self):
        """Test histc correctly counts tokens distributed across all experts."""
        num_experts = 4
        # 2 tokens to expert 0, 3 to expert 1, 1 to expert 2, 0 to expert 3
        indices = torch.tensor([0, 0, 1, 1, 1, 2], device="cuda")

        counts = torch.histc(indices, bins=num_experts, min=0, max=num_experts).to(
            torch.int64
        )

        expected = torch.tensor([2, 3, 1, 0], device="cuda")
        assert torch.equal(counts, expected), f"Expected {expected}, got {counts}"

    def test_histc_counts_single_expert(self):
        """Test histc when all tokens go to a single middle expert."""
        num_experts = 8
        # All tokens go to expert 4
        indices = torch.tensor([4, 4, 4, 4, 4], device="cuda")

        counts = torch.histc(indices, bins=num_experts, min=0, max=num_experts).to(
            torch.int64
        )

        expected = torch.tensor([0, 0, 0, 0, 5, 0, 0, 0], device="cuda")
        assert torch.equal(counts, expected), f"Expected {expected}, got {counts}"
