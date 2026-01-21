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

# Marked as slow to exclude from CI pipeline.
# NOTE: Upgrading PyTorch will resolve this in the future.
pytestmark = pytest.mark.slow


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

    @pytest.mark.skipif(
        not _check_grouped_mm_available(), reason="grouped_mm requires CUDA"
    )
    def test_gradient_flow_with_padding(self):
        """Test gradients flow correctly through the padding wrapper.

        Ensures padding/unpermute operations preserve gradient computation.
        """
        dim = 32
        hidden_dim = 64
        num_experts = 4
        num_tokens = 23  # Unaligned

        torch.manual_seed(42)
        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=True)
        experts.init_weights()
        experts = experts.cuda()

        x = torch.randn(num_tokens, dim, device="cuda", requires_grad=True)
        num_tokens_per_expert = torch.tensor(
            [6, 7, 5, 5], dtype=torch.int64, device="cuda"
        )

        out = experts(x, num_tokens_per_expert)
        loss = out.sum()
        loss.backward()

        # Check that gradients are computed
        assert x.grad is not None, "Input gradient should be computed"
        assert experts.w1.grad is not None, "w1 gradient should be computed"
        assert experts.w2.grad is not None, "w2 gradient should be computed"
        assert experts.w3.grad is not None, "w3 gradient should be computed"

        # Check gradient shapes match parameter shapes
        assert x.grad.shape == x.shape, "Input gradient shape mismatch"
        assert experts.w1.grad.shape == experts.w1.shape, "w1 gradient shape mismatch"
        assert experts.w2.grad.shape == experts.w2.shape, "w2 gradient shape mismatch"
        assert experts.w3.grad.shape == experts.w3.shape, "w3 gradient shape mismatch"

        # Gradients should not be all zeros (sanity check)
        assert x.grad.abs().sum() > 0, "Input gradient should be non-zero"
        assert experts.w1.grad.abs().sum() > 0, "w1 gradient should be non-zero"
        assert experts.w2.grad.abs().sum() > 0, "w2 gradient should be non-zero"
        assert experts.w3.grad.abs().sum() > 0, "w3 gradient should be non-zero"


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

    @pytest.mark.skipif(
        not _check_grouped_mm_available(), reason="grouped_mm requires CUDA"
    )
    def test_grouped_mm_unaligned_tokens(self):
        """Verify grouped_mm with unaligned token counts works via padding wrapper.

        The padding wrapper should be applied automatically when:
        - use_grouped_mm=True
        - Weights are not DTensor OR DTensor doesn't have "ep" in device_mesh
        """
        dim = 32
        hidden_dim = 64
        num_experts = 4
        num_tokens = 23  # Not aligned to 8 (TOKEN_GROUP_ALIGN_SIZE_M)

        torch.manual_seed(42)
        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=True)
        experts.init_weights()
        experts = experts.cuda()

        experts_loop = GroupedExperts(
            dim, hidden_dim, num_experts, use_grouped_mm=False
        )
        experts_loop = experts_loop.cuda()
        with torch.no_grad():
            experts_loop.w1.copy_(experts.w1)
            experts_loop.w2.copy_(experts.w2)
            experts_loop.w3.copy_(experts.w3)

        x = torch.randn(num_tokens, dim, device="cuda")
        # Unaligned token distribution: 6 + 7 + 5 + 5 = 23 (not aligned to 8)
        num_tokens_per_expert = torch.tensor(
            [6, 7, 5, 5], dtype=torch.int64, device="cuda"
        )

        out_grouped = experts(x, num_tokens_per_expert)
        out_loop = experts_loop(x, num_tokens_per_expert)

        # Allow some tolerance due to bf16 in grouped_mm
        assert torch.allclose(out_grouped, out_loop, atol=1e-2, rtol=1e-2)

    @pytest.mark.skipif(
        not _check_grouped_mm_available(), reason="grouped_mm requires CUDA"
    )
    def test_grouped_mm_aligned_tokens(self):
        """Verify grouped_mm with aligned token counts works correctly."""
        dim = 32
        hidden_dim = 64
        num_experts = 4
        num_tokens = 32  # Aligned to 8

        torch.manual_seed(42)
        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=True)
        experts.init_weights()
        experts = experts.cuda()

        experts_loop = GroupedExperts(
            dim, hidden_dim, num_experts, use_grouped_mm=False
        )
        experts_loop = experts_loop.cuda()
        with torch.no_grad():
            experts_loop.w1.copy_(experts.w1)
            experts_loop.w2.copy_(experts.w2)
            experts_loop.w3.copy_(experts.w3)

        x = torch.randn(num_tokens, dim, device="cuda")
        # Aligned token distribution: 8 + 8 + 8 + 8 = 32
        num_tokens_per_expert = torch.tensor(
            [8, 8, 8, 8], dtype=torch.int64, device="cuda"
        )

        out_grouped = experts(x, num_tokens_per_expert)
        out_loop = experts_loop(x, num_tokens_per_expert)

        assert torch.allclose(out_grouped, out_loop, atol=1e-2, rtol=1e-2)

    @pytest.mark.skipif(
        not _check_grouped_mm_available(), reason="grouped_mm requires CUDA"
    )
    def test_ep_detection_logic(self):
        """Test the EP detection logic in GroupedExperts.forward().

        Verifies the condition for applying padding wrapper:
        - Non-DTensor weights -> padding wrapper applied
        - DTensor without "ep" in mesh -> padding wrapper applied
        - DTensor with "ep" in mesh -> padding wrapper NOT applied (EP handles it)
        """
        from unittest.mock import MagicMock, patch

        from torch.distributed.tensor import DTensor

        dim = 16
        hidden_dim = 32
        num_experts = 2

        # Dummy inputs for forward call
        x = torch.randn(10, dim, device="cuda")
        num_tokens_per_expert = torch.tensor([5, 5], device="cuda")

        with (
            patch(
                "areal.experimental.models.archon.moe.grouped_experts._run_experts_grouped_mm"
            ) as mock_run_experts,
            patch(
                "areal.experimental.models.archon.moe.grouped_experts.indices_padding_wrapper"
            ) as mock_padding_wrapper,
        ):
            # Setup mock return values
            mock_wrapped_fn = MagicMock(
                return_value=torch.randn(10, dim, device="cuda")
            )
            mock_padding_wrapper.return_value = mock_wrapped_fn
            mock_run_experts.return_value = torch.randn(10, dim, device="cuda")

            # Test case 1: Non-DTensor weights (regular torch.nn.Parameter)
            experts = GroupedExperts(
                dim, hidden_dim, num_experts, use_grouped_mm=True
            ).cuda()

            experts(x, num_tokens_per_expert)
            mock_padding_wrapper.assert_called_once_with(mock_run_experts)
            mock_wrapped_fn.assert_called_once()

            mock_padding_wrapper.reset_mock()
            mock_wrapped_fn.reset_mock()
            mock_run_experts.reset_mock()

            # Test case 2: Mock DTensor without "ep" in mesh
            mock_dtensor_no_ep = MagicMock(spec=DTensor)
            mock_mesh_no_ep = MagicMock()
            mock_mesh_no_ep.mesh_dim_names = ("dp", "tp")  # No "ep"
            mock_dtensor_no_ep.device_mesh = mock_mesh_no_ep
            mock_dtensor_no_ep.to_local.return_value = torch.randn(
                num_experts, hidden_dim, dim, device="cuda"
            )

            mock_w2_no_ep = MagicMock(spec=DTensor)
            mock_w2_no_ep.to_local.return_value = torch.randn(
                num_experts, dim, hidden_dim, device="cuda"
            )
            mock_w3_no_ep = MagicMock(spec=DTensor)
            mock_w3_no_ep.to_local.return_value = torch.randn(
                num_experts, hidden_dim, dim, device="cuda"
            )

            # Use object.__setattr__ to bypass nn.Module's parameter check
            object.__setattr__(experts, "w1", mock_dtensor_no_ep)
            object.__setattr__(experts, "w2", mock_w2_no_ep)
            object.__setattr__(experts, "w3", mock_w3_no_ep)

            experts(x, num_tokens_per_expert)
            mock_padding_wrapper.assert_called_once_with(mock_run_experts)
            mock_wrapped_fn.assert_called_once()

            mock_padding_wrapper.reset_mock()
            mock_wrapped_fn.reset_mock()
            mock_run_experts.reset_mock()

            # Test case 3: Mock DTensor with "ep" in mesh
            mock_dtensor_with_ep = MagicMock(spec=DTensor)
            mock_mesh_with_ep = MagicMock()
            mock_mesh_with_ep.mesh_dim_names = ("dp", "ep", "tp")  # Has "ep"
            mock_dtensor_with_ep.device_mesh = mock_mesh_with_ep
            mock_dtensor_with_ep.to_local.return_value = torch.randn(
                num_experts, hidden_dim, dim, device="cuda"
            )

            mock_w2_with_ep = MagicMock(spec=DTensor)
            mock_w2_with_ep.to_local.return_value = torch.randn(
                num_experts, dim, hidden_dim, device="cuda"
            )
            mock_w3_with_ep = MagicMock(spec=DTensor)
            mock_w3_with_ep.to_local.return_value = torch.randn(
                num_experts, hidden_dim, dim, device="cuda"
            )

            # Use object.__setattr__ to bypass nn.Module's parameter check
            object.__setattr__(experts, "w1", mock_dtensor_with_ep)
            object.__setattr__(experts, "w2", mock_w2_with_ep)
            object.__setattr__(experts, "w3", mock_w3_with_ep)

            experts(x, num_tokens_per_expert)
            mock_padding_wrapper.assert_not_called()
            mock_run_experts.assert_called_once()


class TestGroupedExpertsIntegration:
    """Integration tests simulating MoE usage."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="permute_tokens uses torch.histc which doesn't support Long dtype on CPU",
    )
    def test_with_permutation(self):
        """Test GroupedExperts with token permutation from routing."""
        from areal.experimental.models.archon.moe.utils import (
            permute_tokens,
            unpermute_tokens,
        )

        device = torch.device("cuda")
        dim = 32
        hidden_dim = 64
        num_experts = 4
        num_tokens = 16
        top_k = 2

        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=False)
        experts.init_weights()
        experts = experts.to(device)

        # Original tokens
        tokens = torch.randn(num_tokens, dim, device=device)

        # Simulate router output
        indices = torch.randint(0, num_experts, (num_tokens, top_k), device=device)

        # Permute tokens by expert
        permuted, sorted_idx, num_per_expert = permute_tokens(
            tokens, indices, num_experts
        )

        # Expert computation
        expert_output = experts(permuted, num_per_expert)

        # Unpermute back
        unpermuted = unpermute_tokens(expert_output, sorted_idx, num_tokens, top_k)

        assert unpermuted.shape == (num_tokens * top_k, dim)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="permute_tokens uses torch.histc which doesn't support Long dtype on CPU",
    )
    def test_batch_sequence_input(self):
        """Test with batch x sequence shaped input."""
        device = torch.device("cuda")
        dim = 64
        hidden_dim = 128
        num_experts = 8
        batch_size = 2
        seq_len = 8
        top_k = 2

        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=False)
        experts.init_weights()
        experts = experts.to(device)

        # Flatten batch x sequence
        tokens = torch.randn(batch_size * seq_len, dim, device=device)

        # Simulate router: each token goes to top_k experts
        from areal.experimental.models.archon.moe.utils import permute_tokens

        indices = torch.randint(
            0, num_experts, (batch_size * seq_len, top_k), device=device
        )
        permuted, _, num_per_expert = permute_tokens(tokens, indices, num_experts)

        output = experts(permuted, num_per_expert)

        assert output.shape[0] == batch_size * seq_len * top_k
        assert output.shape[1] == dim
