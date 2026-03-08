"""Tests for MoE router FP32 dtype feature.

Tests cover:
- RouterGatingLinearFunction autograd behavior (forward dtype, backward grads, saved tensors)
- TokenChoiceTopKRouter with FP32 gate computation (default dtype, forward shapes, indices)
- torch.compile compatibility (fullgraph, no graph breaks)
- Config validation for router_dtype (MoEArgs, ArchonEngineConfig)

Run tests:
    pytest tests/experimental/archon/test_router_fp32.py -v

Note: Requires triton (skipped automatically if not installed).
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton", reason="triton required for archon MoE imports")

from areal.api.cli_args import ArchonEngineConfig
from areal.experimental.models.archon.moe.args import MoEArgs
from areal.experimental.models.archon.moe.router import (
    RouterGatingLinearFunction,
    TokenChoiceTopKRouter,
    router_gating_linear,
)

CUDA_AVAILABLE = torch.cuda.is_available()


class TestRouterGatingLinearFunction:
    """Tests for RouterGatingLinearFunction autograd behavior."""

    def test_forward_output_dtype_is_router_dtype(self):
        """Test forward output is cast to router_dtype (float32)."""
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        out = RouterGatingLinearFunction.apply(x, w, torch.float32)
        assert out.dtype == torch.float32

    def test_forward_numerical_parity_vs_flinear(self):
        """Test result matches F.linear computed in float32."""
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        out = RouterGatingLinearFunction.apply(x, w, torch.float32)
        expected = F.linear(x.to(torch.float32), w.to(torch.float32))
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_backward_grad_input_dtype_matches_input(self):
        """Test gradient for input has same dtype as input (BF16)."""
        x = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(8, 16, dtype=torch.bfloat16, requires_grad=True)
        out = RouterGatingLinearFunction.apply(x, w, torch.float32)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.dtype == torch.bfloat16

    def test_backward_grad_weight_dtype_matches_weight(self):
        """Test gradient for weight has same dtype as weight (BF16)."""
        x = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(8, 16, dtype=torch.bfloat16, requires_grad=True)
        out = RouterGatingLinearFunction.apply(x, w, torch.float32)
        out.sum().backward()
        assert w.grad is not None
        assert w.grad.dtype == torch.bfloat16

    def test_backward_grad_router_dtype_returns_none(self):
        """Test third gradient (for router_dtype arg) is None."""

        class BackwardInspector(RouterGatingLinearFunction):
            backward_result = None

            @staticmethod
            def backward(ctx, *grad_outputs):
                result = RouterGatingLinearFunction.backward(ctx, *grad_outputs)
                BackwardInspector.backward_result = result
                return result

        x = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(8, 16, dtype=torch.bfloat16, requires_grad=True)
        out = BackwardInspector.apply(x, w, torch.float32)
        out.sum().backward()

        assert BackwardInspector.backward_result is not None
        assert len(BackwardInspector.backward_result) == 3
        assert BackwardInspector.backward_result[2] is None

    def test_saved_tensors_are_original_dtype(self):
        """Test saved tensors are in original dtype (BF16) for memory efficiency."""

        class BackwardSavedInspector(RouterGatingLinearFunction):
            saved_during_backward = []

            @staticmethod
            def backward(ctx, *grad_outputs):
                BackwardSavedInspector.saved_during_backward = list(ctx.saved_tensors)
                return RouterGatingLinearFunction.backward(ctx, *grad_outputs)

        x = torch.randn(4, 16, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(8, 16, dtype=torch.bfloat16, requires_grad=True)
        out = BackwardSavedInspector.apply(x, w, torch.float32)
        out.sum().backward()

        assert len(BackwardSavedInspector.saved_during_backward) == 2
        assert BackwardSavedInspector.saved_during_backward[0].dtype == torch.bfloat16
        assert BackwardSavedInspector.saved_during_backward[1].dtype == torch.bfloat16


class TestTokenChoiceTopKRouterFP32:
    """Tests for TokenChoiceTopKRouter FP32 gate computation."""

    def test_default_router_dtype_is_none(self):
        """Test default router_dtype is None (no override, uses model dtype)."""
        router = TokenChoiceTopKRouter(dim=16, num_experts=4, top_k=1)
        assert router.router_dtype is None

    def test_explicit_router_dtype_stored(self):
        """Test explicitly passed router_dtype is stored correctly."""
        router = TokenChoiceTopKRouter(
            dim=16, num_experts=4, top_k=1, router_dtype=torch.float32
        )
        assert router.router_dtype == torch.float32

    def test_none_router_dtype_stays_none(self):
        """Test None router_dtype stays None (no override)."""
        router = TokenChoiceTopKRouter(
            dim=16, num_experts=4, top_k=1, router_dtype=None
        )
        assert router.router_dtype is None

    @pytest.mark.skipif(
        not CUDA_AVAILABLE,
        reason="torch.histc on CPU does not support Long dtype",
    )
    def test_forward_scores_dtype_is_float32(self):
        """Test top_scores returned from forward are float32."""
        router = TokenChoiceTopKRouter(dim=16, num_experts=4, top_k=1).cuda()
        x = torch.randn(8, 16, device="cuda")
        top_scores, _, _ = router(x)
        assert top_scores.dtype == torch.float32

    @pytest.mark.skipif(
        not CUDA_AVAILABLE,
        reason="torch.histc on CPU does not support Long dtype",
    )
    def test_forward_output_shapes(self):
        """Test output tensor shapes from forward."""
        num_tokens, dim, num_experts, top_k = 8, 16, 4, 2
        router = TokenChoiceTopKRouter(
            dim=dim, num_experts=num_experts, top_k=top_k
        ).cuda()
        x = torch.randn(num_tokens, dim, device="cuda")
        top_scores, selected_experts_indices, num_tokens_per_expert = router(x)

        assert top_scores.shape == (num_tokens, top_k)
        assert selected_experts_indices.shape == (num_tokens, top_k)
        assert num_tokens_per_expert.shape == (num_experts,)

    @pytest.mark.skipif(
        not CUDA_AVAILABLE,
        reason="torch.histc on CPU does not support Long dtype",
    )
    def test_forward_selected_indices_in_range(self):
        """Test selected expert indices are in [0, num_experts)."""
        num_experts = 4
        router = TokenChoiceTopKRouter(dim=16, num_experts=num_experts, top_k=2).cuda()
        x = torch.randn(8, 16, device="cuda")
        _, selected_experts_indices, _ = router(x)

        assert (selected_experts_indices >= 0).all()
        assert (selected_experts_indices < num_experts).all()

    @pytest.mark.skipif(
        not CUDA_AVAILABLE,
        reason="torch.histc on CPU does not support Long dtype",
    )
    def test_backward_through_gate(self):
        """Test backward pass through gate produces valid gradients."""
        router = TokenChoiceTopKRouter(dim=16, num_experts=4, top_k=1).cuda()
        x = torch.randn(8, 16, device="cuda")
        top_scores, _, _ = router(x)
        top_scores.sum().backward()

        assert router.gate.weight.grad is not None
        assert router.gate.weight.grad.shape == router.gate.weight.shape


class TestRouterFP32CompileCompat:
    """Tests for torch.compile compatibility of router FP32 logic."""

    def test_compile_fullgraph_no_break(self):
        """Test router_gating_linear compiles with fullgraph=True, no graph break."""
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        compiled_fn = torch.compile(
            router_gating_linear, fullgraph=True, backend="eager"
        )
        out = compiled_fn(x, w, torch.float32)
        assert out.dtype == torch.float32

    def test_compile_function_apply(self):
        """Test RouterGatingLinearFunction.apply compiles with fullgraph=True."""
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        compiled_fn = torch.compile(
            lambda inp, wt: RouterGatingLinearFunction.apply(inp, wt, torch.float32),
            fullgraph=True,
            backend="eager",
        )
        out = compiled_fn(x, w)
        assert out.dtype == torch.float32


class TestConfigThreading:
    """Tests for router_dtype config fields in MoEArgs and ArchonEngineConfig."""

    def test_moe_args_router_dtype_default_is_none(self):
        """Test MoEArgs.router_dtype defaults to None."""
        args = MoEArgs()
        assert args.router_dtype is None

    def test_moe_args_router_dtype_custom(self):
        """Test MoEArgs.router_dtype accepts custom dtype."""
        args = MoEArgs(router_dtype=torch.float32)
        assert args.router_dtype == torch.float32

    def test_archon_engine_config_moe_router_dtype_default(self):
        """Test ArchonEngineConfig.moe_router_dtype defaults to 'fp32'."""
        config = ArchonEngineConfig()
        assert config.moe_router_dtype == "fp32"

    def test_archon_engine_config_moe_router_dtype_none(self):
        """Test ArchonEngineConfig.moe_router_dtype can be set to None."""
        config = ArchonEngineConfig(moe_router_dtype=None)
        assert config.moe_router_dtype is None

    def test_archon_engine_config_moe_router_dtype_invalid_rejected(self):
        """Test ArchonEngineConfig rejects invalid moe_router_dtype values."""
        with pytest.raises(ValueError, match="moe_router_dtype"):
            ArchonEngineConfig(moe_router_dtype="fp64")
