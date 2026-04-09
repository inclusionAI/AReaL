"""Unit tests for LoRALinear module and adapter utilities."""

import pytest
import torch
import torch.nn as nn

from areal.experimental.models.archon.lora.adapter import (
    AdapterModule,
    disable_adapter,
    enable_adapter,
    get_adapter_params,
    get_adapter_state_dict,
    set_trainable_params,
)
from areal.experimental.models.archon.lora.lora_linear import LoRALinear

# Try to import PEFT's LoRA Linear module for comparison tests
try:
    from peft.tuners.lora import Linear as PEFTLoRALinear

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class TestLoRALinear:
    """Test LoRALinear module functionality."""

    def test_initialization(self):
        """Test that LoRALinear is properly initialized with zero LoRA contribution."""
        torch.manual_seed(42)

        lora_linear = LoRALinear(in_dim=64, out_dim=32, rank=8, alpha=16.0)

        # lora_b initialized to zeros
        assert torch.allclose(
            lora_linear.lora_b.weight, torch.zeros_like(lora_linear.lora_b.weight)
        ), "lora_b should be initialized to zeros"

        # Initial forward matches base-only output
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            out_with_lora = lora_linear(x)
            base_out = torch.nn.functional.linear(
                x, lora_linear.weight, lora_linear.bias
            )
            assert torch.allclose(out_with_lora, base_out, atol=1e-6), (
                "Initial output should match base output (zero LoRA contribution)"
            )

    def test_forward_pass(self):
        """Test that forward pass correctly implements LoRA computation."""
        torch.manual_seed(42)

        in_dim, out_dim, rank = 64, 32, 8
        alpha = 16.0
        lora_linear = LoRALinear(
            in_dim=in_dim, out_dim=out_dim, rank=rank, alpha=alpha, dropout=0.0
        )

        with torch.no_grad():
            lora_linear.weight.fill_(0.1)
            lora_linear.lora_a.weight.fill_(0.2)
            lora_linear.lora_b.weight.fill_(0.3)

        x = torch.randn(2, 10, in_dim)
        output = lora_linear(x)

        # Manual calculation
        base_out = torch.nn.functional.linear(x, lora_linear.weight, lora_linear.bias)
        lora_a_out = torch.nn.functional.linear(x, lora_linear.lora_a.weight)
        lora_b_out = torch.nn.functional.linear(lora_a_out, lora_linear.lora_b.weight)
        scaling = alpha / rank
        expected_output = base_out + scaling * lora_b_out

        assert torch.allclose(output, expected_output, atol=1e-5), (
            "Forward pass computation mismatch"
        )

    def test_gradient_flow(self):
        """Test that only LoRA parameters receive gradients."""
        torch.manual_seed(42)

        lora_linear = LoRALinear(in_dim=64, out_dim=32, rank=8, alpha=16.0)

        # Freeze base weights
        lora_linear.weight.requires_grad_(False)
        if lora_linear.bias is not None:
            lora_linear.bias.requires_grad_(False)

        # Set non-zero lora_b so gradients flow to lora_a
        with torch.no_grad():
            lora_linear.lora_b.weight.fill_(0.1)

        x = torch.randn(2, 10, 64, requires_grad=True)
        output = lora_linear(x)
        loss = output.sum()
        loss.backward()

        assert lora_linear.weight.grad is None, "Base weight should not have gradients"
        if lora_linear.bias is not None:
            assert lora_linear.bias.grad is None, "Bias should not have gradients"

        assert lora_linear.lora_a.weight.grad is not None, (
            "lora_a should have gradients"
        )
        assert lora_linear.lora_b.weight.grad is not None, (
            "lora_b should have gradients"
        )
        assert not torch.allclose(
            lora_linear.lora_a.weight.grad,
            torch.zeros_like(lora_linear.lora_a.weight.grad),
        ), "lora_a gradients should be non-zero"

    def test_from_linear(self):
        """Test conversion from nn.Linear to LoRALinear."""
        torch.manual_seed(42)

        linear = nn.Linear(64, 32, bias=True)
        with torch.no_grad():
            linear.weight.fill_(0.5)
            linear.bias.fill_(0.1)

        lora_linear = LoRALinear.from_linear(linear, rank=8, alpha=16.0)

        assert torch.allclose(lora_linear.weight, linear.weight), (
            "Base weight should match original linear weight"
        )
        assert torch.allclose(lora_linear.bias, linear.bias), (
            "Bias should match original bias"
        )
        assert lora_linear.in_dim == linear.in_features
        assert lora_linear.out_dim == linear.out_features

        # Output with disabled LoRA matches original
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            lora_linear.disabled = True
            lora_out = lora_linear(x)
            linear_out = linear(x)

            assert torch.allclose(lora_out, linear_out, atol=1e-6), (
                "Output with disabled LoRA should match original linear"
            )

    def test_adapter_params_protocol(self):
        """Test AdapterModule protocol implementation."""
        lora_linear = LoRALinear(in_dim=64, out_dim=32, rank=8, alpha=16.0)

        assert isinstance(lora_linear, AdapterModule), (
            "LoRALinear should implement AdapterModule"
        )

        adapter_param_names = lora_linear.adapter_params()
        assert adapter_param_names == [
            "lora_a.weight",
            "lora_b.weight",
        ], "adapter_params() should return LoRA parameter names"

    def test_disabled_flag(self):
        """Test that disabled flag correctly disables LoRA contribution."""
        torch.manual_seed(42)

        lora_linear = LoRALinear(
            in_dim=64, out_dim=32, rank=8, alpha=16.0, dropout=0.0
        )

        with torch.no_grad():
            lora_linear.lora_b.weight.fill_(0.1)

        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            lora_linear.disabled = False
            out_enabled = lora_linear(x)

            lora_linear.disabled = True
            out_disabled = lora_linear(x)

            base_out = torch.nn.functional.linear(
                x, lora_linear.weight, lora_linear.bias
            )

        assert torch.allclose(out_disabled, base_out, atol=1e-6), (
            "Disabled LoRA should match base output"
        )
        assert not torch.allclose(out_enabled, base_out, atol=1e-5), (
            "Enabled LoRA should differ from base output"
        )

    def test_repr(self):
        """Test __repr__ output."""
        lora_linear = LoRALinear(
            in_dim=64, out_dim=32, rank=8, alpha=16.0, dropout=0.1, use_bias=True
        )
        repr_str = repr(lora_linear)
        assert "LoRALinear" in repr_str
        assert "in_dim=64" in repr_str
        assert "out_dim=32" in repr_str
        assert "rank=8" in repr_str


class TestAdapterUtilities:
    """Test adapter utility functions."""

    def test_get_adapter_params(self):
        """Test extraction of adapter parameters from model."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora1 = LoRALinear(64, 32, rank=8, alpha=16.0)
                self.lora2 = LoRALinear(32, 16, rank=8, alpha=16.0)
                self.linear = nn.Linear(16, 8)

            def forward(self, x):
                return self.linear(self.lora2(self.lora1(x)))

        model = SimpleModel()
        adapter_params = get_adapter_params(model)

        expected_keys = {
            "lora1.lora_a.weight",
            "lora1.lora_b.weight",
            "lora2.lora_a.weight",
            "lora2.lora_b.weight",
        }
        assert set(adapter_params.keys()) == expected_keys, (
            "Should extract only LoRA parameters"
        )

        for param in adapter_params.values():
            assert isinstance(param, nn.Parameter), "Should return Parameter objects"

    def test_set_trainable_params(self):
        """Test freezing/unfreezing parameters."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora = LoRALinear(64, 32, rank=8, alpha=16.0)
                self.linear = nn.Linear(32, 16)

        model = SimpleModel()
        adapter_param_names = set(get_adapter_params(model).keys())
        set_trainable_params(model, adapter_param_names)

        for name, param in model.named_parameters():
            if name in adapter_param_names:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_get_adapter_state_dict(self):
        """Test filtering state dict to adapter parameters only."""
        state_dict = {
            "model.weight": torch.randn(10, 10),
            "model.bias": torch.randn(10),
            "model.lora_a.weight": torch.randn(8, 10),
            "model.lora_b.weight": torch.randn(10, 8),
            "other.lora_a.weight": torch.randn(8, 10),
            "other.lora_b.weight": torch.randn(10, 8),
        }

        adapter_state_dict = get_adapter_state_dict(state_dict)

        expected_keys = {
            "model.lora_a.weight",
            "model.lora_b.weight",
            "other.lora_a.weight",
            "other.lora_b.weight",
        }
        assert set(adapter_state_dict.keys()) == expected_keys, (
            "Should filter to LoRA params only"
        )

    def test_disable_enable_adapter(self):
        """Test disabling/enabling adapters in model."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora1 = LoRALinear(64, 32, rank=8, alpha=16.0)
                self.lora2 = LoRALinear(32, 16, rank=8, alpha=16.0)

        model = SimpleModel()

        assert not model.lora1.disabled
        assert not model.lora2.disabled

        disable_adapter(model)
        assert model.lora1.disabled
        assert model.lora2.disabled

        enable_adapter(model)
        assert not model.lora1.disabled
        assert not model.lora2.disabled


class TestLoRALinearWithBias:
    """Test LoRALinear with bias enabled."""

    def test_bias_initialization(self):
        """Test bias is properly initialized."""
        lora_linear = LoRALinear(
            in_dim=64, out_dim=32, rank=8, alpha=16.0, use_bias=True
        )

        assert lora_linear.bias is not None, "Bias should exist"
        assert torch.allclose(lora_linear.bias, torch.zeros_like(lora_linear.bias)), (
            "Bias should be initialized to zeros"
        )

    def test_bias_forward(self):
        """Test forward pass with bias."""
        torch.manual_seed(42)

        lora_linear = LoRALinear(
            in_dim=64, out_dim=32, rank=8, alpha=16.0, use_bias=True, dropout=0.0
        )

        with torch.no_grad():
            lora_linear.weight.fill_(0.1)
            lora_linear.bias.fill_(0.5)
            lora_linear.lora_a.weight.fill_(0.2)
            lora_linear.lora_b.weight.fill_(0.3)

        x = torch.randn(2, 10, 64)
        output = lora_linear(x)

        base_out = torch.nn.functional.linear(x, lora_linear.weight, lora_linear.bias)
        lora_a_out = torch.nn.functional.linear(x, lora_linear.lora_a.weight)
        lora_b_out = torch.nn.functional.linear(lora_a_out, lora_linear.lora_b.weight)
        scaling = lora_linear.alpha / lora_linear.rank
        expected_output = base_out + scaling * lora_b_out

        assert torch.allclose(output, expected_output, atol=1e-5), (
            "Forward with bias should match manual calc"
        )


class TestLoRALinearDropout:
    """Test LoRALinear with dropout."""

    def test_dropout_training_mode(self):
        """Test that dropout is active in training mode."""
        torch.manual_seed(42)

        lora_linear = LoRALinear(
            in_dim=64, out_dim=32, rank=8, alpha=16.0, dropout=0.5
        )
        lora_linear.train()

        with torch.no_grad():
            lora_linear.lora_b.weight.fill_(0.1)
            lora_linear.lora_a.weight.fill_(0.1)

        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            out1 = lora_linear(x)
            out2 = lora_linear(x)

        assert not torch.allclose(out1, out2, atol=1e-5), (
            "Dropout should cause different outputs"
        )

    def test_dropout_eval_mode(self):
        """Test that dropout is disabled in eval mode."""
        torch.manual_seed(42)

        lora_linear = LoRALinear(
            in_dim=64, out_dim=32, rank=8, alpha=16.0, dropout=0.5
        )
        lora_linear.eval()

        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            out1 = lora_linear(x)
            out2 = lora_linear(x)

        assert torch.allclose(out1, out2, atol=1e-6), (
            "Eval mode should have deterministic output"
        )


@pytest.mark.skipif(not PEFT_AVAILABLE, reason="PEFT not installed")
class TestPEFTCompatibility:
    """Test compatibility with PEFT library implementation."""

    def test_forward_pass_vs_peft(self):
        """Compare forward pass output with PEFT's LoRA Linear module."""
        torch.manual_seed(42)

        in_dim, out_dim, rank = 128, 128, 16
        alpha = 32.0
        adapter_name = "default"

        base_linear = nn.Linear(in_dim, out_dim, bias=False)

        our_lora = LoRALinear.from_linear(
            base_linear, rank=rank, alpha=alpha, dropout=0.0
        )

        peft_lora = PEFTLoRALinear(
            base_layer=nn.Linear(in_dim, out_dim, bias=False),
            adapter_name=adapter_name,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            init_lora_weights=True,
        )
        peft_lora.base_layer.weight.data.copy_(base_linear.weight.data)

        peft_lora_a = peft_lora.lora_A[adapter_name].weight
        peft_lora_b = peft_lora.lora_B[adapter_name].weight

        with torch.no_grad():
            our_lora.lora_a.weight.copy_(peft_lora_a)
            our_lora.lora_b.weight.copy_(peft_lora_b)

        x = torch.randn(2, 10, in_dim)

        with torch.no_grad():
            our_output = our_lora(x)
            peft_output = peft_lora(x)

        max_diff = (our_output - peft_output).abs().max().item()
        assert torch.allclose(our_output, peft_output, atol=1e-5), (
            f"Output mismatch vs PEFT: max diff = {max_diff}"
        )

    def test_gradient_flow_vs_peft(self):
        """Compare gradient flow with PEFT's implementation."""
        torch.manual_seed(42)

        in_dim, out_dim, rank = 64, 64, 8
        alpha = 16.0
        adapter_name = "default"

        base_linear = nn.Linear(in_dim, out_dim, bias=False)

        our_lora = LoRALinear.from_linear(
            base_linear, rank=rank, alpha=alpha, dropout=0.0
        )
        our_lora.weight.requires_grad_(False)

        peft_lora = PEFTLoRALinear(
            base_layer=nn.Linear(in_dim, out_dim, bias=False),
            adapter_name=adapter_name,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            init_lora_weights=True,
        )
        peft_lora.base_layer.weight.data.copy_(base_linear.weight.data)
        peft_lora.base_layer.weight.requires_grad_(False)

        peft_lora_a = peft_lora.lora_A[adapter_name].weight
        peft_lora_b = peft_lora.lora_B[adapter_name].weight

        with torch.no_grad():
            our_lora.lora_a.weight.copy_(peft_lora_a)
            our_lora.lora_b.weight.copy_(peft_lora_b)
            our_lora.lora_b.weight.fill_(0.1)
            peft_lora_b.fill_(0.1)

        x = torch.randn(2, 10, in_dim)

        our_output = our_lora(x)
        our_output.sum().backward()

        peft_output = peft_lora(x)
        peft_output.sum().backward()

        assert torch.allclose(
            our_lora.lora_a.weight.grad, peft_lora_a.grad, atol=1e-4
        ), "lora_a gradient mismatch vs PEFT"
        assert torch.allclose(
            our_lora.lora_b.weight.grad, peft_lora_b.grad, atol=1e-4
        ), "lora_b gradient mismatch vs PEFT"

    def test_scaling_factor_vs_peft(self):
        """Verify scaling factor matches PEFT's implementation."""
        rank = 16
        alpha = 32.0
        adapter_name = "default"

        our_lora = LoRALinear(in_dim=64, out_dim=64, rank=rank, alpha=alpha)

        peft_lora = PEFTLoRALinear(
            base_layer=nn.Linear(64, 64, bias=False),
            adapter_name=adapter_name,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            init_lora_weights=True,
        )

        expected_scaling = alpha / rank
        peft_scaling = peft_lora.scaling[adapter_name]

        assert our_lora.scaling == expected_scaling
        assert our_lora.scaling == peft_scaling

    def test_initialization_vs_peft(self):
        """Compare initialization strategy with PEFT."""
        torch.manual_seed(42)

        our_lora = LoRALinear(in_dim=64, out_dim=64, rank=16, alpha=32.0)

        # lora_b zeros, lora_a non-zero
        assert torch.allclose(
            our_lora.lora_b.weight, torch.zeros_like(our_lora.lora_b.weight)
        ), "lora_b should be zeros (PEFT convention)"

        assert not torch.allclose(
            our_lora.lora_a.weight, torch.zeros_like(our_lora.lora_a.weight)
        ), "lora_a should be non-zero (kaiming_uniform)"

        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            output = our_lora(x)
            base_output = torch.nn.functional.linear(x, our_lora.weight, our_lora.bias)
            assert torch.allclose(output, base_output, atol=1e-6), (
                "Initial output should match base (PEFT convention)"
            )
