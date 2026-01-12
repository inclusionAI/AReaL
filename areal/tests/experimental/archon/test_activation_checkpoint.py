"""Unit tests for Archon activation checkpointing.

Tests cover:
1. _wait_async_tensor helper function
2. _WaitAsyncWrapper module wrapper
3. ActivationCheckpointConfig validation
4. apply_activation_checkpointing with all modes

Run tests:
    pytest areal/tests/experimental/archon/test_activation_checkpoint.py -v
"""

import pytest
import torch
import torch.nn as nn

from areal.experimental.models.archon.activation_checkpoint import (
    ActivationCheckpointConfig,
    _wait_async_tensor,
    _WaitAsyncWrapper,
    apply_activation_checkpointing,
)

# =============================================================================
# _wait_async_tensor Tests
# =============================================================================


class TestWaitAsyncTensor:
    """Test _wait_async_tensor helper function."""

    def test_regular_tensor_passthrough(self):
        """Regular tensor should pass through unchanged."""
        x = torch.randn(2, 3)
        result = _wait_async_tensor(x)
        assert result is x

    def test_none_passthrough(self):
        """None should pass through unchanged."""
        result = _wait_async_tensor(None)
        assert result is None

    def test_non_tensor_passthrough(self):
        """Non-tensor objects should pass through unchanged."""
        x = [1, 2, 3]
        result = _wait_async_tensor(x)
        assert result is x

    def test_int_passthrough(self):
        """Integer should pass through unchanged."""
        result = _wait_async_tensor(42)
        assert result == 42

    def test_string_passthrough(self):
        """String should pass through unchanged."""
        result = _wait_async_tensor("test")
        assert result == "test"


# =============================================================================
# _WaitAsyncWrapper Tests
# =============================================================================


class TestWaitAsyncWrapper:
    """Test _WaitAsyncWrapper module."""

    def test_wrapper_forwards_to_module(self):
        """Wrapper should forward calls to inner module."""
        inner = nn.Linear(4, 2)
        wrapper = _WaitAsyncWrapper(inner)

        x = torch.randn(3, 4)
        expected = inner(x)
        result = wrapper(x)

        assert torch.allclose(result, expected)

    def test_wrapper_handles_kwargs(self):
        """Wrapper should handle keyword arguments."""

        class DummyModule(nn.Module):
            def forward(self, x, scale=1.0):
                return x * scale

        inner = DummyModule()
        wrapper = _WaitAsyncWrapper(inner)

        x = torch.randn(2, 2)
        result = wrapper(x, scale=2.0)
        expected = x * 2.0

        assert torch.allclose(result, expected)

    def test_wrapper_handles_multiple_args(self):
        """Wrapper should handle multiple positional arguments."""

        class MultiArgModule(nn.Module):
            def forward(self, a, b, c):
                return a + b + c

        inner = MultiArgModule()
        wrapper = _WaitAsyncWrapper(inner)

        a, b, c = torch.randn(2), torch.randn(2), torch.randn(2)
        result = wrapper(a, b, c)
        expected = a + b + c

        assert torch.allclose(result, expected)

    def test_wrapper_preserves_module_parameters(self):
        """Wrapper should preserve inner module's parameters."""
        inner = nn.Linear(4, 2)
        wrapper = _WaitAsyncWrapper(inner)

        # Parameters should be accessible
        params = list(wrapper.parameters())
        inner_params = list(inner.parameters())
        assert len(params) == len(inner_params)

        for p, ip in zip(params, inner_params):
            assert p is ip

    def test_wrapper_preserves_training_mode(self):
        """Wrapper should forward training mode to inner module."""
        inner = nn.Linear(4, 2)
        wrapper = _WaitAsyncWrapper(inner)

        wrapper.train()
        assert inner.training is True

        wrapper.eval()
        assert inner.training is False


# =============================================================================
# ActivationCheckpointConfig Tests
# =============================================================================


class TestActivationCheckpointConfig:
    """Test ActivationCheckpointConfig validation."""

    def test_default_config(self):
        """Default config should have mode='none'."""
        config = ActivationCheckpointConfig()
        assert config.mode == "none"
        assert config.selective_ac_option == "1"
        assert config.preserve_rng_state is False

    def test_full_mode(self):
        """Full mode should be valid."""
        config = ActivationCheckpointConfig(mode="full")
        assert config.mode == "full"

    def test_selective_mode_with_integer(self):
        """Selective mode with integer option should be valid."""
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
        assert config.mode == "selective"
        assert config.selective_ac_option == "2"

    def test_selective_mode_with_op(self):
        """Selective mode with 'op' option should be valid."""
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="op")
        assert config.selective_ac_option == "op"

    def test_preserve_rng_state(self):
        """preserve_rng_state should be configurable."""
        config = ActivationCheckpointConfig(mode="full", preserve_rng_state=True)
        assert config.preserve_rng_state is True

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid AC mode"):
            ActivationCheckpointConfig(mode="invalid")

    def test_invalid_selective_option_raises(self):
        """Invalid selective_ac_option should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid selective_ac_option"):
            ActivationCheckpointConfig(mode="selective", selective_ac_option="bad")

    def test_zero_selective_option_raises(self):
        """selective_ac_option='0' should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid selective_ac_option"):
            ActivationCheckpointConfig(mode="selective", selective_ac_option="0")

    def test_selective_option_not_validated_for_non_selective_mode(self):
        """selective_ac_option should not be validated for non-selective modes."""
        # Should not raise even with invalid selective_ac_option
        config = ActivationCheckpointConfig(mode="full", selective_ac_option="bad")
        assert config.mode == "full"


# =============================================================================
# apply_activation_checkpointing Tests
# =============================================================================


class DummyBlock(nn.Module):
    """Dummy transformer block for testing."""

    def __init__(self, dim=8):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class DummyModel(nn.Module):
    """Dummy model with layers attribute for testing."""

    def __init__(self, num_layers=4, dim=8):
        super().__init__()
        self.layers = nn.ModuleDict(
            {str(i): DummyBlock(dim) for i in range(num_layers)}
        )

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x


class TestApplyActivationCheckpointing:
    """Test apply_activation_checkpointing function."""

    def test_none_mode_no_wrapping(self):
        """None mode should not wrap any layers."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="none")
        apply_activation_checkpointing(model, config)

        for layer in model.layers.values():
            assert isinstance(layer, DummyBlock)
            assert not isinstance(layer, _WaitAsyncWrapper)

    def test_full_mode_wraps_all_layers(self):
        """Full mode should wrap all layers with _WaitAsyncWrapper."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="full")
        apply_activation_checkpointing(model, config)

        for layer in model.layers.values():
            assert isinstance(layer, _WaitAsyncWrapper)

    def test_selective_mode_wraps_every_n_layers(self):
        """Selective mode should wrap every Nth layer."""
        model = DummyModel(num_layers=4)
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
        apply_activation_checkpointing(model, config)

        # With ac_freq=2, the 2nd and 4th layers are checkpointed.
        # This corresponds to layers with index 1 and 3.
        assert not isinstance(model.layers["0"], _WaitAsyncWrapper)
        assert isinstance(model.layers["1"], _WaitAsyncWrapper)
        assert not isinstance(model.layers["2"], _WaitAsyncWrapper)
        assert isinstance(model.layers["3"], _WaitAsyncWrapper)

    def test_selective_mode_every_layer(self):
        """Selective mode with option '1' should wrap every layer."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="1")
        apply_activation_checkpointing(model, config)

        for layer in model.layers.values():
            assert isinstance(layer, _WaitAsyncWrapper)

    def test_selective_op_mode_wraps_all_layers(self):
        """Selective op mode should wrap all layers."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="op")
        apply_activation_checkpointing(model, config)

        for layer in model.layers.values():
            assert isinstance(layer, _WaitAsyncWrapper)

    def test_model_without_layers_raises(self):
        """Model without layers attribute should raise ValueError."""
        model = nn.Linear(4, 2)
        config = ActivationCheckpointConfig(mode="full")
        with pytest.raises(ValueError, match="must have a 'layers' attribute"):
            apply_activation_checkpointing(model, config)

    def test_wrapped_model_forward_works(self):
        """Wrapped model should still produce correct forward output."""
        model = DummyModel(num_layers=3, dim=4)

        # Get reference output before wrapping
        x = torch.randn(2, 4)
        with torch.no_grad():
            ref_output = model(x.clone())

        # Apply AC and verify output matches
        config = ActivationCheckpointConfig(mode="full")
        apply_activation_checkpointing(model, config)

        with torch.no_grad():
            wrapped_output = model(x.clone())

        assert torch.allclose(ref_output, wrapped_output)

    def test_wrapped_model_backward_works(self):
        """Wrapped model should support backward pass."""
        model = DummyModel(num_layers=3, dim=4)
        config = ActivationCheckpointConfig(mode="full")
        apply_activation_checkpointing(model, config)

        x = torch.randn(2, 4, requires_grad=True)
        output = model(x)
        loss = output.sum()

        # Should not raise
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None

    def test_multiple_models_independent(self):
        """Multiple models should be wrapped independently."""
        model1 = DummyModel(num_layers=4)
        model2 = DummyModel(num_layers=4)

        config1 = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
        config2 = ActivationCheckpointConfig(mode="full")

        apply_activation_checkpointing(model1, config1)
        apply_activation_checkpointing(model2, config2)

        # model1: selective, every 2nd layer
        assert not isinstance(model1.layers["0"], _WaitAsyncWrapper)
        assert isinstance(model1.layers["1"], _WaitAsyncWrapper)

        # model2: all layers wrapped
        for layer in model2.layers.values():
            assert isinstance(layer, _WaitAsyncWrapper)
