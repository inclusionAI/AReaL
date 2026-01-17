"""Unit tests for Archon activation checkpointing.

Tests cover:
1. ActivationCheckpointConfig validation
2. apply_activation_checkpointing with all modes

Run tests:
    pytest areal/tests/experimental/archon/test_activation_checkpoint.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from areal.experimental.models.archon.activation_checkpoint import (
    ActivationCheckpointConfig,
    apply_activation_checkpointing,
)

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
            assert not isinstance(layer, CheckpointWrapper)

    def test_full_mode_wraps_all_layers(self):
        """Full mode should wrap all layers with CheckpointWrapper."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="full")
        apply_activation_checkpointing(model, config)

        for layer in model.layers.values():
            assert isinstance(layer, CheckpointWrapper)

    def test_selective_mode_wraps_every_n_layers(self):
        """Selective mode should wrap every Nth layer."""
        model = DummyModel(num_layers=4)
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
        apply_activation_checkpointing(model, config)

        # With ac_freq=2, the 2nd and 4th layers are checkpointed.
        # This corresponds to layers with index 1 and 3.
        assert not isinstance(model.layers["0"], CheckpointWrapper)
        assert isinstance(model.layers["1"], CheckpointWrapper)
        assert not isinstance(model.layers["2"], CheckpointWrapper)
        assert isinstance(model.layers["3"], CheckpointWrapper)

    def test_selective_mode_every_layer(self):
        """Selective mode with option '1' should wrap every layer."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="1")
        apply_activation_checkpointing(model, config)

        for layer in model.layers.values():
            assert isinstance(layer, CheckpointWrapper)

    def test_selective_op_mode_wraps_all_layers(self):
        """Selective op mode should wrap all layers."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="op")
        apply_activation_checkpointing(model, config)

        for layer in model.layers.values():
            assert isinstance(layer, CheckpointWrapper)

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
        assert not isinstance(model1.layers["0"], CheckpointWrapper)
        assert isinstance(model1.layers["1"], CheckpointWrapper)

        # model2: all layers wrapped
        for layer in model2.layers.values():
            assert isinstance(layer, CheckpointWrapper)
