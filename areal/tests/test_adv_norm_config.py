from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch

from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization

# =============================================================================
# NormConfig Tests
# =============================================================================


def test_adv_norm_config_inheritance():
    """Test that NormConfig inherits all properties from NormConfig."""
    adv_config = NormConfig()

    # Verify that NormConfig has expected attributes
    assert hasattr(
        adv_config, "mean_level"
    ), "NormConfig should have mean_level attribute"
    assert hasattr(
        adv_config, "std_level"
    ), "NormConfig should have std_level attribute"
    assert hasattr(
        adv_config, "group_size"
    ), "NormConfig should have group_size attribute"

    # Verify default values
    assert adv_config.mean_level == "batch", "Default mean_level should be 'batch'"
    assert adv_config.std_level == "batch", "Default std_level should be 'batch'"
    assert adv_config.group_size == 1, "Default group_size should be 1"


def test_adv_norm_config_custom_values():
    """Test NormConfig with custom values."""
    adv_config = NormConfig(mean_level=None, std_level="batch", group_size=4)

    assert adv_config.mean_level == None
    assert adv_config.std_level == "batch"
    assert adv_config.group_size == 4


def test_adv_norm_config_asdict():
    """Test conversion of NormConfig to dictionary."""
    adv_config = NormConfig(mean_level="batch", std_level="group", group_size=32)

    config_dict = asdict(adv_config)

    assert config_dict["mean_level"] == "batch"
    assert config_dict["std_level"] == "group"
    assert config_dict["group_size"] == 32


@pytest.mark.parametrize("mean_level", ["batch", "group", None])
@pytest.mark.parametrize("std_level", ["batch", "group", None])
@pytest.mark.parametrize("group_size", [1, 8, 32, 128])
def test_adv_norm_config_parameterized(mean_level, std_level, group_size):
    """Parameterized test for NormConfig with various combinations."""
    adv_config = NormConfig(
        mean_level=mean_level, std_level=std_level, group_size=group_size
    )

    assert adv_config.mean_level == mean_level
    assert adv_config.std_level == std_level
    assert adv_config.group_size == group_size


def test_adv_norm_config_equality():
    """Test equality comparison between NormConfig instances."""
    adv_config1 = NormConfig(mean_level="batch", std_level="batch", group_size=1)
    adv_config2 = NormConfig(mean_level="batch", std_level="batch", group_size=1)
    adv_config3 = NormConfig(mean_level="group", std_level="batch", group_size=1)

    assert adv_config1 == adv_config2
    assert adv_config1 != adv_config3


def test_adv_norm_initialization():
    """Test Normalization initialization with various configurations."""
    # Test with batch normalization
    config = NormConfig(mean_level="batch", std_level="batch", group_size=1)
    adv_norm = Normalization(config)
    assert adv_norm.mean_level == "batch"
    assert adv_norm.std_level == "batch"
    assert adv_norm.group_size == 1

    # Test with group normalization
    config = NormConfig(mean_level="group", std_level="group", group_size=8)
    adv_norm = Normalization(config)
    assert adv_norm.mean_level == "group"
    assert adv_norm.std_level == "group"
    assert adv_norm.group_size == 8

    # Test with mixed normalization
    config = NormConfig(mean_level="batch", std_level="group", group_size=16)
    adv_norm = Normalization(config)
    assert adv_norm.mean_level == "batch"
    assert adv_norm.std_level == "group"
    assert adv_norm.group_size == 16

    # Test with no normalization
    config = NormConfig(mean_level=None, std_level=None, group_size=1)
    adv_norm = Normalization(config)
    assert adv_norm.mean_level == None
    assert adv_norm.std_level == None
    assert adv_norm.group_size == 1


def test_adv_norm_initialization_validation():
    """Test Normalization initialization validation."""
    # Test invalid mean_level
    with pytest.raises(ValueError, match="mean_level must be 'batch', 'group' or None"):
        config = NormConfig(mean_level="invalid", std_level="batch", group_size=1)
        Normalization(config)

    # Test invalid std_level
    with pytest.raises(ValueError, match="std_level must be 'batch', 'group', or None"):
        config = NormConfig(mean_level="batch", std_level="invalid", group_size=1)
        Normalization(config)

    # Test missing group_size for group normalization
    with pytest.raises(
        ValueError, match="group_size must be provided if using group normalization"
    ):
        config = NormConfig(mean_level="group", std_level="batch", group_size=None)
        Normalization(config)


def test_adv_norm_batch_normalization():
    """Test batch normalization functionality."""
    config = NormConfig(mean_level="batch", std_level="batch", group_size=1)
    adv_norm = Normalization(config)

    # Create test data
    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)

    # Apply normalization
    normalized = adv_norm(advantages, loss_mask)

    # Check that normalization was applied
    assert normalized.shape == advantages.shape
    assert not torch.allclose(
        normalized, advantages
    )  # Should be different after normalization


def test_adv_norm_group_normalization():
    """Test group normalization functionality."""
    config = NormConfig(mean_level="group", std_level="group", group_size=2)
    adv_norm = Normalization(config)

    # Create test data with 4 samples (2 groups of 2)
    advantages = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        dtype=torch.float32,
    )

    loss_mask = torch.ones_like(advantages)

    # Apply normalization
    normalized = adv_norm(advantages, loss_mask)

    # Check that normalization was applied
    assert normalized.shape == advantages.shape
    assert not torch.allclose(normalized, advantages)


def test_adv_norm_mixed_normalization():
    """Test mixed normalization (different mean and std levels)."""
    config = NormConfig(mean_level="batch", std_level="group", group_size=2)
    adv_norm = Normalization(config)

    # Create test data
    advantages = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        dtype=torch.float32,
    )

    loss_mask = torch.ones_like(advantages)

    # Apply normalization
    normalized = adv_norm(advantages, loss_mask)

    # Check that normalization was applied
    assert normalized.shape == advantages.shape
    assert not torch.allclose(normalized, advantages)


def test_adv_norm_no_normalization():
    """Test no normalization case."""
    config = NormConfig(mean_level=None, std_level=None, group_size=1)
    adv_norm = Normalization(config)

    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    loss_mask = torch.ones_like(advantages)

    # Apply "normalization" - should return original values
    normalized = adv_norm(advantages, loss_mask)

    # Should be identical to input
    assert torch.allclose(normalized, advantages)


def test_adv_norm_center_only():
    """Test normalization with mean subtraction only (std_level='none')."""
    config = NormConfig(mean_level="batch", std_level=None, group_size=1)
    adv_norm = Normalization(config)

    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    loss_mask = torch.ones_like(advantages)

    # Apply normalization
    normalized = adv_norm(advantages, loss_mask)

    # Should be centered but not scaled
    assert normalized.shape == advantages.shape
    # Mean should be approximately 0
    assert torch.abs(normalized.mean()) < 1e-6


def test_adv_norm_without_mask():
    """Test normalization without providing a mask."""
    config = NormConfig(mean_level="batch", std_level="batch", group_size=1)
    adv_norm = Normalization(config)

    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)

    # Apply normalization without mask
    normalized = adv_norm(advantages)

    # Should still work and normalize
    assert normalized.shape == advantages.shape
    assert not torch.allclose(normalized, advantages)


def test_adv_norm_edge_cases():
    """Test edge cases for Normalization."""
    # Test with all zeros
    config = NormConfig(mean_level="batch", std_level="batch", group_size=1)
    adv_norm = Normalization(config)

    advantages = torch.zeros((2, 3), dtype=torch.float32)
    loss_mask = torch.ones_like(advantages)

    normalized = adv_norm(advantages, loss_mask)
    # Should handle zeros without division by zero
    assert torch.allclose(normalized, torch.zeros_like(advantages))

    # Test with very small values
    advantages = torch.tensor([[1e-10, 2e-10], [3e-10, 4e-10]], dtype=torch.float32)
    normalized = adv_norm(advantages)
    # Should handle small values without numerical issues
    assert normalized.shape == advantages.shape


def test_adv_norm_dtype_preservation():
    """Test that output dtype is preserved as float32."""
    config = NormConfig(mean_level="batch", std_level="batch", group_size=1)
    adv_norm = Normalization(config)

    advantages = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    normalized = adv_norm(advantages)

    # Output should be float32
    assert normalized.dtype == torch.float32


@patch("torch.distributed.is_initialized")
@patch("torch.distributed.all_reduce")
def test_adv_norm_distributed(mock_all_reduce, mock_is_initialized):
    """Test Normalization in distributed setting."""
    mock_is_initialized.return_value = True
    mock_all_reduce.return_value = None

    config = NormConfig(mean_level="batch", std_level="batch", group_size=1)
    adv_norm = Normalization(config)

    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    loss_mask = torch.ones_like(advantages)

    # Apply normalization with reduce_group
    normalized = adv_norm(advantages, loss_mask, reduce_group="dummy_group")

    # Should call all_reduce in distributed mode
    assert mock_all_reduce.called
    assert normalized.shape == advantages.shape


@pytest.mark.parametrize(
    "mean_level,std_level",
    [
        ("batch", "batch"),
        ("group", "group"),
        ("batch", "group"),
        ("group", "batch"),
        ("batch", None),
        (None, "batch"),
    ],
)
def test_adv_norm_parameterized(mean_level, std_level):
    """Parameterized test for different normalization combinations."""
    config = NormConfig(
        mean_level=mean_level,
        std_level=std_level,
        group_size=4 if "group" in [mean_level, std_level] else 1,
    )
    adv_norm = Normalization(config)

    # Create test data
    advantages = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=torch.float32,
    )

    loss_mask = torch.ones_like(advantages)

    # Apply normalization
    normalized = adv_norm(advantages, loss_mask)

    # Basic validation
    assert normalized.shape == advantages.shape
    assert normalized.dtype == torch.float32

    # For non-"none" normalization, values should change
    if mean_level != None or std_level != None:
        assert not torch.allclose(normalized, advantages)


def test_adv_norm_debug_cases():
    """Debug test cases for mixed and no normalization scenarios."""
    # Test mixed normalization (batch mean, group std)
    config = NormConfig(mean_level="batch", std_level="group", group_size=2)
    adv_norm = Normalization(config)

    advantages = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        dtype=torch.float32,
    )

    loss_mask = torch.ones_like(advantages)

    # Should work without shape mismatch errors
    normalized = adv_norm(advantages, loss_mask)
    assert normalized.shape == advantages.shape
    assert not torch.allclose(normalized, advantages)  # Should be normalized

    # Test no normalization case
    config = NormConfig(mean_level=None, std_level=None, group_size=1)
    adv_norm = Normalization(config)

    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    loss_mask = torch.ones_like(advantages)

    # Should return original values unchanged
    normalized = adv_norm(advantages, loss_mask)
    assert torch.allclose(normalized, advantages)
