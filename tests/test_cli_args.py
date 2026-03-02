# Import directly from cli_args module to avoid importing areal package

import pytest

from areal.api.cli_args import PPOActorConfig, PPOConfig


class TestPPOActorConfig:
    """Test PPOActorConfig basic functionality."""

    def test_default_values(self):
        """Test default values for PPOActorConfig."""
        config = PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="/test/path",
        )
        # Test that the new parameters have correct defaults
        assert config.behav_imp_weight_mode == "token_mask"
        assert config.behav_imp_weight_cap == 5.0

    def test_validation_passes(self):
        """Test that PPOActorConfig __post_init__ passes with valid config."""
        config = PPOConfig(
            experiment_name="test",
            trial_name="test",
            actor=PPOActorConfig(
                experiment_name="test",
                trial_name="test",
                path="/test/path",
                use_decoupled_loss=True,
            ),
        )
        # Should not raise - PPOActorConfig __post_init__ is called during construction
        assert config.actor.behav_imp_weight_mode == "token_mask"

    def test_disable_mode_validation(self):
        """Test that disable mode requires behav_imp_weight_cap to be None."""
        # Should pass: disable mode with cap=None
        config = PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="/test/path",
            use_decoupled_loss=True,
            behav_imp_weight_mode="disable",
            behav_imp_weight_cap=None,
        )
        # Should not raise
        assert config.behav_imp_weight_mode == "disable"

        # Should fail: disable mode with cap set
        with pytest.raises(ValueError, match="behav_imp_weight_cap must be None"):
            PPOActorConfig(
                experiment_name="test",
                trial_name="test",
                path="/test/path",
                use_decoupled_loss=True,
                behav_imp_weight_mode="disable",
                behav_imp_weight_cap=5.0,
            )

    def test_non_disable_mode_validation(self):
        """Test that non-disable mode requires behav_imp_weight_cap > 1.0."""
        # Should fail: non-disable mode with cap <= 1.0
        with pytest.raises(ValueError, match="behav_imp_weight_cap must be > 1.0"):
            PPOActorConfig(
                experiment_name="test",
                trial_name="test",
                path="/test/path",
                use_decoupled_loss=True,
                behav_imp_weight_mode="token_mask",
                behav_imp_weight_cap=1.0,
            )
