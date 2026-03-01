# Import directly from cli_args module to avoid importing areal package

import pytest

from areal.api.cli_args import PPOActorConfig, PPOConfig, _validate_cfg


class TestPPOActorConfigEngineIS:
    """Test PPOActorConfig validation for enable_mis_tis_correction."""

    def test_engine_is_requires_decoupled_or_recompute(self):
        """Test that enable_mis_tis_correction=True requires decoupled or recompute."""
        # Create config with invalid combination
        config = PPOConfig(
            experiment_name="test",
            trial_name="test",
            actor=PPOActorConfig(
                experiment_name="test",
                trial_name="test",
                path="/test/path",
                enable_mis_tis_correction=True,
                use_decoupled_loss=False,
                prox_logp_method="loglinear",  # not recompute
            ),
        )
        # Should raise when _validate_cfg is called
        with pytest.raises(ValueError, match="enable_mis_tis_correction=True requires"):
            _validate_cfg(config)

    def test_engine_is_works_with_decoupled(self):
        """Test that enable_mis_tis_correction works with decoupled loss."""
        config = PPOConfig(
            experiment_name="test",
            trial_name="test",
            actor=PPOActorConfig(
                experiment_name="test",
                trial_name="test",
                path="/test/path",
                enable_mis_tis_correction=True,
                use_decoupled_loss=True,
            ),
        )
        _validate_cfg(config)  # Should not raise
        assert config.actor.enable_mis_tis_correction is True

    def test_engine_is_works_with_recompute(self):
        """Test that enable_mis_tis_correction works with recompute."""
        config = PPOConfig(
            experiment_name="test",
            trial_name="test",
            actor=PPOActorConfig(
                experiment_name="test",
                trial_name="test",
                path="/test/path",
                enable_mis_tis_correction=True,
                use_decoupled_loss=False,
                prox_logp_method="recompute",
            ),
        )
        _validate_cfg(config)  # Should not raise
        assert config.actor.enable_mis_tis_correction is True

    def test_engine_is_defaults(self):
        """Test default values for engine_is parameters."""
        config = PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="/test/path",
        )
        assert config.enable_mis_tis_correction is False
        assert config.engine_mismatch_is_mode == "sequence_mask"
        assert config.engine_mismatch_is_cap == 3.0
