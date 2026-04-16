"""Tests for speculative decoding (EAGLE) configuration and E2E training.

This module contains:
- TestSpeculativeDecodingConfig: Unit tests for config field parsing and validation
- TestSpeculativeDecodingE2E: End-to-end tests for speculative decoding training

Run unit tests:
    pytest tests/speculative_decoding/test_speculative_decoding.py -v -k "Config"

Run E2E tests (requires GPUs):
    pytest tests/speculative_decoding/test_speculative_decoding.py -v -k "E2E"
"""

import math
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths to test config files (relative to this file)
# ---------------------------------------------------------------------------
_TEST_DIR = Path(__file__).resolve().parent
_CONFIG_SPEC_ONLY = _TEST_DIR / "config_spec_only.yaml"
_CONFIG_SPEC_WITH_MTP = _TEST_DIR / "config_spec_with_mtp.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return as dict (without OmegaConf resolution)."""
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Unit Tests: Configuration Parsing and Validation
# ============================================================================


class TestSpeculativeDecodingConfig:
    """Unit tests for speculative decoding configuration fields."""

    # ------------------------------------------------------------------
    # SGLang speculative decoding config fields
    # ------------------------------------------------------------------

    def test_sglang_config_has_speculative_fields(self):
        """SGLangConfig dataclass should expose all speculative decoding fields."""
        from areal.api.cli_args import SGLangConfig

        cfg = SGLangConfig()
        assert hasattr(cfg, "speculative_algorithm")
        assert hasattr(cfg, "speculative_draft_model_path")
        assert hasattr(cfg, "speculative_num_steps")
        assert hasattr(cfg, "speculative_eagle_topk")
        assert hasattr(cfg, "speculative_num_draft_tokens")
        assert hasattr(cfg, "speculative_attention_mode")

    def test_sglang_config_defaults(self):
        """Default values should disable speculative decoding."""
        from areal.api.cli_args import SGLangConfig

        cfg = SGLangConfig()
        assert cfg.speculative_algorithm is None
        assert cfg.speculative_draft_model_path is None
        assert cfg.speculative_num_steps == 3
        assert cfg.speculative_eagle_topk == 1
        assert cfg.speculative_num_draft_tokens == 4
        assert cfg.speculative_attention_mode is None

    def test_sglang_config_eagle_values(self):
        """SGLangConfig should accept EAGLE algorithm settings."""
        from areal.api.cli_args import SGLangConfig

        cfg = SGLangConfig(
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=8,
        )
        assert cfg.speculative_algorithm == "EAGLE"
        assert cfg.speculative_num_steps == 5
        assert cfg.speculative_eagle_topk == 2
        assert cfg.speculative_num_draft_tokens == 8

    def test_sglang_config_eagle3_values(self):
        """SGLangConfig should accept EAGLE3 algorithm settings."""
        from areal.api.cli_args import SGLangConfig

        cfg = SGLangConfig(speculative_algorithm="EAGLE3")
        assert cfg.speculative_algorithm == "EAGLE3"

    def test_sglang_config_draft_model_path(self):
        """SGLangConfig should accept an external draft model path."""
        from areal.api.cli_args import SGLangConfig

        cfg = SGLangConfig(
            speculative_algorithm="EAGLE",
            speculative_draft_model_path="/models/eagle-draft",
        )
        assert cfg.speculative_draft_model_path == "/models/eagle-draft"

    def test_sglang_config_enable_draft_weights_cpu_backup(self):
        """SGLangConfig should expose enable_draft_weights_cpu_backup field."""
        from areal.api.cli_args import SGLangConfig

        cfg = SGLangConfig()
        assert hasattr(cfg, "enable_draft_weights_cpu_backup")

    # ------------------------------------------------------------------
    # PPOActorConfig MTP training fields
    # ------------------------------------------------------------------

    def test_actor_config_has_mtp_fields(self):
        """PPOActorConfig should expose MTP training fields."""
        from areal.api.cli_args import PPOActorConfig

        # PPOActorConfig requires certain fields; check class attributes
        assert hasattr(PPOActorConfig, "enable_mtp_training")
        assert hasattr(PPOActorConfig, "mtp_num_layers")
        assert hasattr(PPOActorConfig, "mtp_loss_scaling_factor")

    def test_actor_config_mtp_defaults(self):
        """MTP training should be disabled by default."""
        from areal.api.cli_args import PPOActorConfig

        # Access field defaults from the dataclass
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(PPOActorConfig)}
        assert fields["enable_mtp_training"].default is False
        assert fields["mtp_num_layers"].default == 1
        assert fields["mtp_loss_scaling_factor"].default == 0.1

    def test_actor_config_mtp_validation_num_layers_zero(self):
        """Enabling MTP with mtp_num_layers=0 should raise ValueError."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="mtp_num_layers must be > 0"):
            PPOActorConfig(
                enable_mtp_training=True,
                mtp_num_layers=0,
                mtp_loss_scaling_factor=0.1,
            )

    def test_actor_config_mtp_validation_scaling_factor_out_of_range(self):
        """MTP loss scaling factor outside (0, 1.0] should raise ValueError."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="mtp_loss_scaling_factor must be in"):
            PPOActorConfig(
                enable_mtp_training=True,
                mtp_num_layers=1,
                mtp_loss_scaling_factor=1.5,
            )

    def test_actor_config_mtp_validation_scaling_factor_zero(self):
        """MTP loss scaling factor of 0 should raise ValueError."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="mtp_loss_scaling_factor must be in"):
            PPOActorConfig(
                enable_mtp_training=True,
                mtp_num_layers=1,
                mtp_loss_scaling_factor=0.0,
            )

    def test_actor_config_mtp_validation_negative_layers(self):
        """Negative mtp_num_layers should raise ValueError."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="mtp_num_layers must be > 0"):
            PPOActorConfig(
                enable_mtp_training=True,
                mtp_num_layers=-1,
                mtp_loss_scaling_factor=0.1,
            )

    # ------------------------------------------------------------------
    # MegatronEngineConfig MTP fields
    # ------------------------------------------------------------------

    def test_megatron_config_has_mtp_fields(self):
        """MegatronEngineConfig should have MTP-related fields."""
        from areal.api.cli_args import MegatronEngineConfig

        assert hasattr(MegatronEngineConfig, "mtp_num_layers")
        assert hasattr(MegatronEngineConfig, "mtp_loss_scaling_factor")

    def test_megatron_config_mtp_defaults(self):
        """MegatronEngineConfig MTP defaults should be 0 / 0.1."""
        from areal.api.cli_args import MegatronEngineConfig

        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(MegatronEngineConfig)}
        assert fields["mtp_num_layers"].default == 0
        assert fields["mtp_loss_scaling_factor"].default == 0.1

    # ------------------------------------------------------------------
    # YAML config file parsing
    # ------------------------------------------------------------------

    def test_spec_only_yaml_loads(self):
        """config_spec_only.yaml should load without errors."""
        cfg = _load_yaml(_CONFIG_SPEC_ONLY)
        assert cfg["experiment_name"] == "test-spec-decode-only"
        assert cfg["sglang"]["speculative_algorithm"] == "EAGLE"

    def test_spec_only_yaml_mtp_disabled(self):
        """config_spec_only.yaml should have MTP training disabled."""
        cfg = _load_yaml(_CONFIG_SPEC_ONLY)
        assert cfg["actor"]["enable_mtp_training"] is False

    def test_spec_with_mtp_yaml_loads(self):
        """config_spec_with_mtp.yaml should load without errors."""
        cfg = _load_yaml(_CONFIG_SPEC_WITH_MTP)
        assert cfg["experiment_name"] == "test-spec-decode-mtp"
        assert cfg["sglang"]["speculative_algorithm"] == "EAGLE"

    def test_spec_with_mtp_yaml_mtp_enabled(self):
        """config_spec_with_mtp.yaml should have MTP training enabled."""
        cfg = _load_yaml(_CONFIG_SPEC_WITH_MTP)
        assert cfg["actor"]["enable_mtp_training"] is True
        assert cfg["actor"]["mtp_num_layers"] == 1
        assert cfg["actor"]["mtp_loss_scaling_factor"] == 0.1

    def test_spec_with_mtp_yaml_megatron_mtp(self):
        """config_spec_with_mtp.yaml should have Megatron MTP settings."""
        cfg = _load_yaml(_CONFIG_SPEC_WITH_MTP)
        megatron_cfg = cfg["actor"]["megatron"]
        assert megatron_cfg["mtp_num_layers"] == 1
        assert megatron_cfg["mtp_loss_scaling_factor"] == 0.1

    def test_spec_with_mtp_yaml_draft_cpu_backup(self):
        """config_spec_with_mtp.yaml should enable draft weights CPU backup."""
        cfg = _load_yaml(_CONFIG_SPEC_WITH_MTP)
        assert cfg["sglang"]["enable_draft_weights_cpu_backup"] is True

    def test_spec_only_yaml_no_draft_cpu_backup(self):
        """config_spec_only.yaml should not set draft weights CPU backup."""
        cfg = _load_yaml(_CONFIG_SPEC_ONLY)
        assert "enable_draft_weights_cpu_backup" not in cfg["sglang"]


# ============================================================================
# E2E Tests: Speculative Decoding Training
# ============================================================================


def _has_gpus(min_count: int = 2) -> bool:
    """Check if sufficient GPUs are available."""
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() >= min_count
    except ImportError:
        return False


@pytest.mark.skipif(not _has_gpus(2), reason="Requires at least 2 GPUs")
class TestSpeculativeDecodingE2E:
    """End-to-end tests for speculative decoding training.

    These tests require GPU resources and a model checkpoint. They verify
    that the full training loop runs correctly with speculative decoding
    enabled, producing valid statistics.
    """

    def test_spec_only_e2e(self):
        """E2E test: EAGLE speculative decoding without MTP training.

        Verifies that:
        1. Training completes without errors
        2. Speculative decoding stats are collected
        3. Accept rate is within valid range [0, 1]
        """
        from tests.speculative_decoding.entrypoint import (
            MinimalSpecDecodePPOTrainer,
        )

        trainer = MinimalSpecDecodePPOTrainer(
            config_path=str(_CONFIG_SPEC_ONLY)
        )
        stats = trainer.run(max_steps=2)
        summary = stats.summary()

        # Verify stats were collected
        assert summary["num_steps"] > 0, "Expected at least 1 training step"

        # Accept rate should be in valid range
        if summary["total_draft_tokens"] > 0:
            assert 0.0 <= summary["overall_accept_rate"] <= 1.0, (
                f"Accept rate {summary['overall_accept_rate']} out of range"
            )

        # MTP loss should NOT be present (MTP training disabled)
        assert len(stats.mtp_losses) == 0, (
            "MTP losses should be empty when enable_mtp_training=False"
        )

    def test_spec_with_mtp_e2e(self):
        """E2E test: EAGLE speculative decoding with MTP online training.

        Verifies that:
        1. Training completes without errors
        2. Speculative decoding stats are collected
        3. MTP loss is recorded and is finite
        4. Accept rate is within valid range [0, 1]
        """
        from tests.speculative_decoding.entrypoint import (
            MinimalSpecDecodePPOTrainer,
        )

        trainer = MinimalSpecDecodePPOTrainer(
            config_path=str(_CONFIG_SPEC_WITH_MTP)
        )
        stats = trainer.run(max_steps=2)
        summary = stats.summary()

        # Verify stats were collected
        assert summary["num_steps"] > 0, "Expected at least 1 training step"

        # Accept rate should be in valid range
        if summary["total_draft_tokens"] > 0:
            assert 0.0 <= summary["overall_accept_rate"] <= 1.0, (
                f"Accept rate {summary['overall_accept_rate']} out of range"
            )

        # MTP loss should be present and finite
        assert len(stats.mtp_losses) > 0, (
            "MTP losses should be recorded when enable_mtp_training=True"
        )
        for loss in stats.mtp_losses:
            assert math.isfinite(loss), f"MTP loss is not finite: {loss}"

    def test_spec_decode_rewards_collected(self):
        """Verify that rewards are collected during speculative decoding training."""
        from tests.speculative_decoding.entrypoint import (
            MinimalSpecDecodePPOTrainer,
        )

        trainer = MinimalSpecDecodePPOTrainer(
            config_path=str(_CONFIG_SPEC_WITH_MTP)
        )
        stats = trainer.run(max_steps=3)

        # Rewards should be collected
        assert len(stats.rewards) > 0, "Expected rewards to be collected"
        for reward in stats.rewards:
            assert math.isfinite(reward), f"Reward is not finite: {reward}"


# ============================================================================
# Unit Tests: SpecDecodeStats helper class
# ============================================================================


class TestSpecDecodeStats:
    """Unit tests for the SpecDecodeStats dataclass."""

    def test_empty_stats(self):
        """Empty stats should return 0 accept rate and NaN losses."""
        from tests.speculative_decoding.entrypoint import SpecDecodeStats

        stats = SpecDecodeStats()
        assert stats.overall_accept_rate == 0.0
        assert math.isnan(stats.mean_mtp_loss)
        assert math.isnan(stats.mean_reward)

    def test_accept_rate_calculation(self):
        """Accept rate should be accept_tokens / draft_tokens."""
        from tests.speculative_decoding.entrypoint import SpecDecodeStats

        stats = SpecDecodeStats(total_accept_tokens=75, total_draft_tokens=100)
        assert stats.overall_accept_rate == pytest.approx(0.75)

    def test_mean_mtp_loss(self):
        """Mean MTP loss should average all recorded losses."""
        from tests.speculative_decoding.entrypoint import SpecDecodeStats

        stats = SpecDecodeStats(mtp_losses=[1.0, 2.0, 3.0])
        assert stats.mean_mtp_loss == pytest.approx(2.0)

    def test_mean_reward(self):
        """Mean reward should average all recorded rewards."""
        from tests.speculative_decoding.entrypoint import SpecDecodeStats

        stats = SpecDecodeStats(rewards=[0.5, 1.0, 1.5])
        assert stats.mean_reward == pytest.approx(1.0)

    def test_summary_keys(self):
        """Summary dict should contain all expected keys."""
        from tests.speculative_decoding.entrypoint import SpecDecodeStats

        stats = SpecDecodeStats()
        summary = stats.summary()
        expected_keys = {
            "total_accept_tokens",
            "total_draft_tokens",
            "overall_accept_rate",
            "num_steps",
            "step_accept_rates",
            "mean_mtp_loss",
            "mean_reward",
            "mtp_losses",
            "rewards",
        }
        assert set(summary.keys()) == expected_keys
