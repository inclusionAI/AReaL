"""Unit tests for areal.utils.model module."""

from dataclasses import dataclass

import pytest

from areal.api.cli_args import (
    BaseExperimentConfig,
    ClusterSpecConfig,
    GRPOConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
    RWConfig,
    SFTConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import WeightUpdateMeta
from areal.utils.model import get_model_update_meta


@dataclass
class InvalidConfig(BaseExperimentConfig):
    """Invalid config for testing - missing actor/model attribute."""


def _create_base_cluster_config(fileroot="/tmp/areal_test"):
    """Helper to create a cluster config with common test values."""
    cluster = ClusterSpecConfig()
    cluster.fileroot = fileroot
    return cluster


def create_grpo_config(
    experiment_name="test_experiment",
    trial_name="test_trial",
    allocation_mode="sglang:d4p1t1+d4p1t1",
    weight_update_mode="disk",
    fileroot="/tmp/areal_test",
):
    """Factory to create a GRPOConfig with sensible test defaults."""
    config = GRPOConfig()
    config.experiment_name = experiment_name
    config.trial_name = trial_name
    config.allocation_mode = allocation_mode
    config.cluster = _create_base_cluster_config(fileroot)
    config.actor = PPOActorConfig()
    config.actor.weight_update_mode = weight_update_mode
    return config


def create_ppo_config(
    experiment_name="test_experiment",
    trial_name="test_trial",
    allocation_mode="vllm:d4p1t2+d8p1t1",
    weight_update_mode="disk",
    fileroot="/tmp/areal_test",
):
    """Factory to create a PPOConfig with sensible test defaults."""
    config = PPOConfig()
    config.experiment_name = experiment_name
    config.trial_name = trial_name
    config.allocation_mode = allocation_mode
    config.cluster = _create_base_cluster_config(fileroot)
    config.actor = PPOActorConfig()
    config.actor.weight_update_mode = weight_update_mode
    config.critic = PPOCriticConfig()
    return config


def create_sft_config(
    experiment_name="test_experiment",
    trial_name="test_trial",
    allocation_mode="d8p1t1",
    weight_update_mode="disk",
    fileroot="/tmp/areal_test",
):
    """Factory to create an SFTConfig with sensible test defaults."""
    config = SFTConfig()
    config.experiment_name = experiment_name
    config.trial_name = trial_name
    config.allocation_mode = allocation_mode
    config.cluster = _create_base_cluster_config(fileroot)
    config.actor = TrainEngineConfig()
    config.actor.weight_update_mode = weight_update_mode
    return config


def create_rw_config(
    experiment_name="test_experiment",
    trial_name="test_trial",
    allocation_mode="d8p1t1",
    weight_update_mode="disk",
    fileroot="/tmp/areal_test",
):
    """Factory to create an RWConfig with sensible test defaults."""
    config = RWConfig()
    config.experiment_name = experiment_name
    config.trial_name = trial_name
    config.allocation_mode = allocation_mode
    config.cluster = _create_base_cluster_config(fileroot)
    config.actor = TrainEngineConfig()
    config.actor.weight_update_mode = weight_update_mode
    return config


class TestGetModelUpdateMeta:
    """Tests for get_model_update_meta function."""

    @pytest.mark.parametrize(
        "config_factory,weight_update_mode,expected_type",
        [
            (create_grpo_config, "disk", "disk"),
            (create_ppo_config, "disk", "disk"),
            (create_sft_config, "disk", "disk"),
            (create_rw_config, "disk", "disk"),
            (create_grpo_config, "fsdp_xccl", "xccl"),
            (create_ppo_config, "fsdp_xccl", "xccl"),
            (create_sft_config, "fsdp_xccl", "xccl"),
            (create_rw_config, "fsdp_xccl", "xccl"),
        ],
        ids=[
            "GRPO-disk",
            "PPO-disk",
            "SFT-disk",
            "RW-disk",
            "GRPO-fsdp_xccl",
            "PPO-fsdp_xccl",
            "SFT-fsdp_xccl",
            "RW-fsdp_xccl",
        ],
    )
    def test_get_model_update_meta_modes(
        self, config_factory, weight_update_mode, expected_type
    ):
        """Test get_model_update_meta with various configs and modes."""
        config = config_factory(weight_update_mode=weight_update_mode)

        result = get_model_update_meta(config)

        assert isinstance(result, WeightUpdateMeta)
        assert result.type == expected_type

        if expected_type == "disk":
            assert config.cluster.fileroot in result.path
            assert config.experiment_name in result.path
            assert config.trial_name in result.path

    def test_invalid_config_type(self):
        """Test that passing TrainEngineConfig directly raises TypeError."""
        config = TrainEngineConfig()
        config.experiment_name = "test_experiment"
        config.trial_name = "test_trial"
        config.weight_update_mode = "disk"

        with pytest.raises(TypeError) as exc_info:
            get_model_update_meta(config)

        assert "must be BaseExperimentConfig" in str(exc_info.value)
        assert "TrainEngineConfig" in str(exc_info.value)

    def test_config_without_actor_or_model(self):
        """Test that config without actor or model attribute raises ValueError."""
        config = InvalidConfig()
        config.experiment_name = "test_experiment"
        config.trial_name = "test_trial"
        config.allocation_mode = "d8p1t1"
        config.cluster = ClusterSpecConfig()

        with pytest.raises(ValueError) as exc_info:
            get_model_update_meta(config)

        assert "must have either 'actor' or 'model' attribute" in str(exc_info.value)

    def test_regression_issue_482(self):
        """Regression test for issue #482 - GRPOConfig with vLLM training.

        This test verifies that the bug reported in issue #482 is fixed.
        The bug was: AttributeError: 'GRPOConfig' object has no attribute 'weight_update_mode'
        The fix accesses config.actor.weight_update_mode instead.
        """
        config = create_grpo_config(
            experiment_name="boba_grpo_vllm_16_gpus",
            trial_name="trial_0",
            allocation_mode="vllm:d4p1t2+d8p1t1",
            fileroot="/tmp/areal",
        )

        # This should not raise AttributeError
        result = get_model_update_meta(config)

        assert isinstance(result, WeightUpdateMeta)
        assert result.type == "disk"
