"""Unit tests for FSDP pipeline_parallel.py (no GPU required)."""

import pytest

from areal.api.cli_args import FSDPEngineConfig
from areal.engine.fsdp_utils.pipeline_parallel import (
    generate_hf_fqn_per_model_part,
)


class TestGenerateHFFqnPerModelPart:
    """Test generate_hf_fqn_per_model_part function."""

    def test_single_stage(self):
        """Single stage should contain all modules."""
        result = generate_hf_fqn_per_model_part(num_stages=1, num_layers=4)
        assert len(result) == 1
        assert result[0] == [
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
            "model.norm",
            "lm_head",
        ]

    def test_two_stages_four_layers(self):
        """Two stages with 4 layers should split evenly."""
        result = generate_hf_fqn_per_model_part(num_stages=2, num_layers=4)
        assert len(result) == 2
        # 4 + 1 + 1 = 6 effective layers, 6 // 2 = 3 per stage
        # Stage 0: embedding + 2 layers (effective: 1 + 2 = 3)
        assert result[0] == ["model.embed_tokens", "model.layers.0", "model.layers.1"]
        # Stage 1: 2 layers + output (effective: 2 + 1 = 3)
        assert result[1] == [
            "model.layers.2",
            "model.layers.3",
            "model.norm",
            "lm_head",
        ]

    def test_four_stages_eight_layers(self):
        """Four stages with 8 layers."""
        result = generate_hf_fqn_per_model_part(num_stages=4, num_layers=8)
        assert len(result) == 4
        # 8 + 1 + 1 = 10 effective layers, 10 // 4 = 2 per stage, 2 extra
        # Stage 0: gets 3 effective (embed_tokens counts as 1, so 2 layers)
        assert result[0] == [
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.1",
        ]
        # Stage 1: gets 3 effective (all transformer layers)
        assert result[1] == ["model.layers.2", "model.layers.3", "model.layers.4"]
        # Stage 2: gets 2 effective (all transformer layers)
        assert result[2] == ["model.layers.5", "model.layers.6"]
        # Stage 3: gets 2 effective (norm+lm_head counts as 1, so 1 layer)
        assert result[3] == ["model.layers.7", "model.norm", "lm_head"]

    def test_two_stages_eight_layers(self):
        """Two stages with 8 layers."""
        result = generate_hf_fqn_per_model_part(num_stages=2, num_layers=8)
        assert len(result) == 2
        # 8 + 1 + 1 = 10 effective layers, 10 // 2 = 5 per stage
        # Stage 0: embed_tokens + 4 layers
        assert result[0] == [
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
        ]
        # Stage 1: 4 layers + norm + lm_head
        assert result[1] == [
            "model.layers.4",
            "model.layers.5",
            "model.layers.6",
            "model.layers.7",
            "model.norm",
            "lm_head",
        ]

    def test_three_stages_six_layers(self):
        """Three stages with 6 layers."""
        result = generate_hf_fqn_per_model_part(num_stages=3, num_layers=6)
        assert len(result) == 3
        # 6 + 1 + 1 = 8 effective layers, 8 // 3 = 2 per stage, 2 extra
        # Stage 0: gets 3 effective (embed_tokens + 2 layers)
        assert result[0] == [
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.1",
        ]
        # Stage 1: gets 3 effective (3 layers)
        assert result[1] == ["model.layers.2", "model.layers.3", "model.layers.4"]
        # Stage 2: gets 2 effective (1 layer + norm + lm_head)
        assert result[2] == ["model.layers.5", "model.norm", "lm_head"]

    def test_custom_weights(self):
        """Test with custom first_stage_less_layers/last_stage_less_layers."""
        # first_stage_less_layers=2, last_stage_less_layers=2 means embedding "costs" 2 slots
        result = generate_hf_fqn_per_model_part(
            num_stages=2,
            num_layers=4,
            first_stage_less_layers=2,
            last_stage_less_layers=2,
        )
        assert len(result) == 2
        # 4 + 2 + 2 = 8 effective layers, 8 // 2 = 4 per stage
        # Stage 0: embed_tokens (counts as 2) + 2 layers
        assert result[0] == ["model.embed_tokens", "model.layers.0", "model.layers.1"]
        # Stage 1: 2 layers + norm + lm_head (counts as 2)
        assert result[1] == [
            "model.layers.2",
            "model.layers.3",
            "model.norm",
            "lm_head",
        ]

    def test_zero_first_stage_less_layers(self):
        """Test with zero first_stage_less_layers."""
        result = generate_hf_fqn_per_model_part(
            num_stages=2,
            num_layers=4,
            first_stage_less_layers=0,
            last_stage_less_layers=1,
        )
        assert len(result) == 2
        # 4 + 0 + 1 = 5 effective layers, 5 // 2 = 2 per stage, 1 extra
        # Stage 0: gets 3 effective (embed_tokens counts as 0, so 3 layers)
        assert result[0] == [
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
        ]
        # Stage 1: gets 2 effective (1 layer + norm + lm_head)
        assert result[1] == ["model.layers.3", "model.norm", "lm_head"]

    def test_validation_num_stages_zero(self):
        """Test validation error for num_stages < 1."""
        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            generate_hf_fqn_per_model_part(num_stages=0, num_layers=4)

    def test_validation_num_stages_negative(self):
        """Test validation error for negative num_stages."""
        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            generate_hf_fqn_per_model_part(num_stages=-1, num_layers=4)

    def test_validation_too_many_stages(self):
        """Test validation error when num_stages exceeds effective layers."""
        with pytest.raises(ValueError, match="cannot exceed effective layers"):
            generate_hf_fqn_per_model_part(num_stages=10, num_layers=4)

    def test_validation_first_stage_less_layers_too_large(self):
        """Test validation error when first_stage_less_layers exceeds layers_per_stage."""
        # 4 + 5 + 1 = 10 effective, 10 // 4 = 2 per stage, but first_stage_less_layers=5 > 2
        with pytest.raises(
            ValueError, match="first_stage_less_layers.*exceeds layers_per_stage"
        ):
            generate_hf_fqn_per_model_part(
                num_stages=4,
                num_layers=4,
                first_stage_less_layers=5,
                last_stage_less_layers=1,
            )

    def test_validation_last_stage_less_layers_too_large(self):
        """Test validation error when last_stage_less_layers exceeds layers_per_stage."""
        # 4 + 1 + 5 = 10 effective, 10 // 4 = 2 per stage, but last_stage_less_layers=5 > 2
        with pytest.raises(
            ValueError, match="last_stage_less_layers.*exceeds layers_per_stage"
        ):
            generate_hf_fqn_per_model_part(
                num_stages=4,
                num_layers=4,
                first_stage_less_layers=1,
                last_stage_less_layers=5,
            )

    def test_large_model(self):
        """Test with a large model (e.g., 32 layers, 8 stages)."""
        result = generate_hf_fqn_per_model_part(num_stages=8, num_layers=32)
        assert len(result) == 8

        # Verify all layers are covered exactly once
        all_layer_names = []
        for stage in result:
            all_layer_names.extend([m for m in stage if m.startswith("model.layers.")])
        expected_layers = [f"model.layers.{i}" for i in range(32)]
        assert all_layer_names == expected_layers

        # Verify first stage has model.embed_tokens
        assert result[0][0] == "model.embed_tokens"

        # Verify last stage has model.norm and lm_head
        assert result[-1][-2:] == ["model.norm", "lm_head"]

    def test_critic_model_single_stage(self):
        """Critic model with single stage should use 'score' instead of 'lm_head'."""
        result = generate_hf_fqn_per_model_part(
            num_stages=1, num_layers=4, is_critic=True
        )
        assert len(result) == 1
        assert result[0] == [
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
            "model.norm",
            "score",
        ]

    def test_critic_model_two_stages(self):
        """Critic model with two stages should use 'score' on last stage."""
        result = generate_hf_fqn_per_model_part(
            num_stages=2, num_layers=4, is_critic=True
        )
        assert len(result) == 2
        assert result[0] == [
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.1",
        ]
        assert result[1] == [
            "model.layers.2",
            "model.layers.3",
            "model.norm",
            "score",
        ]

    def test_critic_model_four_stages(self):
        """Critic model with four stages."""
        result = generate_hf_fqn_per_model_part(
            num_stages=4, num_layers=8, is_critic=True
        )
        assert len(result) == 4
        # Last stage should have 'score' instead of 'lm_head'
        assert result[3] == ["model.layers.7", "model.norm", "score"]

    def test_exact_distribution(self):
        """Test when layers divide evenly."""
        # 6 layers + 1 + 1 = 8, 8 // 4 = 2 per stage exactly
        result = generate_hf_fqn_per_model_part(num_stages=4, num_layers=6)
        assert len(result) == 4

        # All stages should have 2 effective layers
        # Stage 0: embed_tokens (1) + 1 layer = 2
        assert result[0] == ["model.embed_tokens", "model.layers.0"]
        # Stage 1: 2 layers = 2
        assert result[1] == ["model.layers.1", "model.layers.2"]
        # Stage 2: 2 layers = 2
        assert result[2] == ["model.layers.3", "model.layers.4"]
        # Stage 3: 1 layer + norm + lm_head (1) = 2
        assert result[3] == ["model.layers.5", "model.norm", "lm_head"]


class TestZBVFqnGeneration:
    """Test FQN generation for ZBV pipeline configurations."""

    def test_zbv_fqn_generation(self):
        """Verify FQN distribution for a typical ZBV config (pp_degree=2, 8 layers)."""
        result = generate_hf_fqn_per_model_part(num_stages=4, num_layers=8)
        assert len(result) == 4

        # ZBV: Rank 0 gets stages (0, 3), rank 1 gets stages (1, 2)
        rank0_modules = result[0] + result[3]
        rank1_modules = result[1] + result[2]

        # Rank 0 has first and last stages
        assert "model.embed_tokens" in rank0_modules
        assert "model.norm" in rank0_modules
        assert "lm_head" in rank0_modules

        # Rank 1 has only middle layers (no embeddings or output head)
        assert all(m.startswith("model.layers.") for m in rank1_modules)

        # All layers covered exactly once
        all_layers = []
        for stage in result:
            all_layers.extend([m for m in stage if m.startswith("model.layers.")])
        assert all_layers == [f"model.layers.{i}" for i in range(8)]


class TestFSDPEngineConfig:
    """Test FSDPEngineConfig pipeline parallelism fields and validation."""

    def test_pp_default_values(self):
        """PP fields should have correct defaults."""
        config = FSDPEngineConfig()
        assert config.pp_schedule == "Interleaved1F1B"
        assert config.pp_layers_per_stage is None
        assert config.pp_first_stage_less_layers == 1
        assert config.pp_last_stage_less_layers == 1

    def test_pp_custom_values(self):
        """PP fields should accept custom values."""
        config = FSDPEngineConfig(
            pp_schedule="1F1B",
            pp_layers_per_stage=4,
            pp_first_stage_less_layers=2,
            pp_last_stage_less_layers=0,
        )
        assert config.pp_schedule == "1F1B"
        assert config.pp_layers_per_stage == 4
        assert config.pp_first_stage_less_layers == 2
        assert config.pp_last_stage_less_layers == 0

    def test_pp_layers_per_stage_zero(self):
        """pp_layers_per_stage=0 should raise ValueError."""
        with pytest.raises(ValueError, match="pp_layers_per_stage must be >= 1"):
            FSDPEngineConfig(pp_layers_per_stage=0)

    def test_pp_layers_per_stage_negative(self):
        """pp_layers_per_stage=-1 should raise ValueError."""
        with pytest.raises(ValueError, match="pp_layers_per_stage must be >= 1"):
            FSDPEngineConfig(pp_layers_per_stage=-1)

    def test_pp_first_stage_less_layers_negative(self):
        """Negative pp_first_stage_less_layers should raise ValueError."""
        with pytest.raises(ValueError, match="pp_first_stage_less_layers must be >= 0"):
            FSDPEngineConfig(pp_first_stage_less_layers=-1)

    def test_pp_last_stage_less_layers_negative(self):
        """Negative pp_last_stage_less_layers should raise ValueError."""
        with pytest.raises(ValueError, match="pp_last_stage_less_layers must be >= 0"):
            FSDPEngineConfig(pp_last_stage_less_layers=-1)
