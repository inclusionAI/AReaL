"""Tests for Qwen3ModelArgs with MoE support.

Run tests:
    pytest areal/tests/experimental/archon/test_qwen3_args.py -v
"""

from areal.experimental.models.archon.moe import MoEArgs
from areal.experimental.models.archon.qwen3.model.args import Qwen3ModelArgs


class TestQwen3ModelArgsDefaults:
    """Tests for Qwen3ModelArgs default values."""

    def test_default_values(self):
        """Test default values for non-MoE model."""
        args = Qwen3ModelArgs()

        assert args.dim == 1024
        assert args.n_layers == 28
        assert args.moe_enabled is False
        assert args.moe_args is None
        assert args.decoder_sparse_step == 1

    def test_moe_disabled_by_default(self):
        """Test that MoE is disabled by default."""
        args = Qwen3ModelArgs()

        assert args.moe_enabled is False
        assert args.moe_args is None


class TestQwen3ModelArgsMoE:
    """Tests for Qwen3ModelArgs with MoE configuration."""

    def test_moe_enabled_with_args(self):
        """Test enabling MoE with explicit MoEArgs."""
        moe_args = MoEArgs(num_experts=64, top_k=8)
        args = Qwen3ModelArgs(
            moe_enabled=True,
            moe_args=moe_args,
            moe_inter_dim=1024,
        )

        assert args.moe_enabled is True
        assert args.moe_args is not None
        assert args.moe_args.num_experts == 64
        assert args.moe_args.top_k == 8
        assert args.moe_inter_dim == 1024

    def test_decoder_sparse_step(self):
        """Test decoder_sparse_step configuration."""
        args = Qwen3ModelArgs(
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=8),
            decoder_sparse_step=2,
        )

        assert args.decoder_sparse_step == 2


class TestQwen3ModelArgsFromHfConfig:
    """Tests for Qwen3ModelArgs.from_hf_config() with MoE."""

    def test_from_hf_config_dense(self):
        """Test from_hf_config for dense (non-MoE) model."""

        class MockDenseConfig:
            hidden_size = 1024
            num_hidden_layers = 24
            num_attention_heads = 16
            num_key_value_heads = 8
            vocab_size = 100000
            intermediate_size = 2816
            rms_norm_eps = 1e-6

        args = Qwen3ModelArgs.from_hf_config(MockDenseConfig())

        assert args.moe_enabled is False
        assert args.moe_args is None
        assert args.dim == 1024
        assert args.n_layers == 24

    def test_from_hf_config_moe_with_num_experts(self):
        """Test from_hf_config for MoE model using num_experts."""

        class MockMoEConfig:
            hidden_size = 2048
            num_hidden_layers = 36
            num_attention_heads = 16
            num_key_value_heads = 4
            vocab_size = 151936
            intermediate_size = 8960
            rms_norm_eps = 1e-5
            # MoE fields
            num_experts = 128
            num_experts_per_tok = 8
            moe_intermediate_size = 1408
            norm_topk_prob = True

        args = Qwen3ModelArgs.from_hf_config(MockMoEConfig())

        assert args.moe_enabled is True
        assert args.moe_args is not None
        assert args.moe_args.num_experts == 128
        assert args.moe_args.top_k == 8
        assert args.moe_args.route_norm is True
        assert args.moe_inter_dim == 1408

    def test_from_hf_config_moe_with_num_local_experts(self):
        """Test from_hf_config for MoE model using num_local_experts."""

        class MockMoEConfig:
            hidden_size = 2048
            num_hidden_layers = 36
            num_attention_heads = 16
            num_key_value_heads = 4
            vocab_size = 151936
            intermediate_size = 8960
            rms_norm_eps = 1e-5
            # MoE fields (alternative naming)
            num_local_experts = 64
            num_experts_per_tok = 4

        args = Qwen3ModelArgs.from_hf_config(MockMoEConfig())

        assert args.moe_enabled is True
        assert args.moe_args is not None
        assert args.moe_args.num_experts == 64
        assert args.moe_args.top_k == 4

    def test_from_hf_config_moe_with_shared_experts(self):
        """Test from_hf_config with shared experts."""

        class MockMoEConfig:
            hidden_size = 2048
            num_hidden_layers = 36
            num_attention_heads = 16
            num_key_value_heads = 4
            vocab_size = 151936
            intermediate_size = 8960
            rms_norm_eps = 1e-5
            num_experts = 64
            num_experts_per_tok = 4
            num_shared_experts = 2

        args = Qwen3ModelArgs.from_hf_config(MockMoEConfig())

        assert args.moe_args.num_shared_experts == 2

    def test_from_hf_config_moe_decoder_sparse_step(self):
        """Test from_hf_config with decoder_sparse_step."""

        class MockMoEConfig:
            hidden_size = 2048
            num_hidden_layers = 36
            num_attention_heads = 16
            num_key_value_heads = 4
            vocab_size = 151936
            intermediate_size = 8960
            rms_norm_eps = 1e-5
            num_experts = 64
            num_experts_per_tok = 4
            decoder_sparse_step = 2  # Every other layer is MoE

        args = Qwen3ModelArgs.from_hf_config(MockMoEConfig())

        assert args.decoder_sparse_step == 2


class TestQwen3ModelArgsQwen3MoE:
    """Tests for Qwen3-MoE specific configurations."""

    def test_qwen3_30b_a3b_config(self):
        """Test configuration matching Qwen3-30B-A3B."""

        class MockQwen3_30B_A3B:
            hidden_size = 2048
            num_hidden_layers = 36
            num_attention_heads = 16
            num_key_value_heads = 4
            vocab_size = 151936
            intermediate_size = 8960
            rms_norm_eps = 1e-5
            num_experts = 128
            num_experts_per_tok = 8
            moe_intermediate_size = 1408
            norm_topk_prob = True

        args = Qwen3ModelArgs.from_hf_config(MockQwen3_30B_A3B())

        assert args.moe_enabled is True
        assert args.moe_args.num_experts == 128
        assert args.moe_args.top_k == 8
        assert args.moe_args.route_norm is True
        assert args.dim == 2048
        assert args.n_layers == 36

    def test_single_expert_not_moe(self):
        """Test that num_experts=1 does not enable MoE."""

        class MockSingleExpert:
            hidden_size = 1024
            num_hidden_layers = 24
            num_attention_heads = 16
            num_key_value_heads = 8
            vocab_size = 100000
            intermediate_size = 2816
            rms_norm_eps = 1e-6
            num_experts = 1

        args = Qwen3ModelArgs.from_hf_config(MockSingleExpert())

        assert args.moe_enabled is False
        assert args.moe_args is None
