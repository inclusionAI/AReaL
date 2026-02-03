"""Tests for MoEArgs dataclass.

Run tests:
    pytest areal/tests/experimental/archon/test_moe_args.py -v
"""

from areal.experimental.models.archon.moe.args import MoEArgs


class TestMoEArgsDefaults:
    """Tests for MoEArgs default values."""

    def test_default_values(self):
        """Test MoEArgs has expected default values."""
        args = MoEArgs()

        assert args.num_experts == 8
        assert args.num_shared_experts == 0
        assert args.top_k == 1
        assert args.score_func == "sigmoid"
        assert args.route_norm is False
        assert args.route_scale == 1.0
        assert args.score_before_experts is True
        assert args.num_expert_groups is None
        assert args.num_limited_groups is None
        assert args.use_grouped_mm is True
        assert args.load_balance_coeff is None
        assert args._debug_force_load_balance is False

    def test_custom_values(self):
        """Test MoEArgs with custom values."""
        args = MoEArgs(
            num_experts=64,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            load_balance_coeff=None,
        )

        assert args.num_experts == 64
        assert args.top_k == 8
        assert args.score_func == "softmax"
        assert args.route_norm is True
        assert args.load_balance_coeff is None


class TestMoEArgsFromHfConfig:
    """Tests for MoEArgs.from_hf_config()."""

    def test_from_hf_config_basic(self):
        """Test creating MoEArgs from a basic HF config."""

        class MockConfig:
            num_experts = 64
            num_experts_per_tok = 8
            norm_topk_prob = True

        args = MoEArgs.from_hf_config(MockConfig())

        assert args.num_experts == 64
        assert args.top_k == 8
        assert args.route_norm is True
        assert args.score_func == "softmax"

    def test_from_hf_config_with_num_local_experts(self):
        """Test creating MoEArgs when HF config uses num_local_experts."""

        class MockConfig:
            num_local_experts = 128
            num_experts_per_tok = 4

        args = MoEArgs.from_hf_config(MockConfig())

        assert args.num_experts == 128
        assert args.top_k == 4

    def test_from_hf_config_defaults(self):
        """Test MoEArgs.from_hf_config uses defaults for missing fields."""

        class MockConfig:
            pass

        args = MoEArgs.from_hf_config(MockConfig())

        assert args.num_experts == 8
        assert args.top_k == 1
        assert args.route_norm is False


class TestMoEArgsQwen3:
    """Tests for Qwen3-MoE specific configurations."""

    def test_qwen3_30b_a3b_config(self):
        """Test config matching Qwen3-30B-A3B."""
        args = MoEArgs(
            num_experts=128,
            num_shared_experts=0,
            top_k=8,
            score_func="softmax",
            route_norm=True,
        )

        assert args.num_experts == 128
        assert args.num_shared_experts == 0
        assert args.top_k == 8

    def test_qwen3_235b_a22b_config(self):
        """Test config matching Qwen3-235B-A22B."""
        args = MoEArgs(
            num_experts=128,
            num_shared_experts=0,
            top_k=8,
            score_func="softmax",
            route_norm=True,
        )

        assert args.num_experts == 128
        assert args.top_k == 8
