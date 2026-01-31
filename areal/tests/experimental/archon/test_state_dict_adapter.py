"""Tests for Archon State Dict Adapter key conversion logic.

These tests verify:
1. Key conversion between HuggingFace and Archon formats
2. Dense and MoE model key conversion
3. Roundtrip conversion consistency
4. MoE expert weight splitting and collection

Run tests:
    pytest areal/tests/experimental/archon/test_state_dict_adapter.py -v

Note: For weight completeness and shape matching tests, see test_weight_sync.py
"""

import pytest
import torch

from areal.experimental.models.archon.qwen3 import Qwen3StateDictAdapter

# =============================================================================
# Mock Configs
# =============================================================================


class MockQwen3Config:
    """Mock Qwen3 model config for testing."""

    model_type = "qwen2"
    num_local_experts = 0  # Dense model


class MockQwen3MoEConfig:
    """Mock Qwen3 MoE model config for testing."""

    model_type = "qwen2_moe"
    num_local_experts = 8


# =============================================================================
# Unit Tests: Key Conversion (No GPU required)
# =============================================================================


class TestQwen3StateDictAdapterDense:
    """Tests for Qwen3StateDictAdapter with dense models."""

    @pytest.fixture
    def adapter(self):
        return Qwen3StateDictAdapter(MockQwen3Config())

    def test_key_conversion_embedding(self, adapter):
        """Test embedding key conversion."""
        hf_key = "model.embed_tokens.weight"
        archon_key = "tok_embeddings.weight"

        # HF -> Archon
        assert adapter._convert_key_from_hf(hf_key) == archon_key
        # Archon -> HF
        assert adapter._convert_key_to_hf(archon_key) == hf_key

    def test_key_conversion_attention(self, adapter):
        """Test attention projection key conversion."""
        test_cases = [
            (
                "model.layers.0.self_attn.q_proj.weight",
                "layers.0.attention.wq.weight",
            ),
            (
                "model.layers.5.self_attn.k_proj.weight",
                "layers.5.attention.wk.weight",
            ),
            (
                "model.layers.10.self_attn.v_proj.weight",
                "layers.10.attention.wv.weight",
            ),
            (
                "model.layers.31.self_attn.o_proj.weight",
                "layers.31.attention.wo.weight",
            ),
        ]

        for hf_key, archon_key in test_cases:
            assert adapter._convert_key_from_hf(hf_key) == archon_key
            assert adapter._convert_key_to_hf(archon_key) == hf_key

    def test_key_conversion_qk_norm(self, adapter):
        """Test Qwen3 Q/K norm key conversion."""
        test_cases = [
            (
                "model.layers.0.self_attn.q_norm.weight",
                "layers.0.attention.q_norm.weight",
            ),
            (
                "model.layers.15.self_attn.k_norm.weight",
                "layers.15.attention.k_norm.weight",
            ),
        ]

        for hf_key, archon_key in test_cases:
            assert adapter._convert_key_from_hf(hf_key) == archon_key
            assert adapter._convert_key_to_hf(archon_key) == hf_key

    def test_key_conversion_mlp(self, adapter):
        """Test MLP key conversion."""
        test_cases = [
            (
                "model.layers.0.mlp.gate_proj.weight",
                "layers.0.feed_forward.w1.weight",
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                "layers.0.feed_forward.w3.weight",
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                "layers.0.feed_forward.w2.weight",
            ),
        ]

        for hf_key, archon_key in test_cases:
            assert adapter._convert_key_from_hf(hf_key) == archon_key
            assert adapter._convert_key_to_hf(archon_key) == hf_key

    def test_key_conversion_layernorm(self, adapter):
        """Test LayerNorm key conversion."""
        test_cases = [
            (
                "model.layers.0.input_layernorm.weight",
                "layers.0.attention_norm.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "layers.0.ffn_norm.weight",
            ),
            ("model.norm.weight", "norm.weight"),
        ]

        for hf_key, archon_key in test_cases:
            assert adapter._convert_key_from_hf(hf_key) == archon_key
            assert adapter._convert_key_to_hf(archon_key) == hf_key

    def test_key_conversion_output(self, adapter):
        """Test output layer key conversion."""
        hf_key = "lm_head.weight"
        archon_key = "output.weight"

        assert adapter._convert_key_from_hf(hf_key) == archon_key
        assert adapter._convert_key_to_hf(archon_key) == hf_key

    def test_skip_rotary_emb(self, adapter):
        """Test that rotary embeddings are skipped."""
        hf_key = "model.layers.0.self_attn.rotary_emb.inv_freq"
        assert adapter._convert_key_from_hf(hf_key) is None

    def test_roundtrip_dense_state_dict(self, adapter):
        """Test full roundtrip conversion for dense model."""
        # Create a minimal HF state dict
        hf_state = {
            "model.embed_tokens.weight": torch.randn(32000, 4096),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(1024, 4096),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(1024, 4096),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(4096, 4096),
            "model.layers.0.self_attn.q_norm.weight": torch.randn(128),
            "model.layers.0.self_attn.k_norm.weight": torch.randn(128),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(11008, 4096),
            "model.layers.0.mlp.up_proj.weight": torch.randn(11008, 4096),
            "model.layers.0.mlp.down_proj.weight": torch.randn(4096, 11008),
            "model.layers.0.input_layernorm.weight": torch.randn(4096),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(4096),
            "model.norm.weight": torch.randn(4096),
            "lm_head.weight": torch.randn(32000, 4096),
        }

        # HF -> Archon
        archon_state = adapter.from_hf(hf_state)

        # Verify Archon keys
        expected_archon_keys = {
            "tok_embeddings.weight",
            "layers.0.attention.wq.weight",
            "layers.0.attention.wk.weight",
            "layers.0.attention.wv.weight",
            "layers.0.attention.wo.weight",
            "layers.0.attention.q_norm.weight",
            "layers.0.attention.k_norm.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w3.weight",
            "layers.0.feed_forward.w2.weight",
            "layers.0.attention_norm.weight",
            "layers.0.ffn_norm.weight",
            "norm.weight",
            "output.weight",
        }
        assert set(archon_state.keys()) == expected_archon_keys

        # Archon -> HF
        roundtrip_state = adapter.to_hf(archon_state)

        # Verify roundtrip
        for key in hf_state:
            if "rotary_emb" in key:
                continue
            assert key in roundtrip_state, f"Missing key: {key}"
            assert torch.allclose(hf_state[key], roundtrip_state[key]), (
                f"Mismatch at {key}"
            )

    def test_convert_single_to_hf(self, adapter):
        """Test single tensor conversion for weight updates."""
        # Test regular weight
        archon_name = "layers.0.attention.wq.weight"
        tensor = torch.randn(4096, 4096)

        result = adapter.convert_single_to_hf(archon_name, tensor)
        assert len(result) == 1
        assert result[0][0] == "model.layers.0.self_attn.q_proj.weight"
        assert torch.equal(result[0][1], tensor)

    def test_qwen3_dense_no_moe_converter(self, adapter):
        """Test that dense Qwen3 adapter does not initialize MoE converter."""
        assert adapter.moe_enabled is False
        assert adapter._moe_converter is None
        assert adapter._moe_state is None

    def test_qwen3_adapter_inherits_base_methods(self, adapter):
        """Test that Qwen3StateDictAdapter inherits from BaseStateDictAdapter."""
        from areal.experimental.models.archon.base import BaseStateDictAdapter

        assert isinstance(adapter, BaseStateDictAdapter)
        assert hasattr(adapter, "get_hf_storage_reader")
        assert hasattr(adapter, "fqn_to_index_mapping")

    def test_qwen3_dense_with_weight_tying(self):
        """Test Qwen3StateDictAdapter handles weight tying correctly for dense model."""

        class MockQwen3DenseConfigTied:
            model_type = "qwen3"
            tie_word_embeddings = True
            moe_enabled = False

        adapter = Qwen3StateDictAdapter(MockQwen3DenseConfigTied())

        archon_state = {
            "tok_embeddings.weight": torch.randn(1000, 64),
            "output.weight": torch.randn(1000, 64),  # Should be skipped
        }

        hf_state = adapter.to_hf(archon_state)

        # output.weight should be skipped when weight tying is enabled
        assert "lm_head.weight" not in hf_state
        assert "model.embed_tokens.weight" in hf_state


class TestQwen3StateDictAdapterMoE:
    """Tests for Qwen3StateDictAdapter with MoE models."""

    @pytest.fixture
    def adapter(self):
        return Qwen3StateDictAdapter(MockQwen3MoEConfig())

    def test_split_moe_experts(self, adapter):
        """Test splitting 3D expert weight into 2D weights."""
        # Create 3D weight: (num_experts=8, out=11008, in=4096)
        weight_3d = torch.randn(8, 11008, 4096)
        archon_key = "layers.0.moe.experts.w1"

        result = adapter._split_moe_experts(archon_key, weight_3d)

        assert len(result) == 8
        for i in range(8):
            hf_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
            assert hf_key in result
            assert result[hf_key].shape == (11008, 4096)
            assert torch.allclose(result[hf_key], weight_3d[i])

    def test_collect_expert_weights(self, adapter):
        """Test collecting 2D expert weights into 3D."""
        state_dict = {}
        buffer = {}

        # Simulate collecting all 8 experts
        for i in range(8):
            hf_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
            weight_2d = torch.randn(11008, 4096)
            adapter._collect_expert_weight(hf_key, weight_2d, buffer, state_dict)

        # After collecting all 8, should have 3D weight in state_dict
        assert "layers.0.moe.experts.w1" in state_dict
        assert state_dict["layers.0.moe.experts.w1"].shape == (8, 11008, 4096)
        assert "layers.0.moe.experts.w1" not in buffer

    def test_moe_roundtrip(self, adapter):
        """Test roundtrip conversion for MoE experts."""
        # Create 3D Archon weight
        archon_state = {
            "layers.0.moe.experts.w1": torch.randn(8, 11008, 4096),
            "layers.0.moe.experts.w2": torch.randn(8, 4096, 11008),
            "layers.0.moe.experts.w3": torch.randn(8, 11008, 4096),
        }

        # Archon -> HF
        hf_state = adapter.to_hf(archon_state)

        # Should have 24 keys (8 experts x 3 projections)
        assert len(hf_state) == 24

        # HF -> Archon
        roundtrip_state = adapter.from_hf(hf_state)

        # Verify roundtrip
        for key in archon_state:
            assert key in roundtrip_state
            assert torch.allclose(archon_state[key], roundtrip_state[key])

    def test_convert_single_moe_to_hf(self, adapter):
        """Test single MoE tensor conversion (splits into multiple)."""
        archon_name = "layers.0.moe.experts.w1"
        tensor = torch.randn(8, 11008, 4096)

        result = adapter.convert_single_to_hf(archon_name, tensor)

        # Should return 8 pairs (one per expert)
        assert len(result) == 8
        for i, (hf_name, hf_tensor) in enumerate(result):
            assert hf_name == f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
            assert hf_tensor.shape == (11008, 4096)

    def test_router_key_conversion(self, adapter):
        """Test MoE router gate key conversion."""
        hf_key = "model.layers.0.mlp.gate.weight"
        archon_key = "layers.0.moe.router.gate.weight"

        assert adapter._convert_key_from_hf(hf_key) == archon_key
        assert adapter._convert_key_to_hf(archon_key) == hf_key

    def test_full_moe_state_dict_roundtrip(self, adapter):
        """Test roundtrip for complete MoE layer state dict including router."""
        # Create full MoE layer state dict
        archon_state = {
            # Expert weights
            "layers.0.moe.experts.w1": torch.randn(8, 1024, 512),
            "layers.0.moe.experts.w2": torch.randn(8, 512, 1024),
            "layers.0.moe.experts.w3": torch.randn(8, 1024, 512),
            # Router
            "layers.0.moe.router.gate.weight": torch.randn(8, 512),
        }

        # Archon -> HF
        hf_state = adapter.to_hf(archon_state)

        # Should have 24 expert keys + 1 router key
        assert len(hf_state) == 25
        assert "model.layers.0.mlp.gate.weight" in hf_state

        # HF -> Archon
        roundtrip_state = adapter.from_hf(hf_state)

        # Verify roundtrip for experts
        for key in [
            "layers.0.moe.experts.w1",
            "layers.0.moe.experts.w2",
            "layers.0.moe.experts.w3",
        ]:
            assert key in roundtrip_state
            assert torch.allclose(archon_state[key], roundtrip_state[key])

        # Verify roundtrip for router
        router_key = "layers.0.moe.router.gate.weight"
        assert router_key in roundtrip_state
        assert torch.allclose(archon_state[router_key], roundtrip_state[router_key])

    def test_activation_checkpoint_wrapper_prefix_stripped(self, adapter):
        """Test that activation checkpoint wrapper prefix is stripped."""
        archon_name = "layers.0._checkpoint_wrapped_module.moe.experts.w1"
        tensor = torch.randn(8, 1024, 512)

        result = adapter.convert_single_to_hf(archon_name, tensor)

        # Should correctly convert despite wrapper prefix
        assert len(result) == 8
        assert result[0][0] == "model.layers.0.mlp.experts.0.gate_proj.weight"

    def test_torch_compile_prefix_stripped(self, adapter):
        """Test that torch.compile wrapper prefix is stripped."""
        archon_name = "layers.0._orig_mod.moe.router.gate.weight"
        tensor = torch.randn(8, 512)

        result = adapter.convert_single_to_hf(archon_name, tensor)

        assert len(result) == 1
        assert result[0][0] == "model.layers.0.mlp.gate.weight"


# =============================================================================
# Integration Tests: MoE Weight Loading (CPU/CUDA)
# =============================================================================


class TestMoEWeightLoadingRoundtrip:
    """Tests for MoE model weight save/load roundtrip."""

    @pytest.fixture
    def moe_model_args(self):
        """Create model args for MoE model."""
        from areal.experimental.models.archon.moe import MoEArgs
        from areal.experimental.models.archon.qwen3 import Qwen3ModelArgs

        return Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            n_layers=4,
            vocab_size=1000,
            max_seq_len=32,
            attn_type="sdpa",
            moe_enabled=True,
            moe_inter_dim=128,
            moe_args=MoEArgs(num_experts=4, top_k=2),
            decoder_sparse_step=2,  # Mixed MoE/dense layers
        )

    @pytest.fixture
    def moe_adapter_config(self):
        """Create mock config for adapter."""

        class MockConfig:
            model_type = "qwen3_moe"
            num_local_experts = 4
            tie_word_embeddings = False

        return MockConfig()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="MoE forward requires CUDA"
    )
    def test_moe_weight_roundtrip_forward_match(
        self, moe_model_args, moe_adapter_config
    ):
        """Test that model forward output matches after save/load roundtrip."""
        from areal.experimental.models.archon.qwen3 import Qwen3Model

        device = torch.device("cuda")

        # Create and initialize model
        model1 = Qwen3Model(moe_model_args)
        model1.init_weights()
        model1.init_buffers(buffer_device=device)
        model1.to(device)

        # Create adapter
        adapter = Qwen3StateDictAdapter(moe_adapter_config)

        # Model -> Archon state dict -> HF state dict
        archon_state = model1.state_dict()
        hf_state = adapter.to_hf(archon_state)

        # HF state dict -> Archon state dict -> Model
        # Note: strict=False because expert_bias is an Archon-only buffer
        # that doesn't exist in HF models (it's initialized to zeros)
        roundtrip_archon_state = adapter.from_hf(hf_state)
        model2 = Qwen3Model(moe_model_args)
        model2.init_buffers(buffer_device=device)
        model2.load_state_dict(roundtrip_archon_state, strict=False)
        model2.to(device)

        # Compare forward outputs
        torch.manual_seed(42)
        tokens = torch.randint(0, 1000, (1, 16), device=device)

        # Create packed input
        cu_seqlens = torch.tensor([0, 16], dtype=torch.int32, device=device)
        max_seqlen = 16
        positions = torch.arange(16, device=device).unsqueeze(0)

        with torch.no_grad():
            out1 = model1(
                tokens, positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
            out2 = model2(
                tokens, positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )

        assert torch.allclose(out1, out2, rtol=1e-5, atol=1e-5), (
            f"Forward output mismatch after roundtrip: max_diff={torch.abs(out1 - out2).max()}"
        )

    def test_moe_weight_keys_preserved(self, moe_model_args, moe_adapter_config):
        """Test that all weight keys are preserved in roundtrip."""
        from areal.experimental.models.archon.qwen3 import Qwen3Model

        model = Qwen3Model(moe_model_args)
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cpu"))

        archon_state = model.state_dict()
        adapter = Qwen3StateDictAdapter(moe_adapter_config)

        # Archon -> HF -> Archon
        hf_state = adapter.to_hf(archon_state)
        roundtrip_state = adapter.from_hf(hf_state)

        # expert_bias is an Archon-only buffer, not present in HF models
        # It's initialized to zeros and is not persisted in HF checkpoints
        archon_keys = {k for k in archon_state.keys() if "expert_bias" not in k}
        roundtrip_keys = set(roundtrip_state.keys())

        # Verify all keys preserved (excluding Archon-only buffers)
        missing_keys = archon_keys - roundtrip_keys
        extra_keys = roundtrip_keys - archon_keys

        assert not missing_keys, f"Missing keys after roundtrip: {missing_keys}"
        assert not extra_keys, f"Extra keys after roundtrip: {extra_keys}"

    def test_moe_weight_values_preserved(self, moe_model_args, moe_adapter_config):
        """Test that all weight values are preserved in roundtrip."""
        from areal.experimental.models.archon.qwen3 import Qwen3Model

        model = Qwen3Model(moe_model_args)
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cpu"))

        archon_state = model.state_dict()
        adapter = Qwen3StateDictAdapter(moe_adapter_config)

        # Archon -> HF -> Archon
        hf_state = adapter.to_hf(archon_state)
        roundtrip_state = adapter.from_hf(hf_state)

        # Verify all values match (excluding Archon-only buffers like expert_bias)
        for key in archon_state:
            if "expert_bias" in key:
                # expert_bias is an Archon-only buffer, not in HF models
                continue
            assert key in roundtrip_state, f"Missing key: {key}"
            original = archon_state[key]
            roundtrip = roundtrip_state[key]
            assert original.shape == roundtrip.shape, (
                f"Shape mismatch for {key}: {original.shape} vs {roundtrip.shape}"
            )
            assert torch.allclose(original, roundtrip), (
                f"Value mismatch for {key}: max_diff={torch.abs(original - roundtrip).max()}"
            )

    def test_mixed_moe_dense_layers_preserved(self, moe_model_args, moe_adapter_config):
        """Test roundtrip with mixed MoE and dense layers."""
        from areal.experimental.models.archon.qwen3 import Qwen3Model

        # decoder_sparse_step=2 means:
        # - layer 0: dense (FFN)
        # - layer 1: MoE
        # - layer 2: dense (FFN)
        # - layer 3: MoE
        model = Qwen3Model(moe_model_args)
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cpu"))

        # Verify layer structure
        assert model.layers["0"].moe is None
        assert model.layers["0"].feed_forward is not None
        assert model.layers["1"].moe is not None
        assert model.layers["1"].feed_forward is None
        assert model.layers["2"].moe is None
        assert model.layers["3"].moe is not None

        archon_state = model.state_dict()
        adapter = Qwen3StateDictAdapter(moe_adapter_config)

        # Verify both dense FFN and MoE keys exist
        dense_ffn_key = "layers.0.feed_forward.w1.weight"
        moe_expert_key = "layers.1.moe.experts.w1"
        moe_router_key = "layers.1.moe.router.gate.weight"

        assert dense_ffn_key in archon_state
        assert moe_expert_key in archon_state
        assert moe_router_key in archon_state

        # Roundtrip
        hf_state = adapter.to_hf(archon_state)
        roundtrip_state = adapter.from_hf(hf_state)

        # Verify both types preserved
        assert dense_ffn_key in roundtrip_state
        assert moe_expert_key in roundtrip_state
        assert moe_router_key in roundtrip_state

        # Verify values
        assert torch.allclose(
            archon_state[dense_ffn_key], roundtrip_state[dense_ffn_key]
        )
        assert torch.allclose(
            archon_state[moe_expert_key], roundtrip_state[moe_expert_key]
        )
        assert torch.allclose(
            archon_state[moe_router_key], roundtrip_state[moe_router_key]
        )


# =============================================================================
# Lightweight Config-Only Tests (No Weight Loading, Fast)
# =============================================================================


@pytest.mark.slow
class TestStateDictAdapterWithRealConfigs:
    """Lightweight tests using real HF configs but no weight loading.

    These tests are fast (seconds) because they only load configs, not weights.
    Use these to verify adapter logic for all supported model types.
    """

    @pytest.fixture(scope="class")
    def model_configs(self):
        """Load configs for all model types that have paths configured."""
        from transformers import AutoConfig

        from areal.tests.experimental.archon.utils import (
            MODEL_PATHS,
        )

        configs = {}
        for model_type, path in MODEL_PATHS.items():
            if path is not None:
                try:
                    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
                    configs[model_type] = config
                except Exception:
                    pass  # Skip if config can't be loaded
        return configs

    def test_adapter_key_mapping_for_all_models(self, model_configs):
        """Test key mapping for all configured model types."""
        from areal.experimental.models.archon import get_model_spec

        for model_type, config in model_configs.items():
            spec = get_model_spec(model_type)
            adapter = spec.state_dict_adapter_class(config)

            # Test basic keys that should work for all models
            basic_hf_keys = [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.norm.weight",
            ]

            for hf_key in basic_hf_keys:
                archon_key = adapter._convert_key_from_hf(hf_key)
                assert archon_key is not None, (
                    f"[{model_type}] Failed to convert HF key: {hf_key}"
                )
                hf_key_back = adapter._convert_key_to_hf(archon_key)
                assert hf_key_back == hf_key, (
                    f"[{model_type}] Roundtrip failed: {hf_key} -> {archon_key} -> {hf_key_back}"
                )

    def test_moe_adapter_expert_split_with_real_config(self, model_configs):
        """Test MoE expert splitting with real config (no weights)."""
        from areal.experimental.models.archon import get_model_spec

        for model_type, config in model_configs.items():
            # Check if MoE
            num_experts = getattr(config, "num_experts", None)
            if num_experts is None:
                num_experts = getattr(config, "num_local_experts", None)
            if num_experts is None or num_experts <= 1:
                continue

            spec = get_model_spec(model_type)
            adapter = spec.state_dict_adapter_class(config)

            # Create small fake 3D weight
            out_dim, in_dim = 16, 32
            fake_weight = torch.randn(num_experts, out_dim, in_dim)

            # Test split
            for weight_name, hf_proj in [
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]:
                archon_key = f"layers.0.moe.experts.{weight_name}"
                result = adapter._split_moe_experts(archon_key, fake_weight)

                assert len(result) == num_experts, (
                    f"[{model_type}] Expected {num_experts} experts, got {len(result)}"
                )

                for expert_id in range(num_experts):
                    expected_key = (
                        f"model.layers.0.mlp.experts.{expert_id}.{hf_proj}.weight"
                    )
                    assert expected_key in result, (
                        f"[{model_type}] Missing {expected_key}"
                    )

    def test_moe_adapter_expert_collect_with_real_config(self, model_configs):
        """Test MoE expert collection with real config (no weights)."""
        from areal.experimental.models.archon import get_model_spec

        for model_type, config in model_configs.items():
            # Check if MoE
            num_experts = getattr(config, "num_experts", None)
            if num_experts is None:
                num_experts = getattr(config, "num_local_experts", None)
            if num_experts is None or num_experts <= 1:
                continue

            spec = get_model_spec(model_type)
            adapter = spec.state_dict_adapter_class(config)

            # Create fake 2D weights and collect them
            out_dim, in_dim = 16, 32

            for hf_proj, archon_weight_name in [
                ("gate_proj", "w1"),
                ("down_proj", "w2"),
                ("up_proj", "w3"),
            ]:
                fake_weights = [
                    torch.randn(out_dim, in_dim) for _ in range(num_experts)
                ]
                buffer = {}
                state_dict = {}

                for expert_id in range(num_experts):
                    hf_key = f"model.layers.0.mlp.experts.{expert_id}.{hf_proj}.weight"
                    adapter._collect_expert_weight(
                        hf_key, fake_weights[expert_id], buffer, state_dict
                    )

                expected_archon_key = f"layers.0.moe.experts.{archon_weight_name}"
                assert expected_archon_key in state_dict, (
                    f"[{model_type}] Expected {expected_archon_key} in state_dict"
                )
                assert state_dict[expected_archon_key].shape == (
                    num_experts,
                    out_dim,
                    in_dim,
                ), f"[{model_type}] Wrong shape for {expected_archon_key}"

    def test_weight_tying_matches_config(self, model_configs):
        """Test that adapter weight tying matches HF config."""
        from areal.experimental.models.archon import get_model_spec

        for model_type, config in model_configs.items():
            spec = get_model_spec(model_type)
            adapter = spec.state_dict_adapter_class(config)

            expected_tying = getattr(config, "tie_word_embeddings", False)
            assert adapter.enable_weight_tying == expected_tying, (
                f"[{model_type}] Weight tying mismatch: "
                f"adapter={adapter.enable_weight_tying}, config={expected_tying}"
            )

    def test_all_layer_indices_handled(self, model_configs):
        """Test that adapter handles all layer indices correctly."""
        from areal.experimental.models.archon import get_model_spec

        for model_type, config in model_configs.items():
            spec = get_model_spec(model_type)
            adapter = spec.state_dict_adapter_class(config)

            num_layers = getattr(config, "num_hidden_layers", 12)
            test_layers = [0, 1, num_layers // 2, num_layers - 1]

            for layer_idx in test_layers:
                hf_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
                archon_key = adapter._convert_key_from_hf(hf_key)

                assert archon_key is not None, (
                    f"[{model_type}] Failed for layer {layer_idx}"
                )
                assert f"layers.{layer_idx}." in archon_key

                hf_key_back = adapter._convert_key_to_hf(archon_key)
                assert hf_key_back == hf_key, (
                    f"[{model_type}] Roundtrip failed for layer {layer_idx}"
                )


# =============================================================================
# Tests for HF Assets Path and Multi-File Checkpoint Support
# =============================================================================


class TestHFAssetsPathSupport:
    """Tests for hf_assets_path parameter and multi-file checkpoint support."""

    @pytest.fixture
    def mock_safetensors_index(self, tmp_path):
        """Create a mock model.safetensors.index.json file."""
        index_data = {
            "metadata": {"total_size": 12345678},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00019.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00019.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00019.safetensors",
                "model.layers.0.self_attn.v_proj.weight": "model-00002-of-00019.safetensors",
                "model.layers.0.self_attn.o_proj.weight": "model-00002-of-00019.safetensors",
                "model.layers.0.mlp.gate_proj.weight": "model-00003-of-00019.safetensors",
                "model.layers.0.mlp.up_proj.weight": "model-00003-of-00019.safetensors",
                "model.layers.0.mlp.down_proj.weight": "model-00004-of-00019.safetensors",
                "model.layers.10.self_attn.q_proj.weight": "model-00010-of-00019.safetensors",
                "model.norm.weight": "model-00019-of-00019.safetensors",
                "lm_head.weight": "model-00019-of-00019.safetensors",
            },
        }
        index_path = tmp_path / "model.safetensors.index.json"
        import json

        with open(index_path, "w") as f:
            json.dump(index_data, f)
        return tmp_path

    def test_init_without_hf_assets_path(self):
        """Test that adapter initializes correctly without hf_assets_path."""
        adapter = Qwen3StateDictAdapter(MockQwen3Config())

        assert adapter.hf_assets_path is None
        assert adapter.fqn_to_index_mapping is None

    def test_init_with_hf_assets_path(self, mock_safetensors_index):
        """Test that adapter initializes correctly with hf_assets_path."""
        adapter = Qwen3StateDictAdapter(
            MockQwen3Config(), hf_assets_path=str(mock_safetensors_index)
        )

        assert adapter.hf_assets_path == str(mock_safetensors_index)
        assert adapter.fqn_to_index_mapping is not None

    def test_load_safetensors_index_mapping(self, mock_safetensors_index):
        """Test that fqn_to_index_mapping is correctly populated."""
        adapter = Qwen3StateDictAdapter(
            MockQwen3Config(), hf_assets_path=str(mock_safetensors_index)
        )

        mapping = adapter.fqn_to_index_mapping
        assert mapping is not None

        # Verify index extraction from filenames
        assert mapping["model.embed_tokens.weight"] == 1
        assert mapping["model.layers.0.self_attn.q_proj.weight"] == 1
        assert mapping["model.layers.0.self_attn.v_proj.weight"] == 2
        assert mapping["model.layers.0.mlp.gate_proj.weight"] == 3
        assert mapping["model.layers.0.mlp.down_proj.weight"] == 4
        assert mapping["model.layers.10.self_attn.q_proj.weight"] == 10
        assert mapping["model.norm.weight"] == 19
        assert mapping["lm_head.weight"] == 19

    def test_load_safetensors_index_missing_file(self, tmp_path):
        """Test that missing index file results in None mapping."""
        adapter = Qwen3StateDictAdapter(MockQwen3Config(), hf_assets_path=str(tmp_path))
        # When index file is missing, fqn_to_index_mapping should be None
        assert adapter.fqn_to_index_mapping is None

    def test_get_hf_storage_reader(self, tmp_path):
        """Test that get_hf_storage_reader returns correct type."""
        from torch.distributed.checkpoint import HuggingFaceStorageReader

        adapter = Qwen3StateDictAdapter(MockQwen3Config())
        reader = adapter.get_hf_storage_reader(str(tmp_path))

        assert isinstance(reader, HuggingFaceStorageReader)

    def test_backward_compatibility_no_hf_assets_path(self):
        """Test that existing code without hf_assets_path still works."""
        # This should work exactly as before
        adapter = Qwen3StateDictAdapter(MockQwen3Config())

        # All existing functionality should work
        hf_key = "model.embed_tokens.weight"
        archon_key = adapter._convert_key_from_hf(hf_key)
        assert archon_key == "tok_embeddings.weight"

        hf_key_back = adapter._convert_key_to_hf(archon_key)
        assert hf_key_back == hf_key

    def test_qwen2_adapter_with_hf_assets_path(self, mock_safetensors_index):
        """Test that Qwen2StateDictAdapter also supports hf_assets_path."""
        from areal.experimental.models.archon.qwen2 import Qwen2StateDictAdapter

        adapter = Qwen2StateDictAdapter(
            MockQwen3Config(), hf_assets_path=str(mock_safetensors_index)
        )

        assert adapter.hf_assets_path == str(mock_safetensors_index)
        assert adapter.fqn_to_index_mapping is not None
        assert adapter.fqn_to_index_mapping["model.embed_tokens.weight"] == 1


# =============================================================================
# Tests for MoEWeightConverter Helper Methods
# =============================================================================


class TestMoEWeightConverter:
    """Tests for MoEWeightConverter methods."""

    @pytest.fixture
    def moe_converter(self):
        from areal.experimental.models.archon.moe_weight_converter import (
            MoEWeightConverter,
        )

        return MoEWeightConverter

    def test_calculate_strided_shard_shard_indices_basic(self, moe_converter):
        """Test calculate_strided_shard_indices with basic inputs."""
        # 8 experts split as [StridedShard(2), Shard(2)]
        # Layout:
        #   StridedShard rank 0, Shard rank 0 -> experts 0,1 (block 0, position 0)
        #   StridedShard rank 1, Shard rank 0 -> experts 2,3 (block 1, position 0)
        #   StridedShard rank 0, Shard rank 1 -> experts 4,5 (block 0, position 1)
        #   StridedShard rank 1, Shard rank 1 -> experts 6,7 (block 1, position 1)

        # GPU (0,0): experts 0,1
        start, end = moe_converter.calculate_strided_shard_indices(
            strided_shard_dim_degree=2,
            strided_shard_dim_rank=0,
            shard_dim_degree=2,
            shard_dim_rank=0,
            dim_size_to_split=8,
        )
        assert start == 0
        assert end == 2

        # GPU (1,0): experts 2,3
        start, end = moe_converter.calculate_strided_shard_indices(
            strided_shard_dim_degree=2,
            strided_shard_dim_rank=1,
            shard_dim_degree=2,
            shard_dim_rank=0,
            dim_size_to_split=8,
        )
        assert start == 2
        assert end == 4

        # GPU (0,1): experts 4,5
        start, end = moe_converter.calculate_strided_shard_indices(
            strided_shard_dim_degree=2,
            strided_shard_dim_rank=0,
            shard_dim_degree=2,
            shard_dim_rank=1,
            dim_size_to_split=8,
        )
        assert start == 4
        assert end == 6

        # GPU (1,1): experts 6,7
        start, end = moe_converter.calculate_strided_shard_indices(
            strided_shard_dim_degree=2,
            strided_shard_dim_rank=1,
            shard_dim_degree=2,
            shard_dim_rank=1,
            dim_size_to_split=8,
        )
        assert start == 6
        assert end == 8

    def test_calculate_strided_shard_shard_indices_uneven_split_error(
        self, moe_converter
    ):
        """Test that uneven split raises ValueError."""
        # 8 experts cannot be evenly split by strided_degree=3 and shard_degree=2
        # because 8 / (3 * 2) = 8/6 is not an integer
        with pytest.raises(ValueError, match="Cannot evenly split"):
            moe_converter.calculate_strided_shard_indices(
                strided_shard_dim_degree=3,
                strided_shard_dim_rank=0,
                shard_dim_degree=2,
                shard_dim_rank=0,
                dim_size_to_split=8,
            )

    def test_calculate_strided_shard_shard_indices_single_shard(self, moe_converter):
        """Test with shard_degree=1 (no additional sharding)."""
        # 4 experts split only by StridedShard(4)
        # GPU 0: experts 0
        # GPU 1: experts 1
        # GPU 2: experts 2
        # GPU 3: experts 3
        for rank in range(4):
            start, end = moe_converter.calculate_strided_shard_indices(
                strided_shard_dim_degree=4,
                strided_shard_dim_rank=rank,
                shard_dim_degree=1,
                shard_dim_rank=0,
                dim_size_to_split=4,
            )
            assert start == rank
            assert end == rank + 1

    def test_calculate_strided_shard_shard_indices_edge_case_64_experts(
        self, moe_converter
    ):
        """Test with 64 experts (common MoE configuration)."""
        # 64 experts split as [StridedShard(4), Shard(4)] across 16 GPUs
        # Block size = 64 / (4 * 4) = 4 experts per GPU

        # GPU (0,0): experts 0-3
        start, end = moe_converter.calculate_strided_shard_indices(
            strided_shard_dim_degree=4,
            strided_shard_dim_rank=0,
            shard_dim_degree=4,
            shard_dim_rank=0,
            dim_size_to_split=64,
        )
        assert start == 0
        assert end == 4

        # GPU (3,3): experts 60-63
        start, end = moe_converter.calculate_strided_shard_indices(
            strided_shard_dim_degree=4,
            strided_shard_dim_rank=3,
            shard_dim_degree=4,
            shard_dim_rank=3,
            dim_size_to_split=64,
        )
        assert start == 60
        assert end == 64


# =============================================================================
# Tests for Qwen2StateDictAdapter Dense Model
# =============================================================================


class TestQwen2StateDictAdapterDense:
    """Tests for Qwen2StateDictAdapter with dense models."""

    @pytest.fixture
    def adapter(self):
        from areal.experimental.models.archon.qwen2 import Qwen2StateDictAdapter

        class MockQwen2Config:
            model_type = "qwen2"
            tie_word_embeddings = False

        return Qwen2StateDictAdapter(MockQwen2Config())

    def test_to_hf_no_full_tensor_call(self, adapter):
        """Test that to_hf() passes tensors through without modification."""
        # Create simple state dict
        archon_state = {
            "tok_embeddings.weight": torch.randn(1000, 64),
            "layers.0.attention.wq.weight": torch.randn(64, 64),
            "norm.weight": torch.randn(64),
            "output.weight": torch.randn(1000, 64),
        }

        hf_state = adapter.to_hf(archon_state)

        # Verify all keys are converted and values are identical (same tensor object)
        assert "model.embed_tokens.weight" in hf_state
        assert torch.equal(
            hf_state["model.embed_tokens.weight"], archon_state["tok_embeddings.weight"]
        )

    def test_convert_single_to_hf_no_full_tensor_call(self, adapter):
        """Test that convert_single_to_hf() returns tensor as-is."""
        name = "layers.0.attention.wq.weight"
        tensor = torch.randn(64, 64)

        result = adapter.convert_single_to_hf(name, tensor)

        assert len(result) == 1
        hf_name, hf_tensor = result[0]
        assert hf_name == "model.layers.0.self_attn.q_proj.weight"
        # Should be the same tensor object (no copy or full_tensor)
        assert hf_tensor is tensor

    def test_roundtrip_preserves_values(self, adapter):
        """Test that roundtrip preserves all weight values."""
        archon_state = {
            "tok_embeddings.weight": torch.randn(1000, 64),
            "layers.0.attention.wq.weight": torch.randn(64, 64),
            "layers.0.attention.wk.weight": torch.randn(16, 64),
            "layers.0.attention.wv.weight": torch.randn(16, 64),
            "layers.0.attention.wo.weight": torch.randn(64, 64),
            "layers.0.feed_forward.w1.weight": torch.randn(128, 64),
            "layers.0.feed_forward.w3.weight": torch.randn(128, 64),
            "layers.0.feed_forward.w2.weight": torch.randn(64, 128),
            "layers.0.attention_norm.weight": torch.randn(64),
            "layers.0.ffn_norm.weight": torch.randn(64),
            "norm.weight": torch.randn(64),
            "output.weight": torch.randn(1000, 64),
        }

        # Archon -> HF -> Archon
        hf_state = adapter.to_hf(archon_state)
        roundtrip_state = adapter.from_hf(hf_state)

        # Verify all values preserved
        for key in archon_state:
            assert key in roundtrip_state, f"Missing key: {key}"
            assert torch.allclose(archon_state[key], roundtrip_state[key]), (
                f"Value mismatch for {key}"
            )

    def test_qwen2_adapter_inherits_base_methods(self, adapter):
        """Test that Qwen2StateDictAdapter inherits from BaseStateDictAdapter."""
        from areal.experimental.models.archon.base import BaseStateDictAdapter

        assert isinstance(adapter, BaseStateDictAdapter)
        assert hasattr(adapter, "get_hf_storage_reader")
        assert hasattr(adapter, "fqn_to_index_mapping")

    def test_qwen2_adapter_with_weight_tying(self):
        """Test Qwen2StateDictAdapter handles weight tying correctly."""
        from areal.experimental.models.archon.qwen2 import Qwen2StateDictAdapter

        class MockQwen2ConfigTied:
            model_type = "qwen2"
            tie_word_embeddings = True

        adapter = Qwen2StateDictAdapter(MockQwen2ConfigTied())

        archon_state = {
            "tok_embeddings.weight": torch.randn(1000, 64),
            "output.weight": torch.randn(1000, 64),  # Should be skipped
        }

        hf_state = adapter.to_hf(archon_state)

        # output.weight should be skipped when weight tying is enabled
        assert "lm_head.weight" not in hf_state
        assert "model.embed_tokens.weight" in hf_state
