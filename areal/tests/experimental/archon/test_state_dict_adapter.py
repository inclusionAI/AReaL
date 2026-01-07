"""Tests for Archon State Dict Adapter and weight conversion.

These tests verify:
1. Key conversion between HuggingFace and Archon formats
2. Dense and MoE model weight conversion
3. Roundtrip conversion consistency
4. Weight values match after loading

Run tests:
    pytest areal/tests/experimental/archon/test_state_dict_adapter.py -v

Note: Some tests require GPU and are marked as slow.
"""

import pytest
import torch

from areal.experimental.models.archon.qwen3.model.state_dict_adapter import (
    Qwen3StateDictAdapter,
)

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


# =============================================================================
# Integration Tests: Weight Loading (GPU required)
# =============================================================================

# Skip if no CUDA available
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@cuda_required
@pytest.mark.slow
def test_archon_weights_match_hf():
    """Verify Archon weight conversion from HuggingFace is correct.

    This test loads actual models and compares weights to ensure
    the state dict adapter correctly converts between formats.
    """
    from areal.tests.experimental.archon.utils import (
        MODEL_PATHS,
        load_archon_model,
        load_hf_model,
        setup_environment,
    )

    setup_environment()

    model_path = MODEL_PATHS["qwen2"]
    dtype = torch.bfloat16

    hf_model = load_hf_model(model_path, dtype=dtype)
    archon_model, _ = load_archon_model(model_path, dtype=dtype)

    # Key mappings: archon_key -> hf_key
    key_mappings = [
        ("tok_embeddings.weight", "model.embed_tokens.weight"),
        ("layers.0.attention.wq.weight", "model.layers.0.self_attn.q_proj.weight"),
        ("layers.0.attention.wk.weight", "model.layers.0.self_attn.k_proj.weight"),
        ("layers.0.attention.wv.weight", "model.layers.0.self_attn.v_proj.weight"),
        ("layers.0.attention.wo.weight", "model.layers.0.self_attn.o_proj.weight"),
        ("layers.0.feed_forward.w1.weight", "model.layers.0.mlp.gate_proj.weight"),
        ("layers.0.feed_forward.w2.weight", "model.layers.0.mlp.down_proj.weight"),
        ("layers.0.feed_forward.w3.weight", "model.layers.0.mlp.up_proj.weight"),
        ("norm.weight", "model.norm.weight"),
        ("output.weight", "lm_head.weight"),
    ]

    archon_params = dict(archon_model.named_parameters())
    hf_params = dict(hf_model.named_parameters())

    for archon_key, hf_key in key_mappings:
        if archon_key not in archon_params or hf_key not in hf_params:
            continue

        archon_w = archon_params[archon_key].data
        hf_w = hf_params[hf_key].data

        assert archon_w.shape == hf_w.shape, (
            f"Shape mismatch for {archon_key}: {archon_w.shape} vs {hf_w.shape}"
        )

        max_diff = (archon_w.float() - hf_w.float()).abs().max().item()
        assert max_diff < 1e-5, f"Weight mismatch for {archon_key}: max_diff={max_diff}"
