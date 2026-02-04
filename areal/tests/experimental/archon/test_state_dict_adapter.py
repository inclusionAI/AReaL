"""Tests for Archon State Dict Adapter roundtrip conversion.

These tests verify key conversion between HuggingFace and Archon formats
by testing full roundtrip conversions (HF -> Archon -> HF).

Run tests:
    pytest areal/tests/experimental/archon/test_state_dict_adapter.py -v

Note: For E2E weight sync tests with real models, see test_weight_sync.py
and test_checkpoint_e2e.py.
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
    tie_word_embeddings = False


class MockQwen3MoEConfig:
    """Mock Qwen3 MoE model config for testing."""

    model_type = "qwen3_moe"
    num_local_experts = 8
    tie_word_embeddings = False


# =============================================================================
# Dense Model Roundtrip Tests
# =============================================================================


class TestQwen3StateDictAdapterDense:
    """Roundtrip tests for Qwen3StateDictAdapter with dense models."""

    @pytest.fixture
    def adapter(self):
        return Qwen3StateDictAdapter(MockQwen3Config())

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


# =============================================================================
# MoE Model Roundtrip Tests
# =============================================================================


class TestQwen3StateDictAdapterMoE:
    """Roundtrip tests for Qwen3StateDictAdapter with MoE models."""

    @pytest.fixture
    def adapter(self):
        return Qwen3StateDictAdapter(MockQwen3MoEConfig())

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


# =============================================================================
# MoE Model Forward Match Test with Adapter Roundtrip
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoEWeightLoadingRoundtrip:
    """Test MoE weight loading roundtrip with forward pass verification."""

    def test_moe_weight_roundtrip_forward_match(self):
        """Test MoE forward output matches after Archon->HF->Archon roundtrip.

        This test verifies that the state dict adapter correctly converts
        MoE weights between Archon and HuggingFace formats by:
        1. Creating a model and running forward pass
        2. Converting state dict: Archon -> HF -> Archon via adapter
        3. Loading converted weights into a new model
        4. Verifying forward outputs match
        """
        from areal.experimental.models.archon.moe import MoEArgs
        from areal.experimental.models.archon.qwen3 import Qwen3Model, Qwen3ModelArgs

        # Create MoE model args
        model_args = Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            moe_inter_dim=96,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=32,
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False),
            decoder_sparse_step=1,  # All layers are MoE
            attn_type="sdpa",
        )

        # Create and initialize first model
        model1 = Qwen3Model(model_args).cuda()
        model1.init_weights()
        model1.init_buffers(buffer_device=torch.device("cuda"))

        # Get original Archon state dict
        archon_state = {k: v.clone() for k, v in model1.state_dict().items()}

        # Create adapter config that matches the model's num_experts
        class AdapterConfig:
            model_type = "qwen3_moe"
            num_local_experts = 4  # Must match MoEArgs(num_experts=4)
            tie_word_embeddings = False

        # Create adapter and perform roundtrip conversion
        adapter = Qwen3StateDictAdapter(AdapterConfig())

        # Archon -> HF -> Archon
        hf_state = adapter.to_hf(archon_state)
        roundtrip_state = adapter.from_hf(hf_state)

        # Create second model and load roundtrip weights
        # Note: strict=False because expert_bias is an Archon-only buffer
        model2 = Qwen3Model(model_args).cuda()
        model2.init_buffers(buffer_device=torch.device("cuda"))
        model2.load_state_dict(roundtrip_state, strict=False)

        # Create test input (packed sequence format)
        torch.manual_seed(42)
        tokens = torch.randint(0, 1000, (1, 16), device="cuda")
        positions = torch.arange(16, device="cuda").unsqueeze(0)
        cu_seqlens = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
        max_seqlen = 16

        # Forward pass on both models
        with torch.no_grad():
            output1 = model1(
                tokens, positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
            output2 = model2(
                tokens, positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )

        # Verify outputs match
        assert torch.allclose(output1, output2, atol=1e-5, rtol=1e-5), (
            f"Forward output mismatch after adapter roundtrip. "
            f"Max diff: {(output1 - output2).abs().max()}"
        )
