"""Tests for Qwen3Model with MoE support.

Run tests:
    pytest areal/tests/experimental/archon/test_qwen3_model_moe.py -v
"""

import pytest
import torch

from areal.experimental.models.archon.moe import MoEArgs
from areal.experimental.models.archon.qwen3 import Qwen3Model, Qwen3ModelArgs
from areal.experimental.models.archon.qwen3.model.model import (
    TransformerBlock,
    _is_moe_layer,
)
from areal.experimental.models.archon.qwen3.model.rope import precompute_rope_cache


def _create_packed_input(
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Create packed sequence inputs from a batch of tokens.

    Args:
        tokens: Token tensor of shape [batch, seq_len]

    Returns:
        Tuple of (tokens_packed, positions, cu_seqlens, max_seqlen):
        - tokens_packed: Shape [1, batch * seq_len]
        - positions: Shape [1, batch * seq_len]
        - cu_seqlens: Shape [batch + 1]
        - max_seqlen: int
    """
    bs, seq_len = tokens.shape
    device = tokens.device
    tokens_packed = tokens.view(1, bs * seq_len)
    # Create positions: each sequence has positions 0, 1, 2, ..., seq_len-1
    positions = (
        torch.arange(seq_len, device=device)
        .unsqueeze(0)
        .repeat(bs, 1)
        .view(1, bs * seq_len)
    )
    cu_seqlens = torch.tensor(
        [i * seq_len for i in range(bs + 1)], dtype=torch.int32, device=device
    )
    return tokens_packed, positions, cu_seqlens, seq_len


class TestIsMoeLayer:
    """Tests for _is_moe_layer helper function."""

    def test_moe_disabled(self):
        """Test that all layers are dense when moe_enabled is False."""
        args = Qwen3ModelArgs(moe_enabled=False, n_layers=10)
        for i in range(10):
            assert _is_moe_layer(i, args) is False

    def test_moe_args_none(self):
        """Test that all layers are dense when moe_args is None."""
        args = Qwen3ModelArgs(moe_enabled=True, moe_args=None, n_layers=10)
        for i in range(10):
            assert _is_moe_layer(i, args) is False

    def test_sparse_step_zero(self):
        """Test that all layers are dense when decoder_sparse_step is 0."""
        args = Qwen3ModelArgs(
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=8),
            decoder_sparse_step=0,
            n_layers=10,
        )
        for i in range(10):
            assert _is_moe_layer(i, args) is False

    def test_sparse_step_negative(self):
        """Test that all layers are dense when decoder_sparse_step is negative."""
        args = Qwen3ModelArgs(
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=8),
            decoder_sparse_step=-1,
            n_layers=10,
        )
        for i in range(10):
            assert _is_moe_layer(i, args) is False

    def test_all_moe_layers(self):
        """Test that all layers are MoE when decoder_sparse_step is 1."""
        args = Qwen3ModelArgs(
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=8),
            decoder_sparse_step=1,
            n_layers=10,
        )
        for i in range(10):
            assert _is_moe_layer(i, args) is True

    def test_alternating_layers(self):
        """Test alternating MoE layers when decoder_sparse_step is 2."""
        args = Qwen3ModelArgs(
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=8),
            decoder_sparse_step=2,
            n_layers=10,
        )
        # Layers 1, 3, 5, 7, 9 are MoE (1-indexed: 2, 4, 6, 8, 10)
        assert _is_moe_layer(0, args) is False
        assert _is_moe_layer(1, args) is True
        assert _is_moe_layer(2, args) is False
        assert _is_moe_layer(3, args) is True
        assert _is_moe_layer(4, args) is False
        assert _is_moe_layer(5, args) is True
        assert _is_moe_layer(6, args) is False
        assert _is_moe_layer(7, args) is True
        assert _is_moe_layer(8, args) is False
        assert _is_moe_layer(9, args) is True

    def test_every_third_layer(self):
        """Test MoE on every third layer when decoder_sparse_step is 3."""
        args = Qwen3ModelArgs(
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=8),
            decoder_sparse_step=3,
            n_layers=12,
        )
        # Layers 2, 5, 8, 11 are MoE (1-indexed: 3, 6, 9, 12)
        expected = [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
        ]
        for i in range(12):
            assert _is_moe_layer(i, args) is expected[i], f"layer {i}"


class TestTransformerBlockMoE:
    """Tests for TransformerBlock with MoE."""

    @pytest.fixture
    def dense_args(self):
        """Model args for dense FFN."""
        return Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=64,
            moe_enabled=False,
            attn_type="sdpa",  # Use SDPA for testing (varlen requires batch=1)
        )

    @pytest.fixture
    def moe_args(self):
        """Model args for MoE."""
        return Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            moe_inter_dim=96,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=64,
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False),
            decoder_sparse_step=1,
            attn_type="sdpa",  # Use SDPA for testing (varlen requires batch=1)
        )

    def test_dense_block_creation(self, dense_args):
        """Test TransformerBlock creates dense FFN when MoE is disabled."""
        block = TransformerBlock(layer_id=0, model_args=dense_args)

        assert block.moe_enabled is False
        assert block.moe is None
        assert block.feed_forward is not None

    def test_moe_block_creation(self, moe_args):
        """Test TransformerBlock creates MoE when MoE is enabled."""
        block = TransformerBlock(layer_id=0, model_args=moe_args)

        assert block.moe_enabled is True
        assert block.moe is not None
        assert block.feed_forward is None

    def test_dense_block_forward(self, dense_args):
        """Test forward pass for dense TransformerBlock."""
        block = TransformerBlock(layer_id=0, model_args=dense_args)
        block.init_weights()
        block.init_buffers(buffer_device=torch.device("cpu"))

        bs, seq_len = 2, 8
        x = torch.randn(bs, seq_len, dense_args.dim)
        # Use proper rope_cache shape: [max_seq_len, head_dim * 2]
        rope_cache = precompute_rope_cache(
            dense_args.head_dim, dense_args.max_seq_len, dense_args.rope_theta
        )
        max_seqlen = seq_len

        # Need to reshape x to [1, bs*seq_len, dim] for packed format
        x_packed = x.view(1, bs * seq_len, dense_args.dim)
        cu_seqlens_packed = torch.tensor(
            [0] + [i * seq_len for i in range(1, bs + 1)], dtype=torch.int32
        )
        # Create positions for packed sequences
        positions = (
            torch.arange(seq_len).unsqueeze(0).repeat(bs, 1).view(1, bs * seq_len)
        )

        out = block(
            x_packed,
            rope_cache,
            positions=positions,
            cu_seqlens=cu_seqlens_packed,
            max_seqlen=max_seqlen,
        )

        assert out.shape == x_packed.shape
        assert not torch.any(torch.isnan(out))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_moe_block_forward(self, moe_args):
        """Test forward pass for MoE TransformerBlock."""
        block = TransformerBlock(layer_id=0, model_args=moe_args).cuda()
        block.init_weights()
        block.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 2, 8
        x = torch.randn(bs, seq_len, moe_args.dim, device="cuda")
        # Use proper rope_cache shape: [max_seq_len, head_dim * 2]
        rope_cache = precompute_rope_cache(
            moe_args.head_dim, moe_args.max_seq_len, moe_args.rope_theta
        ).cuda()
        # Create cu_seqlens for packed sequences
        cu_seqlens_packed = torch.tensor(
            [0] + [i * seq_len for i in range(1, bs + 1)],
            dtype=torch.int32,
            device="cuda",
        )
        max_seqlen = seq_len

        # Reshape x to [1, bs*seq_len, dim] for packed format
        x_packed = x.view(1, bs * seq_len, moe_args.dim)
        # Create positions for packed sequences
        positions = (
            torch.arange(seq_len, device="cuda")
            .unsqueeze(0)
            .repeat(bs, 1)
            .view(1, bs * seq_len)
        )

        out = block(
            x_packed,
            rope_cache,
            positions=positions,
            cu_seqlens=cu_seqlens_packed,
            max_seqlen=max_seqlen,
        )

        assert out.shape == x_packed.shape
        assert not torch.any(torch.isnan(out))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_moe_block_gradient_flow(self, moe_args):
        """Test gradient flow through MoE TransformerBlock."""
        block = TransformerBlock(layer_id=0, model_args=moe_args).cuda()
        block.init_weights()
        block.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 2, 8
        x = torch.randn(bs, seq_len, moe_args.dim, device="cuda", requires_grad=True)
        # Use proper rope_cache shape: [max_seq_len, head_dim * 2]
        rope_cache = precompute_rope_cache(
            moe_args.head_dim, moe_args.max_seq_len, moe_args.rope_theta
        ).cuda()
        # Create cu_seqlens for packed sequences
        cu_seqlens_packed = torch.tensor(
            [0] + [i * seq_len for i in range(1, bs + 1)],
            dtype=torch.int32,
            device="cuda",
        )
        max_seqlen = seq_len

        # Reshape x to [1, bs*seq_len, dim] for packed format
        x_packed = x.view(1, bs * seq_len, moe_args.dim)
        # Create positions for packed sequences
        positions = (
            torch.arange(seq_len, device="cuda")
            .unsqueeze(0)
            .repeat(bs, 1)
            .view(1, bs * seq_len)
        )

        out = block(
            x_packed,
            rope_cache,
            positions=positions,
            cu_seqlens=cu_seqlens_packed,
            max_seqlen=max_seqlen,
        )
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert block.moe.router.gate.weight.grad is not None


class TestQwen3ModelMoE:
    """Tests for full Qwen3Model with MoE."""

    @pytest.fixture
    def dense_model_args(self):
        """Model args for dense model."""
        return Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=64,
            moe_enabled=False,
            attn_type="sdpa",  # Use SDPA for testing
        )

    @pytest.fixture
    def moe_model_args(self):
        """Model args for full MoE model (all layers MoE)."""
        return Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            moe_inter_dim=96,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=64,
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False),
            decoder_sparse_step=1,
            attn_type="sdpa",  # Use SDPA for testing
        )

    @pytest.fixture
    def mixed_model_args(self):
        """Model args for mixed MoE/dense model (alternating layers)."""
        return Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            moe_inter_dim=96,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=64,
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False),
            decoder_sparse_step=2,  # Layers 1, 3 are MoE; layers 0, 2 are dense
            attn_type="sdpa",  # Use SDPA for testing
        )

    def test_dense_model_forward(self, dense_model_args):
        """Test forward pass for dense model."""
        model = Qwen3Model(dense_model_args)
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cpu"))

        bs, seq_len = 2, 16
        tokens = torch.randint(0, dense_model_args.vocab_size, (bs, seq_len))
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        out = model(
            tokens_packed,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        assert out.shape == (1, bs * seq_len, dense_model_args.vocab_size)
        assert not torch.any(torch.isnan(out))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_moe_model_forward(self, moe_model_args):
        """Test forward pass for full MoE model."""
        model = Qwen3Model(moe_model_args).cuda()
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 2, 16
        tokens = torch.randint(
            0, moe_model_args.vocab_size, (bs, seq_len), device="cuda"
        )
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        out = model(
            tokens_packed,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        assert out.shape == (1, bs * seq_len, moe_model_args.vocab_size)
        assert not torch.any(torch.isnan(out))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_mixed_model_forward(self, mixed_model_args):
        """Test forward pass for mixed MoE/dense model."""
        model = Qwen3Model(mixed_model_args).cuda()
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 2, 16
        tokens = torch.randint(
            0, mixed_model_args.vocab_size, (bs, seq_len), device="cuda"
        )
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        out = model(
            tokens_packed,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        assert out.shape == (1, bs * seq_len, mixed_model_args.vocab_size)
        assert not torch.any(torch.isnan(out))

    def test_mixed_model_layer_structure(self, mixed_model_args):
        """Test that mixed model has correct layer structure."""
        model = Qwen3Model(mixed_model_args)

        # With decoder_sparse_step=2:
        # Layer 0: dense (0+1=1, 1%2=1, not 0)
        # Layer 1: MoE (1+1=2, 2%2=0)
        # Layer 2: dense (2+1=3, 3%2=1, not 0)
        # Layer 3: MoE (3+1=4, 4%2=0)
        assert model.layers["0"].moe is None, "Layer 0 should be dense"
        assert model.layers["0"].feed_forward is not None, (
            "Layer 0 should have feed_forward"
        )

        assert model.layers["1"].moe is not None, "Layer 1 should be MoE"
        assert model.layers["1"].feed_forward is None, (
            "Layer 1 should not have feed_forward"
        )

        assert model.layers["2"].moe is None, "Layer 2 should be dense"
        assert model.layers["2"].feed_forward is not None, (
            "Layer 2 should have feed_forward"
        )

        assert model.layers["3"].moe is not None, "Layer 3 should be MoE"
        assert model.layers["3"].feed_forward is None, (
            "Layer 3 should not have feed_forward"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_moe_model_gradient_flow(self, moe_model_args):
        """Test gradient flow through MoE model."""
        model = Qwen3Model(moe_model_args).cuda()
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 2, 16
        tokens = torch.randint(
            0, moe_model_args.vocab_size, (bs, seq_len), device="cuda"
        )
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        out = model(
            tokens_packed,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        loss = out.sum()
        loss.backward()

        # Check gradients flow to embedding
        assert model.tok_embeddings.weight.grad is not None

        # Check gradients flow to MoE layers
        for layer_id, layer in model.layers.items():
            assert layer.moe is not None, f"Layer {layer_id} should have MoE"
            assert layer.moe.router.gate.weight.grad is not None, (
                f"Layer {layer_id} router should have grad"
            )
            assert layer.moe.experts.w1.grad is not None, (
                f"Layer {layer_id} experts.w1 should have grad"
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_mixed_model_gradient_flow(self, mixed_model_args):
        """Test gradient flow through mixed MoE/dense model."""
        model = Qwen3Model(mixed_model_args).cuda()
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 2, 16
        tokens = torch.randint(
            0, mixed_model_args.vocab_size, (bs, seq_len), device="cuda"
        )
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        out = model(
            tokens_packed,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        loss = out.sum()
        loss.backward()

        # Check gradients flow to embedding
        assert model.tok_embeddings.weight.grad is not None

        # Check MoE layers (1, 3) have gradients
        assert model.layers["1"].moe.router.gate.weight.grad is not None
        assert model.layers["3"].moe.router.gate.weight.grad is not None

        # Check dense layers (0, 2) have gradients
        assert model.layers["0"].feed_forward.w1.weight.grad is not None
        assert model.layers["2"].feed_forward.w1.weight.grad is not None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_moe_model_with_positions(self, moe_model_args):
        """Test MoE model with explicit positions."""
        model = Qwen3Model(moe_model_args).cuda()
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 2, 16
        tokens = torch.randint(
            0, moe_model_args.vocab_size, (bs, seq_len), device="cuda"
        )
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        out = model(
            tokens_packed,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        assert out.shape == (1, bs * seq_len, moe_model_args.vocab_size)
        assert not torch.any(torch.isnan(out))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_moe_critic_model(self):
        """Test MoE model in critic mode."""
        model_args = Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            moe_inter_dim=96,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=64,
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False),
            decoder_sparse_step=1,
            is_critic=True,
            num_labels=1,
            attn_type="sdpa",  # Use SDPA for testing
        )
        model = Qwen3Model(model_args).cuda()
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 2, 16
        tokens = torch.randint(0, model_args.vocab_size, (bs, seq_len), device="cuda")
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        out = model(
            tokens_packed,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Critic model outputs (1, bs*seq_len, num_labels)
        assert out.shape == (1, bs * seq_len, model_args.num_labels)
        assert not torch.any(torch.isnan(out))


class TestQwen3MoEConfigurations:
    """Tests for various Qwen3-MoE configurations."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_qwen3_30b_a3b_like_config(self):
        """Test configuration similar to Qwen3-30B-A3B."""
        model_args = Qwen3ModelArgs(
            dim=128,  # Smaller for testing
            hidden_dim=256,
            moe_inter_dim=64,
            n_heads=8,
            n_kv_heads=2,
            n_layers=4,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=64,
            moe_enabled=True,
            moe_args=MoEArgs(
                num_experts=8,  # Smaller for testing (real is 128)
                top_k=2,  # Smaller for testing (real is 8)
                score_func="softmax",
                route_norm=True,
                use_grouped_mm=False,
            ),
            decoder_sparse_step=1,
            attn_type="sdpa",  # Use SDPA for testing
        )

        model = Qwen3Model(model_args).cuda()
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 1, 16
        tokens = torch.randint(0, model_args.vocab_size, (bs, seq_len), device="cuda")
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        out = model(
            tokens_packed,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        assert out.shape == (1, bs * seq_len, model_args.vocab_size)
        assert not torch.any(torch.isnan(out))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_with_shared_experts(self):
        """Test MoE model with shared experts."""
        model_args = Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            moe_inter_dim=96,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=64,
            moe_enabled=True,
            moe_args=MoEArgs(
                num_experts=4,
                top_k=2,
                num_shared_experts=1,
                use_grouped_mm=False,
            ),
            decoder_sparse_step=1,
            attn_type="sdpa",  # Use SDPA for testing
        )

        model = Qwen3Model(model_args).cuda()
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 2, 8
        tokens = torch.randint(0, model_args.vocab_size, (bs, seq_len), device="cuda")
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        out = model(
            tokens_packed,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        assert out.shape == (1, bs * seq_len, model_args.vocab_size)
        assert not torch.any(torch.isnan(out))

        # Check that shared experts exist
        for layer in model.layers.values():
            assert layer.moe is not None
            assert layer.moe.shared_experts is not None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_single_token_inference(self):
        """Test MoE model with single token (generation scenario)."""
        model_args = Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            moe_inter_dim=96,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=64,
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False),
            decoder_sparse_step=1,
            attn_type="sdpa",  # Use SDPA for testing
        )

        model = Qwen3Model(model_args).cuda()
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cuda"))
        model.eval()

        bs = 1
        tokens = torch.randint(0, model_args.vocab_size, (bs, 1), device="cuda")
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        with torch.no_grad():
            out = model(
                tokens_packed,
                positions=positions,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        assert out.shape == (1, 1, model_args.vocab_size)
        assert not torch.any(torch.isnan(out))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for MoE router histc"
    )
    def test_batch_size_1(self):
        """Test MoE model with batch size 1."""
        model_args = Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            moe_inter_dim=96,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=64,
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False),
            decoder_sparse_step=1,
            attn_type="sdpa",  # Use SDPA for testing
        )

        model = Qwen3Model(model_args).cuda()
        model.init_weights()
        model.init_buffers(buffer_device=torch.device("cuda"))

        bs, seq_len = 1, 32
        tokens = torch.randint(0, model_args.vocab_size, (bs, seq_len), device="cuda")
        tokens_packed, positions, cu_seqlens, max_seqlen = _create_packed_input(tokens)

        out = model(
            tokens_packed,
            positions=positions,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        assert out.shape == (1, seq_len, model_args.vocab_size)
        assert not torch.any(torch.isnan(out))
