"""Tests for varlen attention implementation."""

import pytest
import torch

from areal.platforms import current_platform


def setup_environment():
    """Set up test environment."""
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class TestVarlenAttn:
    """Tests for varlen_attn function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        setup_environment()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_basic_forward(self):
        """Test basic forward functionality."""
        from areal.experimental.models.archon.varlen_attention import varlen_attn

        # Setup
        total_tokens, num_heads, head_dim = 45, 8, 64
        device = torch.device(current_platform.device_type)
        q = torch.randn(
            total_tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # cu_seqlens for 3 sequences: 10, 15, 20
        cu_seqlens = torch.tensor([0, 10, 25, 45], dtype=torch.int32, device=device)
        max_seqlen = 20

        # Forward
        out = varlen_attn(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, is_causal=True
        )

        assert out.shape == q.shape
        assert not torch.isnan(out).any()
        assert out.dtype == q.dtype

    def test_single_sequence(self):
        """Test with a single sequence (no packing)."""
        from areal.experimental.models.archon.varlen_attention import varlen_attn

        total_tokens, num_heads, head_dim = 32, 4, 64
        device = torch.device(current_platform.device_type)
        q = torch.randn(
            total_tokens, num_heads, head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        cu_seqlens = torch.tensor([0, 32], dtype=torch.int32, device=device)
        max_seqlen = 32

        out = varlen_attn(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, is_causal=True
        )

        assert out.shape == q.shape
        assert not torch.isnan(out).any()

    def test_gradient(self):
        """Test backward pass correctness."""
        from areal.experimental.models.archon.varlen_attention import varlen_attn

        total_tokens, num_heads, head_dim = 32, 4, 32
        device = torch.device(current_platform.device_type)
        q = torch.randn(
            total_tokens,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        k = torch.randn(
            total_tokens,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        v = torch.randn(
            total_tokens,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )

        cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=device)

        out = varlen_attn(q, k, v, cu_seqlens, cu_seqlens, 16, 16, is_causal=True)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_with_scale(self):
        """Test with custom scale factor."""
        from areal.experimental.models.archon.varlen_attention import varlen_attn

        total_tokens, num_heads, head_dim = 24, 4, 64
        device = torch.device(current_platform.device_type)
        q = torch.randn(
            total_tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        cu_seqlens = torch.tensor([0, 12, 24], dtype=torch.int32, device=device)
        max_seqlen = 12

        # Custom scale
        scale = 0.1
        out = varlen_attn(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            is_causal=True,
            scale=scale,
        )

        assert out.shape == q.shape
        assert not torch.isnan(out).any()


class TestVarlenAttentionWrapper:
    """Tests for VarlenAttentionWrapper."""

    @pytest.fixture(autouse=True)
    def setup(self):
        setup_environment()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_wrapper_shape(self):
        """Test wrapper input/output shapes."""
        from areal.experimental.models.archon.varlen_attention import (
            VarlenAttentionWrapper,
        )

        wrapper = VarlenAttentionWrapper()

        batch, heads, seq_len, head_dim = 1, 8, 45, 64
        device = torch.device(current_platform.device_type)
        q = torch.randn(
            batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        cu_seqlens = torch.tensor([0, 10, 25, 45], dtype=torch.int32, device=device)
        max_seqlen = 20

        out = wrapper(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        assert out.shape == (batch, heads, seq_len, head_dim)
        assert not torch.isnan(out).any()

    def test_wrapper_gradient(self):
        """Test wrapper backward pass."""
        from areal.experimental.models.archon.varlen_attention import (
            VarlenAttentionWrapper,
        )

        wrapper = VarlenAttentionWrapper()

        batch, heads, seq_len, head_dim = 1, 4, 32, 32
        device = torch.device(current_platform.device_type)
        q = torch.randn(
            batch,
            heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        k = torch.randn(
            batch,
            heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        v = torch.randn(
            batch,
            heads,
            seq_len,
            head_dim,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )

        cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=device)
        max_seqlen = 16

        out = wrapper(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_wrapper_batch_must_be_one(self):
        """Test that wrapper raises error for batch > 1."""
        from areal.experimental.models.archon.varlen_attention import (
            VarlenAttentionWrapper,
        )

        wrapper = VarlenAttentionWrapper()

        batch, heads, seq_len, head_dim = 2, 4, 32, 32  # batch=2 should fail
        device = torch.device(current_platform.device_type)
        q = torch.randn(
            batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        cu_seqlens = torch.tensor([0, 32], dtype=torch.int32, device=device)
        max_seqlen = 32

        with pytest.raises(AssertionError, match="batch=1"):
            wrapper(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)


class TestVarlenVsSDPA:
    """Compare varlen_attn vs SDPA outputs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        setup_environment()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_output_similarity_single_seq(self):
        """Test that varlen and SDPA produce similar outputs for single sequence."""
        from areal.experimental.models.archon.attention import SDPAWrapper
        from areal.experimental.models.archon.varlen_attention import (
            VarlenAttentionWrapper,
        )

        batch, heads, seq_len, head_dim = 1, 8, 32, 64
        device = torch.device(current_platform.device_type)

        # Use same random seed for reproducibility
        torch.manual_seed(42)
        q = torch.randn(
            batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Single sequence (no packing)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        # VarlenAttention
        varlen_wrapper = VarlenAttentionWrapper()
        varlen_out = varlen_wrapper(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=seq_len)

        # SDPA
        sdpa_wrapper = SDPAWrapper()
        sdpa_out = sdpa_wrapper(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=seq_len)

        # Compare (allow some numerical tolerance for different backends)
        diff = (varlen_out.float() - sdpa_out.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

        # Flash attention vs SDPA may have some differences
        assert max_diff < 0.5, f"Max diff too large: {max_diff}"
        assert mean_diff < 0.05, f"Mean diff too large: {mean_diff}"

    def test_output_similarity_packed_seqs(self):
        """Test similarity for packed sequences."""
        from areal.experimental.models.archon.attention import SDPAWrapper
        from areal.experimental.models.archon.varlen_attention import (
            VarlenAttentionWrapper,
        )

        batch, heads, seq_len, head_dim = 1, 4, 45, 32
        device = torch.device(current_platform.device_type)

        torch.manual_seed(123)
        q = torch.randn(
            batch, heads, seq_len, head_dim, device=device, dtype=torch.bfloat16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # 3 packed sequences
        cu_seqlens = torch.tensor([0, 10, 25, 45], dtype=torch.int32, device=device)
        max_seqlen = 20

        # VarlenAttention
        varlen_wrapper = VarlenAttentionWrapper()
        varlen_out = varlen_wrapper(
            q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )

        # SDPA
        sdpa_wrapper = SDPAWrapper()
        sdpa_out = sdpa_wrapper(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Compare
        diff = (varlen_out.float() - sdpa_out.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Packed seqs - Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

        # Allow reasonable tolerance
        assert max_diff < 0.5, f"Max diff too large: {max_diff}"
        assert mean_diff < 0.05, f"Mean diff too large: {mean_diff}"
