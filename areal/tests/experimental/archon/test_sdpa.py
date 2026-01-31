"""Tests for SDPA attention wrapper.

These tests verify:
1. create_block_causal_mask_2d generates correct mask patterns
2. Edge cases (single sequence, empty sequences)

Run tests:
    pytest areal/tests/experimental/archon/test_sdpa.py -v
"""

import pytest
import torch

from areal.experimental.models.archon.attention import (
    SDPAWrapper,
    create_block_causal_mask_2d,
)

# Skip if no CUDA available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TestCreateBlockCausalMask2D:
    """Tests for create_block_causal_mask_2d function."""

    def test_single_sequence(self):
        """Test mask for a single sequence - should be lower triangular."""
        cu_seqlens = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
        seq_len = 5
        mask = create_block_causal_mask_2d(
            cu_seqlens, seq_len, device=torch.device("cuda"), dtype=torch.float32
        )

        assert mask.shape == (5, 5)

        # Check lower triangular pattern
        for i in range(5):
            for j in range(5):
                if j <= i:
                    assert mask[i, j] == 0.0, f"Position ({i}, {j}) should be 0.0"
                else:
                    assert mask[i, j] == float("-inf"), (
                        f"Position ({i}, {j}) should be -inf"
                    )

    def test_two_sequences(self):
        """Test mask for two sequences - should be block diagonal."""
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
        seq_len = 5
        mask = create_block_causal_mask_2d(
            cu_seqlens, seq_len, device=torch.device("cuda"), dtype=torch.float32
        )

        assert mask.shape == (5, 5)

        # First block (seq1: positions 0-2)
        for i in range(3):
            for j in range(3):
                if j <= i:
                    assert mask[i, j] == 0.0
                else:
                    assert mask[i, j] == float("-inf")

        # Second block (seq2: positions 3-4)
        for i in range(3, 5):
            for j in range(3, 5):
                if j <= i:
                    assert mask[i, j] == 0.0
                else:
                    assert mask[i, j] == float("-inf")

        # Cross-block should all be -inf
        for i in range(3):
            for j in range(3, 5):
                assert mask[i, j] == float("-inf")
        for i in range(3, 5):
            for j in range(3):
                assert mask[i, j] == float("-inf")

    def test_three_sequences(self):
        """Test mask for three sequences as shown in design doc."""
        cu_seqlens = torch.tensor([0, 3, 5, 7], dtype=torch.int32, device="cuda")
        seq_len = 7
        mask = create_block_causal_mask_2d(
            cu_seqlens, seq_len, device=torch.device("cuda"), dtype=torch.float32
        )

        assert mask.shape == (7, 7)

        # Verify block structure
        # Block 1: [0:3, 0:3]
        # Block 2: [3:5, 3:5]
        # Block 3: [5:7, 5:7]
        blocks = [(0, 3), (3, 5), (5, 7)]

        for block_start, block_end in blocks:
            block_size = block_end - block_start
            for i in range(block_size):
                for j in range(block_size):
                    gi, gj = block_start + i, block_start + j
                    if j <= i:
                        assert mask[gi, gj] == 0.0, (
                            f"Block diagonal ({gi}, {gj}) should be 0.0"
                        )
                    else:
                        assert mask[gi, gj] == float("-inf"), (
                            f"Block upper ({gi}, {gj}) should be -inf"
                        )

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        cu_seqlens = torch.tensor([0, 3], dtype=torch.int32, device="cuda")

        mask_fp32 = create_block_causal_mask_2d(
            cu_seqlens, 3, torch.device("cuda"), torch.float32
        )
        assert mask_fp32.dtype == torch.float32

        mask_bf16 = create_block_causal_mask_2d(
            cu_seqlens, 3, torch.device("cuda"), torch.bfloat16
        )
        assert mask_bf16.dtype == torch.bfloat16


class TestSDPAWrapper:
    """Tests for SDPAWrapper class."""

    @pytest.fixture
    def wrapper(self):
        return SDPAWrapper()

    def test_output_shape(self, wrapper):
        """Test that output shape matches input shape."""
        batch, heads, seq_len, head_dim = 1, 8, 64, 64
        q = torch.randn(
            batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(
            batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
        )
        v = torch.randn(
            batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16
        )
        cu_seqlens = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")

        output = wrapper(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=32)

        assert output.shape == q.shape

    def test_single_sequence_matches_causal(self, wrapper):
        """Test single sequence produces same result as is_causal=True."""
        batch, heads, seq_len, head_dim = 1, 4, 32, 64
        q = torch.randn(
            batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float32
        )
        k = torch.randn(
            batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float32
        )
        v = torch.randn(
            batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float32
        )
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

        # SDPA wrapper output
        sdpa_out = wrapper(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=seq_len)

        # Direct SDPA with is_causal=True
        import torch.nn.functional as F

        causal_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Should be very close
        torch.testing.assert_close(sdpa_out, causal_out, atol=1e-5, rtol=1e-5)
