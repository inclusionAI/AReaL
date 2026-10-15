# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.engine import megatron_engine
from areal.engine.megatron_engine import MegatronEngine
from areal.models.mcore.vocab_parallel_head import ChunkedLMHeadOutput


@pytest.mark.parametrize("padding_length", [0, 4])
def test_chunked_forward_cp_restores_token_order_and_removes_padding(
    monkeypatch, padding_length
):
    engine = object.__new__(MegatronEngine)
    engine.config = SimpleNamespace(is_critic=False)
    engine.enable_tree_training = False
    # Two sequences occupy eight padded positions each. CP rank zero owns
    # their first and last quarters. A final dummy segment is optional.
    full = torch.arange(16 + padding_length, dtype=torch.float32)
    local = full[
        torch.tensor([0, 1, 6, 7, 8, 9, 14, 15] + ([16, 19] if padding_length else []))
    ]
    padded_cu = torch.tensor([0, 8, 16] + ([20] if padding_length else []))
    inputs = {
        "_cp_padded_cu_seqlens": padded_cu,
        "_cp_old_cu_seqlens": torch.tensor([0, 3, 8]),
        "_cp_padding_length": padding_length,
    }

    def reassembled(values, cu_seqlens):
        assert values is local
        assert cu_seqlens is padded_cu
        return full

    # CP communication is covered separately; exercise the consumer's real
    # sequence unpadding after the CP scalar reconstruction boundary.
    monkeypatch.setattr(megatron_engine, "reassemble_cp_packed_logprobs", reassembled)
    output = ChunkedLMHeadOutput(*(local for _ in range(6)))
    result = engine._compute_forward_result(output, inputs)
    torch.testing.assert_close(
        result,
        torch.tensor([0, 1, 2, 8, 9, 10, 11, 12], dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_chunked_forward_without_cp_preserves_already_unpadded_output():
    engine = object.__new__(MegatronEngine)
    engine.config = SimpleNamespace(is_critic=False)
    local = torch.arange(7, dtype=torch.float32)
    output = ChunkedLMHeadOutput(*(local for _ in range(6)))
    assert engine._compute_forward_result(output, {}) is local
