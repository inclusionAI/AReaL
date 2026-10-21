# SPDX-License-Identifier: Apache-2.0

"""GPU numerical gates; compilation and execution require TileLang and Triton."""

import pytest
import torch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GLM-5.3 kernel validation requires a CUDA GPU",
    ),
]


def test_kpool_selection_is_causal_and_respects_packed_boundaries():
    pytest.importorskip("tilelang")
    pytest.importorskip("triton")
    from areal.experimental.ops.glm5_next.indexer import kpool_select_topk

    cu = torch.tensor([0, 3, 12, 82], device="cuda", dtype=torch.int32)
    q = torch.zeros(82, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(82, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(82, 32, device="cuda")
    gates = torch.zeros_like(k)
    ape = torch.zeros(4, 128, device="cuda")
    indices = kpool_select_topk(q, k, weights, gates, ape, cu, 8, 4).cpu()
    for start, end in zip(cu.cpu()[:-1], cu.cpu()[1:], strict=True):
        for token in range(start, end):
            selected = indices[token, 0]
            selected = selected[selected >= 0]
            assert (selected >= start).all() and (selected <= token).all()
            assert selected.unique().numel() == selected.numel()
            if token - start < 8:
                assert set(selected.tolist()) == set(range(start, token + 1))
            # The unpooled causal suffix must always be selected.
            tail_start = start + ((token - start + 1) // 4) * 4
            assert set(range(tail_start, token + 1)).issubset(selected.tolist())


def _assert_bf16_gradient_accuracy(actual, reference, baseline):
    """Bound cancellation error against an independent dense BF16 baseline."""
    actual, reference, baseline = actual.float(), reference.float(), baseline.float()
    error = (actual - reference).abs()
    baseline_error = (baseline - reference).abs()

    def rms(tensor):
        return tensor.square().mean().sqrt()

    reference_rms = rms(reference)
    fp32_eps = torch.finfo(torch.float32).eps
    assert torch.isfinite(actual).all()
    assert error.max() <= 2 * baseline_error.max() + fp32_eps
    assert rms(error) <= 2 * rms(baseline_error) + fp32_eps
    assert rms(error) <= 0.01 * reference_rms + fp32_eps


@pytest.mark.parametrize("seed", [7, 19])
def test_sparse_mla_forward_backward_matches_dense_packed_attention(seed, monkeypatch):
    pytest.importorskip("tilelang")
    from areal.experimental.ops.glm5_next.sparse_attention import (
        SparseAttentionConfig,
        sparse_attention,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(seed)
    q = torch.randn(1, 128, 16, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn(1, 128, 1, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    q[..., 512:] = 0
    kv[..., 512:] = 0
    q.requires_grad_()
    kv.requires_grad_()
    positions = torch.arange(128, device="cuda")
    valid = (positions[:, None] >= positions[None, :]) & (
        positions[:, None] // 64 == positions[None, :] // 64
    )
    indices = torch.full((1, 128, 1, 64), -1, device="cuda", dtype=torch.int32)
    for token in range(128):
        row = positions[valid[token]]
        indices[0, token, 0, : row.numel()] = row.int()
    actual = sparse_attention(
        q,
        kv,
        indices,
        1 / 16,
        d_v=512,
        config=SparseAttentionConfig(forward_backend="tilelang"),
    )

    def dense_attention(query, key_value):
        logits = torch.einsum("bthd,bkd->bhtk", query, key_value[:, :, 0]) / 16
        scores = logits.masked_fill(~valid, -torch.inf).softmax(-1)
        return torch.einsum("bhtk,bkd->bthd", scores, key_value[:, :, 0, :512])

    reference_inputs = [tensor.detach().float().requires_grad_() for tensor in (q, kv)]
    baseline_inputs = [tensor.detach().clone().requires_grad_() for tensor in (q, kv)]
    expected = dense_attention(*reference_inputs)
    baseline = dense_attention(*baseline_inputs)
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.002)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, kv), grad)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad.float())
    baseline_grads = torch.autograd.grad(baseline, baseline_inputs, grad)
    for actual_grad, expected_grad, baseline_grad in zip(
        actual_grads, expected_grads, baseline_grads, strict=True
    ):
        _assert_bf16_gradient_accuracy(actual_grad, expected_grad, baseline_grad)
