# SPDX-License-Identifier: Apache-2.0

"""Exercise the actor's real DSA forward with CPU projections and dense oracle."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn


def stub(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class Projection(nn.Linear):
    def __init__(self, input_size, output_size, norm=False):
        super().__init__(input_size, output_size, bias=False)
        if norm:
            self.layer_norm_weight = nn.Parameter(torch.ones(input_size))

    def forward(self, x):
        if hasattr(self, "layer_norm_weight"):
            x = F.rms_norm(x, (x.shape[-1],), self.layer_norm_weight, 1e-5)
        return F.linear(x, self.weight), None


class CPUMLA(nn.Module):
    def __init__(self, config, submodules, layer_number, **kwargs):
        super().__init__()
        self.config = config
        self.pg_collection = SimpleNamespace(tp=object())
        self.num_attention_heads_per_partition = config.num_attention_heads
        self.linear_q_down_proj = Projection(config.hidden_size, config.q_lora_rank)
        self.linear_q_up_proj = Projection(
            config.q_lora_rank,
            config.num_attention_heads * config.qk_head_dim,
            norm=True,
        )
        self.linear_kv_down_proj = Projection(config.hidden_size, config.kv_lora_rank)
        self.linear_kv_up_proj = Projection(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_head_dim + config.v_head_dim),
            norm=True,
        )
        self.linear_proj = Projection(
            config.num_attention_heads * config.v_head_dim, config.hidden_size
        )


@pytest.fixture
def dsa(monkeypatch):
    stub(monkeypatch, "megatron")
    stub(
        monkeypatch,
        "megatron.core",
        tensor_parallel=SimpleNamespace(
            copy_to_tensor_model_parallel_region=lambda x, group: x
        ),
    )
    stub(monkeypatch, "megatron.core.transformer")
    stub(monkeypatch, "megatron.core.transformer.module", mark_keep_in_fp32=lambda p: p)
    stub(
        monkeypatch,
        "megatron.core.transformer.multi_latent_attention",
        MLASelfAttention=CPUMLA,
    )
    path = Path(__file__).resolve().parents[1] / "areal/models/mcore/glm5_next_dsa.py"
    spec = importlib.util.spec_from_file_location("_glm5_dsa_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Dense causal indices exercise both sequence boundaries and the sentinel.
    def select(q, k, weights, gates, ape, cu, topk, pool):
        indices = torch.full((q.shape[0], 1, q.shape[0]), -1, dtype=torch.int32)
        for start, end in zip(cu[:-1], cu[1:], strict=True):
            for t in range(start, end):
                indices[t, 0, : t - start + 1] = torch.arange(start, t + 1)
        return indices

    def sparse(q, kv, indices, scale, d_v):
        assert q.ndim == kv.ndim == indices.ndim == 4
        assert kv.shape[2] == indices.shape[2] == 1
        assert q.shape[-1] == kv.shape[-1] == d_v + 64
        keys = kv[:, :, 0]
        logits = torch.einsum("bthd,bkd->bhtk", q, keys) * scale
        valid = torch.zeros(q.shape[1], kv.shape[1], dtype=torch.bool)
        for t, row in enumerate(indices[0, :, 0]):
            valid[t, row[row >= 0].long()] = True
        probabilities = logits.masked_fill(~valid, -torch.inf).softmax(-1)
        return torch.einsum("bhtk,bkd->bthd", probabilities, keys[..., :d_v])

    stub(
        monkeypatch,
        "areal.experimental.ops.glm5_next.indexer",
        kpool_select_topk=select,
    )
    stub(
        monkeypatch,
        "areal.experimental.ops.glm5_next.sparse_attention",
        sparse_attention=sparse,
    )
    config = SimpleNamespace(
        context_parallel_size=1,
        qk_pos_emb_head_dim=0,
        hidden_size=8,
        q_lora_rank=4,
        kv_lora_rank=6,
        qk_head_dim=4,
        v_head_dim=3,
        num_attention_heads=2,
        layernorm_epsilon=1e-5,
        params_dtype=torch.float32,
        sequence_parallel=False,
    )
    glm = SimpleNamespace(
        text=dict(index_n_heads=2, index_head_dim=4, index_topk=8, index_kpool=4)
    )
    torch.manual_seed(42)
    return module.Glm5NextDSAAttention(config, None, 1, glm)


def test_dsa_absorption_matches_explicit_mla_output_and_gradient(dsa):
    x = torch.randn(7, 1, 8, requires_grad=True)
    packed = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 7], dtype=torch.int32),
    )
    actual, _ = dsa(x, packed_seq_params=packed)
    q, _ = dsa.linear_q_up_proj(dsa.linear_q_down_proj(x)[0])
    kv, _ = dsa.linear_kv_up_proj(dsa.linear_kv_down_proj(x)[0])
    q = q.reshape(7, 2, 4)
    k, v = kv.reshape(7, 2, 7).split([4, 3], -1)
    logits = torch.einsum("thd,khd->htk", q, k) / 2
    mask = torch.zeros(7, 7, dtype=torch.bool)
    mask[:3, :3] = torch.ones(3, 3, dtype=torch.bool).tril()
    mask[3:, 3:] = torch.ones(4, 4, dtype=torch.bool).tril()
    scores = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    explicit = torch.einsum("htk,khd->thd", scores, v).reshape(7, 1, 6)
    expected, _ = dsa.linear_proj(explicit)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    params = [x] + [p for p in dsa.parameters() if p.requires_grad]
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, params, grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, params, grad)
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=5e-5, atol=3e-6)


def test_dsa_packed_sequences_do_not_leak(dsa):
    x = torch.randn(7, 1, 8)
    packed = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 3, 7]), cu_seqlens_kv=torch.tensor([0, 3, 7])
    )
    before = dsa(x, packed_seq_params=packed)[0]
    x[:3] += 100
    after = dsa(x, packed_seq_params=packed)[0]
    torch.testing.assert_close(before[3:], after[3:], rtol=0, atol=0)


def test_dsa_rejects_unpacked_batches(dsa):
    with pytest.raises(ValueError, match="packed THD"):
        dsa(torch.randn(7, 2, 8))


@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_fast_loader_kda_convolution_is_replicated_across_tp(monkeypatch, tp_rank):
    from tests.test_bailing_v3_hf_load import _load_hf_load_module

    module = _load_hf_load_module(monkeypatch)
    chunks = [torch.arange(24).reshape(8, 1, 3) + i * 24 for i in range(3)]
    result = module._weight_to_mcore_tp(
        hf_config=SimpleNamespace(architectures=["Glm5NextForConditionalGeneration"]),
        mcore_weights_name="decoder.layers.0.self_attention.kda.conv1d.weight",
        mcore_param_shape=[24, 1, 3],
        hf_weights_safe_slice=chunks,
        tp_rank=tp_rank,
        tp_size=4,
    )
    torch.testing.assert_close(result, torch.cat(chunks), rtol=0, atol=0)
