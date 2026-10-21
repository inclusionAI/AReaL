# SPDX-License-Identifier: Apache-2.0

"""CPU architecture and numerical contracts, without importing CUDA runtimes."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def load_module(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config_module = load_module(
    "_glm5_config_test", "areal/models/mcore/glm5_next_config.py"
)
ops = load_module("_glm5_ops_test", "areal/models/mcore/glm5_next_ops.py")


def make_hf_config():
    # Small dimensions with the production layer order, including the last KDA.
    text = dict(
        model_type="glm5_next_text",
        num_hidden_layers=45,
        layer_types=[
            "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
            for i in range(45)
        ],
        first_k_dense_replace=3,
        hidden_size=32,
        intermediate_size=96,
        num_attention_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        rms_norm_eps=1e-5,
        hc_eps=1e-6,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        index_kpool=4,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        index_topk=8,
        index_n_heads=2,
        index_head_dim=4,
        num_experts_per_tok=8,
        n_routed_experts=16,
        moe_intermediate_size=16,
        n_shared_experts=1,
        routed_scaling_factor=2.5,
        swiglu_limit=10.0,
        vocab_size=128,
        max_position_embeddings=256,
        linear_attn_config=dict(
            num_heads=4, head_dim=8, short_conv_kernel_size=4, gate_lower_bound=-5.0
        ),
    )
    return dict(
        model_type="glm5_next",
        architectures=["Glm5NextForConditionalGeneration"],
        text_config=text,
    )


@pytest.fixture
def hf_config():
    return make_hf_config()


def test_config_nested_and_flattened_preserve_last_kda(hf_config):
    original = copy.deepcopy(hf_config)
    config = config_module.Glm5NextConfig.from_hf(hf_config)
    assert len(config.layer_types) == 45
    assert config.layer_types[-1] == "linear_attention"
    assert sum(config.moe_layers) == 42
    assert config.mcore_args()["activation_func_clamp_value"] == 10.0
    assert config.mcore_args()["rotary_scaling_factor"] == 1.0
    assert hf_config == original
    flat = copy.deepcopy(hf_config["text_config"])
    flat.pop("linear_attn_config")
    flat.update(
        linear_num_heads=4,
        linear_head_dim=8,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
    )
    assert config_module.Glm5NextConfig.from_hf(flat).linear_num_heads == 4


def test_config_four_layers_keeps_one_moe_and_one_dsa(hf_config):
    text = hf_config["text_config"]
    text["num_hidden_layers"] = 4
    text["layer_types"] = text["layer_types"][:4]
    config = config_module.Glm5NextConfig.from_hf(hf_config)
    assert config.moe_layers == (0, 0, 0, 1)
    assert config.layer_types[-1] == "deepseek_sparse_attention"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("num_hidden_layers", 44, "layer_types"),
        ("qk_rope_head_dim", 64, "NoPE"),
        ("hc_eps", 1e-5, "hc_eps"),
        ("index_kpool", 3, "power of two"),
        ("index_kpool_always_select_tail", False, "causal tail"),
        ("index_topk", 7, "multiple"),
        ("num_experts_per_tok", 17, "expert count"),
        ("norm_topk_prob", False, "normalized sigmoid"),
    ],
)
def test_config_unsupported_architecture_fails_early(hf_config, field, value, match):
    hf_config["text_config"][field] = value
    with pytest.raises(ValueError, match=match):
        config_module.Glm5NextConfig.from_hf(hf_config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_parallel_size", 2),
        ("mtp_num_layers", 1),
        ("fp8", "hybrid"),
        ("params_dtype", torch.float16),
        ("params_dtype", torch.float32),
        ("recompute_modules", ["mhc"]),
        ("virtual_pipeline_model_parallel_size", 2),
        ("pipeline_model_parallel_layout", "Et|tL"),
    ],
)
def test_runtime_unsupported_mode_fails_before_allocation(field, value):
    config = SimpleNamespace(context_parallel_size=1)
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config_module.validate_glm5_next_runtime(config)


def test_runtime_full_recompute_is_allowed():
    config_module.validate_glm5_next_runtime(
        SimpleNamespace(context_parallel_size=1, recompute_granularity="full")
    )


def test_mhc_mappings_match_independent_float64_reference():
    torch.manual_seed(8)
    n = 4
    # Near-zero inputs distinguish rsqrt(mean(x²)+eps) from 1/(rms(x)+eps).
    x = torch.randn(3, 2, n * 5) * 1e-4
    weight = torch.randn(n * (n + 2), n * 5)
    bias = torch.randn(n * (n + 2))
    scale = torch.tensor([0.1, 0.2, 0.3])
    pre, post, residual = ops.mhc_mappings(x, weight, scale, bias, n, 1e-5, 20)
    a, w, b, s = [v.double().numpy() for v in (x, weight, bias, scale)]
    normalized = a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + 1e-5)
    logits = normalized @ w.T

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    np.testing.assert_allclose(
        pre.numpy(), sigmoid(logits[..., :n] * s[0] + b[:n]) + 1e-6, rtol=2e-6
    )
    np.testing.assert_allclose(
        post.numpy(),
        2 * sigmoid(logits[..., n : 2 * n] * s[1] + b[n : 2 * n]),
        rtol=2e-6,
    )
    matrix = (logits[..., 2 * n :] * s[2] + b[2 * n :]).reshape(3, 2, n, n)
    matrix = np.exp(matrix - matrix.max(axis=-1, keepdims=True))
    matrix = matrix / matrix.sum(axis=-1, keepdims=True) + 1e-6
    for i in range(20):
        if i:
            matrix /= matrix.sum(axis=-1, keepdims=True) + 1e-6
        matrix /= matrix.sum(axis=-2, keepdims=True) + 1e-6
    np.testing.assert_allclose(residual.numpy(), matrix, rtol=3e-6)
    torch.testing.assert_close(
        residual.sum(-1), torch.ones_like(pre), rtol=0, atol=2e-5
    )


def test_mhc_bf16_mixing_preserves_fp32_accumulation_and_gradients():
    torch.manual_seed(5)
    x = torch.randn(7, 1, 12, dtype=torch.bfloat16, requires_grad=True)
    output = torch.randn(7, 1, 3, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(24, 12, requires_grad=True)
    scale = torch.ones(3, requires_grad=True)
    bias = torch.randn(24, requires_grad=True)
    pre, post, residual = ops.mhc_mappings(x, weight, scale, bias, 4, 1e-5, 20)
    actual = ops.mhc_combine(x, output, residual, post)
    streams = x.float().reshape(7, 1, 4, 3)
    expected = torch.einsum("...ij,...id->...jd", residual, streams)
    expected = expected + post[..., None] * output.float().unsqueeze(-2)
    torch.testing.assert_close(actual, expected.flatten(-2).bfloat16(), rtol=0, atol=0)
    aggregate = ops.mhc_aggregate(x, pre)
    torch.testing.assert_close(
        aggregate, (streams * pre[..., None]).sum(-2).bfloat16(), rtol=0, atol=0
    )
    (actual.float().square().sum() + aggregate.float().square().sum()).backward()
    for tensor in (x, output, weight, scale, bias):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_mhc_residual_orientation_and_stream_injection():
    x = torch.tensor([[[1.0, 2.0, 10.0, 20.0]]])
    # Swap streams; a symmetric identity-only test would miss the transpose.
    mapping = torch.tensor([[[[0.2, 0.8], [0.3, 0.7]]]])
    post = torch.tensor([[[2.0, 3.0]]])
    output = torch.tensor([[[5.0, 6.0]]])
    expected = torch.tensor([[[13.2, 18.4, 22.8, 33.6]]])
    torch.testing.assert_close(
        ops.mhc_combine(x, output, mapping, post), expected, rtol=1e-6, atol=1e-6
    )
