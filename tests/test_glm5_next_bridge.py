# SPDX-License-Identifier: Apache-2.0

"""Conversion tests against the actual mbridge and mHC-capable MCore runtime."""

import json

import pytest
import torch

from tests.test_glm5_next import make_hf_config


@pytest.fixture
def bridge():
    pytest.importorskip(
        "mbridge", reason="mbridge is required for actual bridge conversion tests"
    )
    from mbridge.core.parallel_states import ParallelStates
    from megatron.core.transformer import MLATransformerConfig
    from transformers import PretrainedConfig

    from areal.models.mcore.glm5_next import check_glm5_next_capabilities
    from areal.models.mcore.glm5_next_bridge import Glm5NextBridge

    try:
        check_glm5_next_capabilities(MLATransformerConfig)
    except RuntimeError as error:
        pytest.skip(str(error))
    hf_config = make_hf_config()
    hf_config["text_config"].update(num_key_value_heads=4, attention_dropout=0.0)
    return Glm5NextBridge(
        PretrainedConfig.from_dict(hf_config),
        dtype=torch.bfloat16,
        parallel_states=ParallelStates(vpp_size=None),
    )


@pytest.mark.parametrize("site", ["self_attention", "mlp"])
def test_bridge_mhc_scale_roundtrip_preserves_fp32(bridge, site):
    name = f"decoder.layers.3.{site}_hyper_connection.scale"
    original = torch.tensor([1.0001, 1.0002, 1.0003])
    loaded = bridge._weight_to_mcore_format(name, [original])
    names, tensors = bridge._weight_to_hf_format(name, loaded)
    suffix = "attn" if site == "self_attention" else "ffn"
    assert names == [f"model.language_model.layers.3.hc_{suffix}_scale"]
    assert loaded.dtype == tensors[0].dtype == torch.float32
    torch.testing.assert_close(tensors[0], original, rtol=0, atol=0)


def test_bridge_conv_roundtrip_and_tp_replication(bridge):
    name = "decoder.layers.0.self_attention.kda.conv1d.weight"
    original = [
        torch.arange(24, dtype=torch.bfloat16).reshape(8, 1, 3) + i * 24
        for i in range(3)
    ]
    fused = bridge._weight_to_mcore_format(name, original)
    param = torch.nn.Parameter(torch.empty_like(fused))
    pieces = bridge._weight_split_across_tp(name, fused, param, 4)
    assert len(pieces) == 4
    for piece in pieces:
        torch.testing.assert_close(piece, fused, rtol=0, atol=0)
    names, exported = bridge._weight_to_hf_format(
        name, bridge._weight_merge_across_tp(name, pieces, param)
    )
    assert names == [
        f"model.language_model.layers.0.self_attn.{part}_conv1d.weight"
        for part in ("q", "k", "v")
    ]
    for actual, expected in zip(exported, original, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bridge_dense_glu_and_nested_embedding_names(bridge):
    name = "decoder.layers.0.mlp.linear_fc1.weight"
    original = [torch.randn(8, 4).bfloat16() for _ in range(2)]
    fused = bridge._weight_to_mcore_format(name, original)
    names, tensors = bridge._weight_to_hf_format(name, fused)
    assert names == [
        f"model.language_model.layers.0.mlp.{part}_proj.weight"
        for part in ("gate", "up")
    ]
    for a, b in zip(tensors, original, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert bridge._weight_name_mapping_mcore_to_hf(
        "embedding.word_embeddings.weight"
    ) == ["model.language_model.embed_tokens.weight"]


def test_bridge_extra_args_revalidate_runtime(bridge):
    with pytest.raises(ValueError, match="context_parallel_size=1"):
        bridge.set_extra_args(context_parallel_size=2)


def test_runtime_audit_imports_engine_and_builds_bridge(bridge, tmp_path):
    from examples.glm5_next.validate import audit

    (tmp_path / "config.json").write_text(json.dumps(bridge.hf_config.to_dict()))
    result = audit(tmp_path, check_runtime=True)
    assert result["runtime_bridge_layers"] == result["layers"] == 45
    assert result["runtime_imports"] == "passed (GPU kernels not executed)"
