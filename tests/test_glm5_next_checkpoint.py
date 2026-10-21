# SPDX-License-Identifier: Apache-2.0

import __future__

import ast
import json
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.test_glm5_next import (
    config_module,
    load_module,
    make_hf_config,
)

checkpoint = load_module(
    "_glm5_checkpoint_test", "areal/models/mcore/glm5_next_checkpoint.py"
)
validation = load_module("_glm5_validation_test", "examples/glm5_next/validate.py")


def load_function(relative_path, name, namespace):
    # Isolate dispatch from optional CUDA/TE imports, as in test_bailing_v3_nccl.
    path = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(path.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    code = ast.Module(body=[function], type_ignores=[])
    exec(
        compile(code, str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace[name]


def test_fp8_exclusions_preserve_kv_b_and_quantize_q_b():
    config = dict(
        quant_method="fp8",
        modules_to_not_convert=["model.layers.3.self_attn.kv_b_proj", "lm_head"],
    )
    normalized = config_module.glm5_next_quantization_config(config)
    assert config["modules_to_not_convert"] == [
        "model.layers.3.self_attn.kv_b_proj",
        "lm_head",
    ]
    calls = []

    def quantize(name, pairs, config, dtype):
        calls.extend(n for n, _ in pairs)
        return pairs

    wrapper = load_function(
        "areal/engine/megatron_utils/fp8/quantize.py",
        "quantize_params",
        dict(torch=torch, fnmatchcase=fnmatchcase, _quantize_params=quantize),
    )
    names = [
        "model.language_model.layers.3.self_attn.kv_b_proj.weight",
        "model.language_model.layers.3.self_attn.q_b_proj.weight",
        "lm_head.weight",
    ]
    params = [(n, torch.ones(2, 2)) for n in names]
    result = wrapper(
        "decoder.layers.3.self_attention.linear_kv_up_proj.weight", params, normalized
    )
    assert calls == [names[1]]
    assert [n for n, _ in result] == names
    assert result[0][1] is params[0][1]


@pytest.mark.parametrize("exclusions", [None, [], ["unrelated_module"]])
def test_quantization_without_matching_exclusions_preserves_dispatch(exclusions):
    calls = []

    def quantize(name, pairs, config, dtype):
        calls.extend(pairs)
        return pairs

    wrapper = load_function(
        "areal/engine/megatron_utils/fp8/quantize.py",
        "quantize_params",
        dict(torch=torch, fnmatchcase=fnmatchcase, _quantize_params=quantize),
    )
    pairs = [("model.layers.0.mlp.gate_proj.weight", torch.ones(2, 2))]
    result = wrapper(
        "decoder.layers.0.mlp.linear_fc1.weight",
        pairs,
        {"modules_to_not_convert": exclusions},
    )
    assert result[0][1] is pairs[0][1]
    assert calls[0][0] == pairs[0][0]


def test_online_weight_conversion_uses_bridge_and_preserves_packed_scale():
    tensor = torch.tensor([1.0001, 1.0002, 1.0003])
    calls = []
    config = {"quant_method": "fp8"}

    def convert(name, param):
        calls.append(name)
        return ["model.language_model.layers.0.hc_attn_scale"], [param]

    bridge = SimpleNamespace(
        _weight_to_hf_format=convert,
        hf_config=SimpleNamespace(quantization_config=config),
    )

    def quantize(name, pairs, quantization_config):
        assert quantization_config is config
        return pairs

    dispatch = load_function(
        "areal/engine/megatron_utils/megatron.py",
        "convert_to_hf",
        dict(torch=torch, TransformerConfig=object, quantize_params=quantize),
    )
    result = dispatch(
        None,
        "glm5_next",
        "module.module.decoder.layers.0.self_attention_hyper_connection.scale",
        tensor,
        bridge=bridge,
    )
    assert calls == ["decoder.layers.0.self_attention_hyper_connection.scale"]
    torch.testing.assert_close(result[0][1], tensor, rtol=0, atol=0)
    with pytest.raises(ValueError, match="without LoRA"):
        dispatch(None, "glm5_next_lora", "weight", tensor, bridge=bridge)


def test_export_retains_only_vision_and_scrubs_mtp(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    vision_name = "model.visual.patch_embed.proj.weight"
    text_name = "model.language_model.embed_tokens.weight"
    vision = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    save_file(
        {vision_name: vision, text_name: torch.zeros(2, 4)},
        source / "original.safetensors",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    vision_name: "original.safetensors",
                    text_name: "original.safetensors",
                }
            }
        )
    )
    save_file({text_name: torch.ones(2, 4)}, output / "trained.safetensors")
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {text_name: "trained.safetensors"}})
    )
    (output / "config.json").write_text(
        json.dumps(
            dict(
                model_type="glm5_next",
                vision_config={"hidden_size": 4},
                text_config={"model_type": "", "num_nextn_predict_layers": 1},
            )
        )
    )
    for _ in range(2):
        checkpoint.copy_glm5_next_vision_weights(source, output)
        index = json.loads((output / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == {vision_name, text_name}
        assert index["metadata"]["total_size"] == 12 * 2 + 8 * 4
    with safe_open(output / index["weight_map"][vision_name], framework="pt") as handle:
        torch.testing.assert_close(
            handle.get_tensor(vision_name), vision, rtol=0, atol=0
        )
    with safe_open(output / index["weight_map"][text_name], framework="pt") as handle:
        assert handle.get_tensor(text_name).sum() == 8
    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["num_nextn_predict_layers"] == 0
    assert config["text_config"]["model_type"] == "glm5_next_text"
    with pytest.raises(ValueError, match="overwrite"):
        checkpoint.copy_glm5_next_vision_weights(source, source)


def test_smoke_checkpoint_contains_exactly_selected_layers(tmp_path):
    hf_config = make_hf_config()
    source, output = tmp_path / "source", tmp_path / "smoke"
    source.mkdir()
    hf_config["text_config"]["num_nextn_predict_layers"] = 1
    (source / "config.json").write_text(json.dumps(hf_config))
    tensors = {
        f"model.language_model.layers.{i}.hc_attn_scale": torch.ones(3) * i
        for i in range(5)
    }
    tensors["lm_head.weight"] = torch.ones(8, 4)
    tensors["model.visual.proj.weight"] = torch.ones(2, 2)
    save_file(tensors, source / "original.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "original.safetensors" for n in tensors}})
    )
    result = validation.prepare_smoke(source, output)
    assert result["layers"] == 4 and result["tensors"] == 6
    index = json.loads((output / "model.safetensors.index.json").read_text())
    with safe_open(
        output / next(iter(index["weight_map"].values())), framework="pt"
    ) as handle:
        assert set(handle.keys()) == set(index["weight_map"])
        assert "model.language_model.layers.4.hc_attn_scale" not in handle.keys()
    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["num_nextn_predict_layers"] == 0
    assert validation.audit(output)["moe_layers"] == 1
    with pytest.raises(ValueError, match="new directory"):
        validation.prepare_smoke(source, output)


def test_logp_comparison_requires_paired_scored_tokens(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    common = dict(token_ids=np.array([4, 5, 6]), loss_mask=np.array([0, 1, 1]))
    np.savez(a, **common, logprobs=np.array([np.nan, -0.5, -2.0]))
    np.savez(b, **common, logprobs=np.array([np.nan, -0.51, -2.02]))
    report = validation.compare_logps(a, b, 0.03)
    assert report["passed"] and report["tokens"] == 2
    assert not validation.compare_logps(a, b, 0.001)["passed"]
    common["token_ids"][1] = 7
    np.savez(b, **common, logprobs=np.array([0.0, -0.51, -2.02]))
    with pytest.raises(ValueError, match="unpaired token_ids"):
        validation.compare_logps(a, b, 0.03)
