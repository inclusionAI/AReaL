from __future__ import annotations

from areal.engine.sglang_ext.sglang_worker_extension import (
    _inject_awex_parameter_aliases,
)


def test_inject_awex_parameter_aliases_maps_self_attn_to_attention_keys():
    dense = object()
    qkv = object()
    parameters = {
        "model.layers.0.self_attn.o_proj.weight": dense,
        "model.layers.0.self_attn.qkv_proj.weight": qkv,
    }

    added = _inject_awex_parameter_aliases(parameters)

    assert added >= 2
    assert parameters["model.layers.0.attention.dense.weight"] is dense
    assert parameters["model.layers.0.attention.query_key_value_proj.weight"] is qkv


def test_inject_awex_parameter_aliases_maps_attention_to_self_attn_keys():
    dense = object()
    qkv = object()
    parameters = {
        "model.layers.0.attention.dense.weight": dense,
        "model.layers.0.attention.query_key_value_proj.weight": qkv,
    }

    added = _inject_awex_parameter_aliases(parameters)

    assert added >= 2
    assert parameters["model.layers.0.self_attn.o_proj.weight"] is dense
    assert parameters["model.layers.0.self_attn.qkv_proj.weight"] is qkv


def test_inject_awex_parameter_aliases_adds_lm_head_embedding_aliases():
    emb = object()
    parameters = {
        "model.embed_tokens.weight": emb,
    }

    added = _inject_awex_parameter_aliases(parameters)

    assert added >= 1
    assert parameters["lm_head.weight"] is emb


def test_inject_awex_parameter_aliases_adds_layernorm_and_qkv_split_aliases():
    qln = object()
    kln = object()
    q = object()
    k = object()
    v = object()
    parameters = {
        "model.layers.0.self_attn.query_layernorm.weight": qln,
        "model.layers.0.self_attn.key_layernorm.weight": kln,
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.self_attn.k_proj.weight": k,
        "model.layers.0.self_attn.v_proj.weight": v,
    }

    added = _inject_awex_parameter_aliases(parameters)

    assert added >= 5
    assert parameters["model.layers.0.attention.query_layernorm.weight"] is qln
    assert parameters["model.layers.0.attention.key_layernorm.weight"] is kln
    assert parameters["model.layers.0.attention.q_proj.weight"] is q
    assert parameters["model.layers.0.attention.k_proj.weight"] is k
    assert parameters["model.layers.0.attention.v_proj.weight"] is v


def test_inject_awex_parameter_aliases_adds_output_layer_aliases():
    out = object()
    parameters = {
        "model.output_layer.weight": out,
    }

    added = _inject_awex_parameter_aliases(parameters)

    assert added >= 1
    assert parameters["lm_head.weight"] is out
