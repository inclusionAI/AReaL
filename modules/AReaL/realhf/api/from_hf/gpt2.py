# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import os
from typing import *

import torch
import transformers

from realhf.api.core.model_api import ReaLModelConfig, register_hf_family
from realhf.base.testing import (
    TESTING_MODEL_HEAD_DIM,
    TESTING_MODEL_HIDDEN_SIZE,
    TESTING_MODEL_INTERMEDIATE_SIZE,
    TESTING_MODEL_N_HEADS,
    TESTING_MODEL_N_LAYERS,
    TESTING_MODEL_N_POSITIONS,
    TESTING_MODEL_VOCAB_SIZE,
)


def sd_from_gpt2(state_dict: Dict, config: ReaLModelConfig) -> Dict:
    if any(k.startswith("transformer.") for k in state_dict.keys()):
        new_sd = {k.replace("transformer.", ""): v for k, v in state_dict.items()}
        if "lm_head.weight" in new_sd:
            head_w = new_sd.pop("lm_head.weight")
            assert torch.allclose(head_w, new_sd["wte.weight"])
        state_dict = new_sd

    new_sd = {}
    if "wte.weight" in state_dict:
        new_sd["0.wte.weight"] = state_dict["wte.weight"]
    if "wpe.weight" in state_dict:
        new_sd["0.wpe.weight"] = state_dict["wpe.weight"]
    if "lm_head.weight" in state_dict:
        new_sd[f"{config.n_layers + 1}.weight"] = state_dict["lm_head.weight"]
    if "ln_f.weight" in state_dict:
        new_sd[f"{config.n_layers}.ln_f.weight"] = state_dict["ln_f.weight"]
        new_sd[f"{config.n_layers}.ln_f.bias"] = state_dict["ln_f.bias"]
    for i in range(config.n_layers):
        if not any(k.startswith(f"h.{i}.") for k in state_dict):
            continue
        new_sd[f"{i+1}.attn.c_attn.ln.weight"] = state_dict[f"h.{i}.ln_1.weight"]
        new_sd[f"{i+1}.attn.c_attn.ln.bias"] = state_dict[f"h.{i}.ln_1.bias"]

        attn_w = state_dict[f"h.{i}.attn.c_attn.weight"]
        attn_bias = state_dict[f"h.{i}.attn.c_attn.bias"]
        qw, kw, vw = torch.chunk(attn_w, 3, dim=1)
        qb, kb, vb = torch.chunk(attn_bias, 3, dim=0)
        new_sd[f"{i+1}.attn.c_attn.q_attn.weight"] = qw.transpose(0, 1)
        new_sd[f"{i+1}.attn.c_attn.k_attn.weight"] = kw.transpose(0, 1)
        new_sd[f"{i+1}.attn.c_attn.v_attn.weight"] = vw.transpose(0, 1)
        new_sd[f"{i+1}.attn.c_attn.q_attn.bias"] = qb
        new_sd[f"{i+1}.attn.c_attn.k_attn.bias"] = kb
        new_sd[f"{i+1}.attn.c_attn.v_attn.bias"] = vb

        new_sd[f"{i+1}.attn.c_proj.weight"] = state_dict[
            f"h.{i}.attn.c_proj.weight"
        ].transpose(0, 1)
        new_sd[f"{i+1}.attn.c_proj.bias"] = state_dict[f"h.{i}.attn.c_proj.bias"]

        new_sd[f"{i+1}.mlp.ln.weight"] = state_dict[f"h.{i}.ln_2.weight"]
        new_sd[f"{i+1}.mlp.ln.bias"] = state_dict[f"h.{i}.ln_2.bias"]
        new_sd[f"{i+1}.mlp.c_fc.weight"] = state_dict[
            f"h.{i}.mlp.c_fc.weight"
        ].transpose(0, 1)
        new_sd[f"{i+1}.mlp.c_fc.bias"] = state_dict[f"h.{i}.mlp.c_fc.bias"]
        new_sd[f"{i+1}.mlp.c_proj.weight"] = state_dict[
            f"h.{i}.mlp.c_proj.weight"
        ].transpose(0, 1)
        new_sd[f"{i+1}.mlp.c_proj.bias"] = state_dict[f"h.{i}.mlp.c_proj.bias"]
    return new_sd


def sd_to_gpt2(state_dict: Dict, config: ReaLModelConfig) -> Dict:
    max_positions = config.n_positions
    bias = torch.tril(
        torch.ones((max_positions, max_positions), dtype=torch.bool)
    ).view(1, 1, max_positions, max_positions)

    new_sd = {}
    if "0.wte.weight" in state_dict:
        new_sd["wte.weight"] = state_dict["0.wte.weight"]
        new_sd["wpe.weight"] = state_dict["0.wpe.weight"]
    if f"{config.n_layers}.ln_f.weight" in state_dict:
        new_sd["ln_f.weight"] = state_dict[f"{config.n_layers}.ln_f.weight"]
        new_sd["ln_f.bias"] = state_dict[f"{config.n_layers}.ln_f.bias"]
    if f"{config.n_layers + 1}.weight" in state_dict:
        new_sd["lm_head.weight"] = state_dict[f"{config.n_layers + 1}.weight"]
    for i in range(config.n_layers):
        if not any(k.startswith(f"{i+1}.") for k in state_dict):
            continue
        new_sd[f"h.{i}.ln_1.weight"] = state_dict[f"{i+1}.attn.c_attn.ln.weight"]
        new_sd[f"h.{i}.ln_1.bias"] = state_dict[f"{i+1}.attn.c_attn.ln.bias"]

        qw = state_dict[f"{i+1}.attn.c_attn.q_attn.weight"].transpose(0, 1)
        kw = state_dict[f"{i+1}.attn.c_attn.k_attn.weight"].transpose(0, 1)
        vw = state_dict[f"{i+1}.attn.c_attn.v_attn.weight"].transpose(0, 1)
        qb = state_dict[f"{i+1}.attn.c_attn.q_attn.bias"]
        kb = state_dict[f"{i+1}.attn.c_attn.k_attn.bias"]
        vb = state_dict[f"{i+1}.attn.c_attn.v_attn.bias"]
        attn_w = torch.cat([qw, kw, vw], dim=1)
        attn_bias = torch.cat([qb, kb, vb], dim=0)
        new_sd[f"h.{i}.attn.c_attn.weight"] = attn_w
        new_sd[f"h.{i}.attn.c_attn.bias"] = attn_bias

        new_sd[f"h.{i}.attn.c_proj.weight"] = state_dict[
            f"{i+1}.attn.c_proj.weight"
        ].transpose(0, 1)
        new_sd[f"h.{i}.attn.c_proj.bias"] = state_dict[f"{i+1}.attn.c_proj.bias"]

        new_sd[f"h.{i}.ln_2.weight"] = state_dict[f"{i+1}.mlp.ln.weight"]
        new_sd[f"h.{i}.ln_2.bias"] = state_dict[f"{i+1}.mlp.ln.bias"]
        new_sd[f"h.{i}.mlp.c_fc.weight"] = state_dict[
            f"{i+1}.mlp.c_fc.weight"
        ].transpose(0, 1)
        new_sd[f"h.{i}.mlp.c_fc.bias"] = state_dict[f"{i+1}.mlp.c_fc.bias"]
        new_sd[f"h.{i}.mlp.c_proj.weight"] = state_dict[
            f"{i+1}.mlp.c_proj.weight"
        ].transpose(0, 1)
        new_sd[f"h.{i}.mlp.c_proj.bias"] = state_dict[f"{i+1}.mlp.c_proj.bias"]
        new_sd[f"h.{i}.attn.bias"] = bias
    return new_sd


# param name is used to load directly from huggingface checkpoints
def gpt2_embedding_layer_names(config: ReaLModelConfig) -> List[str]:
    return [
        "wte.weight",
        "wpe.weight",
        "transformer.wte.weight",
        "transformer.wpe.weight",
    ]


def gpt2_transformer_block_param_name(config: ReaLModelConfig, idx: int) -> List[str]:
    names = [
        f"h.{idx}.ln_1.weight",
        f"h.{idx}.ln_1.bias",
        # f'h.{idx}.attn.bias',
        f"h.{idx}.attn.c_attn.weight",
        f"h.{idx}.attn.c_attn.bias",
        f"h.{idx}.attn.c_proj.weight",
        f"h.{idx}.attn.c_proj.bias",
        f"h.{idx}.ln_2.weight",
        f"h.{idx}.ln_2.bias",
        f"h.{idx}.mlp.c_fc.weight",
        f"h.{idx}.mlp.c_fc.bias",
        f"h.{idx}.mlp.c_proj.weight",
        f"h.{idx}.mlp.c_proj.bias",
    ]
    if idx == config.n_layers - 1:
        names += ["ln_f.weight", "ln_f.bias"]
    _names = ["transformer." + name for name in names]
    return names + _names


def gpt2_output_head_param_name(config: ReaLModelConfig) -> List[str]:
    return ["wte.weight", "transformer.wte.weight", "lm_head.weight"]


def convert_config_gpt2(
    hf_config: transformers.GPT2Config,
) -> ReaLModelConfig:
    return ReaLModelConfig(
        n_layers=hf_config.n_layer,
        n_kv_heads=hf_config.n_head,
        hidden_dim=hf_config.n_embd,
        head_dim=hf_config.n_embd // hf_config.n_head,
        n_q_heads=hf_config.n_head,
        use_mlp_bias=True,
        intermediate_dim=(
            hf_config.n_inner if hf_config.n_inner is not None else 4 * hf_config.n_embd
        ),
        vocab_size=hf_config.vocab_size,
        n_positions=hf_config.n_positions,
        embd_pdrop=hf_config.embd_pdrop,
        attn_pdrop=hf_config.attn_pdrop,
        resid_pdrop=hf_config.resid_pdrop,
        layer_norm_epsilon=hf_config.layer_norm_epsilon,
        activation_function=hf_config.activation_function,
        use_attention_bias=True,
        use_attn_proj_bias=True,
        scale_attn_by_inverse_layer_idx=hf_config.scale_attn_by_inverse_layer_idx,
        tied_embedding=True,
        scale_attn_weights=hf_config.scale_attn_weights,
    )


def to_gpt2_config(config: ReaLModelConfig) -> transformers.GPT2Config:
    return transformers.GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.hidden_dim,
        n_layer=config.n_layers,
        n_head=config.n_q_heads,
        n_inner=config.intermediate_dim,
        activation_function=config.activation_function,
        embd_pdrop=config.embd_pdrop,
        attn_pdrop=config.attn_pdrop,
        resid_pdrop=config.resid_pdrop,
        layer_norm_epsilon=config.layer_norm_epsilon,
        scale_attn_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
        scale_attn_weights=config.scale_attn_weights,
        architectures=["GPT2LMHeadModel"],
    )


def make_real_config_gpt2() -> ReaLModelConfig:
    hf_config = transformers.GPT2Config(
        vocab_size=TESTING_MODEL_VOCAB_SIZE,
        n_positions=TESTING_MODEL_N_POSITIONS,
        n_embd=TESTING_MODEL_HIDDEN_SIZE,
        n_layer=TESTING_MODEL_N_LAYERS,
        n_head=TESTING_MODEL_N_HEADS,
        n_inner=TESTING_MODEL_INTERMEDIATE_SIZE,
        activation_function="gelu_new",
    )
    return convert_config_gpt2(hf_config)


register_hf_family(
    name="gpt2",
    hf_cls_name="GPT2LMHeadModel",
    config_from_hf_converter=convert_config_gpt2,
    config_to_hf_converter=to_gpt2_config,
    sd_from_hf_converter=sd_from_gpt2,
    sd_to_hf_converter=sd_to_gpt2,
    embedding_param_names=gpt2_embedding_layer_names,
    tblock_param_names=gpt2_transformer_block_param_name,
    head_param_names=gpt2_output_head_param_name,
    real_config_maker=make_real_config_gpt2,
)
