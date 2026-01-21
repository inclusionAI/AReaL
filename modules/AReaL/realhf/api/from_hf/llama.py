# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

from typing import *

import torch
import transformers

from realhf.api.core.model_api import ReaLModelConfig, register_hf_family
from realhf.base.constants import use_te_impl
from realhf.base.testing import (
    TESTING_MODEL_HEAD_DIM,
    TESTING_MODEL_HIDDEN_SIZE,
    TESTING_MODEL_INTERMEDIATE_SIZE,
    TESTING_MODEL_N_HEADS,
    TESTING_MODEL_N_LAYERS,
    TESTING_MODEL_N_POSITIONS,
    TESTING_MODEL_VOCAB_SIZE,
)


def convert_state_dict_llama(state_dict: Dict, config: ReaLModelConfig) -> Dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == "model.embed_tokens.weight":
            new_state_dict["0.wte.weight"] = v
        elif k == "lm_head.weight":
            new_state_dict[f"{config.n_layers + 1}.weight"] = v
        elif k == "model.norm.weight":
            new_state_dict[f"{config.n_layers}.ln_f.weight"] = v
        elif "inv_freq" in k:
            continue
        else:
            block_idx = int(k.split(".")[2])
            name = k.split(".", 3)[3]
            replace_pairs = [
                ("self_attn.", "attn."),
                ("post_attention_layernorm.", "mlp.ln."),
                ("input_layernorm.", "attn.c_attn.ln."),
                ("attn.o_proj.", "attn.c_proj."),
                ("q_proj.", "c_attn.q_attn."),
                ("k_proj.", "c_attn.k_attn."),
                ("v_proj.", "c_attn.v_attn."),
            ]
            for k1, k2 in replace_pairs:
                if k1 in name:
                    name = name.replace(k1, k2)
            new_state_dict[f"{block_idx + 1}.{name}"] = v

    if use_te_impl():
        state_dict = new_state_dict
        new_state_dict = {}
        te_replace_pairs = [
            (".mlp.ln.weight", ".mlp.layer_norm_weight"),
            (".mlp.down_proj.weight", ".mlp.fc2_weight"),
        ]
        for k, v in state_dict.items():
            for k1, k2 in te_replace_pairs:
                if k1 in k:
                    k = k.replace(k1, k2)
            new_state_dict[k] = v

        # fuse gate && up weight
        for i in range(config.n_layers):
            gate_w = new_state_dict[f"{i+1}.mlp.gate_proj.weight"]
            upproj_w = new_state_dict[f"{i+1}.mlp.up_proj.weight"]
            w = torch.cat([gate_w, upproj_w], dim=0)
            new_state_dict[f"{i+1}.mlp.fc1_weight"] = w
            new_state_dict[f"{i+1}.mlp._extra_state"] = None
            new_state_dict.pop(f"{i+1}.mlp.gate_proj.weight")
            new_state_dict.pop(f"{i+1}.mlp.up_proj.weight")
    return new_state_dict


def to_llama_state_dict(
    state_dict: Dict[str, torch.Tensor], config: ReaLModelConfig
) -> Dict:
    if use_te_impl():
        # remove all extra states
        keys = list(state_dict.keys())
        for k in keys:
            if k.endswith("_extra_state"):
                state_dict.pop(k)

        # split gate && up weight
        for i in range(config.n_layers):
            w = state_dict[f"{i + 1}.mlp.fc1_weight"]
            gate_w, upproj_w = w.split((w.shape[0] // 2, w.shape[0] // 2), dim=0)
            state_dict[f"{i + 1}.mlp.gate_proj.weight"] = gate_w.contiguous()
            state_dict[f"{i + 1}.mlp.up_proj.weight"] = upproj_w.contiguous()
            state_dict.pop(f"{i + 1}.mlp.fc1_weight")

        # rename
        new_state_dict = {}
        te_replace_pairs = [
            (".mlp.layer_norm_weight", ".mlp.ln.weight"),
            (".mlp.fc2_weight", ".mlp.down_proj.weight"),
        ]
        for k, v in state_dict.items():
            for k1, k2 in te_replace_pairs:
                if k1 in k:
                    k = k.replace(k1, k2)
            new_state_dict[k] = v
        state_dict = new_state_dict

    _k = list(state_dict.keys())[0]
    device = state_dict[_k].device

    layer_indices = list(set([int(k.split(".")[0]) for k in state_dict.keys()]))

    new_sd = {}
    for i in layer_indices:
        if i == 0:
            new_sd["model.embed_tokens.weight"] = state_dict["0.wte.weight"]
        elif i == config.n_layers + 1:
            if config.is_critic or not config.tied_embedding:
                new_sd["lm_head.weight"] = state_dict[f"{i}.weight"]
        else:
            new_sd[f"model.layers.{i-1}.input_layernorm.weight"] = state_dict[
                f"{i}.attn.c_attn.ln.weight"
            ]
            new_sd[f"model.layers.{i-1}.mlp.down_proj.weight"] = state_dict[
                f"{i}.mlp.down_proj.weight"
            ]
            new_sd[f"model.layers.{i-1}.mlp.gate_proj.weight"] = state_dict[
                f"{i}.mlp.gate_proj.weight"
            ]
            new_sd[f"model.layers.{i-1}.mlp.up_proj.weight"] = state_dict[
                f"{i}.mlp.up_proj.weight"
            ]
            if config.use_mlp_bias:
                new_sd[f"model.layers.{i-1}.mlp.down_proj.bias"] = state_dict[
                    f"{i}.mlp.down_proj.bias"
                ]
                new_sd[f"model.layers.{i-1}.mlp.gate_proj.bias"] = state_dict[
                    f"{i}.mlp.gate_proj.bias"
                ]
                new_sd[f"model.layers.{i-1}.mlp.up_proj.bias"] = state_dict[
                    f"{i}.mlp.up_proj.bias"
                ]
            new_sd[f"model.layers.{i-1}.post_attention_layernorm.weight"] = state_dict[
                f"{i}.mlp.ln.weight"
            ]
            new_sd[f"model.layers.{i-1}.self_attn.k_proj.weight"] = state_dict[
                f"{i}.attn.c_attn.k_attn.weight"
            ]
            new_sd[f"model.layers.{i-1}.self_attn.o_proj.weight"] = state_dict[
                f"{i}.attn.c_proj.weight"
            ]
            new_sd[f"model.layers.{i-1}.self_attn.q_proj.weight"] = state_dict[
                f"{i}.attn.c_attn.q_attn.weight"
            ]
            new_sd[f"model.layers.{i-1}.self_attn.v_proj.weight"] = state_dict[
                f"{i}.attn.c_attn.v_attn.weight"
            ]
            if config.use_attention_bias:
                new_sd[f"model.layers.{i-1}.self_attn.q_proj.bias"] = state_dict[
                    f"{i}.attn.c_attn.q_attn.bias"
                ]
                new_sd[f"model.layers.{i-1}.self_attn.k_proj.bias"] = state_dict[
                    f"{i}.attn.c_attn.k_attn.bias"
                ]
                new_sd[f"model.layers.{i-1}.self_attn.v_proj.bias"] = state_dict[
                    f"{i}.attn.c_attn.v_attn.bias"
                ]
            if config.use_attn_proj_bias:
                new_sd[f"model.layers.{i-1}.self_attn.o_proj.bias"] = state_dict[
                    f"{i}.attn.c_proj.bias"
                ]
            new_sd[f"model.layers.{i-1}.self_attn.rotary_emb.inv_freq"] = 1.0 / (
                config.rotary_base
                ** (
                    torch.arange(
                        0,
                        config.head_dim,
                        2,
                        device=device,
                        dtype=torch.float32,
                    )
                    / config.head_dim
                )
            )
            if i == config.n_layers:
                new_sd["model.norm.weight"] = state_dict[f"{i}.ln_f.weight"]

    return new_sd


# param name is used to load directly from huggingface checkpoints
def llama_embedding_layer_names(config: ReaLModelConfig) -> List[str]:
    return ["model.embed_tokens.weight"]


def llama_transformer_block_param_name(config: ReaLModelConfig, idx: int) -> List[str]:
    names = []
    for k in ["weight", "bias"]:
        names += [
            f"model.layers.{idx}.input_layernorm.{k}",
            f"model.layers.{idx}.mlp.down_proj.{k}",
            f"model.layers.{idx}.mlp.gate_proj.{k}",
            f"model.layers.{idx}.mlp.up_proj.{k}",
            f"model.layers.{idx}.post_attention_layernorm.{k}",
            f"model.layers.{idx}.self_attn.k_proj.{k}",
            f"model.layers.{idx}.self_attn.o_proj.{k}",
            f"model.layers.{idx}.self_attn.q_proj.{k}",
            # f"model.layers.{idx}.self_attn.rotary_emb.inv_freq",
            f"model.layers.{idx}.self_attn.v_proj.{k}",
        ]
        if idx == config.n_layers - 1:
            names += [f"model.norm.{k}"]
    return names


def llama_output_head_param_name(config: ReaLModelConfig) -> List[str]:
    if config.tied_embedding and not config.is_critic:
        return ["model.embed_tokens.weight"]
    else:
        return ["lm_head.weight"]


def convert_config_llama(
    hf_config: transformers.LlamaConfig,
) -> ReaLModelConfig:
    return ReaLModelConfig(
        n_layers=hf_config.num_hidden_layers,
        n_kv_heads=hf_config.num_key_value_heads,
        n_q_heads=hf_config.num_attention_heads,
        hidden_dim=hf_config.hidden_size,
        head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
        intermediate_dim=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        n_positions=hf_config.max_position_embeddings,
        embd_pdrop=0.0,
        attn_pdrop=(
            hf_config.attention_dropout
            if hasattr(hf_config, "attention_dropout")
            else 0.1
        ),
        layer_norm_epsilon=hf_config.rms_norm_eps,
        activation_function=hf_config.hidden_act,
        use_attention_bias=hf_config.attention_bias,
        use_attn_proj_bias=hf_config.attention_bias,
        scale_attn_by_inverse_layer_idx=False,
        layer_norm_type="rms",
        mlp_type="llama",
        apply_rotary=True,
        rotary_base=hf_config.rope_theta,
        rotary_interleaved=False,
        rotary_scaling=(
            None if hf_config.rope_scaling is None else hf_config.rope_scaling["factor"]
        ),
        rotary_scaling_type=(
            None if hf_config.rope_scaling is None else hf_config.rope_scaling["type"]
        ),
    )


def convert_config_back_llama(
    config: ReaLModelConfig,
) -> transformers.LlamaConfig:
    rope_scaling = {}
    if config.rotary_scaling is not None:
        rope_scaling["factor"] = config.rotary_scaling
    if config.rotary_scaling_type is not None:
        rope_scaling["type"] = config.rotary_scaling_type
    return transformers.LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_dim,
        intermediate_size=config.intermediate_dim,
        num_hidden_layers=config.n_layers,
        num_key_value_heads=config.n_kv_heads,
        num_attention_heads=config.n_q_heads,
        max_position_embeddings=config.n_positions,
        rms_norm_eps=config.layer_norm_epsilon,
        hidden_act=config.activation_function,
        attention_dropout=config.attn_pdrop,
        attention_bias=config.use_attention_bias,
        rope_theta=config.rotary_base,
        rope_scaling=rope_scaling if rope_scaling else None,
        architectures=["LlamaForCausalLM"],
    )


def make_real_config_llama():
    hf_config = transformers.LlamaConfig(
        vocab_size=TESTING_MODEL_VOCAB_SIZE,
        max_position_embeddings=TESTING_MODEL_N_POSITIONS,
        hidden_size=TESTING_MODEL_HIDDEN_SIZE,
        intermediate_size=TESTING_MODEL_INTERMEDIATE_SIZE,
        num_hidden_layers=TESTING_MODEL_N_LAYERS,
        num_attention_heads=TESTING_MODEL_N_HEADS,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
    )
    return convert_config_llama(hf_config)


for name in [
    "llama",
    "codellama",
    "deepseek",
]:
    register_hf_family(
        name=name,
        hf_cls_name="LlamaForCausalLM",
        config_from_hf_converter=convert_config_llama,
        config_to_hf_converter=convert_config_back_llama,
        sd_from_hf_converter=convert_state_dict_llama,
        sd_to_hf_converter=to_llama_state_dict,
        embedding_param_names=llama_embedding_layer_names,
        tblock_param_names=llama_transformer_block_param_name,
        head_param_names=llama_output_head_param_name,
        real_config_maker=make_real_config_llama,
    )
