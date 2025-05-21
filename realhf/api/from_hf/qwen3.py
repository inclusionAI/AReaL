# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import os
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

from .llama import (
    convert_state_dict_llama,
    llama_embedding_layer_names,
    llama_output_head_param_name,
    llama_transformer_block_param_name,
    to_llama_state_dict,
)


TRANSFRMER_QWEN3_CONFIG = transformers.Qwen3Config # transformers.Qwen3Config

def convert_config_qwen3(
    hf_config: TRANSFRMER_QWEN3_CONFIG,
) -> ReaLModelConfig:
    return ReaLModelConfig(
        n_layers=hf_config.num_hidden_layers,
        n_kv_heads=hf_config.num_key_value_heads,
        hidden_dim=hf_config.hidden_size,
        n_q_heads=hf_config.num_attention_heads,
        head_dim=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
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
        use_attention_bias=False,
        use_attn_proj_bias=False,
        scale_attn_by_inverse_layer_idx=False,
        layer_norm_type="rms",
        qk_layernorm=True,
        mlp_type="llama",
        apply_rotary=True,
        rotary_base=hf_config.rope_theta,
        rotary_interleaved=False,
        tied_embedding=hf_config.tie_word_embeddings,
    )


def convert_config_back_qwen3(
    config: ReaLModelConfig,
) -> TRANSFRMER_QWEN3_CONFIG:
    return TRANSFRMER_QWEN3_CONFIG(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_dim,
        intermediate_size=config.intermediate_dim,
        num_hidden_layers=config.n_layers,
        num_key_value_heads=config.n_kv_heads,
        num_attention_heads=config.n_q_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.n_positions,
        rms_norm_eps=config.layer_norm_epsilon,
        hidden_act=config.activation_function,
        attention_dropout=config.attn_pdrop,
        rope_theta=config.rotary_base,
        architectures=["Qwen3ForCausalLM"], # ["Qwen3ForCausalLM"],
        tie_word_embeddings=config.tied_embedding,
    )


def qwen3_config_maker():
    hf_config = TRANSFRMER_QWEN3_CONFIG(
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
    return convert_config_qwen3(hf_config)

def convert_state_dict_qwen3(state_dict: Dict, config: ReaLModelConfig) -> Dict:
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
                ("attn.q_norm.", "attn.q_ln."),
                ("attn.k_norm.", "attn.k_ln."),
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

def qwen3_transformer_block_param_name(config: ReaLModelConfig, idx: int) -> List[str]:
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
    # Qwen3
    if config.qk_layernorm:
        names += [
            f"model.layers.{idx}.self_attn.q_norm.weight",
            f"model.layers.{idx}.self_attn.k_norm.weight",
        ]
    return names



register_hf_family(
    name="qwen3",
    hf_cls_name="Qwen3ForCausalLM", # "Qwen3ForCausalLM"
    config_from_hf_converter=convert_config_qwen3,
    config_to_hf_converter=convert_config_back_qwen3,
    sd_from_hf_converter=convert_state_dict_qwen3,
    sd_to_hf_converter=to_llama_state_dict,
    embedding_param_names=llama_embedding_layer_names,
    tblock_param_names=qwen3_transformer_block_param_name,
    head_param_names=llama_output_head_param_name,
    real_config_maker=qwen3_config_maker,
)