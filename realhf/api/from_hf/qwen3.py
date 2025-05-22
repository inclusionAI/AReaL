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

TRANSFRMER_QWEN3_CONFIG = transformers.Qwen3Config  # transformers.Qwen3Config


def convert_config_qwen3(
    hf_config: TRANSFRMER_QWEN3_CONFIG,
) -> ReaLModelConfig:
    return ReaLModelConfig(
        n_layers=hf_config.num_hidden_layers,
        n_kv_heads=hf_config.num_key_value_heads,
        hidden_dim=hf_config.hidden_size,
        n_q_heads=hf_config.num_attention_heads,
        head_dim=getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        ),
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
        architectures=["Qwen3ForCausalLM"],  # ["Qwen3ForCausalLM"],
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
        head_dim=TESTING_MODEL_HIDDEN_SIZE // TESTING_MODEL_N_HEADS,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
    )
    return convert_config_qwen3(hf_config)


def convert_state_dict_qwen3(state_dict: Dict, config: ReaLModelConfig) -> Dict:
    llama_state_dict = convert_state_dict_llama(state_dict, config)
    # model.layers.0.self_attn.k_norm.weight -> 1.attn.k_ln.weight
    new_state_dict = {}
    for k, v in llama_state_dict.items():
        if "k_norm" in k:
            k = k.replace("k_norm", "k_ln")
        if "q_norm" in k:
            k = k.replace("q_norm", "q_ln")
        new_state_dict[k] = v
    return new_state_dict


def convert_state_dict_back_qwen3(state_dict: Dict, config: ReaLModelConfig) -> Dict:
    new_sd = to_llama_state_dict(state_dict, config)
    layer_indices = list(set([int(k.split(".")[0]) for k in state_dict.keys()]))
    for i in layer_indices:
        if i == 0 or i == config.n_layers + 1:
            continue
        new_sd[f"model.layers.{i - 1}.self_attn.k_norm.weight"] = state_dict[
            f"{i}.attn.k_ln.weight"
        ]
        new_sd[f"model.layers.{i - 1}.self_attn.q_norm.weight"] = state_dict[
            f"{i}.attn.q_ln.weight"
        ]
    return new_sd


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
    hf_cls_name="Qwen3ForCausalLM",  # "Qwen3ForCausalLM"
    config_from_hf_converter=convert_config_qwen3,
    config_to_hf_converter=convert_config_back_qwen3,
    sd_from_hf_converter=convert_state_dict_qwen3,
    sd_to_hf_converter=convert_state_dict_back_qwen3,
    embedding_param_names=llama_embedding_layer_names,
    tblock_param_names=qwen3_transformer_block_param_name,
    head_param_names=llama_output_head_param_name,
    real_config_maker=qwen3_config_maker,
)
