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

from .llama import (
    convert_state_dict_llama,
    llama_embedding_layer_names,
    llama_output_head_param_name,
    llama_transformer_block_param_name,
    to_llama_state_dict,
)


def convert_config_gemma(
    hf_config: transformers.GemmaConfig,
) -> ReaLModelConfig:
    if hf_config.hidden_activation is None:
        act = "gelu_pytorch_tanh"
    else:
        act = hf_config.hidden_activation
    return ReaLModelConfig(
        n_layers=hf_config.num_hidden_layers,
        n_kv_heads=hf_config.num_key_value_heads,
        hidden_dim=hf_config.hidden_size,
        n_q_heads=hf_config.num_attention_heads,
        head_dim=hf_config.head_dim,
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
        activation_function=act,  # NOTE: here is different than LLaMA
        use_attention_bias=hf_config.attention_bias,
        use_attn_proj_bias=hf_config.attention_bias,
        scale_attn_by_inverse_layer_idx=False,
        layer_norm_type="gemma",
        mlp_type="llama",
        apply_rotary=True,
        rotary_base=hf_config.rope_theta,
        rotary_interleaved=False,
        tied_embedding=hf_config.tie_word_embeddings,
        normalize_embed=True,
    )


def convert_config_back_gemma(
    config: ReaLModelConfig,
) -> transformers.GemmaConfig:
    return transformers.GemmaConfig(
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
        hidden_activation=config.activation_function,
        attention_bias=config.use_attention_bias,
        attention_dropout=config.attn_pdrop,
        rope_theta=config.rotary_base,
        tie_word_embeddings=config.tied_embedding,
        architectures=["GemmaForCausalLM"],
    )


def gemma_config_maker() -> ReaLModelConfig:
    hf_config = transformers.GemmaConfig(
        attention_bias=False,
        hidden_act="gelu",
        hidden_size=TESTING_MODEL_HIDDEN_SIZE,
        intermediate_size=TESTING_MODEL_INTERMEDIATE_SIZE,
        max_position_embeddings=TESTING_MODEL_N_POSITIONS,
        num_attention_heads=TESTING_MODEL_N_HEADS,
        num_hidden_layers=TESTING_MODEL_N_LAYERS,
        num_key_value_heads=4,
        head_dim=TESTING_MODEL_HEAD_DIM,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        vocab_size=TESTING_MODEL_VOCAB_SIZE,
    )
    return convert_config_gemma(hf_config)


register_hf_family(
    name="gemma",
    hf_cls_name="GemmaForCausalLM",
    config_from_hf_converter=convert_config_gemma,
    config_to_hf_converter=convert_config_back_gemma,
    sd_from_hf_converter=convert_state_dict_llama,
    sd_to_hf_converter=to_llama_state_dict,
    embedding_param_names=llama_embedding_layer_names,
    tblock_param_names=llama_transformer_block_param_name,
    head_param_names=llama_output_head_param_name,
    real_config_maker=gemma_config_maker,
)
