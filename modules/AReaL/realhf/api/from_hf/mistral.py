# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

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


def config_from_mistral(hf_config: transformers.MistralConfig) -> ReaLModelConfig:
    return ReaLModelConfig(
        n_layers=hf_config.num_hidden_layers,
        vocab_size=hf_config.vocab_size,
        hidden_dim=hf_config.hidden_size,
        n_q_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
        intermediate_dim=hf_config.intermediate_size,
        activation_function=hf_config.hidden_act,
        n_positions=hf_config.max_position_embeddings,
        layer_norm_epsilon=hf_config.rms_norm_eps,
        layer_norm_type="rms",
        tied_embedding=hf_config.tie_word_embeddings,
        mlp_type="llama",
        rotary_base=hf_config.rope_theta,
        apply_rotary=True,
        attn_pdrop=hf_config.attention_dropout,
        resid_pdrop=0.0,
        use_attention_bias=False,
        use_attn_proj_bias=False,
        embd_pdrop=0.0,
        sliding_window=hf_config.sliding_window,
        scale_attn_by_inverse_layer_idx=False,
    )


def config_to_mistral(config: ReaLModelConfig) -> transformers.MistralConfig:
    return transformers.MistralConfig(
        num_hidden_layers=config.n_layers,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_dim,
        num_attention_heads=config.n_q_heads,
        num_key_value_heads=config.n_kv_heads,
        intermediate_size=config.intermediate_dim,
        hidden_act=config.activation_function,
        max_position_embeddings=config.n_positions,
        rms_norm_eps=config.layer_norm_epsilon,
        tie_word_embeddings=False,
        rope_theta=config.rotary_base,
        attention_dropout=config.attn_pdrop,
        sliding_window=config.sliding_window,
        architectures=["MistralForCausalLM"],
    )


def get_real_config_mistral() -> ReaLModelConfig:
    hf_config = transformers.MistralConfig(
        vocab_size=TESTING_MODEL_VOCAB_SIZE,
        max_position_embeddings=TESTING_MODEL_N_POSITIONS,
        hidden_size=TESTING_MODEL_HIDDEN_SIZE,
        intermediate_size=TESTING_MODEL_INTERMEDIATE_SIZE,
        num_hidden_layers=TESTING_MODEL_N_LAYERS,
        num_attention_heads=TESTING_MODEL_N_HEADS,
        num_key_value_heads=2,
    )
    return config_from_mistral(hf_config)


register_hf_family(
    "mistral",
    "MistralForCausalLM",
    config_from_hf_converter=config_from_mistral,
    config_to_hf_converter=config_to_mistral,
    sd_from_hf_converter=convert_state_dict_llama,
    sd_to_hf_converter=to_llama_state_dict,
    embedding_param_names=llama_embedding_layer_names,
    tblock_param_names=llama_transformer_block_param_name,
    head_param_names=llama_output_head_param_name,
    real_config_maker=get_real_config_mistral,
)
