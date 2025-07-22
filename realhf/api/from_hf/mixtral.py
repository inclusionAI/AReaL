# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

from typing import *

import torch
import transformers

from realhf.api.core.model_api import ReaLModelConfig, ReaLMoEConfig, register_hf_family
from realhf.base.testing import (
    TESTING_MODEL_HEAD_DIM,
    TESTING_MODEL_HIDDEN_SIZE,
    TESTING_MODEL_INTERMEDIATE_SIZE,
    TESTING_MODEL_N_HEADS,
    TESTING_MODEL_N_LAYERS,
    TESTING_MODEL_N_POSITIONS,
    TESTING_MODEL_VOCAB_SIZE,
)

from .llama import llama_embedding_layer_names, llama_output_head_param_name


def config_from_mixtral(hf_config: transformers.MixtralConfig) -> ReaLModelConfig:
    moe = ReaLMoEConfig(
        num_experts=hf_config.num_local_experts,
        top_k=hf_config.num_experts_per_tok,
        aux_loss_coeff=hf_config.router_aux_loss_coef,
        input_jitter_eps=hf_config.router_jitter_noise,
    )
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
        mlp_type="moe",
        tied_embedding=hf_config.tie_word_embeddings,
        rotary_base=hf_config.rope_theta,
        apply_rotary=True,
        attn_pdrop=hf_config.attention_dropout,
        resid_pdrop=0.0,
        use_attention_bias=False,
        use_attn_proj_bias=False,
        embd_pdrop=0.0,
        sliding_window=hf_config.sliding_window,
        moe=moe,
        scale_attn_by_inverse_layer_idx=False,
    )


def config_to_mixtral(config: ReaLModelConfig) -> transformers.MistralConfig:
    return transformers.MixtralConfig(
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
        num_local_experts=config.moe.num_experts,
        num_experts_per_tok=config.moe.top_k,
        router_aux_loss_coef=config.moe.aux_loss_coeff,
        router_jitter_noise=config.moe.input_jitter_eps,
    )


def convert_state_dict_mixtral(state_dict: Dict, config: ReaLModelConfig) -> Dict:
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
                ("block_sparse_moe.gate.", "mlp.router."),
                ("block_sparse_moe.experts", "mlp.experts.local_experts"),
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ]
            for k1, k2 in replace_pairs:
                if k1 in name:
                    name = name.replace(k1, k2)
            new_state_dict[f"{block_idx + 1}.{name}"] = v
    return new_state_dict


def to_mixtral_state_dict(
    state_dict: Dict[str, torch.Tensor], config: ReaLModelConfig
) -> Dict:
    _k = list(state_dict.keys())[0]
    device = state_dict[_k].device
    num_experts = config.moe.num_experts

    layer_indices = list(set([int(k.split(".")[0]) for k in state_dict.keys()]))

    new_sd = {}
    for i in layer_indices:
        if i == 0:
            new_sd["model.embed_tokens.weight"] = state_dict["0.wte.weight"]
        elif i == config.n_layers + 1:
            if config.is_critic or not config.tied_embedding:
                new_sd["lm_head.weight"] = state_dict[f"{i}.weight"]
        else:
            # moe layer
            new_sd[f"model.layers.{i-1}.block_sparse_moe.gate.weight"] = state_dict[
                f"{i}.mlp.router.weight"
            ]
            for j in range(num_experts):
                new_sd[f"model.layers.{i-1}.block_sparse_moe.experts.{j}.w1.weight"] = (
                    state_dict[f"{i}.mlp.experts.local_experts.{j}.gate_proj.weight"]
                )
                new_sd[f"model.layers.{i-1}.block_sparse_moe.experts.{j}.w2.weight"] = (
                    state_dict[f"{i}.mlp.experts.local_experts.{j}.down_proj.weight"]
                )
                new_sd[f"model.layers.{i-1}.block_sparse_moe.experts.{j}.w3.weight"] = (
                    state_dict[f"{i}.mlp.experts.local_experts.{j}.up_proj.weight"]
                )
            # others
            new_sd[f"model.layers.{i-1}.input_layernorm.weight"] = state_dict[
                f"{i}.attn.c_attn.ln.weight"
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


def mixtral_transformer_block_param_name(
    config: ReaLModelConfig, idx: int
) -> List[str]:
    names = []
    num_experts = config.moe.num_experts
    for k in ["weight", "bias"]:
        names += [
            f"model.layers.{idx}.input_layernorm.{k}",
            f"model.layers.{idx}.block_sparse_moe.gate.{k}",
            f"model.layers.{idx}.post_attention_layernorm.{k}",
            f"model.layers.{idx}.self_attn.k_proj.{k}",
            f"model.layers.{idx}.self_attn.o_proj.{k}",
            f"model.layers.{idx}.self_attn.q_proj.{k}",
            # f"model.layers.{idx}.self_attn.rotary_emb.inv_freq",
            f"model.layers.{idx}.self_attn.v_proj.{k}",
        ]
        for j in range(num_experts):
            names += [
                f"model.layers.{idx}.block_sparse_moe.experts.{j}.w1.{k}",
                f"model.layers.{idx}.block_sparse_moe.experts.{j}.w2.{k}",
                f"model.layers.{idx}.block_sparse_moe.experts.{j}.w3.{k}",
            ]
        if idx == config.n_layers - 1:
            names += [f"model.norm.{k}"]
    return names


def get_real_config_mixtral() -> ReaLModelConfig:
    hf_config = transformers.MixtralConfig(
        vocab_size=TESTING_MODEL_VOCAB_SIZE,
        max_position_embeddings=TESTING_MODEL_N_POSITIONS,
        hidden_size=TESTING_MODEL_HIDDEN_SIZE,
        intermediate_size=TESTING_MODEL_INTERMEDIATE_SIZE,
        num_hidden_layers=TESTING_MODEL_N_LAYERS,
        num_attention_heads=TESTING_MODEL_N_HEADS,
        num_key_value_heads=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_jitter_noise=None,
    )
    return config_from_mixtral(hf_config)


register_hf_family(
    "mixtral",
    "MixtralForCausalLM",
    config_from_hf_converter=config_from_mixtral,
    config_to_hf_converter=config_to_mixtral,
    sd_from_hf_converter=convert_state_dict_mixtral,
    sd_to_hf_converter=to_mixtral_state_dict,
    embedding_param_names=llama_embedding_layer_names,
    tblock_param_names=mixtral_transformer_block_param_name,
    head_param_names=llama_output_head_param_name,
    real_config_maker=get_real_config_mixtral,
)
