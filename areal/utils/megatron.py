import re

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from torch import Tensor
from torch.nn.parameter import Parameter


# Adapted from slime
def all_gather_param(name: str, param: Parameter | Tensor):
    if "expert_bias" in name:
        return param

    assert hasattr(
        param, "tensor_model_parallel"
    ), f"{name} does not have tensor_model_parallel attribute"
    if (
        not param.tensor_model_parallel
        or getattr(param, "parallel_mode", None) == "duplicated"
    ):
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    partition_dim = param.partition_dim
    assert param.partition_stride == 1, "partition_stride != 1 is not supported"
    # TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?
    # TODO: check only GLU is used.
    if "linear_fc1.weight" in name:
        param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
        param_partitions = [p[0] for p in param_partitions] + [
            p[1] for p in param_partitions
        ]
    # this is bug in megatron's grouped moe.
    if "linear_fc2.weight" in name:
        if partition_dim == 0:
            partition_dim = 1
    param = torch.cat(param_partitions, dim=partition_dim)
    return param


# Adapted from slime
def remove_padding(name: str, param: Parameter | Tensor, vocab_size: int):
    if (
        name == "module.module.embedding.word_embeddings.weight"
        or name == "module.module.output_layer.weight"
    ):
        return param[:vocab_size]
    return param


# Adapted from slime
def convert_qwen3moe_to_hf(
    tf_config: TransformerConfig, name: str, param: Parameter | Tensor
):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = (
            tf_config.kv_channels
            if tf_config.kv_channels is not None
            else tf_config.hidden_size // tf_config.num_attention_heads
        )
    except (AttributeError, TypeError):
        head_dim = tf_config.hidden_size // tf_config.num_attention_heads
    value_num_per_group = tf_config.num_attention_heads // tf_config.num_query_groups

    assert (
        tf_config.num_query_groups is not None
    ), "Qwen3-MoE models should have num_query_groups"

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                        up_weight,
                    ),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                        param,
                    ),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
                        up_weight,
                    ),
                ]
            elif rest == "linear_fc2.weight":
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
                        param,
                    )
                ]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":

            param = param.view(
                tf_config.num_query_groups, -1, head_dim, tf_config.hidden_size
            )
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
            )
            q_param = q_param.reshape(-1, tf_config.hidden_size)
            k_param = k_param.reshape(-1, tf_config.hidden_size)
            v_param = v_param.reshape(-1, tf_config.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(tf_config.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[
                    value_num_per_group * head_dim,
                    head_dim,
                    head_dim,
                ],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "pre_mlp_layernorm.weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [
                (f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)
            ]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


# Adapted from slime
def convert_qwen2_to_hf(
    tf_config: TransformerConfig, name: str, param: Parameter | Tensor
):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = (
            tf_config.kv_channels
            if tf_config.kv_channels is not None
            else tf_config.hidden_size // tf_config.num_attention_heads
        )
    except (AttributeError, TypeError):
        head_dim = tf_config.hidden_size // tf_config.num_attention_heads
    value_num_per_group = tf_config.num_attention_heads // tf_config.num_query_groups

    assert (
        tf_config.num_query_groups is not None
    ), "Qwen2 models should have num_query_groups"

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":

            param = param.view(
                tf_config.num_query_groups, -1, head_dim, tf_config.hidden_size
            )
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
            )
            q_param = q_param.reshape(-1, tf_config.hidden_size)
            k_param = k_param.reshape(-1, tf_config.hidden_size)
            v_param = v_param.reshape(-1, tf_config.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(tf_config.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[
                    value_num_per_group * head_dim,
                    head_dim,
                    head_dim,
                ],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


# Adapted from slime
# A registry for conversion functions is more extensible.
_CONVERSION_FN_REGISTRY = {
    "qwen3_moe": convert_qwen3moe_to_hf,
    "qwen2": convert_qwen2_to_hf,
    "qwen3": convert_qwen2_to_hf,
}


def convert_to_hf(
    tf_config: TransformerConfig, model_name: str, name: str, param: Parameter | Tensor
):
    for key, conversion_fn in _CONVERSION_FN_REGISTRY.items():
        if key in model_name:
            return conversion_fn(tf_config, name, param)

    raise ValueError(f"Unsupported model for HF conversion: {model_name}")


def get_named_parameters(model_module, num_experts):
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    if num_experts:
        expert_offset = ep_rank * num_experts // ep_size

    # NOTE: vp_stage is always 0 when VP is not actually enabled
    layer_offset = get_transformer_layer_offset(model_module.config)
    for name, param in model_module.named_parameters():
        # for model without ddp wrap
        if not name.startswith("module.module."):
            name = "module." + name

        decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
        match = re.match(decoder_layers_pattern, name)
        if not match:
            mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
            match = re.match(mtp_layers_pattern, name)
            if not match:
                yield name, param
                continue

            # mtp layer starts from layer 0
            layer_idx, rest = match.groups()
            expert_pattern = r"transformer_layer.mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if not match:
                yield name, param
                continue

            rest, expert_idx = match.groups()
            expert_idx = int(expert_idx) + expert_offset
            yield f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.weight{expert_idx}", param
            continue

        layer_idx, rest = match.groups()
        layer_idx = int(layer_idx) + layer_offset

        # this is hardcoded for te grouped matmul
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            expert_idx = int(expert_idx) + expert_offset
            yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}", param
        else:
            yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

    # treat expert bias as normal parameters
    for name, buffer in model_module.named_buffers():
        if "expert_bias" not in name:
            continue
        # for model without ddp wrap
        if not name.startswith("module.module."):
            name = "module." + name

        decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
        match = re.match(decoder_layers_pattern, name)
        if not match:
            yield name, buffer
        else:
            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset
            yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer
