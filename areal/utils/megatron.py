import re

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from torch import Tensor
from torch.nn.parameter import Parameter

from areal.utils.fp8 import FP8BlockwiseTensorHelper, quantize_params


def _all_gather_and_concat(
    tensor: torch.Tensor,
    tp_size: int,
    tp_group,
    partition_dim: int,
    name: str,
) -> torch.Tensor:
    """All-gather tensor partitions and concatenate along partition dimension."""
    partitions = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(partitions, tensor, group=tp_group)

    # TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?
    # TODO: check only GLU is used.
    if "linear_fc1.weight" in name:
        partitions = [p.chunk(2, dim=0) for p in partitions]
        partitions = [p[0] for p in partitions] + [p[1] for p in partitions]

    # this is bug in megatron's grouped moe.
    partition_dim = (
        1 if "linear_fc2.weight" in name and partition_dim == 0 else partition_dim
    )

    return torch.cat(partitions, dim=partition_dim)


def _all_gather_fp8_tensor_and_concat(
    tensor,
    tp_size: int,
    tp_group,
    partition_dim: int,
    name: str,
    block_size: int = 128,
) -> FP8BlockwiseTensorHelper:
    """All-gather a Float8BlockwiseQTensor along the partition dimension.

    Returns FP8BlockwiseTensorHelper that wraps rowwise_data and rowwise_scale_inv.
    This allows conversion functions to work with FP8 tensors as regular tensors.
    """
    gathered_rowwise_data = _all_gather_and_concat(
        tensor._rowwise_data, tp_size, tp_group, partition_dim, name
    )
    gathered_rowwise_scale_inv = _all_gather_and_concat(
        tensor._rowwise_scale_inv, tp_size, tp_group, partition_dim, name
    )

    return FP8BlockwiseTensorHelper(
        gathered_rowwise_data, gathered_rowwise_scale_inv, block_size
    )


# Adapted from slime
def all_gather_param(
    name: str, param: Parameter | Tensor, fp8_direct_convert: bool = False
) -> torch.Tensor | FP8BlockwiseTensorHelper:
    if "expert_bias" in name:
        return param

    if not hasattr(param, "tensor_model_parallel"):
        raise ValueError(f"{name} does not have tensor_model_parallel attribute")

    param_is_fp8 = is_float8tensor(param)

    if (
        not param.tensor_model_parallel
        or getattr(param, "parallel_mode", None) == "duplicated"
    ):
        # For FP8 tensors, return the tensor directly without accessing .data
        # because accessing .data on QuantizedTensor triggers __torch_dispatch__
        # which dequantizes the tensor to bfloat16
        if param_is_fp8 and fp8_direct_convert:
            return param
        # If param is TE FP8, .data will implicitly convert TE FP8 to bf16,
        # and then be converted to PyTorch FP8 later in convert_to_hf
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    partition_dim = param.partition_dim
    assert param.partition_stride == 1, "partition_stride != 1 is not supported"

    # Handle FP8 tensors specially
    if param_is_fp8 and fp8_direct_convert:
        # Get block_size from quantization config if available
        # Default to 128 if not specified
        block_size = 128  # TODO: get from quantization_config if available
        return _all_gather_fp8_tensor_and_concat(
            param, tp_size, tp_group, partition_dim, name, block_size
        )

    # bf16/fp32
    param = _all_gather_and_concat(param.data, tp_size, tp_group, partition_dim, name)
    return param


# Adapted from slime
def remove_padding(
    name: str, param: Parameter | Tensor | FP8BlockwiseTensorHelper, vocab_size: int
):
    if (
        name == "module.module.embedding.word_embeddings.weight"
        or name == "module.module.output_layer.weight"
    ):
        return param[:vocab_size]
    return param


# Adapted from slime
def convert_qwen3moe_to_hf(
    tf_config: TransformerConfig,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
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

    if tf_config.num_query_groups is None:
        raise ValueError("Qwen3-MoE models should have num_query_groups")

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
    tf_config: TransformerConfig,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
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

    if tf_config.num_query_groups is None:
        raise ValueError("Qwen2 models should have num_query_groups")

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
def convert_deepseekv3_to_hf(
    tf_config: TransformerConfig,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
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
        elif rest == "self_attention.linear_q_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_proj.weight", param)]
        elif rest == "self_attention.linear_q_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_proj.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_b_proj.weight", param)]
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
        elif (
            rest == "self_attention.linear_qkv.layer_norm_weight"
            or rest == "input_layernorm.weight"
        ):
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "self_attention.linear_kv_down_proj.weight":
            return [
                (f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight", param)
            ]
        elif rest == "self_attention.linear_kv_up_proj.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight", param)
            ]
        elif rest == "self_attention.linear_kv_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight", param)]
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

    raise ValueError(f"Unknown parameter name: {name}")


# Adapted from slime
# A registry for conversion functions is more extensible.
_CONVERSION_FN_REGISTRY = {
    "qwen3_moe": convert_qwen3moe_to_hf,
    "qwen2": convert_qwen2_to_hf,
    "qwen3": convert_qwen2_to_hf,
    "deepseekv3": convert_deepseekv3_to_hf,
}


def convert_to_hf(
    tf_config: TransformerConfig,
    model_name: str,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
    quantization_config: dict[str, int | str | list[str]] | None = None,
    fp8_direct_convert: bool = False,
):
    """Convert Megatron parameter to HuggingFace format, optionally with FP8 quantization.

    Args:
        tf_config: Transformer configuration
        model_name: Model name (e.g., "qwen2", "qwen3_moe")
        name: Parameter name in Megatron format
        param: Parameter tensor or FP8BlockwiseTensorHelper
        quantization_config: Optional quantization config dict with keys:
            - quant_method: "fp8"
            - fmt: "e4m3"
            - activation_scheme: "dynamic"
            - weight_block_size: Optional tuple/list of [block_m, block_n] for blockwise quantization
        fp8_direct_convert: If True, directly convert TE FP8 tensors to PyTorch FP8 format.
            If False, dequantize TE FP8 to bf16 first, then quantize to PyTorch FP8.

    Returns:
        List of (name, tensor) tuples in HuggingFace format. For FP8 quantization,
        returns both quantized weight and scale tensors.
    """
    for key, conversion_fn in _CONVERSION_FN_REGISTRY.items():
        if key in model_name:
            converted_named_tensors = conversion_fn(tf_config, name, param)
            if quantization_config:
                if fp8_direct_convert:
                    converted_fp8_named_tensors = []
                    for hf_name, hf_tensor in converted_named_tensors:
                        if isinstance(hf_tensor, FP8BlockwiseTensorHelper):
                            # FP8BlockwiseTensorHelper from all_gather
                            weight, scale_inv = hf_tensor.to_pytorch_fp8()
                            converted_fp8_named_tensors.append((hf_name, weight))
                            scale_inv_name = f"{hf_name}_scale_inv"
                            converted_fp8_named_tensors.append(
                                (scale_inv_name, scale_inv)
                            )
                        else:
                            # Keep non-FP8 or non-weight tensors as is
                            converted_fp8_named_tensors.append((hf_name, hf_tensor))
                    return converted_fp8_named_tensors
                else:
                    # Quantize from bf16 to PyTorch FP8
                    return quantize_params(
                        name, converted_named_tensors, quantization_config
                    )
            return converted_named_tensors

    raise ValueError(f"Unsupported model for HF conversion: {model_name}")


def get_named_parameters(model_module, num_experts):
    def _iter_single(single_module):
        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        if num_experts:
            expert_offset = ep_rank * num_experts // ep_size
        else:
            expert_offset = 0

        config = getattr(single_module, "config", None)
        if config is None and hasattr(single_module, "module"):
            config = getattr(single_module.module, "config", None)
        if config is None:
            raise AttributeError("Megatron module does not expose transformer config")

        vp_stage = getattr(single_module, "virtual_pipeline_model_parallel_rank", None)
        if vp_stage is None and hasattr(single_module, "module"):
            vp_stage = getattr(
                single_module.module, "virtual_pipeline_model_parallel_rank", None
            )
        if vp_stage is None:
            try:
                vp_stage = mpu.get_virtual_pipeline_model_parallel_rank()
            except AssertionError:
                vp_stage = None

        layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for name, param in single_module.named_parameters():
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
                yield (
                    f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.weight{expert_idx}",
                    param,
                )
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield (
                    f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}",
                    param,
                )
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        for name, buffer in single_module.named_buffers():
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

    if isinstance(model_module, (list, tuple)):
        try:
            vp_world = mpu.get_virtual_pipeline_model_parallel_world_size()
            original_vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        except AssertionError:
            original_vp_rank = None
            vp_world = None

        for vpp_rank, single_module in enumerate(model_module):
            if vp_world and vp_world > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            yield from _iter_single(single_module)

        if (
            vp_world
            and vp_world > 1
            and original_vp_rank is not None
            and original_vp_rank >= 0
        ):
            mpu.set_virtual_pipeline_model_parallel_rank(original_vp_rank)
        return

    yield from _iter_single(model_module)
