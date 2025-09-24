from typing import Dict

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers import PretrainedConfig

from areal.api.cli_args import FSDPWrapPolicy, TrainEngineConfig
from areal.utils.fsdp import apply_fsdp2
from areal.utils.fsdp.parallel import NoParallel
from areal.utils.fsdp.parallel_dims import FSDPParallelDims
from areal.utils.model import is_gemma3_model, is_moe_model, is_valid_vision_model


def apply_non_moe_tp(
    model: nn.Module,
    model_config: PretrainedConfig,
    parallel_dims: FSDPParallelDims,
    tp_device_mesh: DeviceMesh,
):
    num_attention_heads: int
    num_key_value_heads: int
    try:
        num_attention_heads, num_key_value_heads = (
            model.config.num_attention_heads,  # type: ignore
            model.config.num_key_value_heads,  # type: ignore
        )
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,  # type: ignore
            model.config.text_config.num_key_value_heads,  # type: ignore
        )

    tensor_parallel_size = parallel_dims.tp

    if (
        num_attention_heads % tensor_parallel_size != 0
        or num_key_value_heads % tensor_parallel_size != 0
    ):
        raise ValueError(
            f"num_attention_heads {num_attention_heads} and num_key_value_heads {num_key_value_heads} must be divisible by tensor_parallel_size {tensor_parallel_size}"
        )

    if not isinstance(model.model, nn.Module):
        raise RuntimeError("Model does not have the required submodule 'model'.")

    # For model or model.language_model
    model_tp_plan: Dict[str, ParallelStyle] = {
        "embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        "layers.*.input_layernorm": SequenceParallel(),
        # All-gather
        "layers.*.self_attn": PrepareModuleInput(
            input_kwarg_layouts={"hidden_states": Shard(1)},
            desired_input_kwarg_layouts={"hidden_states": Replicate()},
        ),
        "layers.*.self_attn.q_proj": ColwiseParallel(),
        "layers.*.self_attn.k_proj": ColwiseParallel(),
        "layers.*.self_attn.v_proj": ColwiseParallel(),
        # special q/k norm for qwen3
        "layers.*.self_attn.q_norm": NoParallel(),
        "layers.*.self_attn.k_norm": NoParallel(),
        # Reduce in RowwiseParallel, Scatter by Shard(1)
        "layers.*.self_attn.o_proj": RowwiseParallel(
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        "layers.*.post_attention_layernorm": SequenceParallel(),
        "norm": SequenceParallel(),
    }

    if not is_moe_model(model_config.model_type):
        model_tp_plan.update(
            {
                # All-gather
                "layers.*.mlp": PrepareModuleInput(
                    input_layouts=Shard(1),
                    desired_input_layouts=Replicate(),
                ),
                "layers.*.mlp.gate_proj": ColwiseParallel(),
                "layers.*.mlp.up_proj": ColwiseParallel(),
                # Reduce in RowwiseParallel, Scatter by Shard(1)
                "layers.*.mlp.down_proj": RowwiseParallel(
                    output_layouts=Shard(1),
                    use_local_output=False,
                ),
            }
        )

    if is_gemma3_model(model_config.model_type):
        model_tp_plan.update(
            {
                "layers.*.pre_feedforward_layernorm": SequenceParallel(),
                "layers.*.post_feedforward_layernorm": SequenceParallel(),
            }
        )

    # For root module
    root_tp_plan: Dict[str, ParallelStyle] = {
        # All-gather
        "lm_head": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate(),
        ),
    }

    if is_valid_vision_model(model_config.model_type):
        if isinstance(model.model.language_model, nn.Module):
            # For vision-language models, avoid sharding the embedding layer because
            # the visual components access it without tensor parallelism support.
            # Instead, configure the first transformer layer to handle input
            # sharding properly.
            model_tp_plan.pop("embed_tokens", None)
            model_tp_plan["layers.0"] = PrepareModuleInput(
                input_layouts=Replicate(),
                desired_input_layouts=Shard(1),
            )

            parallelize_module(
                model.model.language_model,
                device_mesh=tp_device_mesh,
                parallelize_plan=model_tp_plan,
            )
        else:
            raise RuntimeError(
                "Vision model does not have the required submodule 'model.language_model'"
            )
    else:
        parallelize_module(
            model.model,
            device_mesh=tp_device_mesh,
            parallelize_plan=model_tp_plan,
        )

    parallelize_module(
        model,
        device_mesh=tp_device_mesh,
        parallelize_plan=root_tp_plan,
    )


def parallelize_model(
    model: nn.Module,
    config: TrainEngineConfig,
    model_config: PretrainedConfig,
    nd_device_mesh: DeviceMesh,
    parallel_dims: FSDPParallelDims,
    cpu_offload: CPUOffloadPolicy | None = None,
    wrap_policy: FSDPWrapPolicy | None = None,
):
    tp_enabled = parallel_dims.tp_enabled

    if tp_enabled:
        apply_non_moe_tp(model, model_config, parallel_dims, nd_device_mesh["tp"])

    mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, config.dtype),
        reduce_dtype=getattr(torch, config.grad_reduce_dtype),
        cast_forward_inputs=True,
    )
    fsdp_kwargs = {
        # This dim is guaranteed to exist by FSDPParallelDims
        "mesh": nd_device_mesh["dp_sp"],
        "mp_policy": mixed_precision_policy,
        "offload_policy": cpu_offload,
        "reshard_after_forward": True,
    }
    apply_fsdp2(model, fsdp_kwargs, wrap_policy)
