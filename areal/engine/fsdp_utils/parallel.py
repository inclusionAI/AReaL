from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed import ProcessGroup
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

from areal.api.alloc_mode import FSDPParallelStrategy
from areal.api.cli_args import FSDPWrapPolicy, TrainEngineConfig
from areal.engine.core.model import (
    is_gemma3_model,
    is_moe_model,
    is_qwen3_vl_model,
    is_valid_vision_model,
)
from areal.engine.core.topology import DeviceMeshTopology
from areal.engine.fsdp_utils import apply_fsdp2
from areal.models.parallel_styles import ReplicateParallel

__all__ = ["ReplicateParallel", "ParallelHelper", "parallelize_model"]


@dataclass
class ParallelHelper:
    """FSDP parallel helper that delegates to DeviceMeshTopology.

    Maintains backward-compatible sp-based naming (sp_size, sp_group, dp_sp, sp_tp)
    as aliases for the canonical cp-based names.
    """

    _ps: FSDPParallelStrategy
    _topology: DeviceMeshTopology | None = None

    @classmethod
    def from_parallel_strategy(cls, fsdp_ps: FSDPParallelStrategy) -> "ParallelHelper":
        assert fsdp_ps.pp_size == 1, "Pipeline parallelism is not supported in FSDP"

        return cls(_ps=fsdp_ps)

    def __str__(self) -> str:
        _ps = self._ps
        return f"(dp={_ps.dp_size}, sp={_ps.cp_size}, tp={_ps.tp_size}, ep={_ps.ep_size}, etp={_ps.etp_size}, world_size={_ps.world_size})"

    def __post_init__(self):
        self._topology = DeviceMeshTopology(self._ps)

    @property
    def topology(self) -> DeviceMeshTopology:
        if self._topology is None:
            self._topology = DeviceMeshTopology(self._ps)
        return self._topology

    def build_mesh(self) -> DeviceMesh:
        return self.topology.build_mesh()

    @property
    def world_mesh(self) -> DeviceMesh:
        return self.topology.world_mesh

    # Enabled flags
    @property
    def dp_enabled(self) -> bool:
        return self.topology.dp_enabled

    @property
    def sp_enabled(self) -> bool:
        return self.topology.cp_enabled

    @property
    def cp_enabled(self) -> bool:
        return self.topology.cp_enabled

    @property
    def tp_enabled(self) -> bool:
        return self.topology.tp_enabled

    @property
    def ep_enabled(self) -> bool:
        return self.topology.ep_enabled

    @property
    def etp_enabled(self) -> bool:
        return self.topology.etp_enabled

    # Size properties
    @property
    def dp_size(self) -> int:
        return self.topology.dp_size

    @property
    def sp_size(self) -> int:
        """Backward-compatible alias for cp_size."""
        return self.topology.cp_size

    @property
    def cp_size(self) -> int:
        return self.topology.cp_size

    @property
    def tp_size(self) -> int:
        return self.topology.tp_size

    @property
    def ep_size(self) -> int:
        return self.topology.ep_size

    @property
    def etp_size(self) -> int:
        return self.topology.etp_size

    # Process groups (backward-compatible sp naming)
    @property
    def dp_group(self) -> ProcessGroup:
        return self.world_mesh["dp"].get_group()

    @property
    def sp_group(self) -> ProcessGroup:
        """Backward-compatible alias for cp group."""
        return self.world_mesh["sp"].get_group()

    @property
    def cp_group(self) -> ProcessGroup:
        return self.world_mesh["cp"].get_group()

    @property
    def tp_group(self) -> ProcessGroup:
        return self.world_mesh["tp"].get_group()

    @property
    def gradient_div_factor(self) -> int:
        return self.topology.gradient_divide_factor

    @property
    def context_and_model_parallel_size(self) -> int:
        return self.topology.context_and_model_parallel_size

    @property
    def seq_len_divisor(self) -> int:
        return self.topology.seq_len_divisor


def apply_non_moe_tp(
    model: nn.Module,
    model_config: PretrainedConfig,
    parallel_helper: ParallelHelper,
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

    tensor_parallel_size = parallel_helper.tp_size

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
    model_tp_plan: dict[str, ParallelStyle] = {
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
        "layers.*.self_attn.q_norm": ReplicateParallel(),
        "layers.*.self_attn.k_norm": ReplicateParallel(),
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
    root_tp_plan: dict[str, ParallelStyle] = {}
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Module):
        # Implicitly all-gather in ColwiseParallel
        # Output is sharded on the last dimension (Shard(2))
        root_tp_plan["lm_head"] = ColwiseParallel(
            input_layouts=Shard(1),
        )
    if hasattr(model, "score") and isinstance(model.score, nn.Module):
        # For PPO's critic model's score layer:
        # 1. The input is sharded by sequence parallelism (Shard(1))
        # 2. `score` is a linear layer with replicated weights
        # 3. All-gather the output along the sequence dimension to get the full results
        root_tp_plan["score"] = ReplicateParallel(
            input_layout=Shard(1),
            desired_input_layout=Shard(1),
            output_layout=Replicate(),
        )

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

            # For Qwen3 VL, patch _deepstack_process for TP
            if is_qwen3_vl_model(model_config.model_type):
                # NOTE: Lazy import to avoid ImportError when qwen3_vl model is not used.
                # transformers.models.qwen3_vl doesn't exist in all transformers versions,
                # so we only import it when actually needed for Qwen3 VL models.
                from areal.models.transformers.qwen3_vl import (
                    patch_qwen3_vl_deepstack_process_for_tp,
                )

                patch_qwen3_vl_deepstack_process_for_tp(model.model.language_model)

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
    parallel_helper: ParallelHelper,
    cpu_offload: CPUOffloadPolicy | None = None,
    wrap_policy: FSDPWrapPolicy | None = None,
):
    tp_enabled = parallel_helper.tp_enabled

    if tp_enabled:
        apply_non_moe_tp(model, model_config, parallel_helper, nd_device_mesh["tp"])

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
