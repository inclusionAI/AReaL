from dataclasses import dataclass
from functools import partial
from typing import Dict

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Placement
from transformers import PretrainedConfig

from areal.api.alloc_mode import FSDPParallelStrategy
from areal.api.cli_args import FSDPWrapPolicy, TrainEngineConfig
from areal.platforms import current_platform
from areal.utils.fsdp import apply_fsdp2
from areal.utils.model import is_gemma3_model, is_moe_model, is_valid_vision_model

__all__ = ["ReplicateParallel", "ParallelHelper", "parallelize_model"]


# Copied from torchtitan. Used for Qwen3 Q/K norm.
# NOTE: This is to achieve replicate computation on the gate module in the MoE router.
# It does nothing other than (1) setting the module parameters as DTensors on the given mesh
# and (2) inserting hooks to module boundary to change torch.Tensor to DTensor and back.
# The reason we need this wrapping is to ensure all parameters are on the same 1D/2D mesh,
# which is assumed by (1) gradient norm clipping, and (2) optimizer fused implementation.
class ReplicateParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn, self.input_layout, self.desired_input_layout
            ),  # type: ignore[arg-type]
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


@dataclass
class ParallelHelper:
    _ps: FSDPParallelStrategy
    _world_mesh: DeviceMesh | None = None

    @classmethod
    def from_parallel_strategy(cls, fsdp_ps: FSDPParallelStrategy) -> "ParallelHelper":
        assert fsdp_ps.pp_size == 1, "Pipeline parallelism is not supported in FSDP"

        return cls(_ps=fsdp_ps)

    def __str__(self) -> str:
        _ps = self._ps
        return f"(dp={_ps.dp_size}, sp={_ps.cp_size}, tp={_ps.tp_size}, ep={_ps.ep_size}, etp={_ps.etp_size}, world_size={_ps.world_size})"

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp, sp, tp, ep, etp, world_size = (
            self._ps.dp_size,
            self._ps.cp_size,
            self._ps.tp_size,
            self._ps.ep_size,
            self._ps.etp_size,
            self._ps.world_size,
        )
        for d in (sp, tp, ep, etp):
            assert d >= 1, "Parallelism degree should be >= 1"

        assert (
            dp * sp * tp == world_size
        ), f"Invalid parallel dims: dp({dp}) * sp({sp}) * tp({tp}) != WORLD_SIZE({world_size})"

        if ep > 1:
            assert etp == tp or etp == 1, "Currently we only support ETP=TP or ETP=1"
            if etp == tp:
                # ep would borrow all sp and some dp degree
                assert ep % sp == 0 and (dp * sp) % ep == 0
            elif etp == 1:
                # ep would borrow all sp and tp and some dp degree
                assert ep % (sp * tp) == 0 and (dp * sp * tp) % ep == 0

    def build_mesh(self) -> DeviceMesh:
        if self._ps.ep_size > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    def _build_mesh_with_ep(self) -> DeviceMesh:
        dp, sp, tp, ep, etp = (
            self._ps.dp_size,
            self._ps.cp_size,
            self._ps.tp_size,
            self._ps.ep_size,
            self._ps.etp_size,
        )

        # With ep, dp and ep are derived submeshes:
        # dp = dp_mod_ep * dp_in_ep
        if etp == tp:
            # ep = dp_in_ep * sp
            dp_mod_ep = dp * sp // ep
            dp_in_ep = ep // sp
        else:
            assert etp == 1
            # ep = dp_in_ep * sp * tp
            dp_mod_ep = dp * sp * tp // ep
            dp_in_ep = ep // (sp * tp)

        mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(dp_mod_ep, dp_in_ep, sp, tp),
            mesh_dim_names=("dp_mod_ep", "dp_in_ep", "sp", "tp"),
        )

        # Create all the submesh here for process groups
        # Guaranteed dims:
        #     root mesh: dp_mod_ep, dp_in_ep, sp, tp
        #     sub  mesh: dp, dp_sp, sp_tp, ep
        mesh["dp_mod_ep", "dp_in_ep"]._flatten(mesh_dim_name="dp")
        mesh["dp_mod_ep", "dp_in_ep", "sp"]._flatten(mesh_dim_name="dp_sp")
        mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")
        ep_mesh_dim_names = ("dp_in_ep", "sp", "tp") if etp == 1 else ("dp_in_ep", "sp")
        mesh[tuple(ep_mesh_dim_names)]._flatten(mesh_dim_name="ep")

        return mesh

    def _build_mesh_without_ep(self) -> DeviceMesh:
        dp, sp, tp = (self._ps.dp_size, self._ps.cp_size, self._ps.tp_size)

        mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(dp, sp, tp),
            mesh_dim_names=("dp", "sp", "tp"),
        )

        # Create all the submesh here for process groups
        # Guaranteed dims:
        #     root mesh: dp, sp, tp
        #     sub  mesh: dp_sp, sp_tp
        mesh["dp", "sp"]._flatten(mesh_dim_name="dp_sp")
        mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")

        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self) -> bool:
        return self._ps.dp_size > 1

    @property
    def sp_enabled(self) -> bool:
        return self._ps.cp_size > 1

    @property
    def tp_enabled(self) -> bool:
        return self._ps.tp_size > 1

    @property
    def ep_enabled(self) -> bool:
        return self._ps.ep_size > 1

    @property
    def etp_enabled(self) -> bool:
        return self._ps.etp_size > 1

    @property
    def dp_size(self) -> int:
        return self._ps.dp_size

    @property
    def sp_size(self) -> int:
        return self._ps.cp_size

    @property
    def tp_size(self) -> int:
        return self._ps.tp_size

    @property
    def ep_size(self) -> int:
        return self._ps.ep_size

    @property
    def etp_size(self) -> int:
        return self._ps.etp_size

    @property
    def gradient_div_factor(self) -> int:
        # This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        # Although the FSDP sharding of experts is done on a mesh of a different size than
        # other parameters, the gradient division factor should be consistent with data.
        return self._ps.dp_size * self._ps.cp_size

    @property
    def non_data_parallel_size(self) -> int:
        return self._ps.cp_size * self._ps.tp_size

    @property
    def seq_len_divisor(self) -> int:
        # 1. Sequence Parallel requires that seq_len be divisible by TP degree.
        # 2. Ulysses Sequence Parallel requires that seq_len be divisible by SP degree.
        return self._ps.tp_size * self._ps.cp_size


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
    root_tp_plan: Dict[str, ParallelStyle] = {}
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Module):
        # All-gather
        root_tp_plan["lm_head"] = ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate(),
        )
    if hasattr(model, "score") and isinstance(model.score, nn.Module):
        # For PPO's critic model's score layer:
        # 1. The input is sharded by sequence parallelism
        # 2. `score` is a linear layer, but its weight is not a DTensor. Use local input.
        # 3. All-gather the output along the sequence dimension to get the full results.
        root_tp_plan["score"] = PrepareModuleInputOutput(
            input_layouts=Shard(1),
            desired_input_layouts=Shard(1),
            use_local_input=True,
            output_layouts=Shard(1),
            desired_output_layouts=Replicate(),
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
