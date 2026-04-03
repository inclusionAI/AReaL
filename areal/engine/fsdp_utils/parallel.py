from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
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

from areal.api import FSDPParallelStrategy
from areal.api.cli_args import FSDPWrapPolicy, TrainEngineConfig
from areal.engine.core.model import (
    is_gemma3_model,
    is_moe_model,
    is_qwen3_vl_model,
    is_valid_vision_model,
)
from areal.engine.fsdp_utils import apply_fsdp2
from areal.infra.platforms import current_platform
from areal.models.parallel_styles import ReplicateParallel

__all__ = ["ReplicateParallel", "ParallelHelper", "parallelize_model"]


@dataclass
class ParallelHelper:
    _ps: FSDPParallelStrategy
    _world_mesh: DeviceMesh | None = None

    @classmethod
    def from_parallel_strategy(cls, fsdp_ps: FSDPParallelStrategy) -> "ParallelHelper":
        return cls(_ps=fsdp_ps)

    def __str__(self) -> str:
        _ps = self._ps
        s = f"(dp={_ps.dp_size}, sp={_ps.cp_size}, tp={_ps.tp_size}, ep={_ps.ep_size}, etp={_ps.etp_size}"
        if _ps.pp_size > 1:
            s += f", pp={_ps.pp_size}"
        s += f", world_size={_ps.world_size})"
        return s

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp, sp, tp, pp, ep, etp, world_size = (
            self._ps.dp_size,
            self._ps.cp_size,
            self._ps.tp_size,
            self._ps.pp_size,
            self._ps.ep_size,
            self._ps.etp_size,
            self._ps.world_size,
        )
        for d in (sp, tp, pp, ep, etp):
            assert d >= 1, "Parallelism degree should be >= 1"

        if dp * sp * tp * pp != world_size:
            raise ValueError(
                f"Invalid parallel dims: dp({dp}) * sp({sp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({world_size})"
            )

        if pp > 1 and ep > 1:
            raise ValueError(
                "FSDP Engine does not support PP + EP combination. "
                "Use Archon Engine for PP + EP support."
            )

        if ep > 1:
            assert etp == tp or etp == 1, "Currently we only support ETP=TP or ETP=1"
            if etp == tp:
                assert ep % sp == 0 and (dp * sp) % ep == 0
            elif etp == 1:
                assert ep % (sp * tp) == 0 and (dp * sp * tp) % ep == 0

    def build_mesh(self) -> DeviceMesh:
        if self._ps.pp_size > 1:
            return self._build_mesh_with_pp()
        elif self._ps.ep_size > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    def _build_mesh_with_pp(self) -> DeviceMesh:
        """Build device mesh with pipeline parallelism.

        Mesh dimensions (outermost to innermost): pp → dp → sp → tp
        This mirrors torchtitan's approach where pp is the outermost dimension.
        """
        dp, sp, tp, pp = (
            self._ps.dp_size,
            self._ps.cp_size,
            self._ps.tp_size,
            self._ps.pp_size,
        )

        mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(pp, dp, sp, tp),
            mesh_dim_names=("pp", "dp", "sp", "tp"),
        )

        # Create submeshes for process groups
        # dp_sp: used for FSDP sharding (data parallel + sequence parallel)
        mesh["dp", "sp"]._flatten(mesh_dim_name="dp_sp")
        # sp_tp: used for model parallel group (without PP dimension)
        mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")
        # pp_sp_tp: used for full model parallel group (includes PP dimension)
        # This is needed so that only one rank per DP group is the DP head.
        mesh["pp", "sp", "tp"]._flatten(mesh_dim_name="pp_sp_tp")

        return mesh

    def _build_mesh_with_ep(self) -> DeviceMesh:
        dp, sp, tp, ep, etp = (
            self._ps.dp_size,
            self._ps.cp_size,
            self._ps.tp_size,
            self._ps.ep_size,
            self._ps.etp_size,
        )

        if etp == tp:
            dp_mod_ep = dp * sp // ep
            dp_in_ep = ep // sp
        else:
            assert etp == 1
            dp_mod_ep = dp * sp * tp // ep
            dp_in_ep = ep // (sp * tp)

        mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(dp_mod_ep, dp_in_ep, sp, tp),
            mesh_dim_names=("dp_mod_ep", "dp_in_ep", "sp", "tp"),
        )

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

        mesh["dp", "sp"]._flatten(mesh_dim_name="dp_sp")
        mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")

        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def pp_enabled(self) -> bool:
        return self._ps.pp_size > 1

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
    def pp_size(self) -> int:
        return self._ps.pp_size

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
    def dp_group(self) -> ProcessGroup:
        return self.world_mesh["dp"].get_group()

    @property
    def sp_group(self) -> ProcessGroup:
        return self.world_mesh["sp"].get_group()

    @property
    def tp_group(self) -> ProcessGroup:
        return self.world_mesh["tp"].get_group()

    @property
    def pp_group(self) -> ProcessGroup | None:
        """Return PP process group, or None if PP is disabled."""
        if self.pp_enabled:
            return self.world_mesh["pp"].get_group()
        return None

    @property
    def pp_rank(self) -> int:
        """Return this rank's position in the PP dimension."""
        if self.pp_enabled:
            return self.world_mesh["pp"].get_local_rank()
        return 0

    @property
    def gradient_div_factor(self) -> int:
        return self._ps.dp_size * self._ps.cp_size

    @property
    def context_and_model_parallel_size(self) -> int:
        return self._ps.cp_size * self._ps.tp_size

    @property
    def seq_len_divisor(self) -> int:
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
        root_tp_plan["lm_head"] = ColwiseParallel(
            input_layouts=Shard(1),
        )
    if hasattr(model, "score") and isinstance(model.score, nn.Module):
        root_tp_plan["score"] = ReplicateParallel(
            input_layout=Shard(1),
            desired_input_layout=Shard(1),
            output_layout=Replicate(),
        )

    if is_valid_vision_model(model_config.model_type):
        if isinstance(model.model.language_model, nn.Module):
            model_tp_plan.pop("embed_tokens", None)
            model_tp_plan["layers.0"] = PrepareModuleInput(
                input_layouts=Replicate(),
                desired_input_layouts=Shard(1),
            )

            if is_qwen3_vl_model(model_config.model_type):
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
    pp_enabled: bool = False,
):
    """Apply N-D parallelism (TP + FSDP2) to a model or model part.

    When PP is enabled, this function is called per model_part (pipeline stage)
    with reshard_after_forward=False to avoid redundant all-gathers across
    microbatches.

    Args:
        model: The model or model part to parallelize.
        config: Training engine configuration.
        model_config: HuggingFace model configuration.
        nd_device_mesh: N-D device mesh.
        parallel_helper: Parallel configuration helper.
        cpu_offload: CPU offload policy for FSDP.
        wrap_policy: FSDP wrap policy.
        pp_enabled: Whether pipeline parallelism is active.
    """
    tp_enabled = parallel_helper.tp_enabled

    if tp_enabled:
        apply_non_moe_tp(model, model_config, parallel_helper, nd_device_mesh["tp"])

    mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, config.dtype),
        reduce_dtype=getattr(torch, config.grad_reduce_dtype),
        cast_forward_inputs=True,
    )

    # When PP is enabled, keep parameters gathered after forward pass
    # to avoid repeated all-gathers for each microbatch passing through the stage.
    # This is the critical optimization from torchtitan/verl for FSDP2 + PP.
    reshard_after_forward = not pp_enabled

    fsdp_kwargs = {
        "mesh": nd_device_mesh["dp_sp"],
        "mp_policy": mixed_precision_policy,
        "offload_policy": cpu_offload,
        "reshard_after_forward": reshard_after_forward,
    }
    apply_fsdp2(model, fsdp_kwargs, wrap_policy)

    if pp_enabled and torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        free_gb = free_mem / (1024**3)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        print(
            f"[GPU_MEM Rank {rank}] After FSDP wrap (pp_enabled={pp_enabled}, "
            f"reshard_after_forward={reshard_after_forward}): "
            f"allocated={allocated:.2f}GiB, reserved={reserved:.2f}GiB, "
            f"free={free_gb:.2f}GiB, param_bytes={param_bytes / (1024**3):.3f}GiB",
            flush=True,
        )
