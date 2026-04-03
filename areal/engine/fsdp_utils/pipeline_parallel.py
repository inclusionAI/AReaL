# Pipeline Parallelism support for FSDP Engine.
# Adapted from areal/experimental/models/archon/pipeline_parallel.py
# to work with HuggingFace models (nn.ModuleList for layers) used by FSDPEngine.

import copy
import functools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)

from areal.utils import logging

if TYPE_CHECKING:
    from torch.distributed.pipelining.schedules import _PipelineSchedule


@functools.cache
def _get_logger() -> logging.Logger:
    """Get rank-aware logger for this module."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[FSDP PipelineParallel Rank {rank}]")


__all__ = [
    "generate_hf_fqn_per_model_part",
    "pipeline_module_split_hf",
    "build_pipeline_schedule",
    "pipeline_llm_hf",
    "FSDPPipelinedRunner",
    "create_fsdp_runner",
]


def build_pipeline_schedule(
    stages: list[PipelineStage],
    pp_schedule: str,
    n_microbatches: int,
    pp_degree: int = 1,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> "_PipelineSchedule":
    """Build pipeline schedule for FSDP PP.

    Args:
        stages: Pipeline stages for this rank.
        pp_schedule: Schedule name (e.g., "1F1B", "Interleaved1F1B").
        n_microbatches: Number of microbatches.
        pp_degree: Pipeline parallel degree (for bubble warning).
        loss_fn: Loss function (None for eval mode).

    Returns:
        Configured pipeline schedule instance.
    """
    schedule_class = get_schedule_class(pp_schedule)
    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)

    num_total_stages = len(stages) * pp_degree
    if n_microbatches < num_total_stages:
        _get_logger().warning(
            f"n_microbatches ({n_microbatches}) < num_total_stages ({num_total_stages}), "
            "may result in pipeline bubble"
        )

    return schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=False,
    )


def generate_hf_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    first_stage_less_layers: int = 1,
    last_stage_less_layers: int = 1,
    is_critic: bool = False,
) -> list[list[str]]:
    """Generate module FQN lists for each pipeline stage (HuggingFace model layout).

    HuggingFace models typically have:
      - model.embed_tokens (embedding)
      - model.layers.0, model.layers.1, ... (transformer blocks as ModuleList)
      - model.norm (final LayerNorm)
      - lm_head or score (output head)

    This function distributes transformer layers across pipeline stages,
    accounting for the computational cost of embedding and output layers.

    Args:
        num_stages: Number of pipeline stages.
        num_layers: Number of transformer layers in the model.
        first_stage_less_layers: Weight for input modules (embed_tokens), default 1.
        last_stage_less_layers: Weight for output modules (norm + lm_head/score), default 1.
        is_critic: Whether the model is a critic (uses 'score' instead of 'lm_head').

    Returns:
        List of module name lists, one per stage.

    Example:
        >>> generate_hf_fqn_per_model_part(2, 4)
        [['model.embed_tokens', 'model.layers.0', 'model.layers.1'],
         ['model.layers.2', 'model.layers.3', 'model.norm', 'lm_head']]
    """
    output_module = "score" if is_critic else "lm_head"

    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")

    # Single stage: return everything
    if num_stages == 1:
        layer_names = [f"model.layers.{i}" for i in range(num_layers)]
        return [["model.embed_tokens"] + layer_names + ["model.norm", output_module]]

    num_effective_layers = num_layers + first_stage_less_layers + last_stage_less_layers

    if num_stages > num_effective_layers:
        raise ValueError(
            f"num_stages ({num_stages}) cannot exceed effective layers "
            f"({num_effective_layers} = {num_layers} + {first_stage_less_layers} + {last_stage_less_layers})"
        )

    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    if layers_per_stage == 0:
        raise ValueError(
            f"layers_per_stage is 0 with {num_effective_layers} effective layers "
            f"and {num_stages} stages"
        )
    if first_stage_less_layers > layers_per_stage:
        raise ValueError(
            f"first_stage_less_layers ({first_stage_less_layers}) exceeds layers_per_stage ({layers_per_stage})"
        )
    if last_stage_less_layers > layers_per_stage:
        raise ValueError(
            f"last_stage_less_layers ({last_stage_less_layers}) exceeds layers_per_stage ({layers_per_stage})"
        )

    module_names_per_stage: list[list[str]] = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules: list[str] = []

        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        if stage_idx == 0:
            stage_modules.append("model.embed_tokens")
            num_transformer_layers = (
                effective_layers_for_stage - first_stage_less_layers
            )
            for _ in range(num_transformer_layers):
                if current_layer < num_layers:
                    stage_modules.append(f"model.layers.{current_layer}")
                    current_layer += 1
        elif stage_idx == num_stages - 1:
            num_transformer_layers = effective_layers_for_stage - last_stage_less_layers
            for _ in range(num_transformer_layers):
                if current_layer < num_layers:
                    stage_modules.append(f"model.layers.{current_layer}")
                    current_layer += 1
            stage_modules.extend(["model.norm", output_module])
        else:
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"model.layers.{current_layer}")
                    current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def pipeline_module_split_hf(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """Split an HF model into pipeline stages based on module names.

    HuggingFace models use nn.ModuleList for transformer layers (model.layers).
    Unlike Archon's ModuleDict, we handle ModuleList by keeping only the layers
    assigned to this stage and replacing others with None.

    The model's forward() must handle missing layers gracefully — we achieve this
    by wrapping each stage's model_part with a custom forward that only processes
    the layers present.

    Args:
        whole_model: The complete HF model to split.
        pp_mesh: Pipeline parallel device mesh.
        pp_schedule: Schedule type.
        device: Target device for stages.
        module_names_per_stage: Module FQNs for each stage.

    Returns:
        Tuple of (list of PipelineStage, list of model parts).
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()
    num_stages = len(module_names_per_stage)

    def _build_stage_from_modules(
        stage_idx: int,
        module_names: list[str],
        num_stages: int,
    ) -> tuple[PipelineStage, nn.Module]:
        """Build a single pipeline stage from module names for HF models."""
        # Deep copy to create independent model part
        model = copy.deepcopy(whole_model)
        modules_to_keep = set(module_names)

        # Determine which layer indices to keep
        layer_indices_to_keep = set()
        for name in modules_to_keep:
            if name.startswith("model.layers."):
                idx = int(name.split(".")[-1])
                layer_indices_to_keep.add(idx)

        has_embed = "model.embed_tokens" in modules_to_keep
        has_norm = "model.norm" in modules_to_keep
        has_lm_head = "lm_head" in modules_to_keep
        has_score = "score" in modules_to_keep

        # Handle model.embed_tokens
        if (
            not has_embed
            and hasattr(model, "model")
            and hasattr(model.model, "embed_tokens")
        ):
            model.model.embed_tokens = None

        # Handle model.layers (nn.ModuleList)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            original_layers = model.model.layers
            # Replace non-kept layers with None
            new_layers = nn.ModuleList()
            for i, layer in enumerate(original_layers):
                if i in layer_indices_to_keep:
                    new_layers.append(layer)
                else:
                    new_layers.append(None)
            model.model.layers = new_layers

        # Handle model.norm
        if not has_norm and hasattr(model, "model") and hasattr(model.model, "norm"):
            model.model.norm = None

        # Handle lm_head / score
        if not has_lm_head and hasattr(model, "lm_head"):
            model.lm_head = None
        if not has_score and hasattr(model, "score"):
            model.score = None

        # Create a wrapper module that handles the partial forward
        stage_model = _HFPipelineStageModule(
            model=model,
            has_embed=has_embed,
            has_norm=has_norm,
            has_output_head=(has_lm_head or has_score),
            layer_indices=sorted(layer_indices_to_keep),
            is_critic=has_score,
        )

        stage = PipelineStage(
            stage_model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group(),
        )

        return stage, stage_model

    def _get_stage_indices() -> tuple[int, ...]:
        """Get stage indices for this rank based on schedule style."""
        if num_stages % pp_degree != 0:
            raise ValueError(
                f"num_stages ({num_stages}) must be evenly divisible by "
                f"pp_degree ({pp_degree})"
            )
        stages_per_rank = num_stages // pp_degree

        schedule_class = get_schedule_class(pp_schedule)
        v_style_schedules = (ScheduleZBVZeroBubble,)
        try:
            from torch.distributed.pipelining.schedules import ScheduleDualPipeV

            v_style_schedules = (ScheduleZBVZeroBubble, ScheduleDualPipeV)
        except ImportError:
            pass
        style = "v" if schedule_class in v_style_schedules else "loop"

        if style == "v":
            if stages_per_rank != 2:
                raise ValueError(
                    f"V-style schedules require exactly 2 stages per rank, "
                    f"got {stages_per_rank}"
                )
            stage_v_pairs = list(
                zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1))
            )
            return stage_v_pairs[pp_rank]
        else:
            return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))

    stages: list[PipelineStage] = []
    model_parts: list[nn.Module] = []

    for stage_idx in _get_stage_indices():
        stage, model_part = _build_stage_from_modules(
            stage_idx, module_names_per_stage[stage_idx], num_stages
        )
        stages.append(stage)
        model_parts.append(model_part)

        _get_logger().info(
            f"Built stage {stage_idx} (pp_rank={pp_rank}) "
            f"with modules: {module_names_per_stage[stage_idx]}"
        )

    return stages, model_parts


class _HFPipelineStageModule(nn.Module):
    """Wrapper module for an HF model pipeline stage.

    This module handles the partial forward pass for a subset of the model layers.
    Each stage only processes its assigned layers and passes activations to the
    next stage via PipelineStage's send/recv mechanism.

    Args:
        model: The (pruned) HuggingFace model for this stage.
        has_embed: Whether this stage has the embedding layer.
        has_norm: Whether this stage has the final norm layer.
        has_output_head: Whether this stage has lm_head or score.
        layer_indices: Sorted list of transformer layer indices in this stage.
        is_critic: Whether this is a critic model (uses score instead of lm_head).
    """

    def __init__(
        self,
        model: nn.Module,
        has_embed: bool,
        has_norm: bool,
        has_output_head: bool,
        layer_indices: list[int],
        is_critic: bool = False,
    ):
        super().__init__()
        self.model = model
        self.has_embed = has_embed
        self.has_norm = has_norm
        self.has_output_head = has_output_head
        self.layer_indices = layer_indices
        self.is_critic = is_critic

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for this pipeline stage.

        For the first stage, hidden_states is input_ids (Long tensor).
        For middle/last stages, hidden_states is the continuous hidden state
        from the previous stage.

        Args:
            hidden_states: Input tensor — input_ids for first stage,
                          hidden states for subsequent stages.
            **kwargs: Additional keyword arguments (position_ids, attention_mask, etc.)

        Returns:
            Hidden states (for middle stages) or logits (for last stage).
        """
        position_ids = kwargs.get("position_ids", None)

        # First stage: apply embedding
        if self.has_embed:
            hidden_states = self.model.model.embed_tokens(hidden_states)

        # Apply transformer layers
        for idx in self.layer_indices:
            layer = self.model.model.layers[idx]
            if layer is not None:
                layer_outputs = layer(
                    hidden_states,
                    position_ids=position_ids,
                )
                # HF transformer layers return tuples: (hidden_states, ...)
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

        # Last stage: apply norm and output head
        if self.has_norm and self.model.model.norm is not None:
            hidden_states = self.model.model.norm(hidden_states)

        if self.has_output_head:
            if (
                self.is_critic
                and hasattr(self.model, "score")
                and self.model.score is not None
            ):
                hidden_states = self.model.score(hidden_states)
            elif hasattr(self.model, "lm_head") and self.model.lm_head is not None:
                hidden_states = self.model.lm_head(hidden_states)

        return hidden_states


class _NullOutputChunks(list):
    """A list that silently discards appended items to save memory during training."""

    def append(self, item: Any) -> None:
        pass


class FSDPPipelinedRunner:
    """Pipeline-parallel runner for FSDP Engine.

    Handles forward/backward execution using PyTorch's PipelineSchedule API.
    Supports all schedule types: 1F1B, Interleaved1F1B, InterleavedZeroBubble, ZBVZeroBubble.
    """

    def __init__(
        self,
        pp_stages: list[PipelineStage],
        pp_schedule: str,
        pp_group_size: int,
        has_first_stage: bool,
        has_last_stage: bool,
    ):
        self.pp_stages = pp_stages
        self.pp_schedule = pp_schedule
        self.pp_group_size = pp_group_size
        self.has_first_stage = has_first_stage
        self.has_last_stage = has_last_stage

    def _create_schedule(
        self,
        n_microbatches: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    ) -> "_PipelineSchedule":
        return build_pipeline_schedule(
            stages=self.pp_stages,
            pp_schedule=self.pp_schedule,
            n_microbatches=n_microbatches,
            pp_degree=self.pp_group_size,
            loss_fn=loss_fn,
        )

    def _get_output_stage(self) -> PipelineStage:
        for stage in self.pp_stages:
            if stage.is_last:
                return stage
        raise RuntimeError("No last stage found in pp_stages")

    def _patch_skip_output_merge(self, schedule: "_PipelineSchedule") -> None:
        """Patch schedule to skip output merging, halving memory usage."""
        schedule._merge_outputs = lambda output_chunks: None

    def run_train(
        self,
        n_microbatches: int,
        input_ids_chunks: list[torch.Tensor],
        target_chunks: list[torch.Tensor] | None,
        extra_kwargs: dict[str, Any] | None,
        contexts: list[Any],
        process_output_fn: Callable,
    ) -> list[torch.Tensor]:
        """Run forward-backward using PP schedule for training.

        Args:
            n_microbatches: Number of microbatches.
            input_ids_chunks: List of input_ids per microbatch (only used on first stage).
            target_chunks: List of target tensors per microbatch (only used on last stage).
            extra_kwargs: Extra kwargs batched across microbatches.
            contexts: List of context objects per microbatch (for loss computation).
            process_output_fn: Callback to compute loss from logits and context.

        Returns:
            Empty list (losses are computed inside pp_loss_fn).
        """
        pp_loss_fn = self._create_loss_fn(contexts, process_output_fn)
        schedule = self._create_schedule(n_microbatches, loss_fn=pp_loss_fn)
        self._patch_skip_output_merge(schedule)

        # Replace output_chunks with null list to free logits immediately after backward
        output_stage = None
        if self.has_last_stage:
            output_stage = self._get_output_stage()
            output_stage.output_chunks = _NullOutputChunks()

        # schedule.step() expects a single batched tensor per input argument.
        # It internally splits them into n_microbatches chunks via torch.tensor_split.
        if self.has_first_stage and input_ids_chunks:
            batched_input = torch.cat(input_ids_chunks, dim=0)
            args = (batched_input,)
        else:
            args = ()

        if target_chunks:
            batched_target = torch.cat(target_chunks, dim=0)
        else:
            batched_target = None

        schedule.step(*args, target=batched_target, **(extra_kwargs or {}))

        # Restore normal list for subsequent eval calls
        if output_stage is not None:
            output_stage.output_chunks = []

        return []

    def run_eval(
        self,
        n_microbatches: int,
        input_ids_chunks: list[torch.Tensor],
        extra_kwargs: dict[str, Any] | None,
        contexts: list[Any],
        process_output_fn: Callable,
    ) -> list[torch.Tensor] | None:
        """Run forward-only using PP schedule for evaluation.

        Returns:
            List of results from process_output_fn, or None if not on last stage.
        """
        schedule = self._create_schedule(n_microbatches, loss_fn=None)
        self._patch_skip_output_merge(schedule)

        # schedule.eval() expects a single batched tensor, not per-microbatch args.
        if self.has_first_stage and input_ids_chunks:
            batched_input = torch.cat(input_ids_chunks, dim=0)
            args = (batched_input,)
        else:
            args = ()
        schedule.eval(*args, **(extra_kwargs or {}))

        if not self.has_last_stage:
            return None

        output_stage = self._get_output_stage()
        results = []
        for output, ctx in zip(output_stage.output_chunks, contexts, strict=True):
            if output.ndim == 3:
                output = output.squeeze(0)
            result = process_output_fn(output, ctx)
            if result is not None:
                results.append(result.detach())
        output_stage.output_chunks.clear()
        return results

    def _create_loss_fn(
        self,
        contexts: list[Any],
        process_output_fn: Callable,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.has_last_stage:
            ctx_iter = iter(contexts)

            def pp_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                ctx = next(ctx_iter)
                # Dummy microbatches (padded for PP schedule divisibility)
                # should produce zero loss to avoid corrupting gradients.
                if isinstance(ctx, dict) and ctx.get("__pp_dummy__", False):
                    return pred.sum() * 0.0
                if pred.ndim == 3:
                    pred = pred.squeeze(0)
                loss = process_output_fn(pred, ctx)
                if loss is None:
                    return pred.sum() * 0.0
                return loss
        else:

            def pp_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return pred.sum() * 0.0

        return pp_loss_fn


def pipeline_llm_hf(
    model: nn.Module,
    device: torch.device,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    pp_degree: int,
    num_layers: int,
    is_critic: bool = False,
    pp_layers_per_stage: int | None = None,
    pp_first_stage_less_layers: int = 1,
    pp_last_stage_less_layers: int = 1,
) -> tuple[list[PipelineStage], list[nn.Module], bool, bool]:
    """Main entry point for pipeline parallelism with HF models.

    Workflow:
    1. Generate module names for each virtual stage
    2. Split model into stages
    3. Return stages and model parts for FSDP parallelization

    Args:
        model: The complete HF model to pipeline.
        device: Target device.
        pp_mesh: Pipeline parallel device mesh.
        pp_schedule: Schedule name.
        pp_degree: Pipeline parallel degree.
        num_layers: Number of transformer layers.
        is_critic: Whether the model is a critic.
        pp_layers_per_stage: Layers per virtual stage (optional).
        pp_first_stage_less_layers: Weight for embedding overhead.
        pp_last_stage_less_layers: Weight for output overhead.

    Returns:
        Tuple of (stages, model_parts, has_first_stage, has_last_stage).
    """
    schedule_class = get_schedule_class(pp_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)

    if pp_layers_per_stage is not None:
        num_virtual_stages = math.ceil(
            (num_layers + pp_first_stage_less_layers + pp_last_stage_less_layers)
            / pp_layers_per_stage
        )
        if num_virtual_stages % pp_degree != 0:
            raise ValueError(
                f"num_virtual_stages ({num_virtual_stages}) must be divisible by "
                f"pp_degree ({pp_degree})."
            )
        stages_per_rank = num_virtual_stages // pp_degree
        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"{pp_schedule} schedule requires exactly 1 stage per rank, "
                f"but got {stages_per_rank}."
            )
        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"{pp_schedule} schedule requires >= 2 stages per rank, "
                f"but got {stages_per_rank}."
            )
    else:
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = pp_degree * stages_per_rank

    _get_logger().info(
        f"PP setup: schedule={pp_schedule}, stages_per_rank={stages_per_rank}, "
        f"num_virtual_stages={num_virtual_stages}"
    )

    # 1. Generate module names per stage
    module_names_per_stage = generate_hf_fqn_per_model_part(
        num_stages=num_virtual_stages,
        num_layers=num_layers,
        first_stage_less_layers=pp_first_stage_less_layers,
        last_stage_less_layers=pp_last_stage_less_layers,
        is_critic=is_critic,
    )

    _get_logger().info(f"PP module distribution: {module_names_per_stage}")

    # 2. Split model into stages
    stages, model_parts = pipeline_module_split_hf(
        whole_model=model,
        pp_mesh=pp_mesh,
        pp_schedule=pp_schedule,
        device=device,
        module_names_per_stage=module_names_per_stage,
    )

    # Determine first/last stage status
    has_first_stage = any(s.is_first for s in stages)
    has_last_stage = any(s.is_last for s in stages)

    _get_logger().info(
        f"Pipeline setup complete: has_first_stage={has_first_stage}, "
        f"has_last_stage={has_last_stage}"
    )

    return stages, model_parts, has_first_stage, has_last_stage


def create_fsdp_runner(
    *,
    pp_enabled: bool,
    pp_stages: list[PipelineStage] | None = None,
    pp_schedule: str | None = None,
    pp_group_size: int = 1,
    has_first_stage: bool = True,
    has_last_stage: bool = True,
) -> FSDPPipelinedRunner | None:
    """Factory function to create a PP runner for FSDP engine.

    Returns None if PP is not enabled.
    """
    if not pp_enabled:
        return None

    assert pp_stages is not None, "pp_stages required when pp_enabled=True"
    assert pp_schedule is not None, "pp_schedule required when pp_enabled=True"

    return FSDPPipelinedRunner(
        pp_stages=pp_stages,
        pp_schedule=pp_schedule,
        pp_group_size=pp_group_size,
        has_first_stage=has_first_stage,
        has_last_stage=has_last_stage,
    )
