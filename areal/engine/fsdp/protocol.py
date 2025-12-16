from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy
from transformers import PretrainedConfig, PreTrainedTokenizerFast, ProcessorMixin

from areal.api.alloc_mode import FSDPParallelStrategy, ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.utils import logging
from areal.utils.fsdp.parallel import ParallelHelper

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
    from areal.core.dist_rollout import DistRolloutCoordinator
    from areal.utils.data import MicroBatchItem, MicroBatchList


@dataclasses.dataclass
class FSDPTrainContext:
    """Context passed through FSDP forward/backward pipeline.

    Attributes
    ----------
    model_inputs
        The prepared inputs passed to the model (may include Ulysses slicing).
    mb_input
        The original micro-batch dict before any Ulysses transformations.
    pad_length
        Number of padding tokens added for sequence packing.
    ulysses_pad_size
        Extra padding added for Ulysses sequence parallel alignment.
    """

    model_inputs: dict[str, Any]
    mb_input: dict[str, Any]
    pad_length: int = 0
    ulysses_pad_size: int = 0


class FSDPEngineProtocol(Protocol):
    """Protocol defining the complete interface for FSDP Engine.

    This protocol serves as the type contract between FSDP Mixins:
    - Attributes: All shared state that Mixins can read/write
    - Internal Methods: Private methods that may be called across Mixin boundaries

    The FSDPEngine glue class must satisfy this protocol by providing all
    attributes in __init__ and inheriting all methods from Mixins.
    """

    # =========================================================================
    # Shared Attributes
    # =========================================================================

    config: TrainEngineConfig
    optimizer_config: Any
    model: nn.Module
    optimizer: torch.optim.Optimizer
    tokenizer: PreTrainedTokenizerFast
    processor: ProcessorMixin | None
    model_config: PretrainedConfig
    _version: int
    initialized: bool
    own_global_group: bool
    _cpu_group: dist.ProcessGroup
    weight_update_group_initialized: bool
    is_vision_model: bool
    world_size: int
    cpu_offload: CPUOffloadPolicy | None
    rollout_engine: InferenceEngine | None
    rollout_coordinator: DistRolloutCoordinator | None
    parallel_helper: ParallelHelper
    world_mesh: DeviceMesh
    dp_group: dist.ProcessGroup
    sp_group: dist.ProcessGroup
    mp_group: dist.ProcessGroup
    rank: int
    dp_head: int
    dp_rank: int
    is_offload: bool
    device: torch.device
    logger: logging.Logger
    lr_scheduler: Any
    weight_update_group: dist.ProcessGroup

    # =========================================================================
    # DistMixin Internal Methods
    # =========================================================================

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> FSDPParallelStrategy: ...

    # =========================================================================
    # StateMixin Internal Methods
    # =========================================================================

    def _get_model_name_parameters(self) -> Iterator[tuple[str, nn.Parameter]]: ...

    def _get_full_tensor(self, param: nn.Parameter) -> torch.Tensor: ...

    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ) -> None: ...

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta) -> None: ...

    def _update_weights_from_distributed(self, meta: WeightUpdateMeta) -> None: ...

    def _update_weights_from_disk(self, meta: WeightUpdateMeta) -> None: ...

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast | None,
        processor: ProcessorMixin | None,
    ) -> None: ...

    def _load_model_from_hf(self, path: str) -> None: ...

    def _save_to_dcp(self, path: str, with_optim: bool) -> None: ...

    def _load_from_dcp(self, path: str, with_optim: bool) -> None: ...

    def _save_optimizer_state(self, path: str) -> None: ...

    def _load_optimizer_state(self, path: str) -> None: ...

    # =========================================================================
    # ComputeMixin Internal Methods
    # =========================================================================

    def _check_rollout_engine_connected(self) -> None: ...

    def _ensure_ready(self) -> None: ...

    def _create_llm_actor_or_critic(self) -> nn.Module: ...

    def _create_device_model(self) -> None: ...

    def _apply_peft_wrapper(self) -> None: ...

    def _create_optimizer(self, ft_spec: FinetuneSpec) -> None: ...

    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList: ...

    def _prepare_mb_inputs(
        self, mb_item: MicroBatchItem
    ) -> tuple[dict[str, Any], FSDPTrainContext]: ...

    def _sp_all_gather(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def _get_vocab_min_max_logits(
        self, logits: torch.Tensor, ulysses_pad_size: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def _compute_logprobs_entropy(
        self, logits: torch.Tensor, inputs: dict[str, Any], ulysses_pad_size: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def _compute_logprobs(
        self, logits: torch.Tensor, inputs: dict[str, Any], ulysses_pad_size: int = 0
    ) -> torch.Tensor: ...

    def _compute_values(
        self, values: torch.Tensor, ulysses_pad_size: int = 0
    ) -> torch.Tensor: ...

    def _compute_logprobs_and_loss(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor: ...

    def _compute_forward_result(
        self, logits: torch.Tensor, ctx: FSDPTrainContext
    ) -> torch.Tensor: ...

    # =========================================================================
    # StateMixin Public Methods (needed for cross-Mixin calls)
    # =========================================================================

    def onload(self) -> None: ...

    def get_version(self) -> int: ...
