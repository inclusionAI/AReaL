from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import mbridge
import torch
import torch.distributed as dist
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer import TransformerConfig
from torch import nn
from transformers import PretrainedConfig, PreTrainedTokenizer

from areal.api.alloc_mode import MegatronParallelStrategy
from areal.api.cli_args import (
    MegatronEngineConfig,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.api.engine_api import InferenceEngine
from areal.core.dist_rollout import DistRolloutCoordinator
from areal.utils.lock import DistributedLock
from areal.utils.logging import Logger
from areal.utils.megatron_checkpointer import MegatronCheckpointManager


@runtime_checkable
class MegatronEngineProtocol(Protocol):
    """Protocol defining shared state and internal methods for Megatron Engine Mixins.

    This protocol is used for type hinting in Mixin classes. All attributes listed
    here are initialized in the Glue Class (MegatronEngine.__init__) and accessed
    by the Mixin methods.
    """

    # === Configuration ===
    config: TrainEngineConfig
    hf_config: PretrainedConfig
    tf_config: TransformerConfig
    optimizer_config: OptimizerConfig | None
    mcore_config: MegatronEngineConfig
    dtype: torch.dtype
    device: torch.device | None

    # === Model & Optimizer ===
    model: Any | None  # _MegatronModelList
    optimizer: DistributedOptimizer | None
    lr_scheduler: OptimizerParamScheduler | None
    checkpointer: MegatronCheckpointManager | None
    bridge: mbridge.AutoBridge | None
    tokenizer: PreTrainedTokenizer | None

    # === Parallelism ===
    parallel_strategy: MegatronParallelStrategy | None
    rank: int | None
    world_size: int | None
    is_pp_head: bool
    own_global_group: bool

    # === Process Group State ===
    process_group_initialized: bool
    _context_and_model_parallel_group: dist.ProcessGroup | None
    _cpu_group: dist.ProcessGroup

    # === Weight Update ===
    weight_update_group_initialized: bool
    weight_update_group_name: str
    weight_update_group: dist.ProcessGroup

    # === Rollout ===
    rollout_engine: InferenceEngine | None
    rollout_coordinator: DistRolloutCoordinator | None

    # === Versioning & State ===
    _version: int
    seed: int
    is_offload: bool
    engine_lock: DistributedLock

    # === Logger ===
    logger: Logger

    # === Internal Methods (cross-Mixin calls) ===

    def _make_parallel_strategy(
        self, parallel_strategy: Any
    ) -> MegatronParallelStrategy:
        """Create Megatron-specific parallel strategy from base strategy."""
        ...

    def _init_context_and_model_parallel_group(self) -> None:
        """Initialize context and model parallel groups for data distribution."""
        ...

    def _create_optimizer(self, ft_spec: Any) -> None:
        """Create optimizer and LR scheduler based on fine-tune spec."""
        ...

    def _check_rollout_engine_connected(self) -> None:
        """Validate that rollout engine has been connected."""
        ...

    def _ensure_ready(self) -> None:
        """Ensure model is ready (onloaded if needed)."""
        ...

    def _load_model_from_hf(self, path: str) -> None:
        """Load model weights from HuggingFace format."""
        ...

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: Any | None = None,
        processor: Any | None = None,
        base_model_path: str | None = None,
    ) -> None:
        """Save model weights to HuggingFace format."""
        ...

    def _update_bucket_weights_from_distributed(
        self,
        meta: Any,
        converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ) -> None:
        """Update a bucket of weights via distributed broadcast."""
        ...

    def _init_weight_update_from_distributed(self, meta: Any) -> None:
        """Initialize weight update group for distributed weight sync."""
        ...

    def _prepare_mb_list(self, input_: dict[str, Any]) -> Any:
        """Prepare micro-batch list from input."""
        ...

    def offload(self) -> None:
        """Offload model memory to CPU."""
        ...

    def onload(self) -> None:
        """Onload model memory from CPU back to GPU."""
        ...

    # === Properties for parallel info ===

    @property
    def data_parallel_rank(self) -> int:
        """Get data parallel rank."""
        ...

    @property
    def data_parallel_world_size(self) -> int:
        """Get data parallel world size."""
        ...

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        """Get data parallel process group."""
        ...

    @property
    def pipeline_parallel_rank(self) -> int:
        """Get pipeline parallel rank."""
        ...

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        """Get context and model parallel process group."""
        ...

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        """Get CPU process group for barrier synchronization."""
        ...

    def is_pipeline_parallel_head(self) -> bool:
        """Check if current rank is pipeline parallel head."""
        ...

    def is_data_parallel_head(self) -> bool:
        """Check if current rank is data parallel head."""
        ...

    def current_data_parallel_head(self) -> int:
        """Get the rank of the head of the current data parallel group."""
        ...
