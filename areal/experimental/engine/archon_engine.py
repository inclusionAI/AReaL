from __future__ import annotations

import gc
import math
import os
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.pipelining.schedules import (
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
    get_schedule_class,
)
from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import MicroBatchSpec
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import (
    DeviceRuntimeInfo,
    FinetuneSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.engine.core.train_engine import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.experimental.engine.archon_checkpoint import (
    load_from_dcp,
    load_model_from_hf,
    load_optimizer_state,
    save_model_to_hf,
    save_optimizer_state,
    save_to_dcp,
)
from areal.experimental.engine.archon_runner import create_runner
from areal.experimental.engine.archon_weight_sync import (
    WeightSyncState,
    init_weight_update_group,
    update_weights_from_disk,
    update_weights_from_distributed,
)
from areal.experimental.models.archon import (
    ArchonParallelDims,
    BaseStateDictAdapter,
    ModelSpec,
    get_model_spec,
    get_supported_model_types,
    is_supported_model,
)
from areal.experimental.models.archon.activation_checkpoint import (
    ActivationCheckpointConfig,
)
from areal.experimental.models.archon.ulysses import (
    ulysses_gather_output,
    ulysses_slice_inputs,
)
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.platforms import current_platform
from areal.models.tree_attn.functional import (
    _gather_packed_tree_logprobs,
    gather_packed_tree_logprobs_entropy,
    gather_packed_tree_vocab_stats,
    merge_packed_tree_results,
)
from areal.models.tree_attn.module import (
    BLOCK_SIZE,
)
from areal.models.tree_attn.tree import TrieNode, build_packed_tree_batch
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.constants import DEFAULT_PAGE_SIZE_BYTES, DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.data import (
    MicroBatchItem,
    MicroBatchList,
    amend_position_ids,
    broadcast_tensor,
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_mb_list,
)
from areal.utils.distributed import patch_dist_group_timeout
from areal.utils.fsdp import get_cosine_schedule_with_warmup
from areal.utils.fsdp.grad import fsdp2_clip_grad_norm
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.lock import DistributedLock
from areal.utils.offload import is_tms_enabled, torch_memory_saver

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.pipelining import PipelineStage
    from torchdata.stateful_dataloader import StatefulDataLoader

    from areal.api.cli_args import PerfTracerConfig, TrainEngineConfig
    from areal.api.engine_api import InferenceEngine
    from areal.api.scheduler_api import Scheduler
    from areal.api.workflow_api import WorkflowLike
    from areal.experimental.engine.archon_runner import ForwardBackwardRunner


@dataclass
class ArchonTrainContext:
    """Context passed through Archon forward/backward pipeline.

    Attributes:
        mb_input: Original microbatch input.
        labels: Target token ids for loss computation (rolled from input_ids).
        pad_length: Batch-level padding added by pad_mb_list.
        trie_node: The root TrieNode for tree training (if applicable).
    """

    mb_input: dict[str, Any]
    labels: torch.Tensor
    pad_length: int = 0
    trie_node: TrieNode | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict without recursive serialization of trie_node.

        Note: We cannot use dataclasses.asdict() here because it recursively
        converts all nested objects. The trie_node field contains a TrieNode
        with recursive parent/child references, which causes
        "RecursionError: maximum recursion depth exceeded" when asdict()
        attempts to serialize the entire tree structure.
        """
        return {
            "mb_input": self.mb_input,
            "labels": self.labels,
            "pad_length": self.pad_length,
            "trie_node": self.trie_node,
        }


class ArchonEngine(TrainEngine):
    """Archon Engine is a torch-native training backend."""

    def __init__(self, config: TrainEngineConfig):
        # Configuration (immutable after init)
        self.config = config
        self.optimizer_config = config.optimizer
        self.enable_tree_training = config.enable_tree_training

        # Model Configuration (loaded during __init__)
        self.model_config: PretrainedConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self._validate_model_type()

        model_type = getattr(self.model_config, "model_type", "")
        self.spec: ModelSpec = get_model_spec(model_type)

        # Core Components (initialized in initialize())
        self.model: nn.Module
        self.tokenizer: PreTrainedTokenizerFast
        self.optimizer: torch.optim.Optimizer
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        self.state_dict_adapter: BaseStateDictAdapter | None = None
        self.runner: ForwardBackwardRunner

        # Distributed / Parallelism (initialized in create_process_group())
        self.rank: int
        self.world_size: int
        self.parallel_dims: ArchonParallelDims
        self._world_mesh: DeviceMesh
        self._cpu_group: dist.ProcessGroup
        self.own_global_group = False

        # Pipeline Parallelism (initialized in initialize())
        self.pp_stages: list[PipelineStage] = []
        self.model_parts: list[nn.Module] = []
        self.pp_has_first_stage: bool = True
        self.pp_has_last_stage: bool = True

        # Rollout / Inference Integration
        self._weight_sync_state: WeightSyncState
        self.engine_lock: DistributedLock
        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None

        # Runtime State (mutable during training)
        self._version: int = 0
        self._initialized = False
        self.is_offload = False

    def create_process_group(
        self,
        parallel_strategy: ParallelStrategy | None = None,
    ):
        patch_dist_group_timeout(DIST_GROUP_DEFAULT_TIMEOUT)

        backend = current_platform.communication_backend
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            self.own_global_group = True

        self._cpu_group = dist.new_group(
            timeout=DIST_GROUP_DEFAULT_TIMEOUT, backend="gloo"
        )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.logger = logging.getLogger(f"[Archon Engine Rank {self.rank}]")

        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()

        self.parallel_dims = ArchonParallelDims(
            dp_shard=parallel_strategy.data_parallel_size,
            tp=parallel_strategy.tensor_parallel_size,
            cp=parallel_strategy.context_parallel_size,
            pp=parallel_strategy.pipeline_parallel_size,
            ep=parallel_strategy.expert_parallel_size,
            etp=parallel_strategy.expert_tensor_parallel_size,
            world_size=self.world_size,
            device_type=current_platform.device_type,
        )

        self._world_mesh = self.parallel_dims.world_mesh

        # Data parallel rank (for data loading)
        dp_mesh = self.parallel_dims.get_mesh("dp")
        self._dp_rank = dp_mesh.get_local_rank() if dp_mesh is not None else 0

        # Pipeline parallel rank
        if self.parallel_dims.pp_enabled:
            self._pp_rank = self.parallel_dims.get_mesh("pp").get_local_rank()
            # Set in _apply_pipeline_parallelism() after pipeline setup
            self._pp_last_stage_rank = None
        else:
            self._pp_rank = 0
            self._pp_last_stage_rank = None

        # Context and model parallel group (pp_cp_tp)
        self._pp_cp_tp_group = self.parallel_dims.get_group("pp_cp_tp")

        # DP head: the rank that holds the batch for this pp_cp_tp group
        self._dp_head = dist.get_process_group_ranks(self._pp_cp_tp_group)[0]

        # Pipeline parallel head: dp_rank=0 and cp/tp rank=0
        cp_rank_is_zero = (
            not self.parallel_dims.cp_enabled
            or self.parallel_dims.get_mesh("cp").get_local_rank() == 0
        )
        tp_rank_is_zero = (
            not self.parallel_dims.tp_enabled
            or self.parallel_dims.get_mesh("tp").get_local_rank() == 0
        )
        self._is_pipeline_parallel_head = (
            self._dp_rank == 0 and cp_rank_is_zero and tp_rank_is_zero
        )

        self.logger.info(
            f"Initialized Archon engine with parallel dims: "
            f"pp={self.parallel_dims.pp}, dp_shard={self.parallel_dims.dp_shard}, "
            f"tp={self.parallel_dims.tp}, cp={self.parallel_dims.cp} (Ulysses SP), "
            f"ep={self.parallel_dims.ep}, etp={self.parallel_dims.etp}"
        )

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        """Initialize model, optimizer, and apply parallelism."""
        assert addr is None, "ArchonEngine does not support remote initialization."
        assert ft_spec is not None, "ArchonEngine requires FinetuneSpec to initialize."

        # Initialize weight sync primitives
        self._weight_sync_state = WeightSyncState(self._pp_rank)
        self.engine_lock = DistributedLock("train_engine_lock")

        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"

        self._create_device_model()
        self.state_dict_adapter = self._create_state_dict_adapter()

        # Compute page_size: number of tokens that fit in one GPU page
        # based on hidden_size and dtype element size
        hidden_size = self.model_config.hidden_size
        param_dtype = getattr(torch, self.config.dtype)
        element_size = torch.empty([], dtype=param_dtype).element_size()
        self.page_size = max(DEFAULT_PAGE_SIZE_BYTES // hidden_size // element_size, 1)

        ac_config = self._build_ac_config()
        enable_compile = self.config.archon.enable_compile

        # V-style schedules (ZBVZeroBubble, DualPipeV) split backward into
        # I (input grad) and W (weight grad) steps. This is incompatible with:
        # 1. torch.compile — its donated buffer optimization assumes a single
        #    backward pass (retain_graph=False).
        # 2. Op-level selective AC — its per-op cache (storage.pop) is consumed
        #    by the I step, leaving nothing for the W step recompute.
        # 3. memory_budget AC — it depends on torch.compile.
        # Full AC / layer-level selective AC use standard checkpoint_wrapper
        # whose gid-based recompute supports multiple backward passes.
        schedule_class = get_schedule_class(self.config.archon.pp_schedule)
        v_style_schedules = (ScheduleZBVZeroBubble, ScheduleDualPipeV)
        if schedule_class in v_style_schedules:
            schedule_name = self.config.archon.pp_schedule
            if enable_compile:
                self.logger.warning(
                    f"{schedule_name} is incompatible with torch.compile. "
                    "Disabling torch.compile."
                )
                enable_compile = False

            if ac_config is not None and (
                (
                    ac_config.mode == "selective"
                    and ac_config.selective_ac_option == "op"
                )
                or ac_config.mode == "memory_budget"
            ):
                self.logger.warning(
                    f"{schedule_name} is incompatible with {ac_config.mode} AC. "
                    "Falling back to full AC."
                )
                ac_config.mode = "full"

        # Force pad_to_maximum when compile is enabled to avoid dynamic shape issues
        if enable_compile and not self.config.pad_to_maximum:
            self.logger.info(
                "torch.compile is enabled: forcing pad_to_maximum=True to avoid "
                "dynamic shape issues with Inductor. Original pad_to_maximum=False."
            )
            self.config.pad_to_maximum = True

        # Force pad_to_maximum when PP is enabled to avoid shape mismatch
        if self.parallel_dims.pp_enabled and not self.config.pad_to_maximum:
            self.logger.info(
                "Pipeline Parallelism is enabled: forcing pad_to_maximum=True to avoid "
                "shape mismatch across microbatches. Original pad_to_maximum=False."
            )
            self.config.pad_to_maximum = True

        if self.enable_tree_training:
            if self.parallel_dims.pp_enabled or self.parallel_dims.cp_enabled:
                raise ValueError(
                    "Tree training with pipeline parallelism (pp > 1) or context parallelism (cp > 1) is currently not supported."
                )
            # Force pad_to_maximum for tree training
            if not self.config.pad_to_maximum:
                self.logger.info(
                    "Tree training enabled: forcing pad_to_maximum=True for "
                    "block mask alignment. Original pad_to_maximum=False."
                )
                self.config.pad_to_maximum = True

        if self.parallel_dims.pp_enabled:
            tie_word_embeddings = getattr(
                self.model_config, "tie_word_embeddings", False
            )
            if tie_word_embeddings:
                raise ValueError(
                    f"Pipeline Parallelism (PP={self.parallel_dims.pp}) is not supported "
                    f"with weight tying (tie_word_embeddings=True). "
                    f"When PP > 1, tok_embeddings and output layers are on different GPUs "
                    f"and cannot share the same weight tensor. "
                    f"Please either disable PP (set pipeline_parallel_size=1) or use a model "
                    f"without weight tying."
                )

        tik = time.perf_counter()

        self._setup_parallelism(param_dtype, ac_config, enable_compile)

        # Synchronize all ranks after parallelization (especially after torch.compile)
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

        self.logger.info(
            f"Applied parallelism in {time.perf_counter() - tik:.2f} seconds"
        )

        self._materialize_and_load_weights()
        self._create_optimizer(ft_spec)

        self.runner = create_runner(
            pp_enabled=self.parallel_dims.pp_enabled,
            model_parts=self.model_parts,
            prepare_inputs_fn=self._prepare_pipelined_mb_inputs
            if self.parallel_dims.pp_enabled
            else self._prepare_mb_inputs,
            pp_stages=self.pp_stages,
            pp_schedule=self.config.archon.pp_schedule,
            pp_group_size=self.parallel_dims.pp,
            has_first_stage=self.pp_has_first_stage,
            has_last_stage=self.pp_has_last_stage,
        )

        self._initialized = True

    @property
    def world_mesh(self) -> DeviceMesh:
        return self._world_mesh

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        return self.parallel_dims.world_mesh["dp"].get_group()

    @property
    def data_parallel_rank(self) -> int:
        return self._dp_rank

    @property
    def data_parallel_world_size(self) -> int:
        return self.parallel_dims.dp_shard

    def current_data_parallel_head(self) -> int:
        return self._dp_head

    def is_data_parallel_head(self) -> bool:
        return self.rank == self._dp_head

    @property
    def pipeline_parallel_rank(self) -> int:
        return self._pp_rank

    def is_pipeline_parallel_head(self) -> bool:
        return self._is_pipeline_parallel_head

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        assert self._pp_cp_tp_group is not None
        return self._pp_cp_tp_group

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        return self._cpu_group

    @property
    def initialized(self) -> bool:
        return self._initialized

    def destroy(self):
        """Clean up resources."""
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "model_parts"):
            self.model_parts.clear()
        gc.collect()
        current_platform.empty_cache()
        gc.collect()

        if dist.is_initialized() and self.own_global_group:
            dist.destroy_process_group()
        self._initialized = False

    def train(self, mode: bool = True):
        for m in self.model_parts:
            m.train(mode=mode)
        return self

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    def optimizer_zero_grad(self):
        assert self.optimizer is not None
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        grad_norm = fsdp2_clip_grad_norm(
            self._get_all_parameters(),
            max_norm=self.optimizer_config.gradient_clipping,
            fsdp_group=self.data_parallel_group,
            tp_group=self.parallel_dims.get_group("tp")
            if self.parallel_dims.tp_enabled
            else None,
            pp_group=self.parallel_dims.get_group("pp")
            if self.parallel_dims.pp_enabled
            else None,
            offload_params=self.config.archon.offload_params,
        )

        if not math.isfinite(grad_norm):
            self.optimizer_zero_grad()
            update_successful = False
        else:
            self.optimizer.step()
            update_successful = True

        current_lr = self.lr_scheduler.get_last_lr()[0]
        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    def lr_scheduler_step(self):
        assert self.lr_scheduler is not None
        self.lr_scheduler.step()

    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> list[torch.Tensor] | None:
        """Forward and optionally backward through micro-batches."""
        return self.runner.run(mb_list, process_output_fn, forward_only)

    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Train on a batch of data."""
        assert self._initialized
        self.optimizer_zero_grad()

        mb_list = self._prepare_mb_list(input_).to(self.device)

        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.data_parallel_group
        )

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = ArchonTrainContext(**ctx_dict)
            return self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier=self.data_parallel_world_size,
            )

        self.forward_backward_batch(mb_list, process_output, forward_only=False)

        return self.optimizer_step()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate on a batch of data."""
        assert self._initialized

        mb_list = self._prepare_mb_list(input_).to(self.device)

        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.data_parallel_group
        )

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = ArchonTrainContext(**ctx_dict)
            return self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
            )

        losses = self.forward_backward_batch(mb_list, process_output, forward_only=True)

        return aggregate_eval_losses(
            losses if self.pp_has_last_stage else None,
            self.data_parallel_group,
            self.pp_has_last_stage,
            self.parallel_dims.get_group("pp")
            if self.parallel_dims.pp_enabled
            else None,
            self._pp_last_stage_rank,
        )

    @torch.no_grad()
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> torch.Tensor:
        """Forward pass without gradient computation."""
        assert self._initialized

        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None
        batch_size = len(output_seqlens)

        mb_list = self._prepare_mb_list(input_).to(self.device)

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = ArchonTrainContext(**ctx_dict)
            return self._compute_forward_result(logits, ctx)

        outputs = self.forward_backward_batch(
            mb_list, process_output, forward_only=True
        )

        if self.pp_has_last_stage:
            assert outputs is not None
            if self.enable_tree_training:
                res = merge_packed_tree_results(outputs, batch_size)
            else:
                res = reorder_and_pad_outputs(
                    outputs, output_seqlens, mb_list, aggregate_fn
                )
        else:
            res = None
        if self.parallel_dims.pp_enabled:
            assert self._pp_last_stage_rank is not None
            res = broadcast_tensor(
                res,
                src_rank=self._pp_last_stage_rank,
                group=self.parallel_dims.get_group("pp"),
            )
        assert res is not None
        return res

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        """Connect to an inference engine for rollout."""
        if self.rollout_engine is not None and self.rollout_engine != engine:
            self.logger.warning(
                f"Connected rollout engine changed from {self.rollout_engine} to {engine}."
            )
        self.rollout_engine = engine
        self.rollout_coordinator = DistRolloutCoordinator(
            rollout_engine=engine, train_engine=self
        )

        if meta.type == "xccl" and not self._weight_sync_state.group_initialized:
            init_weight_update_group(
                state=self._weight_sync_state,
                meta=meta,
                engine=self,
            )

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> dict[str, Any]:
        """Perform rollout using connected inference engine."""
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.rollout_batch(
            data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=group_size,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> dict[str, Any]:
        """Prepare batch from dataloader with rollout."""
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.prepare_batch(
            dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
        )

    def clear_batches(self, *args):
        """Placeholder method of single-controller API."""

    def update_weights(self, meta: WeightUpdateMeta):
        """Update weights to inference engine."""
        self._check_rollout_engine_connected()
        if meta.type == "xccl":
            assert self._weight_sync_state.group_initialized
            tms_context = (
                torch_memory_saver.disable()
                if self.is_offload and not torch.version.hip
                else nullcontext()
            )
            with tms_context:
                update_weights_from_distributed(
                    state=self._weight_sync_state,
                    meta=meta,
                    engine=self,
                )
        elif meta.type == "disk":
            update_weights_from_disk(
                meta=meta,
                engine=self,
            )

    def save(self, meta: SaveLoadMeta):
        """Save model in HuggingFace or DCP format."""
        if meta.weight_format == "hf":
            save_model_to_hf(self, meta.path, meta.tokenizer, meta.processor)
        elif meta.weight_format == "dcp":
            save_to_dcp(self, meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}.")

        if meta.with_optim and meta.weight_format == "hf":
            save_optimizer_state(self, meta.path)

    def load(self, meta: SaveLoadMeta):
        """Load model from HuggingFace or DCP format."""
        if meta.weight_format == "hf":
            load_model_from_hf(self, meta.path)
        elif meta.weight_format == "dcp":
            load_from_dcp(self, meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}.")

        if meta.with_optim and meta.weight_format == "hf":
            load_optimizer_state(self, meta.path)

    def offload(self) -> None:
        """Offload model memory to CPU using torch_memory_saver."""
        self.get_device_stats().log("before offload model")

        current_platform.clear_memory()
        torch_memory_saver.pause()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after offload model")

        self.is_offload = True

    def onload(self) -> None:
        """Onload model memory from CPU back to GPU using torch_memory_saver."""
        torch_memory_saver.resume()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after onload model")

        self.is_offload = False

    def export_stats(self) -> dict[str, float]:
        assert self._initialized
        data = stats_tracker.export_all(reduce_group=self.data_parallel_group)
        if self.parallel_dims.pp_enabled:
            data_list = [data]
            dist.broadcast_object_list(
                data_list,
                src=self._pp_last_stage_rank,
                group=self.parallel_dims.get_group("pp"),
            )
            data.update(data_list[0])
        return data

    def get_device_stats(self) -> DeviceRuntimeInfo:
        return DeviceRuntimeInfo.get_current()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        perf_tracer.configure(config, rank=rank, role=role)

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _check_rollout_engine_connected(self) -> None:
        if self.rollout_engine is None or self.rollout_coordinator is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    def _validate_model_type(self) -> None:
        model_type = getattr(self.model_config, "model_type", "")
        if not is_supported_model(model_type):
            supported = ", ".join(sorted(get_supported_model_types()))
            raise ValueError(
                f"Archon Engine does not support model type '{model_type}'. "
                f"Supported model types: {supported}. "
                f"Please use FSDPEngine for unsupported models."
            )

    def _setup_parallelism(
        self,
        param_dtype: torch.dtype,
        ac_config: ActivationCheckpointConfig | None,
        enable_compile: bool,
    ) -> None:
        if self.parallel_dims.pp_enabled:
            self._apply_pipeline_parallelism(param_dtype, ac_config, enable_compile)
        else:
            self._apply_parallelism(param_dtype, ac_config, enable_compile)

    def _apply_pipeline_parallelism(
        self,
        param_dtype: torch.dtype,
        ac_config: ActivationCheckpointConfig | None,
        enable_compile: bool,
    ) -> None:
        """Apply pipeline parallelism using pipelining_fn."""
        if self.spec.pipelining_fn is None:
            raise RuntimeError(
                f"Pipeline Parallel is enabled but {self.spec.name} "
                f"does not support pipelining"
            )

        (
            self.pp_stages,
            self.model_parts,
            self.pp_has_first_stage,
            self.pp_has_last_stage,
        ) = self.spec.pipelining_fn(
            model=self.model,
            device=self.device,
            parallel_dims=self.parallel_dims,
            archon_config=self.config.archon,
            parallelize_fn=self.spec.parallelize_fn,
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            loss_parallel=True,
            cpu_offload=self.config.archon.offload_params,
            reshard_after_forward_policy="default",
            ac_config=ac_config,
            enable_compile=enable_compile,
        )

        # Delete original model to free memory
        del self.model

        # Determine which rank holds the last pipeline stage
        pp_group = self.parallel_dims.get_group("pp")
        pp_ranks = dist.get_process_group_ranks(pp_group)
        schedule_class = get_schedule_class(self.config.archon.pp_schedule)
        v_style_schedules = (ScheduleZBVZeroBubble, ScheduleDualPipeV)
        if schedule_class in v_style_schedules:
            # V-style: rank 0 holds stages (0, num_stages-1)
            self._pp_last_stage_rank = pp_ranks[0]
        else:
            # Loop-style: last rank has last stage
            self._pp_last_stage_rank = pp_ranks[-1]

        self.logger.info(
            f"PP enabled: has_first={self.pp_has_first_stage}, "
            f"has_last={self.pp_has_last_stage}"
        )

    def _apply_parallelism(
        self,
        param_dtype: torch.dtype,
        ac_config: ActivationCheckpointConfig | None,
        enable_compile: bool,
    ) -> None:
        """Apply parallelism using parallelize_fn."""
        self.spec.parallelize_fn(
            model=self.model,
            parallel_dims=self.parallel_dims,
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            loss_parallel=True,
            cpu_offload=self.config.archon.offload_params,
            reshard_after_forward_policy="default",
            ac_config=ac_config,
            enable_compile=enable_compile,
        )
        self.model_parts = [self.model]

    def _prepare_mb_inputs(
        self, mb_item: MicroBatchItem
    ) -> tuple[dict[str, Any], ArchonTrainContext]:
        inputs = dict(mb_item.padded_mb)

        # Extract trie_node for tree training (if present)
        trie_node = inputs.pop("trie_node", None)

        # For tree training, labels are computed via trie structure, not via roll
        # Tree training input_ids is 1D (packed format), so torch.roll would fail
        if trie_node is not None:
            labels = None
        else:
            labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)

            if self.parallel_dims.cp_enabled:
                cp_mesh = self.parallel_dims.get_mesh("cp")
                inputs, labels = ulysses_slice_inputs(
                    inputs,
                    labels,
                    cp_mesh.get_local_rank(),
                    self.parallel_dims.cp,
                )

            if labels.ndim == 2 and labels.shape[0] == 1:
                labels = labels.squeeze(0)

        ctx = ArchonTrainContext(
            mb_input=mb_item.orig_mb,
            labels=labels,
            pad_length=mb_item.padding_length,
            trie_node=trie_node,
        )
        return inputs, ctx

    def _prepare_pipelined_mb_inputs(
        self,
        mb_list: MicroBatchList,
    ) -> tuple[tuple, dict, torch.Tensor | None, list[ArchonTrainContext]]:
        """Concatenate microbatch inputs for pipeline scheduler's step()/eval() API."""
        input_ids_list: list[torch.Tensor] = []
        positions_list: list[torch.Tensor] = []
        cu_seqlens_list: list[torch.Tensor] = []
        max_seqlen_list: list[int] = []
        target_list: list[torch.Tensor] = []
        contexts: list[ArchonTrainContext] = []

        def ensure_2d(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.ndim == 1 else t

        for mb_item in mb_list:
            inputs, ctx = self._prepare_mb_inputs(mb_item)
            contexts.append(ctx)

            input_ids_list.append(ensure_2d(inputs["input_ids"]))
            positions_list.append(ensure_2d(inputs["position_ids"]))
            cu_seqlens_list.append(ensure_2d(inputs["cu_seqlens"]))
            max_seqlen_list.append(int(inputs["max_seqlen"]))

            # For tree training, labels are None (computed via trie structure)
            if self.pp_has_last_stage and ctx.labels is not None:
                target_list.append(ensure_2d(ctx.labels))

        # Pad cu_seqlens to same length using last value to create zero-length sequences
        max_cu_len = max(cs.shape[1] for cs in cu_seqlens_list)
        padded_cu_seqlens = [
            torch.cat([cs, cs[:, -1:].expand(-1, max_cu_len - cs.shape[1])], dim=1)
            if cs.shape[1] < max_cu_len
            else cs
            for cs in cu_seqlens_list
        ]

        batched_args = (
            (torch.cat(input_ids_list, dim=0),) if self.pp_has_first_stage else ()
        )
        batched_kwargs = {
            "positions": torch.cat(positions_list, dim=0),
            "cu_seqlens": torch.cat(padded_cu_seqlens, dim=0),
            "max_seqlen": torch.tensor(max_seqlen_list),
        }
        # For tree training, target_list is empty (labels computed via trie)
        batched_target = (
            torch.cat(target_list, dim=0)
            if self.pp_has_last_stage and target_list
            else None
        )

        return batched_args, batched_kwargs, batched_target, contexts

    def _create_state_dict_adapter(self) -> BaseStateDictAdapter | None:
        return self.spec.state_dict_adapter_class(
            self.model_config, hf_assets_path=self.config.path
        )

    def _get_all_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.model_parts for p in m.parameters()]

    def _get_model_name_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for m in self.model_parts:
            yield from m.named_parameters()

    def _create_device_model(self):
        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        self.tokenizer = load_hf_tokenizer(self.config.path)

        tik = time.perf_counter()

        # Meta device mode: create structure only, no memory allocation
        # Parameters exist only as metadata until materialized after FSDP
        with torch.device("meta"):
            model = self._create_model_structure()
        model = model.to(getattr(torch, self.config.dtype))
        self.model = model

        self.logger.info(
            f"Model structure created on meta device in "
            f"{time.perf_counter() - tik:.2f}s"
        )
        self.get_device_stats().log("after create model structure")

    def _build_ac_config(self) -> ActivationCheckpointConfig | None:
        # First check if gradient checkpointing is enabled
        if not self.config.gradient_checkpointing:
            return None

        archon_config = self.config.archon
        mode = archon_config.ac_mode

        if mode == "none":
            return None

        ac_config = ActivationCheckpointConfig(
            mode=mode,
            selective_ac_option=archon_config.selective_ac_option,
            memory_budget=archon_config.ac_memory_budget,
            preserve_rng_state=archon_config.ac_preserve_rng_state,
            debug=archon_config.ac_debug,
        )

        self.logger.info(
            f"Activation checkpointing: mode={ac_config.mode}, "
            f"selective_option={ac_config.selective_ac_option}, "
            f"memory_budget={ac_config.memory_budget}, "
            f"preserve_rng={ac_config.preserve_rng_state}, debug={ac_config.debug}"
        )

        return ac_config

    def _create_model_structure(self) -> nn.Module:
        """Create model structure on meta device without loading weights."""
        # Use tree attention type when tree training is enabled
        attn_type = self.config.archon.attn_type
        if self.enable_tree_training:
            if attn_type != "tree":
                self.logger.warning(
                    f"Tree training enabled, overriding attn_type '{self.config.archon.attn_type}' -> 'tree'"
                )
                attn_type = "tree"
        elif attn_type == "tree":
            self.logger.warning(
                "attn_type is 'tree' but tree training is disabled. Overriding to 'varlen'."
            )
            attn_type = "varlen"

        model_args = self.spec.model_args_class.from_hf_config(
            self.model_config,
            is_critic=self.config.is_critic,
            attn_type=attn_type,
        )
        model = self.spec.model_class(model_args)
        return model

    def _materialize_and_load_weights(self):
        """Materialize meta tensors and load weights after FSDP parallelization."""
        if self.config.archon.offload_params:
            init_device = "cpu"
            buffer_device = current_platform.device_type
        else:
            init_device = current_platform.device_type
            buffer_device = init_device

        tik = time.perf_counter()

        for model in self.model_parts:
            model.to_empty(device=init_device)

        if not self.config.init_from_scratch:
            load_model_from_hf(self, self.config.path)
        else:
            with torch.no_grad():
                for model in self.model_parts:
                    model.init_weights()

        for model in self.model_parts:
            model.init_buffers(buffer_device=buffer_device)

        dist.barrier(group=self.cpu_group)

        self.logger.info(
            f"Materialized and loaded weights in {time.perf_counter() - tik:.2f}s"
        )
        self.get_device_stats().log("after materialize and load weights")

    def _create_optimizer(self, ft_spec: FinetuneSpec):
        if self.optimizer_config is None:
            return

        all_params = self._get_all_parameters()

        tik = time.perf_counter()

        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.weight_decay
        beta1 = self.optimizer_config.beta1
        beta2 = self.optimizer_config.beta2
        eps = self.optimizer_config.eps

        if self.optimizer_config.type == "adam":
            self.optimizer = torch.optim.AdamW(
                all_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
                fused=True,
            )
        elif self.optimizer_config.type == "sgd":
            self.optimizer = torch.optim.SGD(
                all_params,
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.optimizer_config.type}"
            )

        total_train_steps = ft_spec.total_train_steps
        num_warmup_steps = int(
            self.optimizer_config.warmup_steps_proportion * total_train_steps
        )

        if self.optimizer_config.lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                total_train_steps,
                min_lr_ratio=self.optimizer_config.min_lr_ratio,
            )
        elif self.optimizer_config.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                total_train_steps,
            )
        elif self.optimizer_config.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
            )
        else:
            raise ValueError(
                f"Unknown lr scheduler type: {self.optimizer_config.lr_scheduler_type}"
            )

        self.logger.info(f"Created optimizer in {time.perf_counter() - tik:.2f}s")

    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        input_ = input_.copy()

        # Tree training path
        if self.enable_tree_training:
            # Tree training cannot work with CP because CP slices sequences
            if self.parallel_dims.cp_enabled:
                raise ValueError(
                    "Tree training cannot be enabled with context parallelism (cp > 1). "
                    "Tree training requires full sequences on each rank."
                )

            tp_size = self.parallel_dims.tp
            # Build tree inputs
            assert BLOCK_SIZE % tp_size == 0, (
                f"BLOCK_SIZE ({BLOCK_SIZE}) must be divisible by "
                f"tensor parallel size ({tp_size})."
            )
            mb_list = build_packed_tree_batch(
                input_,
                mb_spec=self.config.mb_spec,
                pad_to_maximum=self.config.pad_to_maximum,
                dp_group=self.data_parallel_group,
            )
            self.logger.info(
                f"Packed tree #microbatch: {len(mb_list)}, microbatch #tokens: {mb_list.group_lens}, "
                f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}."
            )
            return mb_list

        input_ = amend_position_ids(input_)

        # Pipeline parallelism requires n_microbatches >= pp_stages
        if self.parallel_dims.pp_enabled:
            pp_size = self.parallel_dims.pp
            n_seqs = input_["attention_mask"].shape[0]
            if n_seqs < pp_size:
                raise RuntimeError(
                    f"Pipeline parallelism requires at least {pp_size} sequences, "
                    f"but got {n_seqs}. Increase batch size or reduce PP degree."
                )
            min_n_mbs = pp_size
            mb_spec = MicroBatchSpec.new(
                self.config.mb_spec,
                n_mbs=max(min_n_mbs, self.config.mb_spec.n_mbs or 1),
                n_mbs_divisor=pp_size,
            )
        else:
            mb_spec = self.config.mb_spec

        mb_list = split_padded_tensor_dict_into_mb_list(input_, mb_spec)
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]

        # LCM ensures page-aligned memory and exact CP slicing without extra padding.
        batch_align_to = math.lcm(self.page_size, self.parallel_dims.seq_len_divisor)
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
            batch_align_to=batch_align_to,
        )

        self.logger.info(
            f"Microbatch #tokens (rank {self.rank}): {mb_list.group_lens}, "
            f"padded to: {mb_list.padded_to_lengths}"
        )

        mb_list = unsqueeze_mb_list(mb_list)

        assert mb_list.padded_mbs is not None
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)

        return mb_list

    def _compute_logprobs_and_loss(
        self,
        logits: torch.Tensor,
        ctx: ArchonTrainContext,
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Compute logprobs/entropy and return scaled loss."""
        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )

        if not self.config.is_critic:
            if self.enable_tree_training:
                # Handle dummy trie (empty tree for DP synchronization)
                # When trie has no sequences, return zero loss with grad connection
                if ctx.trie_node is None or not ctx.trie_node.all_sequence_ids:
                    # Return zero loss that maintains gradient connection to logits
                    # This ensures backward() works correctly for FSDP synchronization
                    return logits.sum() * 0.0

                # For tree training, use gather_packed_tree_vocab_stats to properly
                # unpack vocab stats from tree structure back to per-sequence format.
                vocab_min_logits, vocab_max_logits = gather_packed_tree_vocab_stats(
                    logits, ctx.trie_node
                )
                logprobs, entropy = gather_packed_tree_logprobs_entropy(
                    logits,
                    ctx.trie_node,
                    ctx.mb_input["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=self.parallel_dims.get_group("tp")
                    if self.parallel_dims.tp_enabled
                    else None,
                )
            else:
                logprobs, entropy = self._compute_logprobs_entropy(logits, ctx.labels)
                vocab_min_logits, vocab_max_logits = self._get_vocab_min_max_logits(
                    logits
                )

                if self.parallel_dims.cp_enabled:
                    cp_group = self.parallel_dims.get_group("cp")
                    logprobs = ulysses_gather_output(logprobs, cp_group)
                    entropy = ulysses_gather_output(entropy, cp_group)

                if ctx.pad_length > 0:
                    logprobs = logprobs[: -ctx.pad_length]
                    entropy = entropy[: -ctx.pad_length]
                    vocab_min_logits = vocab_min_logits[: -ctx.pad_length]
                    vocab_max_logits = vocab_max_logits[: -ctx.pad_length]

            loss = loss_fn(
                logprobs,
                entropy,
                ctx.mb_input,
                vocab_min_logits=vocab_min_logits,
                vocab_max_logits=vocab_max_logits,
            )
        else:
            values = logits.squeeze(-1)

            if self.parallel_dims.cp_enabled:
                values = ulysses_gather_output(
                    values, self.parallel_dims.get_group("cp")
                )

            if ctx.pad_length > 0:
                values = values[: -ctx.pad_length]

            loss = loss_fn(values, ctx.mb_input)

        loss_scale = loss_weight_fn(ctx.mb_input) / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _get_vocab_min_max_logits(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get vocab min/max logits for non-tree training path."""
        vocab_min_logits = logits.detach().min(-1).values.float()
        vocab_max_logits = logits.detach().max(-1).values.float()
        return vocab_min_logits, vocab_max_logits

    def _compute_logprobs_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy from logits."""
        logprobs, entropy = gather_logprobs_entropy(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=self.parallel_dims.get_group("tp")
            if self.parallel_dims.tp_enabled
            else None,
        )
        return logprobs, entropy

    def _compute_forward_result(
        self,
        logits: torch.Tensor,
        ctx: ArchonTrainContext,
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        """Compute forward output (logprobs or values)."""
        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )

        if not self.config.is_critic:
            if self.enable_tree_training:
                result = _gather_packed_tree_logprobs(
                    logits,
                    ctx.trie_node,
                    ctx.mb_input["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=self.parallel_dims.get_group("tp")
                    if self.parallel_dims.tp_enabled
                    else None,
                )
                return result
            result = self._compute_logprobs(logits, ctx.labels)
        else:
            result = logits.squeeze(-1)

        if self.parallel_dims.cp_enabled:
            result = ulysses_gather_output(result, self.parallel_dims.get_group("cp"))

        if ctx.pad_length > 0:
            result = result[: -ctx.pad_length]

        return result

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities from logits (without entropy)."""
        logprobs = gather_logprobs(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=self.parallel_dims.get_group("tp")
            if self.parallel_dims.tp_enabled
            else None,
        )
        return logprobs


class ArchonPPOActor(ArchonEngine):
    """PPO Actor implementation using Archon backend."""

    def __init__(self, config):
        from areal.trainer.ppo.actor import PPOActor

        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> dict[str, Any]:
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config, scheduler: Scheduler):
        from areal.trainer.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)


class ArchonPPOCritic(ArchonEngine):
    """PPO Critic implementation using Archon backend."""

    def __init__(self, config):
        from areal.trainer.ppo.critic import PPOCritic

        super().__init__(config)
        self.critic = PPOCritic(config, self)

    @torch.no_grad()
    def compute_values(self, *args, **kwargs) -> torch.Tensor:
        return self.critic.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.critic.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config, scheduler: Scheduler):
        from areal.trainer.ppo.critic import PPOCriticController

        return PPOCriticController(train_engine=cls, config=config, scheduler=scheduler)


class ArchonLMEngine(ArchonEngine):
    """Archon-based LM Engine for SFT training."""

    def __init__(self, config: TrainEngineConfig):
        from areal.trainer.sft.lm_engine import LMEngine

        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)

    @classmethod
    def as_controller(cls, config: TrainEngineConfig, scheduler: Scheduler):
        from areal.trainer.sft.lm_engine import LMController

        return LMController(train_engine=cls, config=config, scheduler=scheduler)
