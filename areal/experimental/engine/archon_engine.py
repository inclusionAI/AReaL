import gc
import math
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import Schedule1F1B
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import MicroBatchSpec, PerfTracerConfig, TrainEngineConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import (
    DeviceRuntimeInfo,
    FinetuneSpec,
    ParamSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Scheduler
from areal.api.workflow_api import WorkflowLike
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
from areal.utils import logging, name_resolve, names, perf_tracer, stats_tracker
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
from areal.utils.distributed import init_custom_process_group, patch_dist_group_timeout
from areal.utils.fsdp import get_cosine_schedule_with_warmup
from areal.utils.fsdp.grad import fsdp2_clip_grad_norm
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.network import find_free_ports, gethostip
from areal.utils.offload import torch_memory_saver
from areal.utils.perf_tracer import trace_perf


@dataclass
class ArchonTrainContext:
    """Context passed through Archon forward/backward pipeline.

    Attributes:
        mb_input: Original microbatch input.
        labels: Target token ids for loss computation (rolled from input_ids).
        pad_length: Batch-level padding added by pad_mb_list.
    """

    mb_input: dict[str, Any]
    labels: torch.Tensor
    pad_length: int = 0


class ArchonEngine(TrainEngine):
    """Archon Engine is a torch-native training backend."""

    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.optimizer_config = config.optimizer

        self.model: nn.Module
        self.optimizer: torch.optim.Optimizer
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler
        self.tokenizer: PreTrainedTokenizerFast
        self.model_config: PretrainedConfig
        self._version: int = 0

        self._initialized = False
        self.own_global_group = False
        self.is_offload = False
        self._cpu_group: dist.ProcessGroup

        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None

        self.weight_update_group_initialized = False
        self.weight_update_group_name: str
        self.weight_update_master_addr: str
        self.weight_update_master_port: int
        self.weight_update_group: dist.ProcessGroup

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )

        self._validate_model_type()

        # Get ModelSpec based on model type
        model_type = getattr(self.model_config, "model_type", "")
        self.spec: ModelSpec = get_model_spec(model_type)

        self.parallel_dims: ArchonParallelDims
        self._world_mesh: DeviceMesh
        self.state_dict_adapter: BaseStateDictAdapter | None = None

        # PP (Pipeline Parallelism) related attributes
        self.pp_stages: list[PipelineStage] = []
        self.model_parts: list[nn.Module] = []
        self.pp_has_first_stage: bool = True
        self.pp_has_last_stage: bool = True

        self.enable_tree_training = config.enable_tree_training

        self.world_size: int
        self.rank: int

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

        tp_size = parallel_strategy.tensor_parallel_size
        dp_size = parallel_strategy.data_parallel_size
        cp_size = parallel_strategy.context_parallel_size
        pp_size = parallel_strategy.pipeline_parallel_size
        ep_size = parallel_strategy.expert_parallel_size
        etp_size = parallel_strategy.expert_tensor_parallel_size

        self.parallel_dims = ArchonParallelDims(
            dp_shard=dp_size,
            tp=tp_size,
            cp=cp_size,
            pp=pp_size,
            ep=ep_size,
            etp=etp_size,
            world_size=self.world_size,
            device_type=current_platform.device_type,
        )

        self._world_mesh = self.parallel_dims.world_mesh
        self._cp_tp_group = self.parallel_dims.get_group("cp_tp")

        # Compute dp_rank: the rank within the dp dimension (for data loading)
        dp_mesh = self.parallel_dims.get_mesh("dp")
        if dp_mesh is not None:
            self._dp_rank = dp_mesh.get_local_rank()
        else:
            self._dp_rank = 0

        # Compute dp_head: the rank that holds the batch for this cp_tp group
        self._dp_head = dist.get_process_group_ranks(self._cp_tp_group)[0]

        # Compute pp_last_stage_rank: global rank of the last PP stage
        if self.parallel_dims.pp_enabled:
            pp_group = self.parallel_dims.get_group("pp")
            self._pp_last_stage_rank = dist.get_process_group_ranks(pp_group)[-1]
        else:
            self._pp_last_stage_rank = None

        self.weight_update_group_name = "update_weight_group"

        self.logger.info(
            f"Initialized Archon engine with parallel dims: "
            f"pp={self.parallel_dims.pp}, dp_shard={self.parallel_dims.dp_shard}, "
            f"tp={self.parallel_dims.tp}, cp={self.parallel_dims.cp} (Ulysses SP), "
            f"ep={self.parallel_dims.ep}, etp={etp_size}"
        )

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        """Initialize model, optimizer, and apply parallelism."""
        assert addr is None, "ArchonEngine does not support remote initialization."
        assert ft_spec is not None, "ArchonEngine requires FinetuneSpec to initialize."

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
            self.logger.warning("Tree training is not supported for Archon engine yet.")

        tik = time.perf_counter()

        # TODO: Refactor to reduce scattered PP if-else branches across the codebase.
        if self.parallel_dims.pp_enabled:
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
                parallel_dims=self.parallel_dims,
                device=self.device,
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
            self.model = None

            self.logger.info(
                f"PP enabled: has_first={self.pp_has_first_stage}, "
                f"has_last={self.pp_has_last_stage}"
            )
        else:
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

        # Synchronize all ranks after parallelization (especially after torch.compile)
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

        self.logger.info(
            f"Applied parallelism in {time.perf_counter() - tik:.2f} seconds"
        )

        self._materialize_and_load_weights()

        self._create_optimizer(ft_spec)

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
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        return self._cp_tp_group

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
        if self.parallel_dims.pp_enabled:
            return self._forward_backward_pipelined(
                mb_list, process_output_fn, forward_only
            )
        else:
            return self._forward_backward_sequential(
                mb_list, process_output_fn, forward_only
            )

    def _forward_backward_pipelined(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> list[torch.Tensor] | None:
        """Pipelined forward/backward using Schedule1F1B."""
        n_microbatches = len(mb_list)

        batched_args, batched_kwargs, batched_target, contexts = (
            self._prepare_pp_batched_inputs(mb_list)
        )

        args = batched_args if self.pp_has_first_stage else ()

        if forward_only:
            return self._forward_backward_pipelined_eval(
                n_microbatches, args, batched_kwargs, contexts, process_output_fn
            )
        return self._forward_backward_pipelined_train(
            n_microbatches,
            args,
            batched_kwargs,
            batched_target,
            contexts,
            process_output_fn,
        )

    def _forward_backward_pipelined_eval(
        self,
        n_microbatches: int,
        args: tuple,
        batched_kwargs: dict[str, Any],
        contexts: list,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
    ) -> list[torch.Tensor] | None:
        """Pipelined forward pass without backward (eval mode)."""
        schedule = Schedule1F1B(
            self.pp_stages[0],
            n_microbatches=n_microbatches,
            loss_fn=None,
            scale_grads=False,
        )

        schedule.eval(*args, **batched_kwargs)

        if not self.pp_has_last_stage:
            return None

        results: list[torch.Tensor] = []
        for output, ctx in zip(self.pp_stages[0].output_chunks, contexts, strict=True):
            # Squeeze batch dim: PP outputs (1, seq_len, vocab) -> (seq_len, vocab)
            if output.ndim == 3:
                output = output.squeeze(0)
            ctx_dict = ctx.__dict__.copy()
            result = process_output_fn(output, ctx_dict)
            if result is not None:
                results.append(result.detach())
        return results

    def _forward_backward_pipelined_train(
        self,
        n_microbatches: int,
        args: tuple,
        batched_kwargs: dict[str, Any],
        batched_target: torch.Tensor | None,
        contexts: list,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
    ) -> list[torch.Tensor]:
        """Pipelined forward and backward pass (training mode)."""
        if self.pp_has_last_stage:
            pp_loss_fn = self._create_pp_loss_fn(contexts, process_output_fn)
        else:
            # Non-last stage: dummy loss that keeps all elements in computation graph
            # so autograd can compute complete pred.grad for upstream stage
            def pp_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return pred.sum() * 0.0

        schedule = Schedule1F1B(
            self.pp_stages[0],
            n_microbatches=n_microbatches,
            loss_fn=pp_loss_fn,
            scale_grads=False,
        )

        schedule.step(*args, target=batched_target, **batched_kwargs)

        # Training does not return any meaningful results
        return []

    def _forward_backward_sequential(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool,
    ) -> list[torch.Tensor]:
        """Sequential forward/backward through each microbatch."""
        results: list[torch.Tensor] = []
        for mb_item in mb_list:
            inputs, ctx = self._prepare_mb_inputs(mb_item)

            logits = self.model(
                inputs["input_ids"],
                inputs["position_ids"],
                cu_seqlens=inputs["cu_seqlens"],
                max_seqlen=int(inputs["max_seqlen"]),
            )
            logits = logits.squeeze(0)

            ctx_dict = ctx.__dict__.copy()
            result = process_output_fn(logits, ctx_dict)

            if result is not None:
                if forward_only:
                    results.append(result.detach())
                else:
                    result.backward()

        # When forward_only is False, this actually returns meaningless empty content
        return results

    def _prepare_pp_batched_inputs(
        self,
        mb_list: MicroBatchList,
    ) -> tuple[tuple, dict, torch.Tensor | None, list[ArchonTrainContext]]:
        """Concatenate microbatch inputs for PP schedule step()/eval() API."""
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

            if self.pp_has_last_stage:
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
        batched_target = (
            torch.cat(target_list, dim=0) if self.pp_has_last_stage else None
        )

        return batched_args, batched_kwargs, batched_target, contexts

    def _create_pp_loss_fn(
        self,
        contexts: list[ArchonTrainContext],
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Create PP-compatible loss_fn from process_output_fn."""
        # Iterator consumed once per microbatch; returned loss_fn is single-use only
        ctx_iter = iter(contexts)

        def pp_loss_fn(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            ctx = next(ctx_iter)

            # Squeeze batch dim: PP outputs (1, seq_len, vocab) -> (seq_len, vocab)
            if pred.ndim == 3:
                pred = pred.squeeze(0)

            ctx_dict = ctx.__dict__.copy()
            loss = process_output_fn(pred, ctx_dict)

            if loss is None:
                return pred.sum() * 0.0

            return loss

        return pp_loss_fn

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

        if meta.type == "xccl" and not self.weight_update_group_initialized:
            self._init_weight_update_from_distributed(meta)
            self.weight_update_group_initialized = True

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
            # TODO: support PP weight sync via XCCL
            if self.parallel_dims.pp_enabled:
                raise NotImplementedError(
                    "PP weight synchronization via XCCL not yet supported. "
                    "Use disk-based weight sync (meta.type='disk') instead."
                )
            assert self.weight_update_group_initialized
            tms_context = (
                torch_memory_saver.disable()
                if self.is_offload and not torch.version.hip
                else nullcontext()
            )
            with tms_context:
                self._update_weights_from_distributed(meta)
        elif meta.type == "disk":
            self._update_weights_from_disk(meta)

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

    def _create_state_dict_adapter(self) -> BaseStateDictAdapter | None:
        return self.spec.state_dict_adapter_class(
            self.model_config, hf_assets_path=self.config.path
        )

    def _get_all_parameters(self) -> list[nn.Parameter]:
        return [p for m in self.model_parts for p in m.parameters()]

    def _get_model_name_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for m in self.model_parts:
            yield from m.named_parameters()

    def _get_full_tensor(self, param: nn.Parameter) -> torch.Tensor:
        """Get full tensor from a parameter, handling DTensor and CPU offload."""
        tensor = param.data
        if isinstance(tensor, DTensor):
            if tensor.device.type != "cpu":
                return tensor.full_tensor()

            temp_dtensor = DTensor.from_local(
                tensor.to_local(),
                device_mesh=tensor.device_mesh,
                placements=tensor.placements,
            )
            return temp_dtensor.full_tensor()
        else:
            if tensor.device.type == "cpu":
                tensor = tensor.to(current_platform.device_type)
            return tensor

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta):
        assert meta.type == "xccl"

        meta.nccl_master_address = self.weight_update_master_addr = gethostip()
        meta.nccl_master_port = self.weight_update_master_port = find_free_ports(1)[0]
        meta.nccl_group_name = self.weight_update_group_name

        # Processes launched with torchrun set TORCHELASTIC_USE_AGENT_STORE=True,
        # which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        if dist.get_rank() == 0:
            assert meta.alloc_mode is not None

            fut = self.rollout_engine.init_weights_update_group(meta)

            self.logger.info(
                f"Initializing weight update group: type={meta.type} "
                f"init_method=tcp://{meta.nccl_master_address}:{meta.nccl_master_port} "
                f"group={meta.nccl_group_name}"
            )
            self.weight_update_group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=meta.alloc_mode.gen.world_size + 1,
                init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
                rank=0,
                group_name=meta.nccl_group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )

            fut.result()

    @trace_perf("archon_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        """Broadcast parameters from rank 0, converting to HF format if needed."""
        meta.nccl_master_address = self.weight_update_master_addr
        meta.nccl_master_port = self.weight_update_master_port
        meta.nccl_group_name = self.weight_update_group_name

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
        main_rank = dist.get_rank() == 0

        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []

        param_iterator = self._get_model_name_parameters()

        for name, param in param_iterator:
            tensor = self._get_full_tensor(param)

            if not main_rank:
                continue

            if self.state_dict_adapter is not None:
                hf_pairs = self.state_dict_adapter.convert_single_to_hf(name, tensor)
            else:
                hf_pairs = [(name, tensor)]

            for hf_name, hf_tensor in hf_pairs:
                tensor_size = hf_tensor.numel() * hf_tensor.element_size()

                if tensor_size + buffer_size > weight_chunked_mem_size:
                    self._update_bucket_weights_from_distributed(meta, named_tensors)
                    buffer_size = 0
                    named_tensors = []

                named_tensors.append((hf_name, hf_tensor))
                buffer_size += tensor_size

        if named_tensors:
            self._update_bucket_weights_from_distributed(meta, named_tensors)

        dist.barrier(group=self.cpu_group)

        if dist.get_rank() == 0:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ):
        if not named_tensors:
            return

        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in named_tensors
        ]

        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        for _, tensor in named_tensors:
            handles.append(
                dist.broadcast(
                    tensor, src=0, group=self.weight_update_group, async_op=True
                )
            )
        for handle in handles:
            handle.wait()

        fut.result()

        named_tensors.clear()

    @trace_perf("archon_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self, meta: WeightUpdateMeta):
        fut = Future()

        if dist.get_rank() == 0:
            fut = self.rollout_engine.update_weights_from_disk(meta)

        assert meta.path is not None
        save_model_to_hf(self, meta.path, self.tokenizer, None)

        if dist.get_rank() == 0:
            update_name = names.update_weights_from_disk(
                self.config.experiment_name,
                self.config.trial_name,
                self.get_version(),
            )
            name_resolve.add(
                update_name, str(datetime.now().timestamp()), keepalive_ttl=120
            )

            fut.result()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

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
        model_args = self.spec.model_args_class.from_hf_config(
            self.model_config,
            is_critic=self.config.is_critic,
            attn_type=self.config.archon.attn_type,
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

    def _prepare_mb_inputs(
        self, mb_item: MicroBatchItem
    ) -> tuple[dict[str, Any], ArchonTrainContext]:
        inputs = dict(mb_item.padded_mb)

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
        )
        return inputs, ctx

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
        if not self.config.is_critic:
            logprobs, entropy = self._compute_logprobs_entropy(logits, ctx.labels)

            if self.parallel_dims.cp_enabled:
                cp_group = self.parallel_dims.get_group("cp")
                logprobs = ulysses_gather_output(logprobs, cp_group)
                entropy = ulysses_gather_output(entropy, cp_group)

            if ctx.pad_length > 0:
                logprobs = logprobs[: -ctx.pad_length]
                entropy = entropy[: -ctx.pad_length]

            loss = loss_fn(logprobs, entropy, ctx.mb_input)
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
    ) -> torch.Tensor:
        """Compute forward output (logprobs or values)."""
        if not self.config.is_critic:
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
        from areal.engine.ppo.actor import PPOActor

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
        from areal.engine.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)


class ArchonPPOCritic(ArchonEngine):
    """PPO Critic implementation using Archon backend."""

    def __init__(self, config):
        from areal.engine.ppo.critic import PPOCritic

        super().__init__(config)
        self.critic = PPOCritic(config, self)

    @torch.no_grad()
    def compute_values(self, *args, **kwargs) -> torch.Tensor:
        return self.critic.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.critic.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config, scheduler: Scheduler):
        from areal.engine.ppo.critic import PPOCriticController

        return PPOCriticController(train_engine=cls, config=config, scheduler=scheduler)


class ArchonLMEngine(ArchonEngine):
    """Archon-based LM Engine for SFT training."""

    def __init__(self, config: TrainEngineConfig):
        from areal.engine.sft.lm_engine import LMEngine

        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)

    @classmethod
    def as_controller(cls, config: TrainEngineConfig, scheduler: Scheduler):
        from areal.engine.sft.lm_engine import LMController

        return LMController(train_engine=cls, config=config, scheduler=scheduler)
