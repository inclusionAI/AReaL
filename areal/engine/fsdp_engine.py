import dataclasses
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.nn.functional as dist_F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.tensor import DTensor
from torch_memory_saver import torch_memory_saver
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    PreTrainedTokenizerFast,
    ProcessorMixin,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from areal.api.alloc_mode import FSDPParallelStrategy, ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import FinetuneSpec, ParamSpec, SaveLoadMeta, WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow
from areal.core.dist_rollout import DistRolloutCoordinator
from areal.engine.base_hf_engine import BaseHFEngine
from areal.models.transformers.ulyssess_patch import apply_monkey_patch
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names, pkg_version
from areal.utils.data import (
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
)
from areal.utils.device import clear_memory, print_memory
from areal.utils.distributed import (
    get_gloo_group,
    init_custom_process_group,
    init_gloo_group,
)
from areal.utils.fsdp import fsdp2_load_full_state_dict, get_cosine_schedule_with_warmup
from areal.utils.fsdp.checkpoint import DCPState
from areal.utils.fsdp.grad import fsdp2_clip_grad_norm
from areal.utils.fsdp.optimizer import AnyPrecisionAdamW
from areal.utils.fsdp.parallel import ParallelHelper, parallelize_model
from areal.utils.nccl import NCCL_DEFAULT_TIMEOUT
from areal.utils.perf_tracer import trace_perf, trace_scope
from areal.utils.save_load import get_state_dict_from_repo_id_or_path
from areal.utils.ulysses import (
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
    ulysses_prepare_inputs,
)


class FSDPEngine(BaseHFEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        # FSDP options
        self.cpu_offload: CPUOffloadPolicy | None = None

        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None

        self.parallel_helper: ParallelHelper
        self.world_mesh: DeviceMesh

        self.dp_group: dist.ProcessGroup
        self.sp_group: dist.ProcessGroup

        self.rank: int
        self.dp_head: int
        self.dp_rank: int

        # Optimizer offload/onload buffers
        self._optimizer_offload_buffer: torch.Tensor | None = None  # CPU buffer
        self._optimizer_onload_buffer: torch.Tensor | None = None  # GPU buffer
        self._optimizer_offload_specs: list[tuple[dict, str, int, int]] = []

        # Model offload/onload buffers
        self._model_offload_buffer: torch.Tensor | None = None  # CPU buffer
        self._model_onload_buffer: torch.Tensor | None = None  # GPU buffer
        self._model_offload_specs: list[
            tuple[torch.nn.Parameter, int, int, dict | None]
        ] = []  # (tensor, start, end, dtensor_meta)

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        return self.dp_group

    @property
    def data_parallel_rank(self) -> int:
        return self.dp_rank

    @property
    def data_parallel_world_size(self) -> int:
        return self.parallel_helper.dp_size

    def current_data_parallel_head(self) -> int:
        return self.dp_head

    def is_data_parallel_head(self) -> bool:
        return self.rank == self.dp_head

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        return self.mp_group

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> FSDPParallelStrategy:
        return FSDPParallelStrategy(
            **dataclasses.asdict(parallel_strategy),
        )

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        super().create_process_group(parallel_strategy)

        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()

        self.logger = logging.getLogger(f"[FSDP Engine Rank {dist.get_rank()}]")

        parallel_strategy = self._make_parallel_strategy(parallel_strategy)

        self.parallel_helper = ParallelHelper.from_parallel_strategy(parallel_strategy)

        self.logger.info(
            f"Initializing device mesh with parallel dims {str(self.parallel_helper)}."
        )

        self.world_mesh = self.parallel_helper.world_mesh

        self.dp_group = self.world_mesh["dp"].get_group()
        self.sp_group = self.world_mesh["sp"].get_group()

        # Sequence and model parallel group (sp+tp)
        self.mp_group = self.world_mesh["sp_tp"].get_group()

        self.rank = dist.get_rank()

        self.dp_head = int(self.world_mesh["sp_tp"].mesh[0].item())
        self.dp_rank = dist.get_rank(self.dp_group)

        self.logger.info(f"Data parallel head {self.dp_head} and rank {self.dp_rank}")

        # Initialize gloo group for CPU-based communication
        # This is needed for barrier synchronization when models are moved to CPU
        init_gloo_group()

    def initialize(
        self,
        addr: str | None,
        ft_spec: FinetuneSpec | None,
    ):
        # Initialize distributed enviroments and load model.
        assert addr is None, "FSDPEngine does not support remote initialization."
        assert ft_spec is not None, "FSDPEngine requires FinetuneSpec to initialize."
        if pkg_version.is_version_less("torch", "2.4.0"):
            raise RuntimeError("areal only supports FSDP2, which requires torch>=2.4.0")

        # Create device model
        self.create_device_model()

        # Monkey patch: replace attention's forward() with Ulysses variant.
        apply_monkey_patch(
            model=self.model,
            ulysses_sp_size=self.parallel_helper.sp_size,
        )

        if self.config.use_lora:
            self._apply_peft_wrapper()

        # sharding_strategy = ShardingStrategy.FULL_SHARD
        # Simple auto wrap policy
        self.cpu_offload = (
            CPUOffloadPolicy() if self.config.fsdp.offload_params else None
        )
        tik = time.perf_counter()
        # Prepare lora weights synchronization
        if self.config.use_lora:
            if dist.get_rank() == 0:
                full_state = self.model.state_dict()
            else:
                full_state = {}
        # NOTE: This applies FSDP2 with N-D parallelism (DP+SP+TP)
        parallelize_model(
            self.model,
            config=self.config,
            model_config=self.model_config,
            nd_device_mesh=self.world_mesh,
            parallel_helper=self.parallel_helper,
            cpu_offload=self.cpu_offload,
            wrap_policy=self.config.fsdp.wrap_policy,
        )
        # Synchronize initialized lora weights
        if self.config.use_lora:
            fsdp2_load_full_state_dict(
                self.model,
                full_state,
                self.cpu_offload,
                tie_word_embeddings=self.model_config.tie_word_embeddings,
            )
        self.logger.info(
            f"Applying FSDP2 with N-D parallelism for {time.perf_counter() - tik:.2f} seconds"
        )

        self.create_optimizer(ft_spec)
        self.initialized = True

        # Offload model after initialization if enabled
        if self.config.offload_train:
            self._initialize_optimizer_offload_buffer()
            self._initialize_model_offload_buffer()
            self.sleep()

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
        elif meta.weight_format == "dcp":
            self._save_to_dcp(meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim and meta.weight_format == "hf":
            self.save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            self._load_from_dcp(meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim and meta.weight_format == "hf":
            self.load_optimizer_state(meta.path)

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast | None,
        processor: ProcessorMixin | None,
    ):
        """Save model in HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        os.makedirs(path, exist_ok=True)

        # FSDP2 checkpoint saving
        # Get full state dict with FSDP2
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(self.model, options=options)

        # save huggingface model on rank 0
        if dist.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.model_config.save_pretrained(path)
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if processor is not None:
                processor.save_pretrained(path)

        dist.barrier(device_ids=[self.device.index])

    def _load_model_from_hf(self, path: str):
        """Load model from HuggingFace format."""
        if dist.get_rank() == 0:
            full_state = get_state_dict_from_repo_id_or_path(path)
        else:
            full_state = {}

        fsdp2_load_full_state_dict(
            self.model,
            full_state,
            self.cpu_offload,
            tie_word_embeddings=self.model_config.tie_word_embeddings,
        )

    def _save_to_dcp(
        self,
        path: str,
        with_optim: bool,
    ):
        """Save model in PyTorch Distributed Checkpoint (DCP) format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        os.makedirs(path, exist_ok=True)

        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.save(state_dict, checkpoint_id=path)

    def _load_from_dcp(self, path: str, with_optim: bool):
        """Load model from Distributed Checkpoint (DCP) format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=path,
        )

    def _apply_peft_wrapper(self):
        config = self.config
        if not config.target_modules or config.target_modules == ["all-linear"]:
            target_modules = "all-linear"
        else:
            target_modules = config.target_modules
        peft_config = {
            "task_type": TaskType.CAUSAL_LM,
            "r": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "target_modules": target_modules,
            "bias": "none",
        }
        if self.config.peft_type == "lora":
            peft_config = LoraConfig(**peft_config)
        else:
            raise NotImplementedError()

        self.model.enable_input_require_grads()
        self.model = get_peft_model(
            self.model,
            peft_config,
            autocast_adapter_dtype=False,
        )

        if self.rank == 0:
            self.model.print_trainable_parameters()

    @trace_perf("fsdp_engine.update_bucket", category="comm")
    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ):
        # Early exit when chunk size is relatively small
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

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta):
        assert meta.type == current_platform.communication_backend

        # NOTE: Processes launched with torchrun will set the following env var to True,
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
                timeout=NCCL_DEFAULT_TIMEOUT,
            )

            fut.result()

    @trace_perf("fsdp_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        """Broadcast parameters (chunked) from rank 0 (FSDP2 compatible)."""

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()

        dist.barrier(device_ids=[self.device.index])

        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

        buffer_size = 0
        named_tensors = []

        for name, param in self.get_model_name_parameters():
            if isinstance(param.data, DTensor):
                tensor = param.data.full_tensor()
            else:
                tensor = param.data

            # Ranks other than 0 only help to get the full tensor
            if dist.get_rank() != 0:
                continue

            tensor_size = tensor.numel() * tensor.element_size()

            if tensor_size + buffer_size > weight_chunked_mem_size:
                self._update_bucket_weights_from_distributed(meta, named_tensors)
                buffer_size = 0

            named_tensors.append((name, tensor))
            buffer_size += tensor_size

        # Only rank-0 CAN contain named tensors here
        if named_tensors:
            self._update_bucket_weights_from_distributed(meta, named_tensors)

        dist.barrier(device_ids=[self.device.index])

        if dist.get_rank() == 0:
            self.rollout_engine.continue_generation()

        dist.barrier(device_ids=[self.device.index])
        current_platform.synchronize()

    @trace_perf("fsdp_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self, meta: WeightUpdateMeta):
        fut = Future()

        if dist.get_rank() == 0:
            fut = self.rollout_engine.update_weights_from_disk(meta)

        self._save_model_to_hf(meta.path, self.tokenizer, self.processor)
        # dist.barrier() are called when _save_model_to_hf finished

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

        dist.barrier(device_ids=[self.device.index])
        current_platform.synchronize()

    def update_weights(self, meta: WeightUpdateMeta):
        self._check_rollout_engine_connected()
        if meta.type == current_platform.communication_backend:
            assert self.weight_update_group_initialized
            # In offload mode, wakes up parameters as needed to perform the update.
            tms_context = (
                torch_memory_saver.disable()
                if self.config.offload_train
                and self.config.offload_train_mode == "tms"
                and not torch.version.hip
                else nullcontext()
            )
            with tms_context:
                self._update_weights_from_distributed(meta)
        elif meta.type == "disk":
            self._update_weights_from_disk(meta)
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        if self.rollout_engine is not None and self.rollout_engine != engine:
            self.logger.warning(
                f"Connected rollout engine changed from {self.rollout_engine} to {engine}."
            )
        self.rollout_engine = engine
        self.rollout_coordinator = DistRolloutCoordinator(
            rollout_engine=engine, train_engine=self
        )

        if (
            meta.type == current_platform.communication_backend
            and not self.weight_update_group_initialized
        ):
            self._init_weight_update_from_distributed(meta)
            self.weight_update_group_initialized = True

        dist.barrier(device_ids=[self.device.index])
        current_platform.synchronize()

    def _check_rollout_engine_connected(self):
        """Validate that rollout engine has been connected via connect_engine()."""
        if self.rollout_engine is None or self.rollout_coordinator is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        granularity: int = 1,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.
        """
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.rollout_batch(
            data,
            granularity=granularity,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        granularity: int = 1,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
    ) -> dict[str, Any]:
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.prepare_batch(
            dataloader,
            granularity=granularity,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
        )

    @trace_perf("fsdp_engine.train_batch", category="compute")
    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Train on a batch using gradient accumulation."""
        # Wake up model if offload is enabled
        if self.config.offload_train:
            self.wake_up()

        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        if self.parallel_helper.sp_size > 1:
            set_ulysses_sequence_parallel_group(self.sp_group)

        self.optimizer.zero_grad()

        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)

        total_loss_weight = (
            torch.stack([loss_weight_fn(mb) for mb in mb_list.mbs])
            .sum()
            .detach()
            .clone()
            .to(dtype=torch.float32, device=self.device)
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight, group=self.dp_group)

        # Process microbatches with gradient accumulation
        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):
            if self.parallel_helper.sp_size > 1:
                input_ids = padded_mb_input["input_ids"]
                position_ids = padded_mb_input.get("position_ids", None)

                if self.is_vision_model:
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        _,
                    ) = ulysses_pad(
                        input_ids, position_ids, sp_size=self.parallel_helper.sp_size
                    )
                else:
                    # Pad and slice the inputs
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        _,
                    ) = ulysses_pad_and_slice_inputs(
                        input_ids,
                        position_ids,
                        sp_size=self.parallel_helper.sp_size,
                    )

                if (
                    ulysses_position_ids is not None
                    and not ulysses_position_ids.is_contiguous()
                ):
                    ulysses_position_ids = ulysses_position_ids.contiguous()

                inputs = ulysses_prepare_inputs(
                    padded_mb_input,
                    ulysses_input_ids,
                    ulysses_position_ids,
                    self.parallel_helper.sp_size,
                )
            else:
                inputs = padded_mb_input

            with trace_scope("fsdp_engine.train_batch.forward"):
                outputs = self.model(**inputs)

            logits = outputs.logits.squeeze(0)
            if self.parallel_helper.sp_size > 1:
                loss = loss_fn(logits, inputs)
            else:
                logits = logits[:-pad_length] if pad_length > 0 else logits
                loss = loss_fn(logits, mb_input)

            loss_scale = loss_weight_fn(mb_input) / total_loss_weight
            # Scale loss for accumulation
            # To reverse the gradient averaging for SP groups
            loss_scale *= self.parallel_helper.dp_size

            loss *= loss_scale
            with trace_scope("fsdp_engine.train_batch.backward"):
                loss.backward()

        grad_norm = fsdp2_clip_grad_norm(
            list(self.model.parameters()),
            self.world_mesh,
            max_norm=self.optimizer_config.gradient_clipping,
        )

        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad()
            update_successful = False
        else:
            with trace_scope("fsdp_engine.train_batch.step"):
                self.optimizer.step()
            update_successful = True

        current_lr = self.lr_scheduler.get_last_lr()[0]
        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    @trace_perf("fsdp_engine.eval_batch", category="compute")
    @torch.no_grad()
    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate on a batch."""
        if self.parallel_helper.sp_size > 1:
            set_ulysses_sequence_parallel_group(self.sp_group)

        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)

        total_loss_weight = (
            torch.stack([loss_weight_fn(mb) for mb in mb_list.mbs])
            .sum()
            .detach()
            .clone()
            .to(dtype=torch.float32)
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight, group=self.dp_group)

        total_loss = torch.zeros(1, device=self.device, dtype=torch.float32)

        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):
            if self.parallel_helper.sp_size > 1:
                input_ids = padded_mb_input["input_ids"]
                position_ids = padded_mb_input.get("position_ids", None)

                if self.is_vision_model:
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        _,
                    ) = ulysses_pad(
                        input_ids, position_ids, sp_size=self.parallel_helper.sp_size
                    )
                else:
                    # Pad and slice the inputs
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        _,
                    ) = ulysses_pad_and_slice_inputs(
                        input_ids,
                        position_ids,
                        sp_size=self.parallel_helper.sp_size,
                    )

                if (
                    ulysses_position_ids is not None
                    and not ulysses_position_ids.is_contiguous()
                ):
                    ulysses_position_ids = ulysses_position_ids.contiguous()

                inputs = ulysses_prepare_inputs(
                    padded_mb_input,
                    ulysses_input_ids,
                    ulysses_position_ids,
                    self.parallel_helper.sp_size,
                )
            else:
                inputs = padded_mb_input

            with trace_scope("fsdp_engine.eval_batch.forward"):
                outputs = self.model(**inputs)

            logits = outputs.logits.squeeze(0)
            if self.parallel_helper.sp_size > 1:
                loss = loss_fn(logits, inputs)
            else:
                logits = logits[:-pad_length] if pad_length > 0 else logits
                loss = loss_fn(logits, mb_input)

            loss_scale = loss_weight_fn(mb_input) / total_loss_weight

            # eval_batch does not run backward, the grad will not be averaged over DP group
            # so we shouldn't multiple dp_size in loss_scale
            total_loss += loss.clone().detach() * loss_scale

        dist.all_reduce(total_loss, group=self.dp_group)

        return total_loss

    @trace_perf("fsdp_engine.forward", category="compute")
    @torch.no_grad()
    def forward(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        post_hook: Callable[[torch.Tensor, dict[str, Any]], Any] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Forward pass with optional post-processing."""
        if self.parallel_helper.sp_size > 1:
            set_ulysses_sequence_parallel_group(self.sp_group)

        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None

        results = []

        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):
            ulysses_pad_size = 0
            if self.parallel_helper.sp_size > 1:
                input_ids = padded_mb_input["input_ids"]
                position_ids = padded_mb_input.get("position_ids", None)

                if self.is_vision_model:
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        ulysses_pad_size,
                    ) = ulysses_pad(
                        input_ids, position_ids, sp_size=self.parallel_helper.sp_size
                    )
                else:
                    # Pad and slice the inputs
                    (
                        ulysses_input_ids,
                        ulysses_position_ids,
                        ulysses_pad_size,
                    ) = ulysses_pad_and_slice_inputs(
                        input_ids,
                        position_ids,
                        sp_size=self.parallel_helper.sp_size,
                    )

                if (
                    ulysses_position_ids is not None
                    and not ulysses_position_ids.is_contiguous()
                ):
                    ulysses_position_ids = ulysses_position_ids.contiguous()

                inputs = ulysses_prepare_inputs(
                    padded_mb_input,
                    ulysses_input_ids,
                    ulysses_position_ids,
                    self.parallel_helper.sp_size,
                )
            else:
                inputs = padded_mb_input

            with trace_scope("fsdp_engine.forward.forward"):
                outputs = self.model(**inputs)

            logits = outputs.logits.squeeze(0)

            if post_hook:
                if self.parallel_helper.sp_size > 1:
                    # When Ulysses SP is enabled, post_hook will gather logits internally.
                    result = post_hook(logits, inputs)
                    # Remove Ulysses padding and original padding
                    result = (
                        result[:-ulysses_pad_size] if ulysses_pad_size > 0 else result
                    )
                    result = result[:-pad_length] if pad_length > 0 else result
                else:
                    # Remove original padding
                    logits = logits[:-pad_length] if pad_length > 0 else logits
                    result = post_hook(logits, mb_input)
                results.append(result)
            else:
                if self.parallel_helper.sp_size > 1:
                    # Gather and remove Ulysses padding
                    gathered_logits = dist_F.all_gather(logits, group=self.sp_group)
                    logits = torch.cat(gathered_logits, dim=0)
                    logits = (
                        logits[:-ulysses_pad_size] if ulysses_pad_size > 0 else logits
                    )
                # Remove original padding
                logits = logits[:-pad_length] if pad_length > 0 else logits
                results.append(logits)

        res = aggregate_fn(results)
        output_seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
        unpacked = unpack_sequence(res, lens=output_seqlens, dim=0)
        reordered = reorder_list(unpacked, mb_list.backward_indices)
        return pad_and_stack_tensors_along_first_dim(reordered)

    def create_optimizer(self, ft_spec: FinetuneSpec):
        if self.optimizer_config is None:
            return
        assert self.model is not None
        # Set up optimizer
        tik = time.perf_counter()
        assert self.optimizer_config.type in [
            "adam",
            "adam_bf16",
            "sgd",
        ], "Only adam/adam_bf16/sgd optimizer is supported in this engine."
        if self.optimizer_config.type in ["sgd", "adam_bf16"]:
            self.logger.warning(
                f"Using the '{self.optimizer_config.type}' optimizer with FSDP may be less stable. Consider using the 'adam' (AdamW) optimizer for improved stability and performance."
            )
        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.weight_decay
        beta1 = self.optimizer_config.beta1
        beta2 = self.optimizer_config.beta2
        eps = self.optimizer_config.eps
        if self.optimizer_config.type == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
                # VLM with tensor parallelism is incompatible with fused AdamW
                fused=not (self.is_vision_model and self.parallel_helper.tp_enabled),
            )
        elif self.optimizer_config.type == "adam_bf16":
            self.optimizer = AnyPrecisionAdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
                momentum_dtype="bfloat16",
                variance_dtype="bfloat16",
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
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
                f"Unknown lr scheduler type {self.optimizer_config.lr_scheduler_type}"
            )
        self.logger.info(f"Create optimizer time: {time.perf_counter() - tik}")

    def _get_optimizer_tensor_and_state_refs(
        self,
    ) -> tuple[list[torch.Tensor], list[tuple[dict, str]]]:
        """Get all tensors and state references from the optimizer."""
        tensors_to_move = []
        state_refs = []
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        tensors_to_move.append(value)
                        state_refs.append((state, key))
        return tensors_to_move, state_refs

    def _initialize_model_offload_buffer(self) -> None:
        """Initialize the model parameter offload buffer."""
        # Collect all parameters and buffers
        params_and_buffers = []
        for param in self.model.parameters():
            params_and_buffers.append(param)
        for buf in self.model.buffers():
            params_and_buffers.append(buf)

        if not params_and_buffers:
            return

        # Calculate total size for local shards only
        # For DTensor (FSDP2 sharded params), use local shard size
        total_numel = 0
        dtype = None
        for t in params_and_buffers:
            if isinstance(t.data, DTensor):
                # Get local shard size for FSDP2 sharded parameters
                local_tensor = t.data.to_local()
                total_numel += local_tensor.numel()
                if dtype is None:
                    dtype = local_tensor.dtype
            else:
                # Regular tensor (not sharded)
                total_numel += t.numel()
                if dtype is None:
                    dtype = t.dtype

        # Allocate CPU buffer for local shards only
        if self._model_offload_buffer is None:
            self._model_offload_buffer = torch.empty(
                total_numel,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self.logger.info(
                f"Allocated model offload buffer (local shards): {total_numel} elements, "
                f"{self._model_offload_buffer.element_size() * total_numel / 1024**2:.2f} MB"
            )

        # Calculate specs for each parameter/buffer's local shard
        # Store: (tensor, start_idx, end_idx, dtensor_meta)
        # dtensor_meta = None for regular tensors
        # dtensor_meta = (local_shape, device_mesh, placements) for DTensors
        self._model_offload_specs = []
        start_idx = 0
        for tensor in params_and_buffers:
            if isinstance(tensor.data, DTensor):
                # Store DTensor metadata for reconstruction
                local_tensor = tensor.data.to_local()
                local_numel = local_tensor.numel()
                end_idx = start_idx + local_numel
                dtensor_meta = {
                    "local_shape": local_tensor.shape,
                    "device_mesh": tensor.data.device_mesh,
                    "placements": tensor.data.placements,
                }
                self._model_offload_specs.append(
                    (tensor, start_idx, end_idx, dtensor_meta)
                )
                start_idx = end_idx
            else:
                # Regular tensor (not sharded)
                end_idx = start_idx + tensor.numel()
                self._model_offload_specs.append((tensor, start_idx, end_idx, None))
                start_idx = end_idx

    def _initialize_optimizer_offload_buffer(self) -> None:
        """Initialize the optimizer offload buffer."""
        tensors_to_move, state_refs = self._get_optimizer_tensor_and_state_refs()
        dtype = tensors_to_move[0].dtype

        total_numel = sum(t.numel() for t in tensors_to_move)

        # Allocate offload buffer
        if self._optimizer_offload_buffer is None:
            self._optimizer_offload_buffer = torch.empty(
                total_numel,
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self.logger.info(
                f"Allocated optimizer offload buffer: {total_numel} elements, "
                f"{self._optimizer_offload_buffer.element_size() * total_numel / 1024**2:.2f} MB"
            )

        # Calculate specs for each tensor
        self._optimizer_offload_specs = []
        start_idx = 0
        for tensor, (state, key) in zip(tensors_to_move, state_refs):
            end_idx = start_idx + tensor.numel()
            self._optimizer_offload_specs.append((state, key, start_idx, end_idx))
            start_idx = end_idx

    @torch.no_grad()
    def _move_model(self, device: str = "cpu") -> None:
        """Move model parameters and buffers to the specified device efficiently.

        Uses pre-allocated contiguous buffers for faster transfer.

        Parameters
        ----------
        device: str
            Target device, either "cpu" or "cuda" ("npu")
        """
        if device == "cpu":
            # Offload to CPU
            offload_stream = current_platform.Stream()
            offload_event = current_platform.Event()

            dummy_tensor = torch.tensor(
                (), device="cuda", dtype=self._model_offload_buffer.dtype
            )
            with current_platform.stream(offload_stream):
                # Copy all parameters and buffers to CPU buffer
                for spec in self._model_offload_specs:
                    tensor, start_idx, end_idx, dtensor_meta = spec
                    # Handle DTensor (FSDP2 sharded params) - only copy local shard

                    if dtensor_meta is not None:
                        local_tensor = tensor.data.to_local()
                    else:
                        local_tensor = tensor.data
                    self._model_offload_buffer[start_idx:end_idx].copy_(
                        local_tensor.view(-1), non_blocking=True
                    )
                    # Update tensor data to point to CPU buffer view
                    tensor.data = dummy_tensor

            offload_event.record(offload_stream)
            current_platform.current_stream().wait_event(offload_event)

            # Free GPU buffer
            self._model_onload_buffer = None

        elif (
            device == "cuda" or device == "npu"
        ) and self._model_offload_buffer is not None:
            # Onload to GPU
            total_numel = self._model_offload_buffer.numel()

            # Allocate GPU buffer
            assert self._model_onload_buffer is None
            self._model_onload_buffer = torch.empty(
                total_numel,
                dtype=self._model_offload_buffer.dtype,
                device=device,
            )
            self.logger.info(
                f"Allocated model onload buffer on {device}: {total_numel} elements, "
                f"{self._model_onload_buffer.element_size() * total_numel / 1024**2:.2f} MB"
            )

            restore_stream = current_platform.Stream()
            restore_event = current_platform.Event()

            with current_platform.stream(restore_stream):
                # Copy entire CPU buffer to GPU buffer
                self._model_onload_buffer.copy_(
                    self._model_offload_buffer,
                    non_blocking=True,
                )

                # Update tensor data to point to GPU buffer views
                for spec in self._model_offload_specs:
                    tensor, start_idx, end_idx, dtensor_meta = spec

                    if dtensor_meta is not None:
                        # Reconstruct DTensor from local shard
                        local_shape = dtensor_meta["local_shape"]
                        device_mesh = dtensor_meta["device_mesh"]
                        placements = dtensor_meta["placements"]

                        # Create local tensor from GPU buffer
                        local_tensor = self._model_onload_buffer[
                            start_idx:end_idx
                        ].view(local_shape)

                        # Reconstruct DTensor
                        tensor.data = DTensor.from_local(
                            local_tensor,
                            device_mesh=device_mesh,
                            placements=placements,
                        )
                    else:
                        # Regular tensor - use original shape
                        tensor.data = self._model_onload_buffer[start_idx:end_idx].view(
                            tensor.shape
                        )

            restore_event.record(restore_stream)
            current_platform.current_stream().wait_event(restore_event)

    @torch.no_grad()
    def _move_optimizer(self, device: str = "cpu") -> None:
        """Move optimizer state tensors to the specified device.

        Uses a pre-allocated contiguous CPU buffer for faster async offloading

        Parameters
        ----------
        device: str
            Target device, either "cpu" or "cuda" ("npu")
        """
        if not self.optimizer.state:
            return

        # Collect all tensors that need to be moved
        tensors_to_move, state_refs = self._get_optimizer_tensor_and_state_refs()

        if not tensors_to_move:
            return

        # Use CUDA/NPU stream for async offloading from CUDA/NPU to CPU
        if device == "cpu":
            offload_stream = current_platform.Stream()
            offload_event = current_platform.Event()

            with current_platform.stream(offload_stream):
                for tensor, (state, key, start_idx, end_idx) in zip(
                    tensors_to_move, self._optimizer_offload_specs
                ):
                    # Copy to the pre-allocated buffer
                    self._optimizer_offload_buffer[start_idx:end_idx].copy_(
                        tensor.view(-1), non_blocking=True
                    )
                    state[key] = self._optimizer_offload_buffer[start_idx:end_idx].view(
                        tensor.shape
                    )

            offload_event.record(offload_stream)
            current_platform.current_stream().wait_event(offload_event)

            # Free GPU buffer
            self._optimizer_onload_buffer = None

        elif (
            device == "cuda" or device == "npu"
        ) and self._optimizer_offload_buffer is not None:
            # Restore from CPU buffer to GPU buffer
            # Strategy: Allocate GPU buffer -> Copy entire CPU buffer to GPU -> Update state pointers
            total_numel = self._optimizer_offload_buffer.numel()

            # Allocate GPU buffer
            assert self._optimizer_onload_buffer is None
            self._optimizer_onload_buffer = torch.empty(
                total_numel,
                dtype=self._optimizer_offload_buffer.dtype,
                device=device,
            )
            self.logger.info(
                f"Allocated optimizer restore buffer on {device}: {total_numel} elements, "
                f"{self._optimizer_onload_buffer.element_size() * total_numel / 1024**2:.2f} MB"
            )

            restore_stream = current_platform.Stream()
            restore_event = current_platform.Event()

            with current_platform.stream(restore_stream):
                # Copy entire CPU buffer to GPU buffer
                self._optimizer_onload_buffer.copy_(
                    self._optimizer_offload_buffer,
                    non_blocking=True,
                )

                # Update state pointers to point to GPU buffer views
                for state, key, start_idx, end_idx in self._optimizer_offload_specs:
                    tensor = state[key]
                    state[key] = self._optimizer_onload_buffer[start_idx:end_idx].view(
                        tensor.shape
                    )

            restore_event.record(restore_stream)
            current_platform.current_stream().wait_event(restore_event)

        else:
            for tensor, (state, key) in zip(tensors_to_move, state_refs):
                state[key] = tensor.to(device, non_blocking=True)

    def sleep(self) -> None:
        """Pause CUDA memory for all tracked tensors.
        Ref https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """
        assert self.config.offload_train

        print_memory("before offload model")

        match self.config.offload_train_mode:
            case "tms":
                # Use torch_memory_saver to pause CUDA memory
                clear_memory()
                torch_memory_saver.pause()
            case "move":
                self._move_model("cpu")
                self._move_optimizer("cpu")
                clear_memory()
            case _:
                raise NotImplementedError(
                    f"Unsupported offload_train_mode: {self.config.offload_train_mode}"
                )

        current_platform.synchronize()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    def wake_up(self) -> None:
        """Resume CUDA memory for all tracked tensors
        Ref https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """
        assert self.config.offload_train

        match self.config.offload_train_mode:
            case "tms":
                torch_memory_saver.resume()
            case "move":
                # TODO: support NPU
                self._move_model("cuda")
                self._move_optimizer("cuda")
            case _:
                raise NotImplementedError(
                    f"Unsupported offload_train_mode: {self.config.offload_train_mode}"
                )

        current_platform.synchronize()
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")
