import dataclasses
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
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
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
from areal.api.cli_args import PerfTracerConfig, TrainEngineConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import (
    DeviceRuntimeInfo,
    FinetuneSpec,
    ParamSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Scheduler
from areal.api.workflow_api import RolloutWorkflow
from areal.core.dist_rollout import DistRolloutCoordinator
from areal.engine.core.train_engine import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.experimental.models.archon import (
    BaseStateDictAdapter,
    ModelSpec,
    get_model_spec,
    get_supported_model_types,
    is_supported_model,
)
from areal.experimental.models.archon.activation_checkpoint import (
    ActivationCheckpointConfig,
)
from areal.experimental.utils.archon.parallel import ArchonParallelDims
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names, perf_tracer, stats_tracker
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.data import (
    MicroBatchItem,
    MicroBatchList,
    amend_position_ids,
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_mb_list,
)
from areal.utils.distributed import init_custom_process_group, patch_dist_group_timeout
from areal.utils.fsdp import fsdp2_load_full_state_dict, get_cosine_schedule_with_warmup
from areal.utils.fsdp.checkpoint import DCPState
from areal.utils.fsdp.grad import fsdp2_clip_grad_norm
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.network import find_free_ports, gethostip
from areal.utils.offload import torch_memory_saver
from areal.utils.perf_tracer import trace_perf
from areal.utils.save_load import get_state_dict_from_repo_id_or_path


@dataclass
class ArchonTrainContext:
    """Context passed through Archon forward/backward pipeline."""

    model_inputs: dict[str, Any]
    mb_input: dict[str, Any]
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
        self.parallel_dims = ArchonParallelDims(
            dp_shard=dp_size,
            tp=tp_size,
            world_size=self.world_size,
            device_type=current_platform.device_type,
        )

        self._world_mesh = self.parallel_dims.world_mesh
        self._mp_group = self.parallel_dims.tp_group

        self.weight_update_group_name = "update_weight_group"

        self.logger.info(
            f"Initialized Archon engine with parallel dims: "
            f"dp_shard={self.parallel_dims.dp_shard}, tp={self.parallel_dims.tp}"
        )

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        """Initialize model, optimizer, and apply parallelism."""
        assert addr is None, "ArchonEngine does not support remote initialization."
        assert ft_spec is not None, "ArchonEngine requires FinetuneSpec to initialize."

        self._create_device_model()
        self.state_dict_adapter = self._create_state_dict_adapter()

        tik = time.perf_counter()
        param_dtype = getattr(torch, self.config.dtype)
        tp_mesh = self.world_mesh["tp"] if self.parallel_dims.tp_enabled else None
        dp_mesh = self.world_mesh["fsdp"]
        ac_config = self._build_ac_config()

        self.spec.parallelize_fn(
            model=self.model,
            tp_mesh=tp_mesh,
            dp_mesh=dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            loss_parallel=True,
            cpu_offload=self.config.archon.offload_params,
            reshard_after_forward=True,
            ac_config=ac_config,
            enable_compile=self.config.archon.enable_compile,
        )

        # Synchronize all ranks after parallelization (especially after torch.compile)
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

        self.logger.info(
            f"Applied parallelism in {time.perf_counter() - tik:.2f} seconds"
        )

        self._create_optimizer(ft_spec)

        self._initialized = True

    @property
    def world_mesh(self) -> DeviceMesh:
        return self._world_mesh

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        return self.world_mesh["fsdp"].get_group()

    @property
    def data_parallel_rank(self) -> int:
        return dist.get_rank(self.data_parallel_group)

    @property
    def data_parallel_world_size(self) -> int:
        return self.parallel_dims.dp_shard

    def current_data_parallel_head(self) -> int:
        tp_rank = dist.get_rank(self._mp_group)
        return self.rank - tp_rank

    def is_data_parallel_head(self) -> bool:
        return dist.get_rank(self._mp_group) == 0

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        return self._mp_group

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
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        current_platform.empty_cache()
        gc.collect()

        if dist.is_initialized() and self.own_global_group:
            dist.destroy_process_group()
        self._initialized = False

    def train(self, mode: bool = True):
        assert self.model is not None
        self.model.train(mode=mode)
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

        tp_group = (
            self.world_mesh["tp"].get_group() if self.parallel_dims.tp_enabled else None
        )
        grad_norm = fsdp2_clip_grad_norm(
            list(self.model.parameters()),
            max_norm=self.optimizer_config.gradient_clipping,
            fsdp_group=self.data_parallel_group,
            tp_group=tp_group,
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
    ) -> None:
        """Forward and optionally backward through micro-batches."""
        for mb_item in mb_list:
            inputs, ctx = self._prepare_mb_inputs(mb_item)

            cu_seqlens = inputs.get("cu_seqlens")
            max_seqlen = (
                int(inputs.get("max_seqlen", 0)) if cu_seqlens is not None else None
            )

            logits = self.model(
                inputs["input_ids"],
                inputs.get("position_ids"),
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            logits = logits.squeeze(0)

            ctx_dict = dataclasses.asdict(ctx)
            loss = process_output_fn(logits, ctx_dict)

            if not forward_only and loss is not None:
                loss.backward()

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

        losses: list[torch.Tensor] = []

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = ArchonTrainContext(**ctx_dict)
            loss = self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
            )
            losses.append(loss.detach())
            return loss

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        return aggregate_eval_losses(losses, self.data_parallel_group)

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

        outputs: list[torch.Tensor] = []

        def process_output(logits: torch.Tensor, ctx_dict: dict[str, Any]) -> None:
            ctx = ArchonTrainContext(**ctx_dict)
            result = self._compute_forward_result(logits, ctx)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        return reorder_and_pad_outputs(outputs, output_seqlens, mb_list, aggregate_fn)

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
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
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
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
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
            self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
        elif meta.weight_format == "dcp":
            self._save_to_dcp(meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}.")

        if meta.with_optim and meta.weight_format == "hf":
            self._save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        """Load model from HuggingFace or DCP format."""
        if meta.weight_format == "hf":
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            self._load_from_dcp(meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}.")

        if meta.with_optim and meta.weight_format == "hf":
            self._load_optimizer_state(meta.path)

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
        return stats_tracker.export_all(reduce_group=self.data_parallel_group)

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
        return self.spec.state_dict_adapter_class(self.model_config)

    def _get_model_name_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        return self.model.named_parameters()

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
        self._save_model_to_hf(meta.path, self.tokenizer, None)

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

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast | None,
        processor=None,
    ):
        """Save model in HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        os.makedirs(path, exist_ok=True)

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(self.model, options=options)
        if self.state_dict_adapter is not None:
            state_dict = self.state_dict_adapter.to_hf(state_dict)

        if dist.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            model_path = os.path.join(path, "model.safetensors")
            try:
                save_file(state_dict, model_path)
            except ImportError:
                model_path = os.path.join(path, "pytorch_model.bin")
                torch.save(state_dict, model_path)

            self.model_config.save_pretrained(path)
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if processor is not None:
                processor.save_pretrained(path)
        dist.barrier(group=self.cpu_group)

    def _load_model_from_hf(self, path: str):
        """Load model from HuggingFace format."""
        if dist.get_rank() == 0:
            full_state = get_state_dict_from_repo_id_or_path(path)
            if self.state_dict_adapter is not None:
                full_state = self.state_dict_adapter.from_hf(full_state)
        else:
            full_state = {}

        cpu_offload = self.config.archon.offload_params
        fsdp2_load_full_state_dict(
            self.model,
            full_state,
            cpu_offload,
            tie_word_embeddings=self.model_config.tie_word_embeddings,
        )

    def _save_to_dcp(self, path: str, with_optim: bool):
        if self.model is None:
            raise RuntimeError("Model not initialized")

        os.makedirs(path, exist_ok=True)

        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.save(state_dict, checkpoint_id=path)

    def _load_from_dcp(self, path: str, with_optim: bool):
        if self.model is None:
            raise RuntimeError("Model not initialized")

        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.load(state_dict=state_dict, checkpoint_id=path)

    def _save_optimizer_state(self, path: str):
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        state_dict = self.optimizer.state_dict()
        torch.save(state_dict, shard_path)
        dist.barrier(group=self.cpu_group)

    def _load_optimizer_state(self, path: str):
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        optimizer_state_dict = torch.load(shard_path, weights_only=False)
        self.optimizer.load_state_dict(optimizer_state_dict)
        dist.barrier(group=self.cpu_group)

    def _create_device_model(self):
        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        self.tokenizer = load_hf_tokenizer(self.config.path)

        tik = time.perf_counter()
        with torch.device(current_platform.device_type):
            model = self._create_llm_model()

        self.logger.info(f"Model creation time: {time.perf_counter() - tik:.2f}s")
        self.model = model

    def _build_ac_config(self) -> ActivationCheckpointConfig | None:
        # First check if gradient checkpointing is enabled
        if not self.config.gradient_checkpointing:
            return None

        archon_config = self.config.archon
        granularity = archon_config.recompute_granularity

        if granularity == "none":
            return None

        if granularity == "selective" and archon_config.recompute_num_layers == 0:
            selective_option = "op"
        else:
            selective_option = str(archon_config.recompute_num_layers)

        ac_config = ActivationCheckpointConfig(
            mode=granularity,
            selective_ac_option=selective_option,
            preserve_rng_state=False,
        )

        self.logger.info(
            f"Activation checkpointing enabled: mode={granularity}, "
            f"selective_option={selective_option}"
        )

        return ac_config

    def _create_llm_model(self) -> nn.Module:
        model_args = self.spec.model_args_class.from_hf_config(
            self.model_config,
            is_critic=self.config.is_critic,
            attn_type=self.config.archon.attn_type,
        )
        model = self.spec.model_class(model_args)

        if not self.config.init_from_scratch:
            self._load_hf_weights_to_archon_model(model)
        else:
            model.init_weights()

        dtype = getattr(torch, self.config.dtype)
        model = model.to(dtype)

        return model

    def _load_hf_weights_to_archon_model(self, model: nn.Module):
        self.logger.info(f"Loading weights from {self.config.path}")

        hf_state_dict = get_state_dict_from_repo_id_or_path(self.config.path)

        adapter = self.spec.state_dict_adapter_class(self.model_config)

        archon_state_dict = adapter.from_hf(hf_state_dict)

        missing_keys, unexpected_keys = model.load_state_dict(
            archon_state_dict, strict=False
        )

        if missing_keys:
            expected_missing = {"rope_cache"}
            if not self.config.is_critic:
                expected_missing.add("score.weight")
            else:
                expected_missing.add("output.weight")

            actual_missing = [k for k in missing_keys if k not in expected_missing]
            if actual_missing:
                self.logger.warning(
                    f"Missing keys when loading weights: {actual_missing}"
                )

        if unexpected_keys:
            self.logger.warning(
                f"Unexpected keys when loading weights: {unexpected_keys}"
            )

        self.logger.info("Successfully loaded weights into Archon model")

    def _create_optimizer(self, ft_spec: FinetuneSpec):
        if self.optimizer_config is None:
            return

        assert self.model is not None

        tik = time.perf_counter()

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
                fused=True,
            )
        elif self.optimizer_config.type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
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

        mb_list = split_padded_tensor_dict_into_mb_list(input_, self.config.mb_spec)
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
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
        inputs = mb_item.padded_mb
        ctx = ArchonTrainContext(
            model_inputs=inputs,
            mb_input=mb_item.orig_mb,
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
            logprobs, entropy = self._compute_logprobs_entropy(logits, ctx.model_inputs)
            if ctx.pad_length > 0:
                logprobs = logprobs[: -ctx.pad_length]
                entropy = entropy[: -ctx.pad_length]
            loss = loss_fn(logprobs, entropy, ctx.mb_input)
        else:
            values = logits.squeeze(-1)
            if ctx.pad_length > 0:
                values = values[: -ctx.pad_length]
            loss = loss_fn(values, ctx.mb_input)

        loss_scale = loss_weight_fn(ctx.mb_input) / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_logprobs_entropy(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy from logits."""
        labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        tp_group = (
            self.world_mesh["tp"].get_group() if self.parallel_dims.tp_enabled else None
        )
        logprobs, entropy = gather_logprobs_entropy(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=tp_group,
        )
        return logprobs, entropy

    def _compute_forward_result(
        self,
        logits: torch.Tensor,
        ctx: ArchonTrainContext,
    ) -> torch.Tensor:
        """Compute forward output (logprobs or values)."""
        if not self.config.is_critic:
            result = self._compute_logprobs(logits, ctx.model_inputs)
        else:
            result = logits.squeeze(-1)
        if ctx.pad_length > 0:
            result = result[: -ctx.pad_length]
        return result

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        """Compute log probabilities from logits (without entropy)."""
        labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        tp_group = (
            self.world_mesh["tp"].get_group() if self.parallel_dims.tp_enabled else None
        )
        logprobs = gather_logprobs(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=tp_group,
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
