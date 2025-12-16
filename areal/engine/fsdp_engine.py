import dataclasses
import gc
import math
import os
import time
from collections.abc import Callable, Iterator
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
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    ProcessorMixin,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from areal.api.alloc_mode import FSDPParallelStrategy, ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import FinetuneSpec, ParamSpec, SaveLoadMeta, WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow
from areal.core.dist_rollout import DistRolloutCoordinator
from areal.engine.core import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.models.transformers.ulyssess_patch import apply_monkey_patch
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names, pkg_version, stats_tracker
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
from areal.utils.device import clear_memory, log_gpu_stats
from areal.utils.distributed import init_custom_process_group, patch_dist_group_timeout
from areal.utils.fsdp import fsdp2_load_full_state_dict, get_cosine_schedule_with_warmup
from areal.utils.fsdp.checkpoint import DCPState
from areal.utils.fsdp.grad import fsdp2_clip_grad_norm
from areal.utils.fsdp.optimizer import AnyPrecisionAdamW
from areal.utils.fsdp.parallel import ParallelHelper, parallelize_model
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_processor_and_tokenizer, load_hf_tokenizer
from areal.utils.model import (
    disable_dropout_in_model,
    is_gemma3_model,
    is_qwen3_moe_model,
    is_qwen3_vl_model,
    is_qwen_vl_model,
    is_valid_vision_model,
)
from areal.utils.offload import is_tms_enabled
from areal.utils.perf_tracer import trace_perf, trace_scope
from areal.utils.save_load import get_state_dict_from_repo_id_or_path
from areal.utils.ulysses import (
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
    ulysses_prepare_inputs,
)


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


class FSDPEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.optimizer_config = config.optimizer

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.tokenizer: PreTrainedTokenizerFast
        self.processor: ProcessorMixin | None = None
        self.model_config: PretrainedConfig
        self._version: int = 0

        self.initialized = False
        self.own_global_group = False
        self._cpu_group: dist.ProcessGroup
        self.weight_update_group_initialized = False

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.is_vision_model = is_valid_vision_model(self.model_config.model_type)
        self.world_size = int(os.environ["WORLD_SIZE"])

        # FSDP-specific initialization
        self.cpu_offload: CPUOffloadPolicy | None = None

        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None

        self.parallel_helper: ParallelHelper
        self.world_mesh: DeviceMesh

        self.dp_group: dist.ProcessGroup
        self.sp_group: dist.ProcessGroup
        self.mp_group: dist.ProcessGroup

        self.rank: int
        self.dp_head: int
        self.dp_rank: int

        self.is_offload: bool = False

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
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

        # FSDP-specific process group setup
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

        self.dp_head = dist.get_process_group_ranks(self.mp_group)[0]
        self.dp_rank = dist.get_rank(self.dp_group)

        self.logger.info(f"Data parallel head {self.dp_head} and rank {self.dp_rank}")

    def initialize(
        self,
        addr: str | None,
        ft_spec: FinetuneSpec,
    ):
        # Initialize distributed enviroments and load model.
        assert addr is None, "FSDPEngine does not support remote initialization."
        assert ft_spec is not None, "FSDPEngine requires FinetuneSpec to initialize."
        if pkg_version.is_version_less("torch", "2.4.0"):
            raise RuntimeError("areal only supports FSDP2, which requires torch>=2.4.0")

        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"

        # Create device model
        self._create_device_model()

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

        self._create_optimizer(ft_spec)
        self.initialized = True

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        return self.dp_group

    @property
    def data_parallel_rank(self) -> int:
        return self.dp_rank

    @property
    def data_parallel_world_size(self) -> int:
        return self.parallel_helper.dp_size

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        return self.mp_group

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        assert self.initialized
        return self._cpu_group

    def destroy(self):
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        current_platform.empty_cache()
        gc.collect()
        # NOTE: if `own_global_group` is true, we assume that
        # no communications are needed after `destroy`, so we
        # directly destroy all groups. Otherwise, process group
        # handles still exist and we expect another engine to
        # clean up these groups.
        if dist.is_initialized() and self.own_global_group:
            dist.destroy_process_group()
        self.initialized = False

    def current_data_parallel_head(self) -> int:
        return self.dp_head

    def is_data_parallel_head(self) -> bool:
        return self.rank == self.dp_head

    def train(self, mode: bool = True):
        assert self.model is not None
        self.model.train(mode=mode)
        return self

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

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        granularity: int = 1,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

    def update_weights(self, meta: WeightUpdateMeta):
        self._check_rollout_engine_connected()
        if meta.type == current_platform.communication_backend:
            assert self.weight_update_group_initialized
            # In offload mode, wakes up parameters as needed to perform the update.
            tms_context = (
                torch_memory_saver.disable()
                if self.is_offload and not torch.version.hip
                else nullcontext()
            )
            with tms_context:
                self._update_weights_from_distributed(meta)
        elif meta.type == "disk":
            self._update_weights_from_disk(meta)
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
        elif meta.weight_format == "dcp":
            self._save_to_dcp(meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim and meta.weight_format == "hf":
            self._save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            self._load_from_dcp(meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim and meta.weight_format == "hf":
            self._load_optimizer_state(meta.path)

    def optimizer_zero_grad(self):
        assert self.optimizer is not None
        self.optimizer.zero_grad()

    def optimizer_step(self):
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        grad_norm = fsdp2_clip_grad_norm(
            list(self.model.parameters()),
            self.world_mesh,
            max_norm=self.optimizer_config.gradient_clipping,
            offload_params=self.config.fsdp.offload_params,
        )

        if not math.isfinite(grad_norm):
            self.optimizer_zero_grad()
            update_successful = False
        else:
            with trace_scope("fsdp_engine.step"):
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
        for mb_item in mb_list:
            inputs, ctx = self._prepare_mb_inputs(mb_item)

            with trace_scope("fsdp_engine.forward"):
                outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)

            ctx_dict = dataclasses.asdict(ctx)
            loss = process_output_fn(logits, ctx_dict)

            if not forward_only and loss is not None:
                with trace_scope("fsdp_engine.backward"):
                    loss.backward()

    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        self._ensure_ready()
        self.optimizer_zero_grad()

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.dp_group
        )

        # Step 3: Forward-backward using process_output_fn callback
        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = FSDPTrainContext(**ctx_dict)
            return self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier=self.parallel_helper.dp_size,
            )

        self.forward_backward_batch(mb_list, process_output, forward_only=False)

        # Step 4: Optimizer step
        return self.optimizer_step()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        self._ensure_ready()

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.dp_group
        )

        # Step 3: Forward using process_output_fn callback, collecting losses
        losses: list[torch.Tensor] = []

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = FSDPTrainContext(**ctx_dict)
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

        # Step 4: Aggregate losses
        return aggregate_eval_losses(losses, self.dp_group)

    @torch.no_grad()
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> torch.Tensor:
        self._ensure_ready()

        # Step 1: Prepare sequence lengths
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()

        # Step 2: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 3: Forward using process_output_fn callback, collecting results
        outputs: list[torch.Tensor] = []

        def process_output(logits: torch.Tensor, ctx_dict: dict[str, Any]) -> None:
            ctx = FSDPTrainContext(**ctx_dict)
            result = self._compute_forward_result(logits, ctx)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate and reorder outputs
        return reorder_and_pad_outputs(outputs, output_seqlens, mb_list, aggregate_fn)

    def export_stats(self) -> dict[str, float]:
        return stats_tracker.export_all(reduce_group=self.data_parallel_group)

    def offload(self) -> None:
        """Offload model memory to CPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """

        log_gpu_stats("before offload model")

        # Use torch_memory_saver to pause CUDA memory
        clear_memory()
        torch_memory_saver.pause()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        log_gpu_stats("after offload model")

        self.is_offload = True

    def onload(self) -> None:
        """Onload model memory from CPU back to GPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """

        torch_memory_saver.resume()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        log_gpu_stats("after onload model")

        self.is_offload = False

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> FSDPParallelStrategy:
        return FSDPParallelStrategy(
            **dataclasses.asdict(parallel_strategy),
        )

    def _create_llm_actor_or_critic(self):
        dtype = getattr(torch, self.config.dtype)

        if self.config.is_critic:
            model_class = AutoModelForTokenClassification
            model_kwargs = {"num_labels": 1}
        else:
            model_class = AutoModelForCausalLM
            model_kwargs = {}

        common_kwargs = {
            "dtype": dtype,
            "attn_implementation": self.config.attn_impl,
        }
        model_kwargs.update(common_kwargs)

        if self.config.init_from_scratch:
            # initialize model from config
            # NOTE: VLM cannot directly load state dict using this random initialized model
            model = model_class.from_config(
                self.model_config,
                **model_kwargs,
            )
        else:
            model = model_class.from_pretrained(
                pretrained_model_name_or_path=self.config.path,
                trust_remote_code=True,
                **model_kwargs,
            )
        return model

    def _create_device_model(self):
        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        dtype = getattr(torch, self.config.dtype)

        if self.is_vision_model:
            if dtype == torch.float16:
                raise ValueError(
                    "Vision models do not support float16 dtype. Please use bfloat16."
                )
            if self.config.init_from_scratch:
                raise ValueError(
                    "Vision models do not support initialization from scratch. Please use a pretrained model."
                )
            self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
                self.config.path
            )

            tik = time.perf_counter()
            device = current_platform.device_type
            with torch.device(device):
                model = AutoModelForImageTextToText.from_pretrained(
                    pretrained_model_name_or_path=self.config.path,
                    trust_remote_code=True,
                    dtype=dtype,
                    attn_implementation=self.config.attn_impl,
                )
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)
        else:
            self.tokenizer = load_hf_tokenizer(self.config.path)
            self.processor = None
            tik = time.perf_counter()
            with torch.device(current_platform.device_type):
                model = self._create_llm_actor_or_critic()
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.logger.info(
            f"Model creation and loading time: {time.perf_counter() - tik}"
        )
        self.model = model

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

    def _create_optimizer(self, ft_spec: FinetuneSpec) -> None:
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

    def _check_rollout_engine_connected(self) -> None:
        """Validate that rollout engine has been connected via connect_engine()."""
        if self.rollout_engine is None or self.rollout_coordinator is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    def _ensure_ready(self) -> None:
        if self.is_offload:
            self.onload()

        if self.parallel_helper.sp_size > 1:
            set_ulysses_sequence_parallel_group(self.sp_group)

    def _get_model_name_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        name_params_iterator = self.model.named_parameters()
        if self.is_vision_model and is_qwen_vl_model(self.model_config.model_type):
            for name, value in name_params_iterator:
                new_name = name.replace("model.", "", 1).replace(
                    "language_model", "model"
                )
                yield new_name, value
        elif self.is_vision_model and is_gemma3_model(self.model_config.model_type):
            for name, value in name_params_iterator:
                new_name = name.replace("model.", "", 1)
                if new_name.startswith("language_model."):
                    new_name = new_name.replace(
                        "language_model.", "language_model.model.", 1
                    )
                elif new_name.startswith("lm_head."):
                    new_name = new_name.replace(
                        "lm_head.", "language_model.lm_head.", 1
                    )
                yield new_name, value
        else:
            yield from name_params_iterator

    def _get_full_tensor(self, param: nn.Parameter) -> torch.Tensor:
        """Get full tensor from a parameter, handling DTensor and CPU offloaded tensors."""
        tensor = param.data
        if isinstance(tensor, DTensor):
            # For non-offloaded DTensor, directly call full_tensor()
            if tensor.device.type != "cpu":
                return tensor.full_tensor()

            # Handle CPU offloaded DTensor: reconstruct DTensor from local tensor
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

        if self.config.use_lora:
            if not self.config.target_modules or self.config.target_modules == [
                "all-linear"
            ]:
                target_modules = "all-linear"
            else:
                target_modules = self.config.target_modules

            meta.peft_config = {
                "r": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": target_modules,
                "bias": "none",
            }

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
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )

            fut.result()

    @trace_perf("fsdp_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        """Broadcast parameters (chunked) from rank 0 (FSDP2 compatible)."""

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
        main_rank = dist.get_rank() == 0

        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []

        if self.config.use_lora:
            # For LoRA, only iterate over trainable LoRA parameters
            param_iterator = (
                (name, param)
                for name, param in self.model.named_parameters()
                if param.requires_grad
            )
        else:
            # For full model, iterate over all parameters
            param_iterator = self.get_model_name_parameters()

        for name, param in param_iterator:
            tensor = self._get_full_tensor(param)

            # Ranks other than 0 only help to get the full tensor
            if not main_rank:
                continue

            tensor_size = tensor.numel() * tensor.element_size()

            if tensor_size + buffer_size > weight_chunked_mem_size:
                self._update_bucket_weights_from_distributed(meta, named_tensors)
                buffer_size = 0

            named_tensors.append((name, tensor))
            buffer_size += tensor_size

        # Process remaining parameters
        if named_tensors:
            self._update_bucket_weights_from_distributed(meta, named_tensors)

        dist.barrier(group=self.cpu_group)

        if dist.get_rank() == 0:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    @trace_perf("fsdp_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self, meta: WeightUpdateMeta):
        fut = Future()

        if dist.get_rank() == 0:
            fut = self.rollout_engine.update_weights_from_disk(meta)

        assert meta.path is not None
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

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

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
        dist.barrier(group=self.cpu_group)

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

    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        input_ = input_.copy()

        if is_qwen_vl_model(self.model_config.model_type):
            attn_mask = input_["attention_mask"]
            input_ids = input_["input_ids"]
            image_grid_thw = None
            video_grid_thw = None
            if "multi_modal_input" in input_:
                multi_modal_input = input_["multi_modal_input"]
                image_grid_thw_list = [
                    m["image_grid_thw"]
                    for m in multi_modal_input
                    if "image_grid_thw" in m
                ]
                if image_grid_thw_list:
                    image_grid_thw = torch.cat(image_grid_thw_list)
                video_grid_thw_list = [
                    m["video_grid_thw"]
                    for m in multi_modal_input
                    if "video_grid_thw" in m
                ]
                if video_grid_thw_list:
                    video_grid_thw = torch.cat(video_grid_thw_list)

            position_ids, _ = self.model.model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attn_mask
            )
            position_ids = torch.einsum("ijk->jki", position_ids)
            input_["position_ids"] = position_ids
        else:
            input_ = amend_position_ids(input_)

        mb_list = split_padded_tensor_dict_into_mb_list(input_, self.config.mb_spec)
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
        )
        self.logger.info(
            f"Microbatch #tokens (rank {dist.get_rank()}): {mb_list.group_lens}, "
            f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}"
        )
        mb_list = unsqueeze_mb_list(mb_list)
        if is_qwen_vl_model(self.model_config.model_type):
            assert mb_list.padded_mbs is not None
            for mb in mb_list.padded_mbs:
                mb["position_ids"] = torch.einsum("ijk->kij", mb["position_ids"])

        assert mb_list.padded_mbs is not None
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)
        for mb, padded_mb in zip(mb_list.mbs, mb_list.padded_mbs):
            mb["max_length_q"] = mb["max_length_k"] = mb["max_seqlen"] = int(
                mb["max_seqlen"]
            )
            padded_mb["max_length_q"] = padded_mb["max_length_k"] = padded_mb[
                "max_seqlen"
            ] = int(padded_mb["max_seqlen"])
            mb["cu_seq_lens_q"] = mb["cu_seq_lens_k"] = mb["cu_seqlens"]
            padded_mb["cu_seq_lens_q"] = padded_mb["cu_seq_lens_k"] = padded_mb[
                "cu_seqlens"
            ]
            mb["use_cache"] = False
            padded_mb["use_cache"] = False
            if is_qwen3_moe_model(self.model_config.model_type) or is_qwen3_vl_model(
                self.model_config.model_type
            ):
                mb["attention_mask"] = None
                padded_mb["attention_mask"] = None
            else:
                mb["attention_mask"] = dict(full_attention=None, sliding_attention=None)
                padded_mb["attention_mask"] = dict(
                    full_attention=None, sliding_attention=None
                )
            if "multi_modal_input" in mb:
                image_grid_thw_list = [
                    item["image_grid_thw"]
                    for item in mb["multi_modal_input"]
                    if "image_grid_thw" in item
                ]
                if image_grid_thw_list:
                    mb["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
                    padded_mb["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
                pixel_values_list = [
                    item["pixel_values"]
                    for item in mb["multi_modal_input"]
                    if "pixel_values" in item
                ]
                if pixel_values_list:
                    mb["pixel_values"] = torch.cat(pixel_values_list, dim=0)
                    padded_mb["pixel_values"] = torch.cat(pixel_values_list, dim=0)
                video_grid_thw_list = [
                    item["video_grid_thw"]
                    for item in mb["multi_modal_input"]
                    if "video_grid_thw" in item
                ]
                if video_grid_thw_list:
                    mb["video_grid_thw"] = torch.cat(video_grid_thw_list, dim=0)
                    padded_mb["video_grid_thw"] = torch.cat(video_grid_thw_list, dim=0)
        return mb_list

    def _prepare_mb_inputs(
        self, mb_item: MicroBatchItem
    ) -> tuple[dict[str, Any], FSDPTrainContext]:
        """Prepare micro-batch inputs with Ulysses sequence parallel handling.

        This method handles Ulysses SP padding and slicing, returning both
        the prepared model inputs and a context object for later processing.
        """
        if self.parallel_helper.sp_size > 1:
            input_ids = mb_item.padded_mb["input_ids"]
            position_ids = mb_item.padded_mb.get("position_ids", None)

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
                mb_item.padded_mb,
                ulysses_input_ids,
                ulysses_position_ids,
                self.parallel_helper.sp_size,
            )
        else:
            inputs = mb_item.padded_mb
            ulysses_pad_size = 0

        ctx = FSDPTrainContext(
            model_inputs=inputs,
            mb_input=mb_item.orig_mb,
            pad_length=mb_item.padding_length,
            ulysses_pad_size=ulysses_pad_size,
        )
        return inputs, ctx

    def _sp_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        gathered = dist_F.all_gather(tensor, group=self.sp_group)
        return torch.cat(gathered, dim=-1)

    def _get_vocab_min_max_logits(
        self,
        logits: torch.Tensor,
        ulysses_pad_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_min_logits = logits.detach().min(-1).values.float()
        vocab_max_logits = logits.detach().max(-1).values.float()
        if self.parallel_helper.sp_size > 1:
            vocab_min_logits = self._sp_all_gather(vocab_min_logits)
            vocab_max_logits = self._sp_all_gather(vocab_max_logits)
            if ulysses_pad_size > 0:
                vocab_min_logits = vocab_min_logits[:-ulysses_pad_size]
                vocab_max_logits = vocab_max_logits[:-ulysses_pad_size]
        return vocab_min_logits, vocab_max_logits

    def _compute_logprobs_entropy(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Try to get rolled_input_ids (if Ulysses SP is enabled)
        labels = inputs.get(
            "rolled_input_ids",
            torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
        )
        # inputs (padded_mbs) has batch dim (1, seq_len), squeeze to match logits (seq_len,)
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        logprobs, entropy = gather_logprobs_entropy(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=self.parallel_helper.tp_group
            if self.parallel_helper.tp_size > 1
            else None,
        )
        if self.parallel_helper.sp_size > 1:
            logprobs = self._sp_all_gather(logprobs)
            entropy = self._sp_all_gather(entropy)
            if ulysses_pad_size > 0:
                logprobs = logprobs[:-ulysses_pad_size]
                entropy = entropy[:-ulysses_pad_size]
        return logprobs, entropy

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
    ) -> torch.Tensor:
        # Try to get rolled_input_ids (if Ulysses SP is enabled)
        labels = inputs.get(
            "rolled_input_ids",
            torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
        )
        # inputs (padded_mbs) has batch dim (1, seq_len), squeeze to match logits (seq_len,)
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        logprobs = gather_logprobs(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=self.parallel_helper.tp_group
            if self.parallel_helper.tp_size > 1
            else None,
        )
        if self.parallel_helper.sp_size > 1:
            logprobs = self._sp_all_gather(logprobs)
            if ulysses_pad_size > 0:
                logprobs = logprobs[:-ulysses_pad_size]
        return logprobs

    def _compute_values(
        self,
        values: torch.Tensor,
        ulysses_pad_size: int = 0,
    ) -> torch.Tensor:
        if self.parallel_helper.sp_size > 1:
            values = self._sp_all_gather(values)
            if ulysses_pad_size > 0:
                values = values[:-ulysses_pad_size]
        return values

    def _compute_logprobs_and_loss(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Compute logprobs/entropy and return scaled loss."""
        if not self.config.is_critic:
            logprobs, entropy = self._compute_logprobs_entropy(
                logits, ctx.model_inputs, ctx.ulysses_pad_size
            )
            vocab_min_logits, vocab_max_logits = self._get_vocab_min_max_logits(
                logits, ctx.ulysses_pad_size
            )
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
            values = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
            if ctx.pad_length > 0:
                values = values[: -ctx.pad_length]
            loss = loss_fn(values, ctx.mb_input)

        loss_scale = loss_weight_fn(ctx.mb_input) / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_forward_result(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
    ) -> torch.Tensor:
        """Compute forward output (logprobs or values)."""
        if not self.config.is_critic:
            result = self._compute_logprobs(
                logits, ctx.model_inputs, ctx.ulysses_pad_size
            )
        else:
            result = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
        if ctx.pad_length > 0:
            result = result[: -ctx.pad_length]
        return result
