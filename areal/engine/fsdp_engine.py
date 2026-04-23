# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import gc
import math
import os
import json
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from datetime import datetime
from shutil import rmtree
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.nn.functional as dist_F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from safetensors.torch import save_file as safetensors_save_file
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.tensor import DTensor
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

from areal.api import (
    FinetuneSpec,
    FSDPParallelStrategy,
    InferenceEngine,
    ParallelStrategy,
    ParamSpec,
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import PerfTracerConfig, TrainEngineConfig
from areal.api.io_struct import DeviceRuntimeInfo
from areal.engine.core import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.engine.core.distributed import (
    init_custom_process_group,
    patch_dist_group_timeout,
)
from areal.engine.core.model import (
    disable_dropout_in_model,
    is_gemma3_model,
    is_qwen3_5_model,
    is_qwen3_moe_model,
    is_qwen3_vl_model,
    is_qwen_vl_model,
    is_valid_vision_model,
)
from areal.engine.fsdp_utils import (
    fsdp2_load_full_state_dict,
    get_cosine_schedule_with_warmup,
)
from areal.engine.fsdp_utils.checkpoint import DCPState
from areal.engine.fsdp_utils.grad import fsdp2_clip_grad_norm
from areal.engine.fsdp_utils.optimizer import AnyPrecisionAdamW, PerLayerOptimWrapper
from areal.engine.fsdp_utils.parallel import ParallelHelper, parallelize_model
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.platforms import current_platform
from areal.models.fsdp.ulysses import (
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
    ulysses_prepare_inputs,
)
from areal.models.transformers.ulyssess_patch import apply_monkey_patch
from areal.models.tree_attn.functional import (
    _gather_packed_tree_logprobs,
    gather_packed_tree_logprobs_entropy,
    gather_packed_tree_vocab_stats,
    merge_packed_tree_results,
)
from areal.models.tree_attn.module import (
    build_tree_attn_kwargs,
    patch_fsdp_for_tree_training,
)
from areal.models.tree_attn.tree import TrieNode, build_packed_tree_batch
from areal.utils import (
    logging,
    name_resolve,
    names,
    perf_tracer,
    pkg_version,
    stats_tracker,
)
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.data import (
    MicroBatchItem,
    MicroBatchList,
    amend_position_ids,
    concat_batch,
    pack_tensor_dict,
    pad_mb_list,
    split_batch,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_mb_list,
)
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_processor_and_tokenizer, load_hf_tokenizer
from areal.utils.network import find_free_ports, format_host_for_url, gethostip
from areal.utils.offload import is_tms_enabled, torch_memory_saver
from areal.utils.perf_tracer import trace_perf, trace_scope
from areal.utils.save_load import get_state_dict_from_repo_id_or_path

if TYPE_CHECKING:
    from areal.api import Scheduler
    from areal.api.cli_args import PPOActorConfig, PPOCriticConfig


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
    trie_node
        The root TrieNode for tree training (if applicable).
    """

    model_inputs: dict[str, Any]
    mb_input: dict[str, Any]
    pad_length: int = 0
    ulysses_pad_size: int = 0
    trie_node: TrieNode | None = None

    def to_dict(self) -> dict[str, Any]:
        """Shallow dict conversion (avoids ``dataclasses.asdict`` which would
        recurse into TrieNode and hit ``RecursionError``).
        """
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclasses.dataclass
class _PendingWeightUpdateBucket:
    handles: list[Any]
    fut: Future[None]
    named_tensors: list[tuple[str, torch.Tensor]]
    stream: torch.cuda.Stream | None = None


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

        self._initialized = False
        self.own_global_group = False
        self._cpu_group: dist.ProcessGroup
        self.weight_update_group_initialized = False
        self.weight_update_group_name: str
        self.weight_update_master_addr: str
        self.weight_update_master_port: int

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.is_vision_model = is_valid_vision_model(self.model_config.model_type)

        # FSDP-specific initialization
        self.cpu_offload: CPUOffloadPolicy | None = None

        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None

        self.parallel_helper: ParallelHelper
        self.world_mesh: DeviceMesh

        self.dp_group: dist.ProcessGroup
        self.sp_group: dist.ProcessGroup
        self.mp_group: dist.ProcessGroup

        self.world_size: int
        self.rank: int
        self.dp_head: int
        self.dp_rank: int

        self.is_offload: bool = False
        self._per_layer_optim_wrapper: PerLayerOptimWrapper | None = None
        self.enable_tree_training: bool = self.config.enable_tree_training

        # LoRA delta sync state: tracks whether base model weights have been
        # sent to the inference engine at least once.  When lora_delta_sync
        # is enabled, subsequent update_weights calls only transmit adapter
        # parameters, skipping the much larger base-model parameters.
        self._base_sync_done: bool = False
        # Tracks the versioned name of the most recently loaded LoRA adapter
        # on SGLang, so we can unload it (by the correct name) before loading
        # a new version.  None means no adapter has been loaded yet.
        self._last_loaded_lora_name: str | None = None

        # --- Delta Sync cumulative metrics for wandb ---

    @property
    def _delta_sync_dir(self) -> str:
        """Return a shared directory for delta sync artifacts.

        Uses ``config.delta_sync_dir`` if set, otherwise falls back to
        :func:`areal.utils.fs.get_user_tmp`.  The returned path is
        expected to be on a shared filesystem accessible by both training
        and inference nodes in multi-node setups.
        """
        custom_dir = getattr(self.config, "delta_sync_dir", None)
        if custom_dir:
            os.makedirs(custom_dir, exist_ok=True)
            return custom_dir
        from areal.utils.fs import get_user_tmp
        return get_user_tmp()

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

        self.logger = logging.getLogger(f"[FSDPEngine Rank {dist.get_rank()}]")

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
        self.world_size = dist.get_world_size()

        self.dp_head = dist.get_process_group_ranks(self.mp_group)[0]
        self.dp_rank = dist.get_rank(self.dp_group)

        self.logger.info(f"Data parallel head {self.dp_head} and rank {self.dp_rank}")

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        # Initialize distributed enviroments and load model.
        assert addr is None, "FSDPEngine does not support remote initialization."
        assert ft_spec is not None, "FSDPEngine requires FinetuneSpec to initialize."
        if pkg_version.is_version_less("torch", "2.4.0"):
            raise RuntimeError("areal only supports FSDP2, which requires torch>=2.4.0")

        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"
        self.weight_update_group_name = "update_weight_group"

        # Create device model
        self._create_device_model()

        if self.enable_tree_training and self.parallel_helper.sp_size > 1:
            raise ValueError(
                "Tree training currently cannot be enabled with sp_size > 1."
            )
        # Monkey patch: replace attention's forward() with Ulysses variant.
        apply_monkey_patch(
            model=self.model,
            ulysses_sp_size=self.parallel_helper.sp_size,
            shard_vision_across_sp=self.config.fsdp.shard_vision_across_sp,
        )
        # Monkey patch: replace attention's forward() with tree attention.
        patch_fsdp_for_tree_training(enable=self.enable_tree_training)

        if self.config.use_lora:
            self._apply_peft_wrapper()

        # sharding_strategy = ShardingStrategy.FULL_SHARD
        # Simple auto wrap policy
        self.cpu_offload = (
            CPUOffloadPolicy() if self.config.fsdp.offload_params else None
        )
        tik = time.perf_counter()

        full_state = None
        need_broadcast = False

        is_llm_cpu_load = (
            self.config.fsdp.memory_efficient_load
            and not self.config.init_from_scratch
            and not self.is_vision_model
        )

        if is_llm_cpu_load or self.config.use_lora:
            need_broadcast = True
            if dist.get_rank() == 0:
                if is_llm_cpu_load:
                    pretrained_state = get_state_dict_from_repo_id_or_path(
                        self.config.path
                    )
                    missing, unexpected = self.model.load_state_dict(
                        pretrained_state, strict=False
                    )
                    if missing:
                        self.logger.warning(
                            f"Missing keys when loading pretrained weights: {missing}"
                        )
                    if unexpected:
                        self.logger.warning(
                            f"Unexpected keys when loading pretrained weights: {unexpected}"
                        )
                    del pretrained_state
                    gc.collect()
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

        if need_broadcast:
            broadcast_tik = time.perf_counter()
            fsdp2_load_full_state_dict(
                self.model,
                full_state,
                self.cpu_offload,
                tie_word_embeddings=self.model_config.tie_word_embeddings,
            )
            self.logger.info(
                f"Broadcasting model weights took {time.perf_counter() - broadcast_tik:.2f} seconds"
            )

        self.logger.info(
            f"Applying FSDP2 with N-D parallelism for {time.perf_counter() - tik:.2f} seconds"
        )

        self._create_optimizer(ft_spec)

        if self.config.fsdp.per_layer_optim_step:
            if self.optimizer_config.type != "adam":
                raise ValueError(
                    f"per_layer_optim_step only supports 'adam' optimizer, got '{self.optimizer_config.type}'."
                )
            self._per_layer_optim_wrapper = PerLayerOptimWrapper(
                model=self.model,
                optimizer=self.optimizer,
                device_id=self.device,
                prefetch_layers=self.config.fsdp.optim_step_prefetch_layers,
            )

        self._initialized = True

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
        assert self._initialized
        return self._cpu_group

    def destroy(self):
        self._initialized = False
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            del self.model
        if self._per_layer_optim_wrapper is not None:
            del self._per_layer_optim_wrapper
            self._per_layer_optim_wrapper = None
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

    @property
    def initialized(self) -> bool:
        return self._initialized

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
    ) -> list[dict[str, Any]]:
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
    ) -> list[dict[str, Any]]:
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.prepare_batch(
            dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
        )

    def update_weights(self, meta: WeightUpdateMeta):
        self._check_rollout_engine_connected()
        with self._offload_aware_context():
            if self.config.use_lora and self.config.lora_delta_sync:
                self._update_weights_delta_sync_disk(meta)
            elif meta.type == "xccl":
                assert self.weight_update_group_initialized
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
        with self._offload_aware_context():
            if meta.weight_format == "hf":
                self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
            elif meta.weight_format == "dcp":
                self._save_to_dcp(meta.path, meta.with_optim)
            else:
                raise ValueError(f"Unknown weight format {meta.weight_format}. ")

            if meta.with_optim and meta.weight_format == "hf":
                self._save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        with self._offload_aware_context():
            if meta.weight_format == "hf":
                self._load_model_from_hf(meta.path)
            elif meta.weight_format == "dcp":
                self._load_from_dcp(meta.path, meta.with_optim)
            else:
                raise ValueError(f"Unknown weight format {meta.weight_format}. ")

            if meta.with_optim and meta.weight_format == "hf":
                self._load_optimizer_state(meta.path)

            # Checkpoint load replaces optimizer state tensor objects, losing
            # pinning and normalization established by PerLayerOptimWrapper.__init__.
            if meta.with_optim and self._per_layer_optim_wrapper is not None:
                self._per_layer_optim_wrapper.refresh_states()

    @contextmanager
    def _offload_aware_context(self):
        """Temporarily onload parameters for offload-unsafe operations."""
        if not self.is_offload:
            with nullcontext():
                yield
            return

        self.onload()
        try:
            yield
        finally:
            self.offload()

    def optimizer_zero_grad(self):
        assert self.optimizer is not None
        self.optimizer.zero_grad()

    def optimizer_step(self):
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        grad_norm = fsdp2_clip_grad_norm(
            list(self.model.parameters()),
            max_norm=self.optimizer_config.gradient_clipping,
            fsdp_group=self.world_mesh["dp_sp"].get_group(),
            tp_group=self.world_mesh["tp"].get_group(),
            offload_params=self.config.fsdp.offload_params,
        )

        if not math.isfinite(grad_norm):
            self.optimizer_zero_grad()
            update_successful = False
        elif self.config.fsdp.per_layer_optim_step:
            assert self._per_layer_optim_wrapper is not None
            with trace_scope("fsdp_engine.step"):
                self._per_layer_optim_wrapper.step()
            update_successful = True
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

            # Lazily create tree attention metadata just before forward.
            # The returned dict keys are prefixed with "tree_" to avoid collisions
            # with HuggingFace's own kwargs. The patched _tree_attn_fwd_func in
            # module_fsdp.py reads these keys from the **kwargs that transformers
            # forwards through.
            tree_attn_keys: list[str] = []
            if self.enable_tree_training and ctx.trie_node is not None:
                padded_size = mb_item.padded_to_length
                assert padded_size is not None
                tree_kwargs = build_tree_attn_kwargs(
                    ctx.trie_node, padded_size, self.device
                )
                inputs.update(tree_kwargs)
                tree_attn_keys = list(tree_kwargs.keys())

            with trace_scope("fsdp_engine.forward"):
                outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)

            # Release tree attention metadata after forward pass
            for key in tree_attn_keys:
                del inputs[key]

            ctx_dict = ctx.to_dict()
            loss = process_output_fn(logits, ctx_dict)

            if not forward_only and loss is not None:
                with trace_scope("fsdp_engine.backward"):
                    loss.backward()

    def train_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        self._ensure_ready()
        self.optimizer_zero_grad()

        input_batched, _ = self._normalize_batch_input(input_)

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_batched).to(self.device)

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
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        self._ensure_ready()

        input_batched, _ = self._normalize_batch_input(input_)

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_batched).to(self.device)

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
        input_: list[dict[str, Any]] | dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[torch.Tensor]], torch.Tensor] = torch.cat,
    ) -> torch.Tensor | list[torch.Tensor]:
        self._ensure_ready()

        input_batched, meta = self._normalize_batch_input(input_)

        # Step 1: Prepare sequence lengths
        if meta is not None:
            assert isinstance(input_, list)
            inferred_seqlens = [d["attention_mask"].shape[-1] for d in input_]
            if output_seqlens is not None and output_seqlens != inferred_seqlens:
                raise ValueError(
                    f"output_seqlens mismatch for list input: "
                    f"given {output_seqlens}, "
                    f"inferred {inferred_seqlens} from attention_mask shapes."
                )
            output_seqlens = inferred_seqlens
        cu_seqlens = pack_tensor_dict(input_batched)["cu_seqlens"]
        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None
        batch_size = len(output_seqlens)

        # Step 2: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        # Step 3: Forward using process_output_fn callback, collecting results
        outputs: list[torch.Tensor] = []

        def process_output(logits: torch.Tensor, ctx_dict: dict[str, Any]) -> None:
            ctx = FSDPTrainContext(**ctx_dict)
            result = self._compute_forward_result(logits, ctx)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate and reorder outputs
        if self.enable_tree_training:
            result = merge_packed_tree_results(outputs, batch_size)
        else:
            result = reorder_and_pad_outputs(
                outputs, output_seqlens, mb_list, aggregate_fn
            )

        if meta is None:
            return result
        return split_batch(result, meta)

    def export_stats(self) -> dict[str, float]:
        with self._offload_aware_context():
            return stats_tracker.export_all(
                reduce_group=self.data_parallel_group,
            )

    def offload(self) -> None:
        """Offload model memory to CPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """
        if not is_tms_enabled():
            raise RuntimeError(
                "torch_memory_saver requires `enable_offload=True` in yaml config."
            )

        self.get_device_stats().log("before offload model")

        # Use torch_memory_saver to pause CUDA memory
        current_platform.clear_memory()
        torch_memory_saver.pause()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after offload model")

        self.is_offload = True

    def onload(self) -> None:
        """Onload model memory from CPU back to GPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """

        torch_memory_saver.resume()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after onload model")

        self.is_offload = False

    def clear_batches(self, *args):
        """Placeholder method of single-controller API."""

    def get_device_stats(self) -> DeviceRuntimeInfo:
        return DeviceRuntimeInfo.get_current()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)

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

        if self.config.init_from_scratch or self.config.fsdp.memory_efficient_load:
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
        current_platform.set_numa_affinity(int(os.environ["LOCAL_RANK"]))
        if current_platform.device_type == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        dtype = getattr(torch, self.config.dtype)

        if self.config.fsdp.memory_efficient_load:
            # Only rank 0 loads on CPU; other ranks use meta device (zero memory)
            # to avoid CPU OOM when multiple workers share a node.
            # Weights are broadcast from rank 0 after FSDP sharding in initialize().
            # Note: meta device optimization only applies to LLM (not VLM), because
            # VLM uses from_pretrained() which doesn't support meta device context.
            if not self.is_vision_model and dist.get_rank() != 0:
                loading_device = "meta"
            else:
                loading_device = "cpu"
        else:
            loading_device = current_platform.device_type

        self.get_device_stats().log("before model creation/loading")

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
            # VLM: Use from_pretrained() on loading_device.
            with torch.device(loading_device):
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
            # LLM: Decide between from_config() or from_pretrained() based on memory_efficient_load
            with torch.device(loading_device):
                model = self._create_llm_actor_or_critic()
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)

        self.get_device_stats().log("after model creation/loading")

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if self.config.use_kernels:
            model.use_kernels = True
        self.logger.info(
            f"Model creation and loading time: {time.perf_counter() - tik:.2f}s"
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

    @staticmethod
    def _normalize_batch_input(
        input_: list[dict[str, Any]] | dict[str, Any],
    ) -> tuple[dict[str, Any], Any | None]:
        """Normalize list/dict batch input to a single batched dict.

        Returns ``(batched_input, meta)`` where ``meta`` is non-None only when
        input is list-based and can be used to split forward outputs back into
        per-trajectory results.
        """
        if isinstance(input_, list):
            return concat_batch(input_)
        return input_, None

    def _normalize_peft_param_name(self, name: str) -> str:
        """Convert PEFT parameter name to HuggingFace original model format.

        PEFT wraps model parameters with extra prefixes and infixes:
          'base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight'
        becomes:
          'model.layers.0.self_attn.q_proj.weight'
        """
        # Remove FSDP wrapper prefix if present
        name = name.replace("_fsdp_wrapped_module.", "")
        # Remove 'base_model.model.' prefix (added by PEFT wrapper)
        name = name.replace("base_model.model.", "", 1)
        # Remove residual 'base_model.' prefix (some PEFT versions)
        name = name.replace("base_model.", "", 1)
        # Remove '.base_layer' infix (PEFT marks original weights this way)
        name = name.replace(".base_layer", "")
        return name

    def _collect_lora_params(
        self, meta: WeightUpdateMeta, base_sync_done: bool
    ) -> list[tuple[str, torch.Tensor]]:
        """Collect parameters for LoRA delta sync.

        When base_sync_done is False, collects all base model parameters
        (excluding LoRA-specific weights like lora_A, lora_B) for the
        initial full-model sync via disk.

        When base_sync_done is True, collects only the trainable LoRA
        adapter parameters for incremental sync via disk.

        All ranks must call this method because ``_get_full_tensor``
        triggers FSDP all-gather collectives internally.  However, only
        rank 0 accumulates the resulting tensors in the returned list.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Weight update metadata (used for name remapping via
            _get_model_name_parameters).
        base_sync_done : bool
            If False, collect base model weights (skip lora_ params).
            If True, collect only trainable LoRA adapter weights.

        Returns
        -------
        list[tuple[str, torch.Tensor]]
            List of (name, tensor) pairs.  On rank 0 these contain the
            full tensors ready for disk saving; on other ranks the list
            is empty.
        """
        collected = []
        main_rank = dist.get_rank() == 0

        if base_sync_done:
            # Collect only trainable LoRA parameters (lora_A, lora_B weights)
            self.logger.debug(
                "[LoRA Delta Sync] Collecting LoRA adapter parameters only "
                "(base_sync_done=True)"
            )
            logged_adapter_examples = 0
            trainable_count = 0
            frozen_count = 0
            for name, param in self._get_model_name_parameters(meta):
                if param.requires_grad:
                    trainable_count += 1
                else:
                    frozen_count += 1
                    continue
                tensor = self._get_full_tensor(param)
                if main_rank:
                    if logged_adapter_examples < 5:
                        self.logger.debug(
                            f"[LoRA Delta Sync] Adapter param [{logged_adapter_examples}]: "
                            f"'{name}' shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                        )
                        logged_adapter_examples += 1
                    collected.append((name, tensor))
            if main_rank:
                self.logger.debug(
                    f"[LoRA Delta Sync] Parameter grad stats: "
                    f"requires_grad=True: {trainable_count}, "
                    f"requires_grad=False: {frozen_count}"
                )
        else:
            # Collect base model parameters, skipping LoRA-specific ones
            lora_keywords = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
            self.logger.debug(
                "[LoRA Delta Sync] Collecting base model parameters "
                f"(base_sync_done=False, skipping params matching: {lora_keywords})"
            )
            logged_examples = 0
            skipped_lora_count = 0
            for name, param in self._get_model_name_parameters(meta):
                if any(kw in name for kw in lora_keywords):
                    skipped_lora_count += 1
                    continue
                # Normalize PEFT-wrapped names back to HuggingFace format so
                # they match SGLang's params_dict keys.
                original_name = name
                name = self._normalize_peft_param_name(name)
                if "lora_" in name or ".adapter_" in name:
                    skipped_lora_count += 1
                    if main_rank:
                        self.logger.warning(
                            f"[LoRA Delta Sync] Secondary filter caught "
                            f"residual LoRA/adapter param after normalization: '{name}'"
                        )
                    continue
                tensor = self._get_full_tensor(param)
                if main_rank:
                    if logged_examples < 5:
                        self.logger.debug(
                            f"[LoRA Delta Sync] Base param name mapping: "
                            f"'{original_name}' -> '{name}'"
                        )
                        logged_examples += 1
                    collected.append((name, tensor))
            if main_rank:
                self.logger.debug(
                    f"[LoRA Delta Sync] Skipped {skipped_lora_count} "
                    f"LoRA-specific params during base collection"
                )

        if main_rank:
            total_params = len(collected)
            total_bytes = sum(t.numel() * t.element_size() for _, t in collected)
            total_mb = total_bytes / 1024 / 1024
            self.logger.debug(
                f"[LoRA Delta Sync] Collected {total_params} params, "
                f"total size: {total_mb:.2f} MB, "
                f"base_sync_done={base_sync_done}"
            )
            # Report param collection metrics to wandb
            if base_sync_done:
                stats_tracker.scalar(**{"delta_sync/adapter_mb": total_mb})
            else:
                stats_tracker.scalar(**{"delta_sync/base_mb": total_mb})

        return collected

    def _get_model_name_parameters(
        self, meta: WeightUpdateMeta
    ) -> Iterator[tuple[str, nn.Parameter]]:
        name_params_iterator = self.model.named_parameters()
        if self.is_vision_model and is_qwen_vl_model(self.model_config.model_type):
            for name, value in name_params_iterator:
                if meta.gen_allocation.backend == "sglang":
                    # SGLang 0.5.9 branch
                    # LLM part: "model.language_model.norm.weight" -> "model.norm.weight"
                    # Vision part: "model.visual.blocks.5.mlp.gate_proj.weight" -> "visual.blocks.5.mlp.gate_proj.weight"
                    new_name = name.replace("language_model.", "", 1)
                    if new_name.startswith("model.visual."):
                        new_name = new_name.replace("model.", "", 1)
                    yield new_name, value
                    continue
                # vLLM 0.17.0 branch
                new_name = name.replace("model.", "", 1)
                if new_name.startswith("language_model."):
                    new_name = new_name.replace(
                        "language_model.", "language_model.model.", 1
                    )
                elif new_name.startswith("lm_head."):
                    new_name = f"language_model.{new_name}"
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

    def _update_bucket_weights_from_distributed_async(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        stream: torch.cuda.Stream | None = None,
    ) -> _PendingWeightUpdateBucket | None:
        # Early exit when chunk size is relatively small
        if not named_tensors:
            return None

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
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
            context = torch.cuda.stream(stream)
        else:
            context = nullcontext()

        with context:
            for _, tensor in named_tensors:
                handles.append(
                    dist.broadcast(
                        tensor, src=0, group=self.weight_update_group, async_op=True
                    )
                )

        return _PendingWeightUpdateBucket(
            handles=handles,
            fut=fut,
            named_tensors=named_tensors,
            stream=stream,
        )

    def _wait_pending_weight_update_bucket(
        self, pending_bucket: _PendingWeightUpdateBucket | None
    ):
        if pending_bucket is None:
            return

        for handle in pending_bucket.handles:
            handle.wait()

        pending_bucket.fut.result()
        pending_bucket.named_tensors.clear()

    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ):
        pending_bucket = self._update_bucket_weights_from_distributed_async(
            meta, named_tensors
        )
        self._wait_pending_weight_update_bucket(pending_bucket)

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta):
        assert meta.type == "xccl"

        # Reset weight weight meta with local info
        meta.nccl_master_address = self.weight_update_master_addr = gethostip()
        meta.nccl_master_port = self.weight_update_master_port = find_free_ports(1)[0]
        meta.nccl_group_name = self.weight_update_group_name

        # NOTE: Processes launched with torchrun will set the following env var to True,
        # which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        if dist.get_rank() == 0:
            assert meta.gen_allocation is not None

            fut = self.rollout_engine.init_weights_update_group(meta)

            gen_world_size = meta.gen_allocation.parallel.world_size
            init_method = f"tcp://{format_host_for_url(meta.nccl_master_address)}:{meta.nccl_master_port}"
            self.logger.info(
                f"Initializing weight update group: type={meta.type} "
                f"init_method={init_method} "
                f"group={meta.nccl_group_name}"
            )
            self.weight_update_group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=gen_world_size + 1,
                init_method=init_method,
                rank=0,
                group_name=meta.nccl_group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )

            fut.result()

    def _broadcast_params_bucketed(
        self,
        meta: WeightUpdateMeta,
        param_list: list[tuple[str, torch.Tensor]],
        main_rank: bool,
    ):
        """Broadcast a list of (name, tensor) pairs using bucketed pipelining.

        This is the inner loop extracted from _update_weights_from_distributed
        so it can be called separately for base weights vs adapter weights
        during LoRA delta sync.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Weight update metadata (includes NCCL group info and peft_config).
        param_list : list[tuple[str, torch.Tensor]]
            Pre-collected (name, tensor) pairs to broadcast. On non-main
            ranks this list is empty and the method only participates in the
            NCCL broadcast collectives.
        main_rank : bool
            Whether this process is the main rank (rank 0).
        """
        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
        broadcast_stream = None

        if (
            main_rank
            and current_platform.device_type == "cuda"
            and torch.cuda.is_available()
        ):
            broadcast_stream = torch.cuda.Stream()

        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []
        pending_bucket: _PendingWeightUpdateBucket | None = None

        try:
            for name, tensor in param_list:
                tensor_size = tensor.numel() * tensor.element_size()
                bucket_overflow = (
                    buffer_size > 0
                    and tensor_size + buffer_size > weight_chunked_mem_size
                )
                if bucket_overflow:
                    if pending_bucket is not None:
                        self._wait_pending_weight_update_bucket(pending_bucket)
                        pending_bucket = None

                    pending_bucket = self._update_bucket_weights_from_distributed_async(
                        meta,
                        named_tensors,
                        stream=broadcast_stream,
                    )

                    named_tensors = []
                    buffer_size = 0

                buffer_size += tensor_size
                named_tensors.append((name, tensor))

            if pending_bucket:
                self._wait_pending_weight_update_bucket(pending_bucket)
                pending_bucket = None

            if buffer_size > 0:
                self._update_bucket_weights_from_distributed(meta, named_tensors)
        finally:
            if main_rank and pending_bucket is not None:
                self._wait_pending_weight_update_bucket(pending_bucket)
                pending_bucket = None


    def _save_and_load_lora_adapter(
        self,
        meta: WeightUpdateMeta,
        adapter_params: list[tuple[str, torch.Tensor]],
    ):
        """Save LoRA adapter tensors to a temp directory and load via SGLang.

        This method is called on rank 0 only.  It:
          1. Builds a safetensors state dict from the gathered adapter tensors.
          2. Writes adapter_model.safetensors and adapter_config.json.
          3. Unloads any previously loaded adapter (tolerates failure).
          4. Calls ``/load_lora_adapter`` on the SGLang server.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Weight update metadata (must have lora_name, version, peft_config).
        adapter_params : list[tuple[str, torch.Tensor]]
            Pre-gathered (name, tensor) pairs for the adapter weights.
        """
        from areal.api.io_struct import get_versioned_lora_name
        from safetensors import safe_open

        overall_start = time.monotonic()

        lora_name = meta.lora_name or "default_lora"
        version = meta.version if meta.version is not None else 0
        versioned_name = get_versioned_lora_name(lora_name, version)

        # Determine a temp directory for saving the adapter.
        # Use a persistent temp dir under /tmp so SGLang server can access it.
        adapter_dir = os.path.join(
            self._delta_sync_dir, "areal_lora_adapters", versioned_name
        )
        os.makedirs(adapter_dir, exist_ok=True)

        # --- 1. Save adapter weights as safetensors ---
        step_start = time.monotonic()
        # PEFT's named_parameters() includes the active adapter name in keys,
        # e.g. "...lora_A.default.weight".  The standard PEFT adapter file
        # format (as produced by get_peft_model_state_dict / save_pretrained)
        # strips this adapter name, yielding "...lora_A.weight".  SGLang's
        # /load_lora_adapter expects the stripped format.
        adapter_name = "default"  # PEFT default adapter name
        state_dict = {}
        for name, tensor in adapter_params:
            # Strip the adapter name segment: ".default." -> "."
            # e.g. "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
            #   -> "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            name = name.replace(f".{adapter_name}.", ".")
            state_dict[name] = tensor.contiguous().cpu()

        safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        safetensors_save_file(state_dict, safetensors_path)
        file_size_mb = os.path.getsize(safetensors_path) / 1024 / 1024
        save_elapsed = time.monotonic() - step_start
        self.logger.debug(
            f"[LoRA Delta Sync] Saved {len(state_dict)} adapter tensors "
            f"to {safetensors_path} ({file_size_mb:.2f} MB) "
            f"in {save_elapsed:.3f}s"
        )
        stats_tracker.scalar(**{"delta_sync/disk_save_s": save_elapsed})

        # Verification: re-read and verify the saved file
        try:
            with safe_open(safetensors_path, framework="pt") as f:
                saved_keys = list(f.keys())
                if len(saved_keys) != len(state_dict):
                    self.logger.error(
                        f"[LoRA Delta Sync] Verification FAILED: saved "
                        f"{len(saved_keys)} tensors but expected {len(state_dict)}"
                    )
                else:
                    self.logger.debug(
                        f"[LoRA Delta Sync] Verification OK: safetensors "
                        f"contains {len(saved_keys)} tensors"
                    )
        except Exception as e:
            self.logger.error(
                f"[LoRA Delta Sync] Verification read failed: {type(e).__name__}: {e}"
            )

        # --- 2. Save adapter_config.json ---
        step_start = time.monotonic()
        # Build a PEFT-compatible config dict
        config = self.config
        if not config.target_modules or config.target_modules == ["all-linear"]:
            target_modules_val = "all-linear"
        else:
            target_modules_val = config.target_modules

        adapter_config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": meta.base_model_name or "",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_to_transform": None,
            "layers_pattern": None,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": 0.0,
            "modules_to_save": None,
            "r": config.lora_rank,
            "revision": None,
            "target_modules": target_modules_val,
            "task_type": "CAUSAL_LM",
        }
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)
        self.logger.debug(
            f"[LoRA Delta Sync] Saved adapter_config.json to {config_path} "
            f"in {time.monotonic() - step_start:.3f}s, "
            f"r={adapter_config['r']}, lora_alpha={adapter_config['lora_alpha']}, "
            f"target_modules={adapter_config['target_modules']}"
        )

        example_names = [n for n, _ in adapter_params[:5]]
        self.logger.debug(
            f"[LoRA Delta Sync] First 5 original adapter param names: {example_names}"
        )

        # --- 3. Load new adapter via rollout engine ---
        step_start = time.monotonic()
        # Use the dedicated load_lora_adapter method which sends HTTP requests
        # directly to SGLang servers, bypassing the name_resolve-based disk
        # update flow (which is designed for full model checkpoints).
        prev_lora_name = self._last_loaded_lora_name
        self.logger.debug(
            f"[LoRA Delta Sync] Loading adapter '{versioned_name}' from "
            f"{adapter_dir} via /load_lora_adapter"
            f" (previous adapter: {prev_lora_name!r})"
        )
        fut = self.rollout_engine.load_lora_adapter(
            versioned_name, adapter_dir, prev_lora_name=prev_lora_name,
        )
        fut.result()
        http_load_elapsed = time.monotonic() - step_start
        self.logger.info(
            f"[LoRA Delta Sync] Successfully loaded adapter '{versioned_name}' "
            f"in {http_load_elapsed:.3f}s"
        )
        stats_tracker.scalar(**{"delta_sync/http_load_adapter_s": http_load_elapsed})

        # Track the loaded adapter name for future unloads
        old_loaded_name = self._last_loaded_lora_name
        self._last_loaded_lora_name = versioned_name

        # Clean up the previous version's adapter directory from disk
        if old_loaded_name is not None and old_loaded_name != versioned_name:
            old_adapter_dir = os.path.join(
                self._delta_sync_dir, "areal_lora_adapters", old_loaded_name
            )
            if os.path.isdir(old_adapter_dir):
                try:
                    rmtree(old_adapter_dir)
                    self.logger.debug(
                        f"[LoRA Delta Sync] Cleaned up old adapter directory: "
                        f"{old_adapter_dir}"
                    )
                except OSError as e:
                    self.logger.warning(
                        f"[LoRA Delta Sync] Failed to clean up "
                        f"{old_adapter_dir}: {e}"
                    )

        total_elapsed = time.monotonic() - overall_start
        self.logger.debug(
            f"[LoRA Delta Sync] _save_and_load_lora_adapter total: {total_elapsed:.3f}s"
        )

        # --- Delta Sync per-step metrics for wandb ---
        stats_tracker.scalar(**{"delta_sync/save_and_load_total_s": total_elapsed})
        stats_tracker.scalar(
            **{"delta_sync/is_incremental": 1.0 if self._base_sync_done else 0.0}
        )

    @trace_perf("fsdp_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        """Broadcast parameters with single-pending-bucket pipelining.

        This method handles the NCCL/XCCL-based weight sync path.
        When ``use_lora=True`` without ``lora_delta_sync``, only trainable
        (LoRA) parameters are broadcast.  When ``use_lora=False``, all
        parameters are broadcast.

        Note: when ``lora_delta_sync`` is enabled, the caller dispatches to
        :meth:`_update_weights_delta_sync_disk` instead of this method.
        """

        # Reset weight meta with local info
        meta.nccl_master_address = self.weight_update_master_addr
        meta.nccl_master_port = self.weight_update_master_port
        meta.nccl_group_name = self.weight_update_group_name

        main_rank = dist.get_rank() == 0
        if main_rank:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        # ---------- Weight sync (non-delta-sync) ----------
        if True:
            weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
            broadcast_stream = None

            if (
                main_rank
                and current_platform.device_type == "cuda"
                and torch.cuda.is_available()
            ):
                broadcast_stream = torch.cuda.Stream()

            buffer_size = 0
            named_tensors: list[tuple[str, torch.Tensor]] = []
            pending_bucket: _PendingWeightUpdateBucket | None = None

            if self.config.use_lora:
                # For LoRA, only iterate over trainable LoRA parameters
                param_iterator = (
                    (name, param)
                    for name, param in self._get_model_name_parameters(meta)
                    if param.requires_grad
                )
            else:
                # For full model, iterate over all parameters
                param_iterator = self._get_model_name_parameters(meta)

            try:
                for name, param in param_iterator:
                    # Ranks other than 0 only help to get the full tensor
                    tensor = self._get_full_tensor(param)
                    if not main_rank:
                        continue

                    tensor_size = tensor.numel() * tensor.element_size()
                    bucket_overflow = (
                        buffer_size > 0
                        and tensor_size + buffer_size > weight_chunked_mem_size
                    )
                    if bucket_overflow:
                        # Only middle buckets need drain+align before the next all-gather.
                        if pending_bucket is not None:
                            self._wait_pending_weight_update_bucket(pending_bucket)
                            pending_bucket = None

                        pending_bucket = self._update_bucket_weights_from_distributed_async(
                            meta,
                            named_tensors,
                            stream=broadcast_stream,
                        )

                        named_tensors = []
                        buffer_size = 0

                    buffer_size += tensor_size
                    named_tensors.append((name, tensor))

                if pending_bucket:
                    self._wait_pending_weight_update_bucket(pending_bucket)
                    pending_bucket = None

                # Process remaining parameters
                if buffer_size > 0:
                    self._update_bucket_weights_from_distributed(meta, named_tensors)
            finally:
                if main_rank and pending_bucket is not None:
                    self._wait_pending_weight_update_bucket(pending_bucket)
                    pending_bucket = None

        dist.barrier(group=self.cpu_group)
        if main_rank:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def _save_base_model_for_delta_sync(
        self,
        save_path: str,
        base_params: list[tuple[str, torch.Tensor]],
    ):
        """Save base-model parameters (excluding LoRA) in HuggingFace format.

        This method is called on rank 0 only during delta sync Phase 1a.
        It constructs a state dict from the pre-collected base parameters
        (which have already been all-gathered from FSDP on all ranks) and
        saves them as safetensors alongside the model config.

        Parameters
        ----------
        save_path : str
            Directory to save the HuggingFace checkpoint into.
        base_params : list[tuple[str, torch.Tensor]]
            List of (name, tensor) pairs for base-model weights only.
            Names should already be normalised to HuggingFace format
            (i.e. without PEFT prefixes).
        """
        os.makedirs(save_path, exist_ok=True)
        state_dict = {name: t.contiguous().cpu() for name, t in base_params}

        safetensors_path = os.path.join(save_path, "model.safetensors")
        safetensors_save_file(state_dict, safetensors_path)
        file_size_mb = os.path.getsize(safetensors_path) / 1024 / 1024

        self.model_config.save_pretrained(save_path)

        self.logger.debug(
            f"[LoRA Delta Sync] Saved {len(state_dict)} base-model params "
            f"to {safetensors_path} ({file_size_mb:.2f} MB)"
        )

    @trace_perf("fsdp_engine.update_weights_delta_sync_disk", category="io")
    def _update_weights_delta_sync_disk(self, meta: WeightUpdateMeta):
        """Disk-based LoRA delta sync: both base model and adapter via disk.

        Phase 1 (first call, ``_base_sync_done=False``):
          - Phase 1a: All ranks gather base-model params via FSDP all-gather.
            Rank 0 saves them in HuggingFace safetensors format.  SGLang
            loads via ``/update_weights_from_disk``.
          - Phase 1b: All ranks gather adapter params via FSDP all-gather.
            Rank 0 saves as PEFT safetensors + adapter_config.json.  SGLang
            loads via ``/load_lora_adapter``.

        Phase 2 (subsequent calls, ``_base_sync_done=True``):
          - Only adapter params are gathered and loaded (same as Phase 1b).
          - Base-model weights are already on SGLang, skipped entirely.

        The name_resolve signaling mechanism coordinates the train side
        (which saves files) with the inference side (which loads them).
        """
        main_rank = dist.get_rank() == 0
        if main_rank:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        delta_sync_start = time.monotonic()
        if main_rank:
            self.logger.debug(
                f"[LoRA Delta Sync] Starting disk-based delta sync, "
                f"_base_sync_done={self._base_sync_done}, "
                f"lora_name='{meta.lora_name}', version={meta.version}"
            )

        if not self._base_sync_done:
            phase1a_start = time.monotonic()

            base_params = self._collect_lora_params(meta, base_sync_done=False)
            if main_rank:
                self.logger.debug(
                    f"[LoRA Delta Sync] Phase 1a: saving "
                    f"{len(base_params)} base-model params to disk"
                )

            base_save_path = os.path.join(
                self._delta_sync_dir,
                "areal_lora_delta_sync_base",
                f"v{meta.version or 0}",
            )

            if main_rank:
                disk_meta = WeightUpdateMeta(
                    type="disk",
                    path=base_save_path,
                    use_lora=False,
                    clear_checkpoint_after_load=True,
                    version=meta.version,
                )
                fut = self.rollout_engine.update_weights_from_disk(disk_meta)

                self._save_base_model_for_delta_sync(base_save_path, base_params)

                update_name = names.update_weights_from_disk(
                    self.config.experiment_name,
                    self.config.trial_name,
                    self.get_version(),
                )
                name_resolve.add(
                    update_name,
                    str(datetime.now().timestamp()),
                    keepalive_ttl=120,
                )

                fut.result()
                self.logger.info(
                    f"[LoRA Delta Sync] Phase 1a completed in "
                    f"{time.monotonic() - phase1a_start:.3f}s"
                )
            else:
                pass

            dist.barrier(group=self.cpu_group)

        phase_adapter_start = time.monotonic()
        adapter_params = self._collect_lora_params(meta, base_sync_done=True)
        phase_label = "1b" if not self._base_sync_done else "2"

        if main_rank:
            self.logger.debug(
                f"[LoRA Delta Sync] Phase {phase_label}: "
                f"saving {len(adapter_params)} LoRA adapter params to disk "
                f"for /load_lora_adapter"
            )
            self._save_and_load_lora_adapter(meta, adapter_params)

        if main_rank:
            self.logger.info(
                f"[LoRA Delta Sync] Phase {phase_label} completed in "
                f"{time.monotonic() - phase_adapter_start:.3f}s"
            )

        self._base_sync_done = True

        if main_rank:
            self.logger.info(
                f"[LoRA Delta Sync] Total disk-based delta sync elapsed: "
                f"{time.monotonic() - delta_sync_start:.3f}s"
            )

        dist.barrier(group=self.cpu_group)
        if main_rank:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    @trace_perf("fsdp_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self, meta: WeightUpdateMeta):
        fut = Future()

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()
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
            self.rollout_engine.continue_generation()

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
            if tokenizer is not None and not self.config.use_lora:
                tokenizer.save_pretrained(path)
            if processor is not None and not self.config.use_lora:
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

        # Tree training path
        if self.enable_tree_training:
            mb_list = build_packed_tree_batch(
                input_,
                mb_spec=self.config.mb_spec,
                pad_to_maximum=self.config.pad_to_maximum,
                dp_group=self.data_parallel_group,
                parallel_size=self.parallel_helper.tp_size
                * self.parallel_helper.sp_size,
            )
            self.logger.info(
                f"Packed tree #microbatch: {len(mb_list)}, microbatch #tokens: {mb_list.group_lens}, "
                f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}."
            )
            return mb_list

        if is_qwen_vl_model(self.model_config.model_type):
            attn_mask = input_["attention_mask"]
            input_ids = input_["input_ids"]
            # NOTE: Qwen-VL get_rope_index performs indexed assignment where
            # source positions are int64 and position_ids inherits input_ids.dtype.
            # Ensure input_ids uses int64 so destination/source dtypes align and
            # avoid "Index put requires the source and destination dtypes match".
            if input_ids.dtype != torch.long:
                input_ids = input_ids.to(torch.long)
                input_["input_ids"] = input_ids
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
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attn_mask,
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
            if (
                is_qwen3_moe_model(self.model_config.model_type)
                or is_qwen3_vl_model(self.model_config.model_type)
                or is_qwen3_5_model(self.model_config.model_type)
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
        trie_node = None
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
            trie_node = inputs.pop("trie_node", None)
            ulysses_pad_size = 0

        ctx = FSDPTrainContext(
            model_inputs=inputs,
            mb_input=mb_item.orig_mb,
            pad_length=mb_item.padding_length,
            ulysses_pad_size=ulysses_pad_size,
            trie_node=trie_node,
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
        local_weight = loss_weight_fn(ctx.mb_input)
        if local_weight == 0:
            return logits.mean() * 0.0

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
                    return logits.mean() * 0.0

                # For tree training, use gather_packed_tree_vocab_stats to properly
                # unpack vocab stats from tree structure back to per-sequence format.
                # This is necessary because the logits are in packed tree format where
                # multiple sequences share prefix positions.
                vocab_min_logits, vocab_max_logits = gather_packed_tree_vocab_stats(
                    logits, ctx.trie_node
                )
                logprobs, entropy = gather_packed_tree_logprobs_entropy(
                    logits,
                    ctx.trie_node,
                    ctx.mb_input["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=self.parallel_helper.tp_group
                    if self.parallel_helper.tp_size > 1
                    else None,
                )
            else:
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

        loss_scale = local_weight / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_forward_result(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
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
                    tp_group=self.parallel_helper.tp_group
                    if self.parallel_helper.tp_size > 1
                    else None,
                )
                return result
            result = self._compute_logprobs(
                logits, ctx.model_inputs, ctx.ulysses_pad_size
            )
        else:
            result = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
        if ctx.pad_length > 0:
            result = result[: -ctx.pad_length]
        return result


# =============================================================================
# Algorithm-specific FSDP Engines
# =============================================================================


class FSDPPPOActor(FSDPEngine):
    """PPO Actor implementation using FSDP backend."""

    def __init__(self, config: PPOActorConfig):
        from areal.trainer.ppo.actor import PPOActor

        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> list[torch.Tensor] | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> list[dict[str, Any]]:
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config: PPOActorConfig, scheduler: Scheduler):
        if config._version == "v2":
            from areal.trainer.ppo.actor import PPOActorControllerV2

            return PPOActorControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)


class FSDPPPOCritic(FSDPEngine):
    """PPO Critic implementation using FSDP backend."""

    def __init__(self, config: PPOCriticConfig):
        from areal.trainer.ppo.critic import PPOCritic

        super().__init__(config)
        self.critic = PPOCritic(config, self)

    @torch.no_grad()
    def compute_values(self, *args, **kwargs) -> torch.Tensor:
        return self.critic.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.critic.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config: PPOCriticConfig, scheduler: Scheduler):
        if config._version == "v2":
            from areal.trainer.ppo.critic import PPOCriticControllerV2

            return PPOCriticControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.ppo.critic import PPOCriticController

        return PPOCriticController(train_engine=cls, config=config, scheduler=scheduler)


class FSDPLMEngine(FSDPEngine):
    """Language model engine for SFT using FSDP backend."""

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
        if config._version == "v2":
            from areal.trainer.sft.lm_engine import LMControllerV2

            return LMControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.sft.lm_engine import LMController

        return LMController(train_engine=cls, config=config, scheduler=scheduler)


class FSDPRWEngine(FSDPEngine):
    """Reward model engine using FSDP backend."""

    def __init__(self, config: TrainEngineConfig):
        from copy import deepcopy

        from areal.trainer.rw.rw_engine import RWEngine

        super().__init__(config)
        self.rw_engine = RWEngine(self)
        if self.config.mb_spec.granularity != 2:
            logger = logging.getLogger("RWEngine")
            logger.warning("mb_spec.granularity must be 2 for reward modeling")
            self.config = deepcopy(self.config)
            self.config.mb_spec.granularity = 2

    def train_rw(self, data):
        return self.rw_engine.train_rw(data)

    def evaluate_rw(self, data):
        return self.rw_engine.evaluate_rw(data)

    @classmethod
    def as_controller(cls, config: TrainEngineConfig, scheduler: Scheduler):
        if config._version == "v2":
            from areal.trainer.rw.rw_engine import RWControllerV2

            return RWControllerV2(train_engine=cls, config=config, scheduler=scheduler)

        from areal.trainer.rw.rw_engine import RWController

        return RWController(train_engine=cls, config=config, scheduler=scheduler)
