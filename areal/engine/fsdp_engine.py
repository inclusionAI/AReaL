from __future__ import annotations

import dataclasses
import gc
import math
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime
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
from areal.engine.fsdp_utils.pipeline_parallel import (
    FSDPPipelinedRunner,
    create_fsdp_runner,
    pipeline_llm_hf,
)
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
    pack_tensor_dict,
    pad_mb_list,
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
        self.pp_group: dist.ProcessGroup | None = None

        # Pipeline parallelism state
        self._pp_runner: FSDPPipelinedRunner | None = None
        self._pp_stages: list | None = None
        self._pp_model_parts: list | None = None
        self._pp_has_first_stage: bool = True
        self._pp_has_last_stage: bool = True

        self.world_size: int
        self.rank: int
        self.dp_head: int
        self.dp_rank: int

        self.is_offload: bool = False
        self._per_layer_optim_wrapper: PerLayerOptimWrapper | None = None
        self.enable_tree_training: bool = self.config.enable_tree_training

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

        # Sequence and model parallel group
        # When PP is enabled, include PP in the model parallel group so that
        # only one rank per DP group is identified as the DP head.
        if self.parallel_helper.pp_enabled:
            self.mp_group = self.world_mesh["pp_sp_tp"].get_group()
        else:
            self.mp_group = self.world_mesh["sp_tp"].get_group()

        # Pipeline parallel group (None if PP disabled)
        self.pp_group = self.parallel_helper.pp_group

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dp_head = dist.get_process_group_ranks(self.mp_group)[0]
        self.dp_rank = dist.get_rank(self.dp_group)

        if self.parallel_helper.pp_enabled:
            self.logger.info(
                f"Pipeline parallelism enabled: pp_size={self.parallel_helper.pp_size}, "
                f"pp_rank={self.parallel_helper.pp_rank}"
            )

        # --- [PP_DIAG] 完整 mesh 拓扑校验 (兼容 DP+TP+PP) ---
        _dp_ranks = dist.get_process_group_ranks(self.dp_group)
        _mp_ranks = dist.get_process_group_ranks(self.mp_group)
        _sp_ranks = dist.get_process_group_ranks(self.sp_group)
        _pp_ranks = dist.get_process_group_ranks(self.pp_group) if self.pp_group else []
        _dp_sp_ranks = dist.get_process_group_ranks(self.world_mesh["dp_sp"].get_group())
        self.logger.info(
            f"[PP_DIAG] Mesh topology: global_rank={self.rank}, "
            f"world_size={self.world_size}, "
            f"dp_rank={self.dp_rank}, dp_head={self.dp_head}"
        )
        self.logger.info(
            f"[PP_DIAG] Group ranks: "
            f"dp={_dp_ranks}, sp={_sp_ranks}, "
            f"mp(model_parallel)={_mp_ranks}, "
            f"dp_sp(fsdp_shard)={_dp_sp_ranks}, "
            f"pp={_pp_ranks if _pp_ranks else 'disabled'}"
        )
        if self.parallel_helper.pp_enabled:
            # 校验: dp_head 在整个 DP group 中应该只有一个
            # 当 PP 开启时, mp_group = pp_sp_tp, 确保每个DP group只有1个dp_head
            self.logger.info(
                f"[PP_DIAG] PP validation: pp_size={self.parallel_helper.pp_size}, "
                f"pp_rank={self.parallel_helper.pp_rank}, "
                f"dp_size={self.parallel_helper.dp_size}, "
                f"tp_size={self.parallel_helper._ps.tp_size}, "
                f"sp_size={self.parallel_helper._ps.cp_size}, "
                f"product(dp*sp*tp*pp)="
                f"{self.parallel_helper.dp_size * self.parallel_helper._ps.cp_size * self.parallel_helper._ps.tp_size * self.parallel_helper.pp_size}, "
                f"world_size={self.world_size}"
            )

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

        # PP requires all ranks to load actual weights (via from_pretrained)
        # because pipeline_module_split does deepcopy. Override config early
        # so _create_llm_actor_or_critic uses from_pretrained, not from_config.
        if self.parallel_helper.pp_enabled and self.config.fsdp.memory_efficient_load:
            self.logger.info(
                "Overriding memory_efficient_load=False for pipeline parallelism."
            )
            # dataclass field override — safe because it's read-only after init
            self.config.fsdp.memory_efficient_load = False

        # --- [PP_DIAG] PP 配置校验 ---
        if self.parallel_helper.pp_enabled:
            self.logger.info(
                f"[PP_DIAG] PP config: schedule={self.config.fsdp.pp_schedule}, "
                f"layers_per_stage={self.config.fsdp.pp_layers_per_stage}, "
                f"first_stage_less={self.config.fsdp.pp_first_stage_less_layers}, "
                f"last_stage_less={self.config.fsdp.pp_last_stage_less_layers}, "
                f"memory_efficient_load={self.config.fsdp.memory_efficient_load}, "
                f"per_layer_optim={self.config.fsdp.per_layer_optim_step}, "
                f"gradient_checkpointing={self.config.gradient_checkpointing}"
            )

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

        # PP requires all ranks to have correct weights before the pipeline split
        # (which does deepcopy). Disable memory_efficient_load for PP so that
        # all ranks load via from_pretrained and the deepcopy picks up real weights.
        if self.parallel_helper.pp_enabled and is_llm_cpu_load:
            self.logger.warning(
                "memory_efficient_load is not compatible with pipeline parallelism. "
                "All ranks will load model weights independently. "
                "Consider disabling memory_efficient_load when using PP."
            )
            is_llm_cpu_load = False

        if self.parallel_helper.pp_enabled and self.config.use_lora:
            raise ValueError(
                "LoRA + Pipeline Parallelism is not currently supported. "
                "Please disable either LoRA or PP."
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

        # NOTE: Apply parallelism — either PP + FSDP2 or plain FSDP2
        if self.parallel_helper.pp_enabled:
            self._apply_pipeline_parallelism()
            # Log memory after PP setup
            import torch

            device = self.device
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            free_gb = free_mem / (1024**3)
            total_gb = total_mem / (1024**3)
            self.logger.info(
                f"[GPU_MEM] After PP setup: allocated={allocated:.2f}GiB, "
                f"reserved={reserved:.2f}GiB, free={free_gb:.2f}GiB, total={total_gb:.2f}GiB"
            )
            for i, mp in enumerate(self._pp_model_parts):
                param_bytes = sum(
                    p.numel() * p.element_size() for p in mp.model.parameters()
                )
                self.logger.info(
                    f"[GPU_MEM] PP stage {i} model_part param memory: {param_bytes / (1024**3):.3f} GiB"
                )
        else:
            parallelize_model(
                self.model,
                config=self.config,
                model_config=self.model_config,
                nd_device_mesh=self.world_mesh,
                parallel_helper=self.parallel_helper,
                cpu_offload=self.cpu_offload,
                wrap_policy=self.config.fsdp.wrap_policy,
                pp_enabled=False,
            )

        if need_broadcast:
            broadcast_tik = time.perf_counter()
            if self.parallel_helper.pp_enabled and self._pp_model_parts is not None:
                # PP: load state into each model part's inner HF model
                for part in self._pp_model_parts:
                    inner_model = part.model if hasattr(part, "model") else part
                    fsdp2_load_full_state_dict(
                        inner_model,
                        full_state,
                        self.cpu_offload,
                        tie_word_embeddings=self.model_config.tie_word_embeddings,
                    )
                self.logger.info(
                    f"Broadcasting model weights (PP) took "
                    f"{time.perf_counter() - broadcast_tik:.2f} seconds"
                )
            else:
                fsdp2_load_full_state_dict(
                    self.model,
                    full_state,
                    self.cpu_offload,
                    tie_word_embeddings=self.model_config.tie_word_embeddings,
                )
                self.logger.info(
                    f"Broadcasting model weights took "
                    f"{time.perf_counter() - broadcast_tik:.2f} seconds"
                )

        self.logger.info(
            f"Applying FSDP2 with N-D parallelism for {time.perf_counter() - tik:.2f} seconds"
        )

        self._create_optimizer(ft_spec)

        if self.config.fsdp.per_layer_optim_step:
            if self.parallel_helper.pp_enabled:
                self.logger.warning(
                    "per_layer_optim_step is not compatible with pipeline parallelism. "
                    "Disabling per_layer_optim_step."
                )
            elif self.optimizer_config.type != "adam":
                raise ValueError(
                    f"per_layer_optim_step only supports 'adam' optimizer, got '{self.optimizer_config.type}'."
                )
            else:
                self._per_layer_optim_wrapper = PerLayerOptimWrapper(
                    model=self.model,
                    optimizer=self.optimizer,
                    device_id=self.device,
                    prefetch_layers=self.config.fsdp.optim_step_prefetch_layers,
                )

        # Create PP runner if pipeline parallelism is enabled
        if self.parallel_helper.pp_enabled:
            self._pp_runner = create_fsdp_runner(
                pp_enabled=True,
                pp_stages=self._pp_stages,
                pp_schedule=self.config.fsdp.pp_schedule,
                pp_group_size=self.parallel_helper.pp_size,
                has_first_stage=self._pp_has_first_stage,
                has_last_stage=self._pp_has_last_stage,
                stage_wrappers=self._pp_model_parts,
            )
            self.logger.info(
                f"PP runner created: schedule={self.config.fsdp.pp_schedule}, "
                f"has_first_stage={self._pp_has_first_stage}, "
                f"has_last_stage={self._pp_has_last_stage}"
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
    def pp_enabled(self) -> bool:
        """Whether pipeline parallelism is enabled."""
        return self.parallel_helper.pp_enabled

    @property
    def pp_has_last_stage(self) -> bool:
        """Whether this rank contains the last PP stage (which produces outputs)."""
        return self._pp_has_last_stage

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
        # PP: also set train/eval mode on all model parts
        if self._pp_model_parts is not None:
            for part in self._pp_model_parts:
                part.train(mode=mode)
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
        if meta.type == "xccl":
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

        # Checkpoint load replaces optimizer state tensor objects, losing
        # pinning and normalization established by PerLayerOptimWrapper.__init__.
        if meta.with_optim and self._per_layer_optim_wrapper is not None:
            self._per_layer_optim_wrapper.refresh_states()

    def optimizer_zero_grad(self):
        assert self.optimizer is not None
        self.optimizer.zero_grad()

    def optimizer_step(self):
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        # Collect parameters from all model parts when PP is enabled
        if self._pp_model_parts is not None:
            all_params = []
            for part in self._pp_model_parts:
                all_params.extend(list(part.parameters()))
        else:
            all_params = list(self.model.parameters())

        # --- [PP_DIAG] 优化器步骤: 参数与梯度校验 ---
        _n_total = len(all_params)
        _n_with_grad = sum(1 for p in all_params if p.grad is not None)
        _n_requires_grad = sum(1 for p in all_params if p.requires_grad)
        _grad_mem = sum(
            p.grad.numel() * p.grad.element_size()
            for p in all_params if p.grad is not None
        ) / (1024**3)
        self.logger.info(
            f"[PP_DIAG] optimizer_step: n_total_params={_n_total}, "
            f"n_requires_grad={_n_requires_grad}, "
            f"n_with_grad={_n_with_grad}, "
            f"grad_mem={_grad_mem:.3f}GiB, "
            f"pp_enabled={self._pp_model_parts is not None}, "
            f"n_model_parts={len(self._pp_model_parts) if self._pp_model_parts else 1}"
        )
        if _n_with_grad == 0:
            self.logger.warning(
                "[PP_DIAG] WARNING: No parameters have gradients! "
                "This likely means backward pass did not flow through this rank."
            )
        elif _n_with_grad < _n_requires_grad:
            self.logger.warning(
                f"[PP_DIAG] WARNING: Only {_n_with_grad}/{_n_requires_grad} "
                f"grad-requiring params have gradients."
            )

        grad_norm = fsdp2_clip_grad_norm(
            all_params,
            max_norm=self.optimizer_config.gradient_clipping,
            fsdp_group=self.world_mesh["dp_sp"].get_group(),
            tp_group=self.world_mesh["tp"].get_group(),
            pp_group=self.pp_group,
            offload_params=self.config.fsdp.offload_params,
        )

        # --- [PP_DIAG] grad_norm ---
        self.logger.info(
            f"[PP_DIAG] optimizer_step: grad_norm={grad_norm:.6f}, "
            f"lr={self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else 'N/A'}"
        )
        if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
            self.logger.warning(
                f"[PP_DIAG] WARNING: grad_norm is {grad_norm}! "
                "Check for NaN/Inf in gradients."
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

        # ── Sync tied weights across PP ranks after successful optimizer step ──
        # When tie_word_embeddings=True and PP splits embed_tokens and lm_head
        # onto different ranks, the deepcopy breaks Python-level weight tying.
        # Without this sync, the two copies diverge every step, eventually
        # corrupting generation quality (SGLang uses embed_tokens as lm_head
        # when PP=1 + tied, so embed_tokens must carry output projection grads).
        if update_successful:
            self._sync_tied_weights_pp()

        current_lr = self.lr_scheduler.get_last_lr()[0]
        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    def _sync_tied_weights_pp(self):
        """Synchronize tied weights (embed_tokens ↔ lm_head) across PP ranks.

        When tie_word_embeddings=True and PP is enabled, the deepcopy-based
        pipeline split breaks Python-level weight tying. After each optimizer
        step, embed_tokens.weight (on first PP rank) and lm_head.weight (on
        last PP rank) diverge because they receive different gradients:
          - embed_tokens gets input embedding gradients only
          - lm_head gets output projection gradients only

        Strategy A (weight sync after optimizer.step()):
          We broadcast lm_head.weight from the last PP rank to the first PP
          rank's embed_tokens.weight. lm_head is canonical because:
          1. Output projection gradients are more critical for generation quality
          2. SGLang (PP=1) uses embed_tokens as lm_head, skipping lm_head.weight
             during load_weights — so embed_tokens must carry the output gradients

        FSDP2 DTensor handling:
          In FSDP2, each parameter is a DTensor. The actual local shard data
          lives in param._local_tensor. We broadcast the local shards directly
          (avoiding full_tensor() which would trigger expensive all-gather).
          This is safe because FSDP2 shards tied weights identically when the
          shapes match (same DTensor spec across PP ranks for same-shaped params).
        """
        # Guard: only needed when PP + tie_word_embeddings
        if self._pp_model_parts is None or self.pp_group is None:
            return
        if not getattr(self.model_config, "tie_word_embeddings", False):
            return

        pp_group = self.pp_group
        pp_global_ranks = dist.get_process_group_ranks(pp_group)
        first_pp_global = pp_global_ranks[0]   # rank owning embed_tokens
        last_pp_global = pp_global_ranks[-1]    # rank owning lm_head
        global_rank = dist.get_rank()

        # If first and last PP rank are the same, no sync needed
        if first_pp_global == last_pp_global:
            return

        import time as _time_sync
        _sync_t0 = _time_sync.perf_counter()

        # ── Locate parameters on this rank ──
        embed_param = None  # embed_tokens.weight on first PP rank
        lm_head_param = None  # lm_head.weight on last PP rank

        if self._pp_has_first_stage:
            # First PP rank: find embed_tokens.weight
            for part in self._pp_model_parts:
                inner = part.model if hasattr(part, "model") else part
                for name, param in inner.named_parameters():
                    if "embed_tokens.weight" in name:
                        embed_param = param
                        break
                if embed_param is not None:
                        break

        if self._pp_has_last_stage:
            # Last PP rank: find lm_head.weight
            for part in self._pp_model_parts:
                inner = part.model if hasattr(part, "model") else part
                for name, param in inner.named_parameters():
                    if "lm_head.weight" in name:
                        lm_head_param = param
                        break
                if lm_head_param is not None:
                        break

        # ── Extract local shard from DTensor ──
        # In FSDP2, parameters are DTensors. _local_tensor gives the actual
        # local shard without triggering an all-gather.
        def _get_local_tensor(param):
            if param is None:
                return None
            if isinstance(param, DTensor):
                return param._local_tensor
            if isinstance(param.data, DTensor):
                return param.data._local_tensor
            return param.data

        embed_local = _get_local_tensor(embed_param)
        lm_head_local = _get_local_tensor(lm_head_param)

        # ── Determine buffer shape for broadcast ──
        # Both ranks need to agree on the buffer shape. Since FSDP2 shards
        # identically for same-shaped params, local shards have the same shape.
        if lm_head_local is not None:
            # Last PP rank: use lm_head's local shard shape
            local_shape = lm_head_local.shape
            local_dtype = lm_head_local.dtype
            local_device = lm_head_local.device
        elif embed_local is not None:
            # First PP rank: use embed_tokens's local shard shape
            local_shape = embed_local.shape
            local_dtype = embed_local.dtype
            local_device = embed_local.device
        else:
            # This rank has neither param (middle PP rank) — still participate
            # in broadcast but with a dummy buffer. For PP=2, this won't happen.
            self.logger.info(
                f"[TIED_SYNC] Rank {global_rank}: neither embed_tokens nor "
                f"lm_head found, skipping sync (middle PP rank)"
            )
            return

        # ── Pre-sync diagnostics ──
        if lm_head_local is not None:
            _lm_norm = lm_head_local.float().norm().item()
            self.logger.info(
                f"[TIED_SYNC] Rank {global_rank} (last PP): "
                f"lm_head local_shard shape={list(local_shape)}, "
                f"norm={_lm_norm:.6f}"
            )
        if embed_local is not None:
            _embed_norm_before = embed_local.float().norm().item()
            self.logger.info(
                f"[TIED_SYNC] Rank {global_rank} (first PP): "
                f"embed_tokens local_shard shape={list(embed_local.shape)}, "
                f"norm_before={_embed_norm_before:.6f}"
            )

        # ── Broadcast lm_head local shard from last PP rank → all PP ranks ──
        sync_buffer = torch.empty(
            local_shape, dtype=local_dtype, device=local_device
        )
        if lm_head_local is not None:
            sync_buffer.copy_(lm_head_local)

        dist.broadcast(sync_buffer, src=last_pp_global, group=pp_group)

        # ── Copy received data into embed_tokens's local shard ──
        if embed_local is not None:
            embed_local.copy_(sync_buffer)
            _embed_norm_after = embed_local.float().norm().item()
            self.logger.info(
                f"[TIED_SYNC] Rank {global_rank} (first PP): "
                f"embed_tokens updated, norm_after={_embed_norm_after:.6f}, "
                f"delta_norm={abs(_embed_norm_after - _embed_norm_before):.6f}"
            )

        del sync_buffer
        _sync_elapsed = _time_sync.perf_counter() - _sync_t0
        self.logger.info(
            f"[TIED_SYNC] Rank {global_rank}: "
            f"tied weight sync completed in {_sync_elapsed:.4f}s"
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
    ) -> list | None:
        # PP path: use pipeline schedule
        if self._pp_runner is not None:
            return self._forward_backward_batch_pp(
                mb_list, process_output_fn, forward_only
            )

        # Non-PP path: sequential microbatch execution
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

        return None

    def _forward_backward_batch_pp(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> list | None:
        """Pipeline-parallel forward/backward using PP schedule.

        Prepares chunked inputs for each microbatch and delegates to the
        PP runner which handles the schedule-based execution (1F1B, etc.).
        """
        assert self._pp_runner is not None

        n_microbatches = len(mb_list)
        if n_microbatches == 0:
            return None

        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        free_mem, total_mem = torch.cuda.mem_get_info(self.device)
        free_gb = free_mem / (1024**3)
        self.logger.info(
            f"[GPU_MEM] PP fwd/bwd START: forward_only={forward_only}, n_mb={n_microbatches}, "
            f"allocated={allocated:.2f}GiB, reserved={reserved:.2f}GiB, free={free_gb:.2f}GiB"
        )

        # Prepare inputs and contexts for all microbatches
        input_ids_chunks: list[torch.Tensor] = []
        contexts: list[dict[str, Any]] = []
        target_chunks: list[torch.Tensor] = []

        for mb_item in mb_list:
            inputs, ctx = self._prepare_mb_inputs(mb_item)
            # For PP, the first stage needs input_ids
            input_ids = inputs.get("input_ids", inputs.get("inputs_embeds", None))
            if input_ids is not None:
                input_ids_chunks.append(input_ids)
            contexts.append(ctx.to_dict())
            # Dummy target — schedule.step() will split the batched target into
            # n_microbatches chunks, so each chunk must have matching batch dim.
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            target_chunks.append(
                torch.zeros(batch_size, device=self.device, dtype=torch.long)
            )

        # --- Pad all microbatch input_ids to uniform sequence length ---
        # PP schedule's step()/eval() calls torch.tensor_split on a single batched
        # tensor, which requires all chunks to have the same shape on non-batch dims.
        # Since pad_mb_list pads each microbatch to its nearest bucket (not global max),
        # different microbatches can have different sequence lengths. We must pad them
        # to a uniform length before torch.cat.
        if input_ids_chunks and len(input_ids_chunks) > 1:
            max_seqlen = max(chunk.shape[-1] for chunk in input_ids_chunks)
            padded_chunks = []
            for chunk in input_ids_chunks:
                seqlen = chunk.shape[-1]
                if seqlen < max_seqlen:
                    pad_size = max_seqlen - seqlen
                    # Pad on the right side of the last dimension with zeros
                    chunk = torch.nn.functional.pad(chunk, (0, pad_size), value=0)
                padded_chunks.append(chunk)
            input_ids_chunks = padded_chunks

        # Pad microbatch count for PP schedule divisibility
        pp_group_size = self._pp_runner.pp_group_size
        remainder = n_microbatches % pp_group_size
        if remainder != 0:
            n_pad = pp_group_size - remainder
            self.logger.info(
                f"PP schedule requires n_microbatches divisible by pp_group_size={pp_group_size}. "
                f"Padding {n_pad} dummy microbatch(es) (original={n_microbatches}, "
                f"padded={n_microbatches + n_pad})."
            )
            if input_ids_chunks:
                dummy_shape = input_ids_chunks[0].shape
                dummy_dtype = input_ids_chunks[0].dtype
            else:
                dummy_shape = (1, 1)
                dummy_dtype = torch.long

            for _ in range(n_pad):
                if input_ids_chunks:
                    input_ids_chunks.append(
                        torch.zeros(dummy_shape, device=self.device, dtype=dummy_dtype)
                    )
                dummy_ctx = contexts[0].copy()
                dummy_ctx["__pp_dummy__"] = True
                contexts.append(dummy_ctx)
                target_chunks.append(
                    torch.zeros(
                        dummy_shape[0] if len(dummy_shape) > 0 else 1,
                        device=self.device,
                        dtype=torch.long,
                    )
                )
            n_microbatches = n_microbatches + n_pad

        # --- [PP_DIAG] 微批次准备完成 ---
        self.logger.info(
            f"[PP_DIAG] PP fwd/bwd prepared: "
            f"n_mb={n_microbatches}, forward_only={forward_only}, "
            f"n_input_chunks={len(input_ids_chunks)}, "
            f"n_contexts={len(contexts)}, "
            f"n_targets={len(target_chunks)}"
        )
        if input_ids_chunks:
            _shapes = [tuple(c.shape) for c in input_ids_chunks]
            # 只打印前3个和最后1个, 避免过长
            if len(_shapes) > 4:
                _shapes_str = f"{_shapes[:3]} ... {_shapes[-1:]}"
            else:
                _shapes_str = str(_shapes)
            self.logger.info(
                f"[PP_DIAG] PP input_shapes: {_shapes_str}, "
                f"dtype={input_ids_chunks[0].dtype}"
            )
            # 校验: 所有 input_ids_chunks 的 seq_len 维度应一致 (padding后)
            _seq_lens = set(c.shape[-1] for c in input_ids_chunks)
            if len(_seq_lens) > 1:
                self.logger.warning(
                    f"[PP_DIAG] WARNING: input_ids_chunks have inconsistent "
                    f"seq_lens after padding: {_seq_lens}"
                )

        with trace_scope("fsdp_engine.pp_forward_backward"):
            if forward_only:
                results = self._pp_runner.run_eval(
                    n_microbatches=n_microbatches,
                    input_ids_chunks=input_ids_chunks,
                    extra_kwargs=None,
                    contexts=contexts,
                    process_output_fn=process_output_fn,
                )
            else:
                results = self._pp_runner.run_train(
                    n_microbatches=n_microbatches,
                    input_ids_chunks=input_ids_chunks,
                    target_chunks=target_chunks,
                    extra_kwargs=None,
                    contexts=contexts,
                    process_output_fn=process_output_fn,
                )

        return results

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
            # Strip PP dummy marker before constructing context
            ctx_dict.pop("__pp_dummy__", None)
            ctx = FSDPTrainContext(**ctx_dict)
            # Truncate logits if PP uniform padding made them longer than original
            original_seqlen = ctx.model_inputs["input_ids"].shape[-1]
            if logits.shape[0] > original_seqlen:
                # --- [PP_DIAG] ---
                self.logger.info(
                    f"[PP_DIAG] train process_output: truncating logits "
                    f"{logits.shape[0]} -> {original_seqlen}, "
                    f"logits_shape={tuple(logits.shape)}"
                )
                logits = logits[:original_seqlen]
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
            ctx_dict.pop("__pp_dummy__", None)
            ctx = FSDPTrainContext(**ctx_dict)
            original_seqlen = ctx.model_inputs["input_ids"].shape[-1]
            if logits.shape[0] > original_seqlen:
                logits = logits[:original_seqlen]
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

        # --- [PP_DIAG] eval losses 聚合前 ---
        self.logger.info(
            f"[PP_DIAG] eval_batch: n_losses={len(losses)}, "
            f"pp_group={'enabled' if self.pp_group else 'disabled'}, "
            f"loss_values={[l.item() if isinstance(l, torch.Tensor) else l for l in losses[:5]]}"
        )

        # Step 4: Aggregate losses (with PP broadcast if enabled)
        return aggregate_eval_losses(losses, self.dp_group, pp_group=self.pp_group)

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
        assert output_seqlens is not None
        batch_size = len(output_seqlens)

        # Step 2: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 3: Forward using process_output_fn callback, collecting results
        outputs: list[torch.Tensor] = []

        def process_output(logits: torch.Tensor, ctx_dict: dict[str, Any]) -> None:
            ctx_dict.pop("__pp_dummy__", None)
            ctx = FSDPTrainContext(**ctx_dict)
            original_seqlen = ctx.model_inputs["input_ids"].shape[-1]
            if logits.shape[0] > original_seqlen:
                logits = logits[:original_seqlen]
            result = self._compute_forward_result(logits, ctx)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # --- PP broadcast for forward_batch results ---
        # In PP mode, only the last stage computes outputs via process_output_fn.
        # Other stages have empty outputs. Broadcast results from last stage to all.
        if self._pp_runner is not None and self.pp_group is not None:
            pp_ranks = dist.get_process_group_ranks(self.pp_group)
            last_pp_global_rank = pp_ranks[-1]
            device = self.device

            # --- [PP_DIAG] forward_batch PP broadcast ---
            self.logger.info(
                f"[PP_DIAG] forward_batch broadcast START: "
                f"pp_ranks={pp_ranks}, last_pp_global={last_pp_global_rank}, "
                f"has_last_stage={self._pp_has_last_stage}, "
                f"n_outputs_local={len(outputs)}"
            )

            if self._pp_has_last_stage:
                # Last stage: send number of results, their sizes, and flat data
                n_results = torch.tensor([len(outputs)], device=device, dtype=torch.long)
                dist.broadcast(n_results, src=last_pp_global_rank, group=self.pp_group)
                if len(outputs) > 0:
                    sizes = torch.tensor([r.numel() for r in outputs], device=device, dtype=torch.long)
                    dist.broadcast(sizes, src=last_pp_global_rank, group=self.pp_group)
                    flat = torch.cat([r.reshape(-1).float() for r in outputs])
                    dist.broadcast(flat, src=last_pp_global_rank, group=self.pp_group)

                # 在 last_stage 分支的3个broadcast全部完成后:
                self.logger.info(
                    f"[PP_DIAG] forward_batch broadcast: last_stage sent "
                    f"n_results={len(outputs)}, "
                    f"flat_size={flat.numel() if len(outputs) > 0 else 0}"
                )
            else:
                # Non-last stages: receive results
                n_results = torch.tensor([0], device=device, dtype=torch.long)
                dist.broadcast(n_results, src=last_pp_global_rank, group=self.pp_group)
                n = n_results.item()
                if n > 0:
                    sizes = torch.empty(n, device=device, dtype=torch.long)
                    dist.broadcast(sizes, src=last_pp_global_rank, group=self.pp_group)
                    total_size = sizes.sum().item()
                    flat = torch.empty(total_size, device=device, dtype=torch.float32)
                    dist.broadcast(flat, src=last_pp_global_rank, group=self.pp_group)
                    outputs = list(flat.split(sizes.tolist()))

                self.logger.info(
                    f"[PP_DIAG] forward_batch broadcast: non-last received "
                    f"n_results={n}, "
                    f"output_shapes={[tuple(o.shape) for o in outputs[:3]]}"
                )

        # Step 4: Aggregate and reorder outputs
        if self.enable_tree_training:
            return merge_packed_tree_results(outputs, batch_size)
        return reorder_and_pad_outputs(outputs, output_seqlens, mb_list, aggregate_fn)

    def export_stats(self) -> dict[str, float]:
        return stats_tracker.export_all(reduce_group=self.data_parallel_group)

    def offload(self) -> None:
        """Offload model memory to CPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """

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

    def _apply_pipeline_parallelism(self) -> None:
        """Apply pipeline parallelism: split model into stages, parallelize each part.

        This method:
        1. Determines the number of layers in the model
        2. Calls pipeline_llm_hf to split the model into PP stages
        3. Applies FSDP2 + TP to each model part
        4. Creates optimizer for all model parts
        """
        num_layers = self.model_config.num_hidden_layers
        is_critic = self.config.is_critic

        pp_config = self.config.fsdp
        pp_mesh = self.world_mesh["pp"]

        self.logger.info(
            f"Applying pipeline parallelism: pp_size={self.parallel_helper.pp_size}, "
            f"num_layers={num_layers}, schedule={pp_config.pp_schedule}"
        )

        # Split model into pipeline stages
        stages, model_parts, has_first, has_last = pipeline_llm_hf(
            model=self.model,
            device=self.device,
            pp_mesh=pp_mesh,
            pp_schedule=pp_config.pp_schedule,
            pp_degree=self.parallel_helper.pp_size,
            num_layers=num_layers,
            is_critic=is_critic,
            pp_layers_per_stage=pp_config.pp_layers_per_stage,
            pp_first_stage_less_layers=pp_config.pp_first_stage_less_layers,
            pp_last_stage_less_layers=pp_config.pp_last_stage_less_layers,
        )

        # Apply TP + FSDP2 to each model part's inner HF model.
        # We parallelize model_part.model (the pruned HuggingFace model) instead of
        # the _HFPipelineStageModule wrapper, because apply_fsdp2 and apply_non_moe_tp
        # expect a HuggingFace model with _no_split_modules, config, etc.
        for i, model_part in enumerate(model_parts):
            inner_hf_model = model_part.model  # The pruned AutoModelForCausalLM
            parallelize_model(
                inner_hf_model,
                config=self.config,
                model_config=self.model_config,
                nd_device_mesh=self.world_mesh,
                parallel_helper=self.parallel_helper,
                cpu_offload=self.cpu_offload,
                wrap_policy=self.config.fsdp.wrap_policy,
                pp_enabled=True,
            )

        # AFTER FSDP wrapping: patch the HF model's forward with the stage-aware forward
        for model_part in model_parts:
            model_part._patch_model_forward()

        self._pp_stages = stages
        self._pp_model_parts = model_parts
        self._pp_has_first_stage = has_first
        self._pp_has_last_stage = has_last

        # Replace self.model with the first model part for compatibility
        # (optimizer creation, state_dict, etc.)
        self.model = model_parts[0] if len(model_parts) == 1 else model_parts[0]

        self.logger.info(
            f"Pipeline parallelism applied: {len(stages)} stages, "
            f"has_first_stage={has_first}, has_last_stage={has_last}"
        )

        # --- [PP_DIAG] 各stage详细信息 ---
        for _i, (_stg, _mp) in enumerate(zip(stages, model_parts)):
            _inner = _mp.model if hasattr(_mp, 'model') else _mp
            _n_params = sum(p.numel() for p in _inner.parameters())
            _param_bytes = sum(p.numel() * p.element_size() for p in _inner.parameters())
            _layer_names = []
            if hasattr(_inner, 'model') and hasattr(_inner.model, 'layers'):
                for _li, _layer in enumerate(_inner.model.layers):
                    if _layer is not None:
                        _layer_names.append(str(_li))
            self.logger.info(
                f"[PP_DIAG] Stage[{_i}]: stage_idx={_stg.stage_index}, "
                f"is_first={_stg.is_first}, is_last={_stg.is_last}, "
                f"n_params={_n_params:,}, param_mem={_param_bytes/(1024**3):.3f}GiB, "
                f"layers_present=[{','.join(_layer_names)}], "
                f"has_embed={_mp.has_embed if hasattr(_mp, 'has_embed') else 'N/A'}, "
                f"has_norm={_mp.has_norm if hasattr(_mp, 'has_norm') else 'N/A'}, "
                f"has_output_head={_mp.has_output_head if hasattr(_mp, 'has_output_head') else 'N/A'}"
            )
        # 校验: 所有stage的参数量总和应接近完整模型参数量
        _total_stage_params = 0
        for _mp in model_parts:
            _inner = _mp.model if hasattr(_mp, 'model') else _mp
            _total_stage_params += sum(p.numel() for p in _inner.parameters())
        _full_model_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(
            f"[PP_DIAG] Stage params total={_total_stage_params:,}, "
            f"full_model_params={_full_model_params:,}, "
            f"ratio={_total_stage_params/_full_model_params:.4f}"
        )

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

        # When PP is enabled, collect parameters from ALL model parts
        # (each rank may hold multiple stages in interleaved schedules).
        if self._pp_model_parts is not None and len(self._pp_model_parts) > 1:
            all_pp_params = []
            for part in self._pp_model_parts:
                all_pp_params.extend(list(part.parameters()))
            _optim_params = all_pp_params
        else:
            _optim_params = self.model.parameters()
        # --- [PP_DIAG] 优化器参数收集 ---
        _optim_param_list = list(_optim_params) if not isinstance(_optim_params, list) else _optim_params
        self.logger.info(
            f"[PP_DIAG] _create_optimizer: "
            f"n_optim_params={len(_optim_param_list)}, "
            f"n_model_parts={len(self._pp_model_parts) if self._pp_model_parts else 1}, "
            f"optim_type={self.optimizer_config.type}"
        )
        # 注意: 如果 _optim_params 是 generator，上面 list() 会消耗它
        # 所以需要用 _optim_param_list 替代后续的 _optim_params 使用
        _optim_params = _optim_param_list

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
                _optim_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
                # VLM with tensor parallelism is incompatible with fused AdamW
                fused=not (self.is_vision_model and self.parallel_helper.tp_enabled),
            )
        elif self.optimizer_config.type == "adam_bf16":
            self.optimizer = AnyPrecisionAdamW(
                _optim_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
                momentum_dtype="bfloat16",
                variance_dtype="bfloat16",
            )
        else:
            self.optimizer = torch.optim.SGD(
                _optim_params,
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

    def _map_weight_update_name(
        self, name: str, meta: WeightUpdateMeta
    ) -> str:
        """Map HF parameter name to inference engine parameter name."""
        if self.is_vision_model and is_qwen_vl_model(self.model_config.model_type):
            if meta.gen_allocation.backend == "sglang":
                new_name = name.replace("language_model.", "", 1)
                if new_name.startswith("model.visual."):
                    new_name = new_name.replace("model.", "", 1)
                return new_name
            new_name = name.replace("model.", "", 1)
            if new_name.startswith("language_model."):
                new_name = new_name.replace(
                    "language_model.", "language_model.model.", 1
                )
            elif new_name.startswith("lm_head."):
                new_name = f"language_model.{new_name}"
            return new_name
        elif self.is_vision_model and is_gemma3_model(self.model_config.model_type):
            new_name = name.replace("model.", "", 1)
            if new_name.startswith("language_model."):
                new_name = new_name.replace(
                    "language_model.", "language_model.model.", 1
                )
            elif new_name.startswith("lm_head."):
                new_name = new_name.replace(
                    "lm_head.", "language_model.lm_head.", 1
                )
            return new_name
        return name

    def _get_model_name_parameters(
        self, meta: WeightUpdateMeta
    ) -> Iterator[tuple[str, nn.Parameter]]:
        # ---- PP-aware: iterate ALL local model parts ----
        if self._pp_model_parts is not None and len(self._pp_model_parts) > 1:
            seen_names: set[str] = set()
            for model_part in self._pp_model_parts:
                inner = model_part.model if hasattr(model_part, "model") else model_part
                for name, param in inner.named_parameters():
                    if name in seen_names:
                        continue
                    seen_names.add(name)
                    yield self._map_weight_update_name(name, meta), param
            # --- [PP_DIAG] PP参数遍历 (首次调用时) ---
            self.logger.info(
                f"[PP_DIAG] _get_model_name_parameters PP: "
                f"n_model_parts={len(self._pp_model_parts)}, "
                f"n_unique_params={len(seen_names)}, "
                f"sample_names={list(seen_names)[:3]}"
            )
            return

        # ---- Original non-PP path (unchanged) ----
        name_params_iterator = self.model.named_parameters()
        if self.is_vision_model and is_qwen_vl_model(self.model_config.model_type):
            for name, value in name_params_iterator:
                if meta.gen_allocation.backend == "sglang":
                    new_name = name.replace("language_model.", "", 1)
                    if new_name.startswith("model.visual."):
                        new_name = new_name.replace("model.", "", 1)
                    yield new_name, value
                    continue
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

    def _gather_pp_full_state_dict(
        self,
        options: StateDictOptions | None = None,
    ) -> dict[str, torch.Tensor]:
        """Gather full state dict from ALL PP ranks via NCCL collective.

        Protocol:
          1. Each PP rank does FSDP all-gather for its local model parts.
          2. Metadata exchange via dist.gather_object (param names, shapes, dtypes).
          3. Each rank flattens local tensors into a contiguous byte buffer.
          4. Size exchange via dist.all_gather, then data exchange via dist.all_gather
             (collective, avoids NCCL P2P communicator initialization issues).
          5. pp_rank=0 unflattens received bytes into tensors using the metadata.

        Args:
            options: StateDictOptions for get_model_state_dict. Defaults to
                     full_state_dict=True, cpu_offload=False.
                     NOTE: cpu_offload MUST be False here. When cpu_offload=True,
                     PyTorch's _iterate_state_dict uses ranks_only=(0,) which only
                     returns results on global rank 0 and discards results on all
                     other ranks. This causes PP ranks != global rank 0 to lose
                     their local parameters.

        Returns:
            On pp_rank=0: complete merged state dict (CPU tensors).
            On other pp_ranks: local (partial) state dict (CPU tensors).
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions
        import logging

        logger = logging.getLogger("FSDPEngine")

        if options is None:
            # CRITICAL: cpu_offload must be False!
            # When cpu_offload=True, PyTorch internally sets ranks_only=(0,) in
            # _maybe_full_or_cpu_state_dict(), which means only global rank 0
            # receives the state dict; all other ranks get an empty dict {}.
            # In a PP setup, this causes non-global-rank-0 PP ranks to contribute
            # 0 parameters, resulting in an incomplete merged state dict.
            options = StateDictOptions(full_state_dict=True, cpu_offload=False)

        pp_rank = self.parallel_helper.pp_rank
        pp_size = self.parallel_helper.pp_size
        pp_group = self.pp_group
        device = self.device
        global_rank = dist.get_rank()

        # ── Step 1: FSDP all-gather for local model parts ──────────────────────
        logger.info(
            f"[FSDPEngine Rank {global_rank}] _gather_pp_full_state_dict: "
            f"Step 1 start (pp_rank={pp_rank}, pp_size={pp_size})"
        )
        local_state_dict: dict[str, torch.Tensor] = {}
        for i, part in enumerate(self._pp_model_parts):
            inner = part.model if hasattr(part, "model") else part
            part_state = get_model_state_dict(inner, options=options)
            logger.info(
                f"[FSDPEngine Rank {global_rank}] Part {i}: "
                f"got {len(part_state)} parameters from get_model_state_dict"
            )
            local_state_dict.update(part_state)

        logger.info(
            f"[FSDPEngine Rank {global_rank}] Step 1 done: "
            f"local_state_dict has {len(local_state_dict)} parameters"
        )

        # --- [PP_DIAG] Step 1 参数名抽样 ---
        _sample_keys = list(local_state_dict.keys())[:5]
        logger.info(
            f"[PP_DIAG][Rank {global_rank}] gather Step 1: "
            f"n_local_params={len(local_state_dict)}, "
            f"sample_keys={_sample_keys}, "
            f"dtypes={set(str(v.dtype) for v in list(local_state_dict.values())[:5])}"
        )

        # Synchronize CUDA after get_model_state_dict ─────────────────
        # get_model_state_dict may use internal NCCL streams (for FSDP unshard).
        # Without synchronization, subsequent default-stream operations might read
        # stale tensor data. This is critical when DP > 1.
        current_platform.synchronize()

        if pp_size <= 1:
            return local_state_dict

        # ── Step 2: Metadata exchange ──────────────────────────────────────────
        param_names = list(local_state_dict.keys())
        local_meta = [
            (name, tuple(local_state_dict[name].shape), str(local_state_dict[name].dtype))
            for name in param_names
        ]

        # gather_object dst uses global rank; get the global rank of pp_rank=0
        pp_global_ranks = dist.get_process_group_ranks(pp_group)
        pp_rank0_global = pp_global_ranks[0]

        gathered_meta: list[list[tuple] | None] | None
        if pp_rank == 0:
            gathered_meta = [None] * pp_size
            dist.gather_object(local_meta, gathered_meta, dst=pp_rank0_global, group=pp_group)
        else:
            dist.gather_object(local_meta, None, dst=pp_rank0_global, group=pp_group)
            gathered_meta = None

        logger.info(f"[FSDPEngine Rank {global_rank}] Step 2 done: metadata exchanged")

        # ── Step 3: Flatten local tensors into a contiguous byte buffer ────────
        byte_chunks: list[torch.Tensor] = []
        for name in param_names:
            t = local_state_dict[name]
            if t.device.type == "cpu":
                t = t.to(device, non_blocking=False)
            # reshape(-1) first to flatten to 1D, then view as uint8.
            # Without reshape(-1), a 2D weight [768,768] becomes 2D uint8 [768,1536]
            # while a 1D bias [768] becomes 1D uint8 [1536], and torch.cat fails.
            byte_chunks.append(t.contiguous().reshape(-1).view(torch.uint8))

        if byte_chunks:
            flat_bytes = torch.cat(byte_chunks)
        else:
            flat_bytes = torch.empty(0, dtype=torch.uint8, device=device)

        local_nbytes = flat_bytes.numel()
        logger.info(
            f"[FSDPEngine Rank {global_rank}] Step 3 done: "
            f"flat_bytes = {local_nbytes} bytes ({local_nbytes / 1024 / 1024:.2f} MB)"
        )

        # ── Step 3.5: Exchange buffer sizes ────────────────────────────────────
        size_tensor = torch.tensor([local_nbytes], dtype=torch.long, device=device)
        all_sizes = [
            torch.zeros(1, dtype=torch.long, device=device) for _ in range(pp_size)
        ]
        dist.all_gather(all_sizes, size_tensor, group=pp_group)

        size_list = [s.item() for s in all_sizes]
        logger.info(
            f"[FSDPEngine Rank {global_rank}] Step 3.5 done: "
            f"all_sizes = {size_list}"
        )

        # ── Step 4: Collective data exchange via all_gather ────────────────────
        # Using all_gather (collective) instead of send/recv (P2P) to avoid
        # NCCL P2P communicator lazy initialization deadlock.
        max_nbytes = max(size_list)

        if max_nbytes == 0:
            logger.warning(
                f"[FSDPEngine Rank {global_rank}] "
                "All PP ranks have empty state dicts, skipping gather"
            )
            return local_state_dict

        # Pad local buffer to uniform max size (required by all_gather)
        padded = torch.zeros(max_nbytes, dtype=torch.uint8, device=device)
        if local_nbytes > 0:
            padded[:local_nbytes] = flat_bytes
        del flat_bytes  # Free GPU memory

        # all_gather: every rank gets all padded buffers
        gathered_buffers = [
            torch.empty(max_nbytes, dtype=torch.uint8, device=device)
            for _ in range(pp_size)
        ]

        import time as _time_s4
        _s4_t0 = _time_s4.perf_counter()
        logger.info(
            f"[PP_DIAG][Rank {global_rank}] gather Step 4: all_gather start, "
            f"buffer={max_nbytes/(1024**2):.1f}MB, "
            f"all_sizes_MB=[{', '.join(f'{s/(1024**2):.1f}' for s in size_list)}]"
        )
        dist.all_gather(gathered_buffers, padded, group=pp_group)

        # Synchronize CUDA streams before reading buffers ─
        # dist.all_gather enqueues the NCCL kernel on the NCCL stream but
        # returns to the CPU before the GPU-side transfer finishes.
        # Without synchronization, subsequent .view()/.cpu() on the default
        # CUDA stream may read stale/partially-written gathered_buffers,
        # causing intermittent weight corruption in the inference engine.
        current_platform.synchronize()

        del padded
        _s4_elapsed = _time_s4.perf_counter() - _s4_t0

        logger.info(
            f"[PP_DIAG][Rank {global_rank}] gather Step 4 done: "
            f"all_gather took {_s4_elapsed:.3f}s, "
            f"effective_bw={max_nbytes * pp_size / max(_s4_elapsed, 1e-9) / (1024**3):.2f}GB/s"
        )

        # ── Step 5: pp_rank=0 unflattens received bytes ───────────────────────
        if pp_rank == 0:
            merged = dict(local_state_dict)

            for src_pp in range(1, pp_size):
                remote_nbytes = size_list[src_pp]
                if remote_nbytes == 0:
                    logger.warning(
                        f"[FSDPEngine Rank {global_rank}] "
                        f"pp_rank {src_pp} has 0 bytes, skipping unflatten"
                    )
                    continue

                recv_data = gathered_buffers[src_pp][:remote_nbytes]

                # Walk through metadata and slice the byte buffer
                offset = 0
                param_count = 0
                for name, shape, dtype_str in gathered_meta[src_pp]:
                    dtype = getattr(torch, dtype_str.replace("torch.", ""))
                    elem_size = torch.tensor([], dtype=dtype).element_size()
                    numel = 1
                    for s in shape:
                        numel *= s
                    nbytes = numel * elem_size

                    tensor = (
                        recv_data[offset : offset + nbytes]
                        .view(dtype)
                        .reshape(shape)
                        .cpu()
                    )
                    offset += nbytes
                    merged[name] = tensor
                    param_count += 1

                logger.info(
                    f"[FSDPEngine Rank {global_rank}] "
                    f"Unflattened {param_count} params from pp_rank {src_pp}"
                )

            del gathered_buffers

            # CPU offload the merged state dict (since we disabled cpu_offload
            # in StateDictOptions, tensors are still on GPU at this point)
            for key in merged:
                if merged[key].device.type != "cpu":
                    merged[key] = merged[key].cpu()

            # ── Reconcile tied weights (safety net for weight update path) ──
            # Even though _sync_tied_weights_pp() keeps weights aligned during
            # training, this provides a second line of defense: ensure the
            # merged state dict sent to SGLang has consistent tied weights.
            #
            # SGLang behavior:
            #   PP=1 + tied: lm_head = embed_tokens (same object), load_weights
            #     skips "lm_head.weight" → only embed_tokens.weight is loaded.
            #   PP>1: lm_head = ParallelLMHead (separate), both weights loaded
            #     independently.
            #
            # Strategy: copy lm_head.weight → embed_tokens.weight (lm_head is
            # canonical because it carries output projection gradients).
            # Also ensure lm_head.weight exists for SGLang PP>1 compatibility.
            _tie_word_embeddings = getattr(self.model_config, "tie_word_embeddings", False)
            if _tie_word_embeddings:
                _embed_key = None
                _lm_head_key = None
                for k in merged:
                    if "embed_tokens.weight" in k and _embed_key is None:
                        _embed_key = k
                    if "lm_head.weight" in k and _lm_head_key is None:
                        _lm_head_key = k

                if _embed_key and _lm_head_key:
                    # Both exist: check divergence and reconcile
                    _embed_w = merged[_embed_key]
                    _lm_head_w = merged[_lm_head_key]
                    _divergence_norm = (_embed_w - _lm_head_w).float().norm().item()
                    _embed_norm = _embed_w.float().norm().item()
                    _lm_head_norm = _lm_head_w.float().norm().item()
                    _cos_sim = (
                        torch.nn.functional.cosine_similarity(
                            _embed_w.float().reshape(1, -1),
                            _lm_head_w.float().reshape(1, -1),
                        ).item()
                    )
                    logger.info(
                        f"[TIED_RECONCILE][Rank {global_rank}] "
                        f"embed_key={_embed_key}, lm_head_key={_lm_head_key}, "
                        f"embed_norm={_embed_norm:.6f}, lm_head_norm={_lm_head_norm:.6f}, "
                        f"divergence_norm={_divergence_norm:.6f}, "
                        f"cosine_sim={_cos_sim:.6f}"
                    )
                    if _divergence_norm > 1e-6:
                        logger.warning(
                            f"[TIED_RECONCILE][Rank {global_rank}] "
                            f"Tied weights diverged (norm={_divergence_norm:.6f})! "
                            f"Overwriting embed_tokens with lm_head (canonical)."
                        )
                    # Always reconcile: lm_head → embed_tokens
                    merged[_embed_key] = _lm_head_w.clone()
                    logger.info(
                        f"[TIED_RECONCILE][Rank {global_rank}] "
                        f"Reconciled: {_embed_key} = {_lm_head_key}"
                    )

                elif _lm_head_key and not _embed_key:
                    # Only lm_head exists (unusual): create embed_tokens entry
                    _expected_embed_key = "model.embed_tokens.weight"
                    merged[_expected_embed_key] = merged[_lm_head_key].clone()
                    logger.info(
                        f"[TIED_RECONCILE][Rank {global_rank}] "
                        f"Created {_expected_embed_key} from {_lm_head_key} "
                        f"(lm_head only, embed missing)"
                    )

                elif _embed_key and not _lm_head_key:
                    # Only embed_tokens exists: create lm_head entry
                    # This ensures SGLang PP>1 can load lm_head.weight
                    _expected_lm_key = "lm_head.weight"
                    merged[_expected_lm_key] = merged[_embed_key].clone()
                    logger.info(
                        f"[TIED_RECONCILE][Rank {global_rank}] "
                        f"Created {_expected_lm_key} from {_embed_key} "
                        f"(embed only, lm_head missing)"
                    )
                else:
                    logger.warning(
                        f"[TIED_RECONCILE][Rank {global_rank}] "
                        f"tie_word_embeddings=True but neither embed_tokens.weight "
                        f"nor lm_head.weight found in merged state dict! "
                        f"Keys sample: {list(merged.keys())[:10]}"
                    )

            logger.info(
                f"[FSDPEngine Rank {global_rank}] Step 5 done: "
                f"merged state dict has {len(merged)} parameters"
            )

            # --- [PP_DIAG] 完整性校验 ---
            _n_unique = len(set(merged.keys()))
            logger.info(
                f"[PP_DIAG][Rank {global_rank}] gather Step 5: "
                f"merged n_params={len(merged)}, n_unique={_n_unique}, "
                f"duplicates={len(merged) - _n_unique}"
            )

            return merged

        else:
            del gathered_buffers
            # CPU offload local state dict
            for key in local_state_dict:
                if local_state_dict[key].device.type != "cpu":
                    local_state_dict[key] = local_state_dict[key].cpu()
            return local_state_dict

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

    @trace_perf("fsdp_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        """Broadcast parameters with single-pending-bucket pipelining.

        PP-aware: when PP is enabled, gathers parameters from all PP ranks
        via _gather_pp_full_state_dict before broadcasting to inference engine.
        """
        meta.nccl_master_address = self.weight_update_master_addr
        meta.nccl_master_port = self.weight_update_master_port
        meta.nccl_group_name = self.weight_update_group_name

        main_rank = dist.get_rank() == 0
        if main_rank:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

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

        # ================================================================
        # PP-aware path: gather full state_dict from all PP ranks via NCCL,
        # then rank 0 broadcasts the complete model to inference engine.
        # ================================================================
        if self._pp_model_parts is not None and self.pp_group is not None:
            # --- [PP_DIAG] 权重同步 PP 路径 ---
            self.logger.info(
                f"[PP_DIAG] weight_update PP path: "
                f"main_rank={main_rank}, "
                f"n_model_parts={len(self._pp_model_parts)}, "
                f"pp_size={self.parallel_helper.pp_size}"
            )
            import time as _time_wu
            _wu_t0 = _time_wu.perf_counter()

            options = StateDictOptions(full_state_dict=True, cpu_offload=False)
            full_state_dict = self._gather_pp_full_state_dict(options)

            _wu_elapsed = _time_wu.perf_counter() - _wu_t0
            self.logger.info(
                f"[PP_DIAG] weight_update: gather took {_wu_elapsed:.3f}s, "
                f"state_dict_keys={len(full_state_dict)}"
            )

            if main_rank:
                # Apply name mapping and broadcast
                mapped_params = [
                    (self._map_weight_update_name(name, meta), tensor)
                    for name, tensor in full_state_dict.items()
                    if not self.config.use_lora or "lora" in name.lower()
                ]
                # --- [PP_DIAG] ---
                self.logger.info(
                    f"[PP_DIAG] weight_update: broadcasting "
                    f"{len(mapped_params)} mapped params to inference engine"
                )
                try:
                    for name, tensor in mapped_params:
                        # _gather_pp_full_state_dict returns CPU tensors
                        # (cpu_offload=True), but downstream dist.broadcast
                        # in _update_bucket_weights_from_distributed_async
                        # requires GPU tensors for the NCCL backend.
                        if tensor.device.type == "cpu":
                            tensor = tensor.to(self.device)

                        tensor_size = tensor.numel() * tensor.element_size()
                        bucket_overflow = (
                            buffer_size > 0
                            and tensor_size + buffer_size > weight_chunked_mem_size
                        )
                        if bucket_overflow:
                            if pending_bucket is not None:
                                self._wait_pending_weight_update_bucket(pending_bucket)
                                pending_bucket = None
                            pending_bucket = (
                                self._update_bucket_weights_from_distributed_async(
                                    meta, named_tensors, stream=broadcast_stream,
                                )
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
                    if pending_bucket is not None:
                        self._wait_pending_weight_update_bucket(pending_bucket)

        # ================================================================
        # Original non-PP path (unchanged)
        # ================================================================
        else:
            if self.config.use_lora:
                param_iterator = (
                    (name, param)
                    for name, param in self._get_model_name_parameters(meta)
                    if param.requires_grad
                )
            else:
                param_iterator = self._get_model_name_parameters(meta)

            try:
                for name, param in param_iterator:
                    tensor = self._get_full_tensor(param)
                    if not main_rank:
                        continue

                    tensor_size = tensor.numel() * tensor.element_size()
                    bucket_overflow = (
                        buffer_size > 0
                        and tensor_size + buffer_size > weight_chunked_mem_size
                    )
                    if bucket_overflow:
                        if pending_bucket is not None:
                            self._wait_pending_weight_update_bucket(pending_bucket)
                            pending_bucket = None
                        pending_bucket = (
                            self._update_bucket_weights_from_distributed_async(
                                meta, named_tensors, stream=broadcast_stream,
                            )
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

        dist.barrier(group=self.cpu_group)
        if main_rank:
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
        """Save model in HuggingFace format. PP-aware."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        os.makedirs(path, exist_ok=True)

        if self._pp_model_parts is not None and self.pp_group is not None:
            # PP mode: gather from ALL PP ranks
            options = StateDictOptions(full_state_dict=True, cpu_offload=False)
            state_dict = self._gather_pp_full_state_dict(options)
        elif self._pp_model_parts is not None:
            # PP model parts on single rank (pp_size=1 edge case)
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = {}
            for part in self._pp_model_parts:
                inner = part.model if hasattr(part, "model") else part
                part_state = get_model_state_dict(inner, options=options)
                state_dict.update(part_state)
        else:
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.model, options=options)

        if dist.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            save_model = self.model
            if self._pp_model_parts is not None and hasattr(
                self._pp_model_parts[0], "model"
            ):
                save_model = self._pp_model_parts[0].model
            save_model.save_pretrained(path, state_dict=state_dict)
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

        if self._pp_model_parts is not None:
            # PP: load into each model part's inner HF model
            for part in self._pp_model_parts:
                inner_model = part.model if hasattr(part, "model") else part
                fsdp2_load_full_state_dict(
                    inner_model,
                    full_state,
                    self.cpu_offload,
                    tie_word_embeddings=self.model_config.tie_word_embeddings,
                )
        else:
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

        dcp_model = self.model
        if self._pp_model_parts is not None:
            self.logger.warning(
                "DCP checkpoint saving with PP currently only saves the first model part. "
                "Consider using HF format for PP checkpoints."
            )
        dcp_state = DCPState(dcp_model, self.optimizer if with_optim else None)
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

        loss_scale = loss_weight_fn(ctx.mb_input) / total_loss_weight * loss_multiplier
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
        from areal.trainer.rw.rw_engine import RWController

        return RWController(train_engine=cls, config=config, scheduler=scheduler)
