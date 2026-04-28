from __future__ import annotations

import dataclasses
import functools
import gc
import math
import os
import random
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any

import mbridge
import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge as MegatronBridgeAutoBridge
from megatron.bridge.peft.lora import LoRA as MegatronBridgeLoRA
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_model_config
from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PretrainedConfig

import areal.models.mcore.bailing_moe_bridge  # noqa: F401  # register bridge
from areal.api import (
    FinetuneSpec,
    InferenceEngine,
    MegatronParallelStrategy,
    ParallelStrategy,
    ParamSpec,
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import MicroBatchSpec, PerfTracerConfig, TrainEngineConfig
from areal.api.io_struct import DeviceRuntimeInfo
from areal.engine.core import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.engine.core.distributed import init_custom_process_group
from areal.engine.core.model import disable_dropout_in_model
from areal.engine.megatron_utils.checkpointer import MegatronCheckpointManager
from areal.engine.megatron_utils.deterministic import set_deterministic_algorithms
from areal.engine.megatron_utils.fp8 import FP8BlockwiseTensorHelper
from areal.engine.megatron_utils.megatron import (
    all_gather_param,
    convert_to_hf,
    get_named_parameters,
    remove_padding,
)
from areal.engine.megatron_utils.megatron_lora import get_vllm_lora_target_modules
from areal.engine.megatron_utils.packed_context_parallel import (
    packed_context_parallel_forward,
)
from areal.engine.megatron_utils.pipeline_parallel import (
    configure_pipeline_layer_splits,
)
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.platforms import current_platform
from areal.models.mcore.hf_load import load_weights_from_hf_with_mbridge_fast
from areal.models.mcore.hf_save import save_weights_to_hf_with_mbridge_fast
from areal.models.mcore.registry import make_hf_and_mcore_config, make_mcore_model
from areal.models.tree_attn.functional import (
    _gather_packed_tree_logprobs,
    gather_packed_tree_logprobs_entropy,
    gather_packed_tree_vocab_stats,
    merge_packed_tree_results,
)
from areal.models.tree_attn.module import (
    build_tree_attn_kwargs,
    patch_bridge_for_tree_training,
)
from areal.models.tree_attn.tree import build_packed_tree_batch
from areal.utils import logging, name_resolve, names, perf_tracer, stats_tracker
from areal.utils.constants import (
    DEFAULT_VECTORIZED_ALIGNMENT_BYTES,
    DIST_GROUP_DEFAULT_TIMEOUT,
)
from areal.utils.data import (
    MicroBatchItem,
    MicroBatchList,
    amend_position_ids,
    broadcast_tensor,
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unpad_logits,
)
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.lock import DistributedLock
from areal.utils.network import find_free_ports, format_host_for_url, gethostip
from areal.utils.offload import is_tms_enabled, torch_memory_saver
from areal.utils.perf_tracer import trace_perf, trace_scope
from areal.utils.seeding import get_seed

if TYPE_CHECKING:
    from areal.api import Scheduler
    from areal.api.cli_args import PPOActorConfig, PPOCriticConfig


class _MegatronModelList(list):
    """List wrapper that exposes module-like helpers for Megatron model chunks."""

    def forward(self, *args, **kwargs) -> Any:
        if len(self) == 1:
            return self[0](*args, **kwargs)
        raise RuntimeError(
            "Direct forward calls are only supported for single-chunk model list."
        )

    def named_parameters(self, *args, **kwargs) -> Iterator[tuple[str, nn.Parameter]]:
        for module in self:
            yield from module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_parameters(*args, **kwargs):
            yield parameter


class MegatronEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.logger = logging.getLogger("[MegatronEngine]")
        self.hf_config: PretrainedConfig
        self.tf_config: TransformerConfig
        self.model: _MegatronModelList | None = None
        self.dtype = getattr(torch, self.config.dtype)
        self.device = None
        self.optimizer_config = config.optimizer
        self.mcore_config = config.megatron
        self.parallel_strategy = None
        self.optimizer = None
        self.lr_scheduler = None
        self.bridge = None
        self.process_group_initialized = False
        self._initialized = False
        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None
        self.weight_update_group_initialized: bool = False
        self.weight_update_group_name: str
        self.weight_update_master_addr: str
        self.weight_update_master_port: int
        self._version: int = 0
        self.rank: int | None = None
        self.is_pp_head: bool
        self.world_size: int | None = None
        self.rank_generator: mpu.RankGenerator | None = None
        self.checkpointer: MegatronCheckpointManager | None = None
        self.lr_scheduler: OptimizerParamScheduler | None = None
        self.seed: int = 0
        self.own_global_group: bool = False
        self.is_offload: bool = False
        self.enable_tree_training: bool = self.config.enable_tree_training
        # FP8 configuration
        self.fp8_config = self.mcore_config.fp8_config
        self.enable_fp8: bool = self.fp8_config is not None
        self.fp8_direct_convert: bool = (
            self.fp8_config.direct_convert if self.enable_fp8 else False
        )
        self.quantization_config: dict[str, int | str | list[str]] | None = None
        self.bridge_cls: str = getattr(self.mcore_config, "bridge_type", "mbridge")
        self.bridge_lora: MegatronBridgeLoRA | None = None

        # MTP (Multi-Token Prediction) configuration
        self.enable_mtp_training: bool = getattr(
            self.config, "enable_mtp_training", False
        )
        self.mtp_num_layers: int = getattr(self.config, "mtp_num_layers", 0)
        self.mtp_loss_scaling_factor: float = getattr(
            self.config, "mtp_loss_scaling_factor", 0.1
        )
        self.mtp_detach_heads: bool = getattr(self.config, "mtp_detach_heads", True)
        self._mtp_loss_value: float = 0.0
        self._mtp_layers_verified: bool = False
        self._mtp_tensor_update_warned: bool = False
        if self.enable_mtp_training:
            self.logger.info(
                f"[MTPTrain] MTP online training ENABLED: "
                f"num_layers={self.mtp_num_layers}, "
                f"loss_scaling_factor={self.mtp_loss_scaling_factor}, "
                f"detach_heads={self.mtp_detach_heads}"
            )
            try:
                import megatron.core.transformer.multi_token_prediction  # noqa: F401

                self.logger.info(
                    "[MTPTrain] Verified megatron-core MTP module available. "
                    "Gradient isolation is handled by AReaL monkey-patches: "
                    "MTPLossAutoScaler passthrough (backbone), direct output_layer call (lm_head), "
                    "decoder_input.detach (embedding) when mtp_detach_heads=True."
                )
            except ImportError:
                self.logger.error(
                    "[MTPTrain] megatron-core MTP module not found! "
                    "MTP training requires megatron-core >= 0.12.0. "
                    "Gradient isolation will NOT be applied, which corrupts RL training."
                )

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        import time as _time

        _t0 = _time.time()
        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()
        self.parallel_strategy = self._make_parallel_strategy(parallel_strategy)
        backend = current_platform.communication_backend

        if not dist.is_initialized():
            self.logger.info(
                "[DiagInit] create_process_group: calling dist.init_process_group "
                f"(backend={backend}, RANK={os.environ.get('RANK')}, "
                f"WORLD_SIZE={os.environ.get('WORLD_SIZE')})..."
            )
            _t1 = _time.time()
            dist.init_process_group(
                backend=backend,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            self.logger.info(
                f"[DiagInit] create_process_group: dist.init_process_group done in "
                f"{_time.time() - _t1:.2f}s"
            )

            vpp_size = self.parallel_strategy.virtual_pipeline_parallel_size
            self.logger.info(
                f"[DiagInit] create_process_group: calling mpu.initialize_model_parallel "
                f"(tp={self.parallel_strategy.tensor_parallel_size}, "
                f"pp={self.parallel_strategy.pipeline_parallel_size}, "
                f"cp={self.parallel_strategy.context_parallel_size}, "
                f"ep={self.parallel_strategy.expert_parallel_size}, "
                f"etp={self.parallel_strategy.expert_tensor_parallel_size}, "
                f"vpp={vpp_size})..."
            )
            _t2 = _time.time()
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.parallel_strategy.tensor_parallel_size,
                pipeline_model_parallel_size=self.parallel_strategy.pipeline_parallel_size,
                virtual_pipeline_model_parallel_size=vpp_size if vpp_size > 1 else None,
                use_sharp=False,
                order="tp-cp-ep-dp-pp",
                context_parallel_size=self.parallel_strategy.context_parallel_size,
                expert_model_parallel_size=self.parallel_strategy.expert_parallel_size,
                expert_tensor_parallel_size=self.parallel_strategy.expert_tensor_parallel_size,
                distributed_timeout_minutes=int(
                    DIST_GROUP_DEFAULT_TIMEOUT.seconds / 60
                ),
            )
            self.logger.info(
                f"[DiagInit] create_process_group: mpu.initialize_model_parallel done in "
                f"{_time.time() - _t2:.2f}s"
            )

            tensor_parallel.model_parallel_cuda_manual_seed(self.seed)
            self.own_global_group = True
        else:
            self.logger.info(
                "[DiagInit] create_process_group: dist already initialized, skipping init_process_group"
            )

        self.logger = logging.getLogger(f"[MegatronEngine Rank {dist.get_rank()}]")
        self._context_and_model_parallel_group = None
        self._init_context_and_model_parallel_group()
        self._cpu_group = dist.new_group(
            timeout=DIST_GROUP_DEFAULT_TIMEOUT, backend="gloo"
        )
        self.process_group_initialized = True
        self.logger.info(
            f"[DiagInit] create_process_group: COMPLETED in {_time.time() - _t0:.2f}s"
        )

    def _apply_megatron_bridge_lora(self) -> None:
        assert self.model is not None, "Model must be initialized before applying LoRA."
        assert self.bridge_cls == "megatron-bridge"

        target_modules = list(self.config.target_modules or [])
        if not target_modules or "all-linear" in target_modules:
            # Expand all-linear to explicit Megatron-Bridge linear module targets.
            target_modules = [
                "linear_qkv",
                "linear_proj",
                "linear_fc1",
                "linear_fc2",
            ]
        self.bridge_lora = MegatronBridgeLoRA(
            target_modules=target_modules,
            dim=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=0.0,
        )
        self.model = _MegatronModelList(self.bridge_lora(self.model, training=True))
        self.bridge_lora.set_params_to_save(self.model)

        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        self.logger.info(
            "Applied Megatron Bridge LoRA: target_modules=%s, rank=%s, alpha=%s, trainable=%s/%s (%.4f%%)",
            target_modules,
            self.config.lora_rank,
            self.config.lora_alpha,
            trainable_params,
            total_params,
            100.0 * trainable_params / max(total_params, 1),
        )

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        import time as _time

        _t0 = _time.time()
        self.logger.info("[DiagInit] initialize: ENTERED")

        try:
            self.seed = get_seed()
        except ValueError:
            self.logger.warning("Seed not set, using default seed 42.")
            self.seed = 42

        assert addr is None, "FSDPEngine does not support remote initialization."

        self._normalize_adam_bf16_config()

        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"

        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        current_platform.set_numa_affinity(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.is_pp_head = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
        )
        self.weight_update_group_name = (
            f"update_weight_group_{mpu.get_pipeline_model_parallel_rank()}"
        )
        self.engine_lock = DistributedLock("train_engine_lock")
        self.logger.info(
            f"[DiagInit] initialize: rank={self.rank}, world_size={self.world_size}, "
            f"device={self.device}, is_pp_head={self.is_pp_head}"
        )

        if self.config.use_lora and self.bridge_cls != "megatron-bridge":
            raise NotImplementedError(
                "MegatronEngine LoRA POC currently only supports bridge_type='megatron-bridge'. "
                "mbridge does not support LoRA in this path."
            )

        self.logger.info("[DiagInit] initialize: loading tokenizer...")
        _t1 = _time.time()
        self.tokenizer = load_hf_tokenizer(self.config.path)
        self.logger.info(f"[DiagInit] initialize: tokenizer loaded in {_time.time() - _t1:.2f}s")

        self.logger.info("[DiagInit] initialize: building HF/Megatron bridge...")
        _t2 = _time.time()
        with patch_bridge_for_tree_training(
            self.enable_tree_training and self.bridge_cls == "mbridge"
        ):
            self.bridge = self._build_hf_mcore_bridge()
            self.logger.info(f"[DiagInit] initialize: bridge built in {_time.time() - _t2:.2f}s")

            self.logger.info("[DiagInit] initialize: making HF and mcore config...")
            _t3 = _time.time()
            self.hf_config, self.tf_config = make_hf_and_mcore_config(
                self.config.path,
                dtype=self.dtype,
                bridge=self.bridge,
                bridge_type=self.bridge_cls,
            )
            self.tf_config = configure_pipeline_layer_splits(
                self.parallel_strategy, self.hf_config, self.tf_config
            )
            self.logger.info(f"[DiagInit] initialize: configs made in {_time.time() - _t3:.2f}s")

            self.quantization_config = getattr(
                self.hf_config, "quantization_config", None
            )

            self._check_and_apply_fp8_config()
            self._validate_fp8_consistency()

            if self.enable_mtp_training:
                self.tf_config.mtp_num_layers = self.mtp_num_layers
                self.tf_config.mtp_loss_scaling_factor = self.mtp_loss_scaling_factor
                if hasattr(self.tf_config, "mtp_detach_heads"):
                    self.tf_config.mtp_detach_heads = self.mtp_detach_heads
                self.logger.info(
                    f"[MTPTrain] Propagated MTP config to tf_config: "
                    f"mtp_num_layers={self.mtp_num_layers}, "
                    f"mtp_loss_scaling_factor={self.mtp_loss_scaling_factor}, "
                    f"mtp_detach_heads={self.mtp_detach_heads}"
                )
            else:
                _orig_mtp = getattr(self.tf_config, "mtp_num_layers", None)
                if _orig_mtp is not None and _orig_mtp > 0:
                    self.tf_config.mtp_num_layers = None
                    self.logger.info(
                        f"[MTPConfig] Cleared tf_config.mtp_num_layers "
                        f"(was {_orig_mtp}) because enable_mtp_training=False. "
                        f"MTP layers will NOT be created in GPTModel."
                    )

            self.logger.info("[DiagInit] initialize: creating Megatron model...")
            _t4 = _time.time()
            with self.device:
                models = make_mcore_model(
                    hf_config=self.hf_config,
                    tf_config=self.tf_config,
                    mcore_config=self.mcore_config,
                    bridge=self.bridge,
                    bridge_type=self.bridge_cls,
                    is_critic=self.config.is_critic,
                    use_lora=self.config.use_lora,
                    enable_mtp=self.enable_mtp_training,
                )
            self.logger.info(f"[DiagInit] initialize: Megatron model created in {_time.time() - _t4:.2f}s")

        self.model = _MegatronModelList(models)

        if self.config.use_lora:
            self.logger.info("[DiagInit] initialize: applying Megatron Bridge LoRA...")
            _t_lora = _time.time()
            self._apply_megatron_bridge_lora()
            self.logger.info(f"[DiagInit] initialize: LoRA applied in {_time.time() - _t_lora:.2f}s")

        self.logger.info("[DiagInit] initialize: loading model weights from HF...")
        _t5 = _time.time()
        with self.device:
            self._load_model_from_hf(self.config.path)
        self.logger.info(f"[DiagInit] initialize: HF weights loaded in {_time.time() - _t5:.2f}s")

        # NOTE: Clear high_precision_init_val for FP8 parameters.
        #
        # Background: When using distributed optimizer, Megatron uses
        # high_precision_init_val to initialize optimizer's main parameters.
        # TransformerEngine (TE) provides this via get_high_precision_init_val().
        #
        # Problem with publicly available HF FP8 models:
        # - Megatron sets preserve_high_precision_init_val=True when loading FP8 models
        # - This causes TE (transformer_engine/pytorch/module/base.py) to use the
        #   init_method's random initialization as high_precision_init_val
        # - But for pre-trained HF models, we load actual weights AFTER initialization,
        #   so high_precision_init_val still holds the random init values, not the
        #   loaded weights
        #
        # Solution: Clear high_precision_init_val here after loading HF weights.
        # The optimizer will then use the actual FP8 weights (upcast to high precision)
        # instead of stale random initialization values.
        for model in self.model:
            for _, param in model.named_parameters():
                if hasattr(param, "get_high_precision_init_val"):
                    param.clear_high_precision_init_val()
                    delattr(param, "get_high_precision_init_val")
                    delattr(param, "clear_high_precision_init_val")

        assert self.model, "Megatron models failed to initialize."
        modules = [m.module if isinstance(m, DDP) else m for m in self.model]
        total_params = sum(
            param.numel() for module in modules for param in module.parameters()
        )
        self.logger.info(
            f"Model parameter count: {total_params / 1e6:.2f}M, pp_stage={mpu.get_pipeline_model_parallel_rank()}, vpp_chunks={len(self.model)}"
        )

        if self.config.disable_dropout:
            for model in self.model:
                disable_dropout_in_model(model)

        primary_model = self.model[0]
        model_config = get_model_config(primary_model)

        # NOTE: It is recommended to set this option to True for RL training on MoE models for stability.
        if self.mcore_config.use_deterministic_algorithms:
            set_deterministic_algorithms(model_config)

        # Set vp_stage for DDP models
        for i, model_chunk in enumerate(self.model):
            if (
                isinstance(model_chunk, DDP)
                and self.mcore_config.virtual_pipeline_parallel_size > 1
            ):
                vp_stage = getattr(model_chunk.module, "vp_stage", None)
                self.logger.info(f"Setting vp_stage {vp_stage} for model chunk {i}.")
                setattr(model_chunk, "vp_stage", vp_stage)

        if self.mcore_config.ddp.overlap_grad_reduce and isinstance(primary_model, DDP):
            model_config.no_sync_func = [
                model_chunk.no_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                model_config.no_sync_func = model_config.no_sync_func[0]

        if (
            self.mcore_config.ddp.overlap_param_gather
            and self.mcore_config.ddp.align_param_gather
        ):
            model_config.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                model_config.param_sync_func = model_config.param_sync_func[0]
        model_config.finalize_model_grads_func = finalize_model_grads
        self.logger.info("[DiagInit] initialize: creating optimizer...")
        _t6 = _time.time()
        self._create_optimizer(ft_spec)
        self.logger.info(f"[DiagInit] initialize: optimizer created in {_time.time() - _t6:.2f}s")

        if self.enable_mtp_training and not self._mtp_layers_verified:
            mtp_param_count = 0
            mtp_param_names = []
            for module in modules:
                for name, param in module.named_parameters():
                    if ".mtp." in name:
                        mtp_param_count += param.numel()
                        if len(mtp_param_names) < 5:
                            mtp_param_names.append(name)

            # With pipeline parallelism, MTP layers only exist on the last stage.
            # Non-last stages legitimately have 0 MTP params.
            is_last_stage = True
            try:
                if (
                    mpu.is_initialized()
                    and mpu.get_pipeline_model_parallel_world_size() > 1
                ):
                    is_last_stage = mpu.is_pipeline_last_stage()
            except Exception:
                pass

            if mtp_param_count == 0:
                if not is_last_stage:
                    self._mtp_layers_verified = True
                    self.logger.info(
                        "[MTPTrain] This rank is NOT on the last pipeline stage; "
                        "MTP parameters are expected only on the last stage. "
                        "Skipping MTP param verification on this rank."
                    )
                else:
                    self.logger.error(
                        "[MTPTrain] enable_mtp_training=True but NO MTP parameters found "
                        "on the LAST pipeline stage! "
                        "Possible causes: 1) mtp_num_layers=0 in model config; "
                        "2) Model checkpoint does not contain MTP layers; "
                        "3) mbridge did not pass mtp_block_spec to GPTModel. "
                        "MTP loss will NOT be computed."
                    )
            else:
                self._mtp_layers_verified = True
                self.logger.info(
                    f"[MTPTrain] Verified MTP parameters in model: "
                    f"total_mtp_params={mtp_param_count / 1e6:.2f}M, "
                    f"sample_params={mtp_param_names}"
                )

        self._initialized = True
        self.logger.info(
            f"[DiagInit] initialize: COMPLETED in {_time.time() - _t0:.2f}s total"
        )

    def _build_hf_mcore_bridge(self):
        if self.bridge_cls == "mbridge":
            self.bridge = mbridge.AutoBridge.from_pretrained(
                self.config.path, trust_remote_code=True
            )
            self.bridge.dtype = self.dtype
            if self.config.gradient_checkpointing:
                self.bridge.set_extra_args(
                    recompute_granularity=self.mcore_config.recompute_granularity,
                    recompute_method=self.mcore_config.recompute_method,
                    recompute_num_layers=self.mcore_config.recompute_num_layers,
                    distribute_saved_activations=self.mcore_config.distribute_saved_activations,
                    recompute_modules=self.mcore_config.recompute_modules,
                )
            self.logger.info(
                "Using mbridge to create models and hf model save/load in MegatronEngine."
            )

        elif self.bridge_cls == "megatron-bridge":
            if self.enable_tree_training:
                raise NotImplementedError(
                    "Tree training is not supported with bridge_type='megatron-bridge'."
                )
            self.bridge = MegatronBridgeAutoBridge.from_hf_pretrained(
                self.config.path,
                trust_remote_code=True,
                dtype=self.config.dtype,
            )
            self.logger.info(
                "Using megatron-bridge to create models and hf model save/load in MegatronEngine."
            )

        else:
            self.logger.info(
                "Not using bridge to create models and hf model save/load in MegatronEngine."
            )
            self.bridge = None
        return self.bridge

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def data_parallel_rank(self) -> int:
        assert self.process_group_initialized
        return mpu.get_data_parallel_rank()

    @property
    def data_parallel_world_size(self) -> int:
        assert self.process_group_initialized
        return mpu.get_data_parallel_world_size()

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return mpu.get_data_parallel_group()

    def current_data_parallel_head(self) -> int:
        """Get the rank of the head of the current data parallel group."""
        assert self.process_group_initialized
        ranks = dist.get_process_group_ranks(self.context_and_model_parallel_group)
        return ranks[0]

    def is_data_parallel_head(self) -> bool:
        assert self.process_group_initialized
        ranks = dist.get_process_group_ranks(self.context_and_model_parallel_group)
        return ranks[0] == self.rank

    @property
    def pipeline_parallel_rank(self) -> int:
        assert self.process_group_initialized
        return mpu.get_pipeline_model_parallel_rank()

    def is_pipeline_parallel_head(self) -> bool:
        assert self.process_group_initialized
        return self.is_pp_head

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return self._context_and_model_parallel_group

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return self._cpu_group

    def destroy(self):
        self._initialized = False
        self.process_group_initialized = False
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            self.model = None
        gc.collect()
        current_platform.empty_cache()
        gc.collect()
        # NOTE: if `own_global_group` is true, we assume that
        # no communications are needed after `destroy`, so we
        # directly destroy all groups. Otherwise, process group
        # handles still exist and we expect another engine to
        # clean up these groups.
        if dist.is_initialized() and self.own_global_group:
            mpu.destroy_model_parallel()
            dist.destroy_process_group()
            self.own_global_group = False

    def train(self, mode: bool = True):
        assert self.model is not None
        for model in self.model:
            model.train(mode=mode)
        return self

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        if self.rollout_engine is not None and self.rollout_engine != engine:
            self.logger.warning(
                f"Connected rollout engine changed from {self.rollout_engine} to {engine}."
            )
        self.rollout_engine = engine
        # Check if engine supports tensor weight updates (MTP draft sync).
        self._engine_supports_tensor_update = hasattr(
            engine, "update_weights_from_tensor"
        )
        if self.enable_mtp_training and self._engine_supports_tensor_update:
            self.logger.info(
                "[MTPTrain] Inference engine supports update_weights_from_tensor. "
                "MTP draft model weights will be synced via tensor update path."
            )
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
            if meta.with_optim:
                raise ValueError(
                    "HF format does not support optimizer state saving, please use DCP format instead."
                )
            self._save_model_to_hf(
                meta.path,
                tokenizer=meta.tokenizer,
                processor=meta.processor,
                base_model_path=meta.base_model_path,
            )
        elif meta.weight_format == "dcp":
            if self.checkpointer is None:
                raise NotImplementedError(
                    "DCP checkpoint save is not available for this Megatron configuration "
                    "(e.g., LoRA path without distributed optimizer support). "
                    "Please use weight_format='hf' for adapter/full-model export."
                )
            self.checkpointer.save_checkpoint(meta.path, with_optimizer=meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            if meta.with_optim:
                raise ValueError(
                    "HF format does not support optimizer state loading, please use DCP format instead."
                )
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            if self.checkpointer is None:
                raise NotImplementedError(
                    "DCP checkpoint load is not available for this Megatron configuration "
                    "(e.g., LoRA path without distributed optimizer support). "
                    "Please use weight_format='hf' for adapter/full-model load."
                )
            self.checkpointer.load_checkpoint(meta.path, with_optimizer=meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def optimizer_zero_grad(self):
        assert self.optimizer is not None, "Optimizer is not initialized."
        self.optimizer.zero_grad()
        for model in self.model:
            model.zero_grad_buffer()

    @staticmethod
    def _roll_tensor_packed(
        tensor: torch.Tensor, shift: int, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """Roll tensor within each packed sequence boundary.

        In sequence packing mode, multiple sequences are concatenated. A naive
        torch.roll would leak tokens across sequence boundaries. This function
        rolls within each sequence independently and zeros out boundary positions.
        """
        result = torch.zeros_like(tensor)
        num_seqs = cu_seqlens.shape[0] - 1
        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_slice = tensor[..., start:end]
            rolled = torch.roll(seq_slice, shifts=shift, dims=-1)
            if shift < 0:
                rolled[..., shift:] = 0  # zero out wrapped-around positions at end
            else:
                rolled[..., :shift] = 0
            result[..., start:end] = rolled
        return result

    def _collect_mtp_loss(self) -> dict[str, float]:
        """Collect MTP loss from Megatron-Core's MTPLossLoggingHelper after forward-backward.

        The MTP loss is computed during the forward pass and added directly to the
        RL loss in _compute_logprobs_and_loss (bypassing MTPLossAutoScaler, which
        fails under Megatron DDP/TP). This function only collects the loss VALUE
        for logging and monitoring purposes.

        IMPORTANT: All CP ranks must participate in the all-reduce to avoid deadlock.
        The gate condition uses is_pipeline_last_stage() instead of
        is_mp_src_rank_with_outputs() to ensure all CP ranks enter the all-reduce.
        """
        mtp_stats = {}
        try:
            from megatron.core.transformer.multi_token_prediction import (
                MTPLossLoggingHelper,
            )

            tracker = MTPLossLoggingHelper.tracker
            if tracker and "values" in tracker:
                values = tracker["values"]

                is_last_pp_stage = mpu.is_pipeline_last_stage(ignore_virtual=True)

                if tracker.get("reduce_group") is not None:
                    import torch.distributed

                    torch.distributed.all_reduce(values, group=tracker["reduce_group"])
                if tracker.get("avg_group") is not None:
                    import torch.distributed

                    torch.distributed.all_reduce(
                        values,
                        group=tracker["avg_group"],
                        op=torch.distributed.ReduceOp.AVG,
                    )

                mtp_loss_value = values.sum().item()
                self._mtp_loss_value = mtp_loss_value

                if is_last_pp_stage:
                    mtp_stats["mtp_loss"] = mtp_loss_value

                if math.isnan(mtp_loss_value) or math.isinf(mtp_loss_value):
                    self.logger.error(
                        f"[MTPTrain] MTP loss is NaN/Inf! value={mtp_loss_value}. "
                        f"Check MTP label construction and model configuration."
                    )
                else:
                    # Note: mtp_loss_value is the SUM of per-micro-batch
                    # average MTP losses (accumulated via += in the tracker).
                    # This is by design in Megatron-Core.  For N micro-batches
                    # the value ≈ N * per_token_mtp_loss.
                    self.logger.info(
                        f"[MTPTrain] MTP loss (accumulated)={mtp_loss_value:.6f}, "
                        f"scaling_factor={self.mtp_loss_scaling_factor}, "
                        f"is_last_pp_stage={is_last_pp_stage}"
                    )

                    # Log gradient norms for MTP vs non-MTP parameters
                    # to verify gradient isolation is working correctly.
                    if is_last_pp_stage and self.mtp_detach_heads:
                        try:
                            from megatron.core.transformer.multi_token_prediction import (
                                MTPLossAutoScaler,
                            )
                            mtp_g = 0.0
                            non_mtp_g = 0.0
                            mtp_n = 0
                            non_mtp_n = 0
                            emb_g = 0.0
                            lmh_g = 0.0
                            total_params = 0
                            no_grad_params = 0
                            # Per-MTP-param diagnostics for debugging
                            mtp_param_details = []
                            for module in self.model:
                                for name, param in module.named_parameters():
                                    total_params += 1
                                    # Megatron DDP stores grads in main_grad
                                    has_main_grad = hasattr(param, "main_grad") and param.main_grad is not None
                                    has_grad = param.grad is not None
                                    grad = None
                                    grad_source = "none"
                                    if has_main_grad:
                                        grad = param.main_grad
                                        grad_source = "main_grad"
                                    elif has_grad:
                                        grad = param.grad
                                        grad_source = "grad"
                                    if grad is None:
                                        no_grad_params += 1
                                        if ".mtp." in name:
                                            mtp_param_details.append(
                                                f"  {name}: NO GRAD (main_grad={has_main_grad}, grad={has_grad})"
                                            )
                                        continue
                                    g = grad.data.float().norm() ** 2
                                    if ".mtp." in name:
                                        mtp_g += g.item()
                                        mtp_n += 1
                                        # Log per-param detail for MTP params
                                        g_norm = g.item() ** 0.5
                                        mtp_param_details.append(
                                            f"  {name}: norm={g_norm:.8f} src={grad_source}"
                                        )
                                        # Also check if param.grad has gradient
                                        # when main_grad is zero (diagnostic)
                                        if g_norm == 0.0 and has_main_grad and has_grad:
                                            alt_g = param.grad.data.float().norm().item()
                                            mtp_param_details[-1] += f" ALT_grad_norm={alt_g:.8f}"
                                    else:
                                        non_mtp_g += g.item()
                                        non_mtp_n += 1
                                    if "embedding" in name and ".mtp." not in name:
                                        emb_g += g.item()
                                    if "output_layer" in name and ".mtp." not in name:
                                        lmh_g += g.item()

                            # Log MTPLossAutoScaler backward scale for debugging
                            try:
                                scale_val = MTPLossAutoScaler.main_loss_backward_scale
                                if hasattr(scale_val, "item"):
                                    scale_str = f"{scale_val.item():.6f}"
                                else:
                                    scale_str = str(scale_val)
                            except Exception:
                                scale_str = "N/A"

                            self.logger.info(
                                f"[MTPDetach] Gradient norms: "
                                f"mtp={mtp_g**0.5:.6f}({mtp_n} params), "
                                f"non_mtp={non_mtp_g**0.5:.6f}({non_mtp_n} params), "
                                f"emb={emb_g**0.5:.6f}, lmh={lmh_g**0.5:.6f}, "
                                f"total={total_params}, no_grad={no_grad_params}, "
                                f"mtp_backward_scale={scale_str}"
                            )
                            # Log per-MTP-param details
                            if mtp_param_details:
                                self.logger.info(
                                    "[MTPGradDiag] Per-MTP-param gradient norms:\n"
                                    + "\n".join(mtp_param_details)
                                )
                            # Additional diagnostic: check if any MTP param
                            # has .grad (not main_grad) with nonzero value,
                            # which would indicate gradient accumulation fusion
                            # mismatch between .grad and .main_grad
                            if mtp_g == 0.0:
                                alt_grad_found = False
                                for module in self.model:
                                    for name, param in module.named_parameters():
                                        if ".mtp." not in name:
                                            continue
                                        if param.grad is not None and param.grad.data.float().norm().item() > 0:
                                            alt_grad_found = True
                                            self.logger.warning(
                                                f"[MTPGradDiag] ALERT: {name} has nonzero .grad "
                                                f"(norm={param.grad.data.float().norm().item():.8f}) "
                                                f"but zero .main_grad! This indicates gradient "
                                                f"accumulation fusion mismatch."
                                            )
                                if not alt_grad_found:
                                    self.logger.warning(
                                        "[MTPGradDiag] All MTP params have zero gradient "
                                        "in BOTH .main_grad and .grad. The MTP backward "
                                        "path is completely broken. Check: "
                                        "1) MTP loss was stored in _mtp_loss_for_backward, "
                                        "2) MTP loss was added to RL loss in _compute_logprobs_and_loss, "
                                        "3) mtp_loss requires_grad=True, "
                                        "4) _mtp_hs requires_grad=True."
                                    )
                                self.logger.info(
                                    "[MTPGradDiag] Deep chain check: examining "
                                    "MTP param registration in grad buffer...")
                                for module in self.model:
                                    for name, param in module.named_parameters():
                                        if ".mtp." not in name:
                                            continue
                                        _has_mg = hasattr(param, "main_grad") and param.main_grad is not None
                                        _has_g = param.grad is not None
                                        _flag_v = getattr(param, "grad_added_to_main_grad", "N/A")
                                        _mg_ptr = param.main_grad.data_ptr() if _has_mg else 0
                                        self.logger.info(
                                            "[MTPGradDiag]   %s: "
                                            "has_main_grad=%s, has_grad=%s, "
                                            "grad_added_flag=%s, rg=%s, mg_ptr=%s",
                                            name, _has_mg, _has_g,
                                            _flag_v, param.requires_grad, _mg_ptr)
                            mtp_stats["mtp_grad_norm"] = mtp_g**0.5
                            self._last_mtp_grad_norm = mtp_g**0.5
                            mtp_stats["non_mtp_grad_norm"] = non_mtp_g**0.5
                            mtp_stats["mtp_backward_scale"] = (
                                float(scale_str) if scale_str != "N/A" else 0.0)
                        except Exception as e:
                            self.logger.warning(
                                f"[MTPDetach] Grad norm logging failed: {e}"
                            )

                MTPLossLoggingHelper.clean_loss_in_tracker()
            else:
                if self.enable_mtp_training:
                    self.logger.warning(
                        "[MTPTrain] MTP loss tracker is empty after forward-backward "
                        "even though enable_mtp_training=True. Possible causes: "
                        "1) Model does not have MTP layers; "
                        "2) mtp_kwargs were not passed correctly; "
                        "3) Megatron-Core version mismatch. "
                        "Verify model architecture and mtp_num_layers config."
                    )

        except ImportError:
            self.logger.warning(
                "[MTPTrain] Cannot import MTPLossLoggingHelper from megatron.core. "
                "MTP loss collection disabled. Ensure megatron-core >= 0.12.0 "
                "for MTP with gradient isolation support."
            )
        except Exception as e:
            self.logger.error(
                f"[MTPTrain] Error collecting MTP loss: {e}", exc_info=True
            )

        return mtp_stats

    def optimizer_step(self):
        with trace_scope("megatron_engine.step"):
            update_successful, grad_norm, _ = self.optimizer.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Log MTP lr if using separate param group
        _mtp_lr = None
        if self.enable_mtp_training and len(self.optimizer.param_groups) > 1:
            for _pg in self.optimizer.param_groups:
                if _pg.get('max_lr', None) != self.optimizer.param_groups[0].get('max_lr', None):
                    _mtp_lr = _pg['lr']
                    break


        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
            mtp_lr=_mtp_lr if _mtp_lr is not None else current_lr,
        )

    def lr_scheduler_step(self):
        assert self.lr_scheduler is not None, "LR Scheduler is not initialized."
        self.lr_scheduler.step(1)

    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> None:
        self._ensure_ready()

        def forward_step(batch_iter, model):
            mb_input: MicroBatchItem = next(batch_iter)

            cu_seqlens = mb_input.padded_mb.get("cu_seqlens", None)

            # Lazily create tree attention metadata just before forward.
            # dense_mask=True because Megatron's gradient checkpointing uses
            # save_for_backward() which can only save torch.Tensor objects;
            # BlockMask is recreated inside PytorchFlexAttention.forward().
            tree_attn_keys: list[str] = []
            if self.enable_tree_training:
                trie_node = mb_input.padded_mb.get("trie_node", None)
                # Ensure trie_node is also in orig_mb for _compute_logprobs_and_loss
                if trie_node is not None and "trie_node" not in mb_input.orig_mb:
                    mb_input.orig_mb["trie_node"] = trie_node
                padded_size = mb_input.padded_to_length
                if trie_node is not None:
                    assert padded_size is not None
                    tree_kwargs = build_tree_attn_kwargs(
                        trie_node,
                        padded_size,
                        mb_input.padded_mb["input_ids"].device,
                        dense_mask=True,
                    )
                    mb_input.padded_mb.update(tree_kwargs)
                    tree_attn_keys = list(tree_kwargs.keys())

            # ---- MTP handling in GPTModel._postprocess() ----
            #
            # megatron-core 0.16.x _postprocess() behaviour:
            #
            #   if mtp_in_postprocess:
            #       hidden_states = self.mtp(...)       # MTP forward
            #   if config.mtp_num_layers is not None:
            #       <MTP loss computed, attached via MTPLossAutoScaler>
            #   logits = self.output_layer(hidden_states)
            #   if labels is None: return logits
            #   return compute_language_model_loss(labels, logits)
            #
            # Inference (forward_only=True):
            #   AReaL does NOT pass labels, so labels.clone() crashes.
            #   MTP forward is also unnecessary for logprob collection.
            #   → Disable MTP entirely so _postprocess returns logits.
            #
            # Training (forward_only=False):
            #   We NEED MTP loss in the autograd graph for draft-model
            #   training, but AReaL also needs logits (not CE loss) for
            #   its RL loss pipeline.  Strategy:
            #     1. Keep MTP enabled so _postprocess runs mtp forward
            #        and computes MTP loss (MTPLossAutoScaler).
            #     2. Pass labels & loss_mask via extra_block_kwargs.
            #     3. Monkey-patch compute_language_model_loss: the LAST
            #        call (main CE) returns logits instead of loss;
            #        earlier calls (per-MTP-layer) use real CE.
            extra_block_kwargs = None
            _mtp_restore = None
            _clm_loss_restore = None
            _postprocess_restore = None  # for _postprocess gradient isolation patch
            _mtp_get_emb_restore = []  # for _get_embeddings gradient isolation patch
            _mtp_ckpt_restore = []  # (layer, orig_method) pairs

            # Defensive guard: even when enable_mtp_training=False, the
            # model may still have MTP artefacts (e.g. config.mtp_num_layers
            # leaked from HF/mbridge config, or MTP layers loaded from a
            # checkpoint).  During inference this causes _postprocess() to
            # enter the MTP loss path and crash on labels.clone() when
            # labels is None.  Disable MTP at runtime in this case.
            if not self.enable_mtp_training and forward_only:
                _unwrapped_def = model
                while hasattr(_unwrapped_def, "module"):
                    _unwrapped_def = _unwrapped_def.module
                _def_mtp = getattr(_unwrapped_def, "mtp", None)
                _def_mtp_process = getattr(_unwrapped_def, "mtp_process", False)
                _def_mtp_layers = getattr(_unwrapped_def.config, "mtp_num_layers", None)
                if (
                    _def_mtp is not None
                    or _def_mtp_process
                    or _def_mtp_layers is not None
                ):
                    _unwrapped_def.mtp = None
                    _unwrapped_def.mtp_process = False
                    _unwrapped_def.config.mtp_num_layers = None
                    _mtp_restore = (
                        _unwrapped_def,
                        _def_mtp,
                        _def_mtp_process,
                        _def_mtp_layers,
                    )
                    self.logger.debug(
                        f"[MTPGuard] Disabled MTP for inference "
                        f"(enable_mtp_training=False but model had "
                        f"mtp={_def_mtp is not None}, "
                        f"mtp_process={_def_mtp_process}, "
                        f"mtp_num_layers={_def_mtp_layers})"
                    )

            if self.enable_mtp_training:
                _engine_ref = self
                self._mtp_loss_for_backward = []
                # MTP loss EMA for adaptive clipping (prevents loss spikes)
                if not hasattr(self, '_mtp_loss_ema'):
                    self._mtp_loss_ema = None  # Will be initialized on first MTP loss
                    self._mtp_loss_clip_count = 0
                    self._mtp_loss_total_count = 0
                # [v5-F6] Hint SpecDec v2 env toggle for throughput (idempotent,
                # rank-0 only to avoid N-rank log spam, print once only).
                import os as _os_v5
                try:
                    _rank_v5 = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                except Exception:
                    _rank_v5 = 0
                if _rank_v5 == 0 and _os_v5.environ.get("SGLANG_ENABLE_SPEC_V2", "") == "":
                    if not getattr(self, '_mtp_env_hint_printed', False):
                        self._mtp_env_hint_printed = True
                        self.logger.info(
                            "[MTPEnvHint] SGLANG_ENABLE_SPEC_V2 not set; "
                            "consider exporting SGLANG_ENABLE_SPEC_V2=True to "
                            "enable overlap scheduler for speculative decoding."
                        )

                _unwrapped = model
                while hasattr(_unwrapped, "module"):
                    _unwrapped = _unwrapped.module

                if forward_only:
                    # -- Inference: disable MTP to avoid crash --
                    _saved_mtp = getattr(_unwrapped, "mtp", None)
                    _saved_mtp_process = getattr(_unwrapped, "mtp_process", None)
                    _saved_mtp_layers = getattr(
                        _unwrapped.config, "mtp_num_layers", None
                    )
                    if (
                        _saved_mtp is not None
                        or _saved_mtp_process
                        or _saved_mtp_layers is not None
                    ):
                        _unwrapped.mtp = None
                        _unwrapped.mtp_process = False
                        _unwrapped.config.mtp_num_layers = None
                        _mtp_restore = (
                            _unwrapped,
                            _saved_mtp,
                            _saved_mtp_process,
                            _saved_mtp_layers,
                        )
                    self.logger.debug(
                        "[MTPTrain] Disabled MTP in _postprocess for "
                        "inference (forward_only=True)"
                    )
                else:
                    # -- Training: enable MTP with labels & loss_mask --
                    # Pass raw input_ids as MTP labels (NOT pre-shifted).
                    # Megatron-Core _postprocess() calls roll_tensor(labels, -1)
                    # internally for each MTP layer, so MTP layer k predicts
                    # token at position i+(k+1).  This matches the slime
                    # implementation which passes batch["tokens"] directly.
                    _input_ids = mb_input.padded_mb["input_ids"]
                    _mtp_labels = _input_ids
                    # loss_mask carried through pack/pad pipeline;
                    # fall back to None → megatron uses ones_like.
                    _mtp_loss_mask = mb_input.padded_mb.get("loss_mask", None)
                    extra_block_kwargs = {"labels": _mtp_labels}
                    if _mtp_loss_mask is not None:
                        extra_block_kwargs["loss_mask"] = _mtp_loss_mask

                    # In Megatron-Core 0.16.0, MTP CE loss gradient leaks to
                    # backbone through 3 paths:
                    #
                    # Path 1: MTP loss → hidden_states → backbone
                    #   ANALYSIS: MTPLossAutoScaler.backward() returns
                    #   (grad_output, ones*scale) — grad_output is the main
                    #   loss gradient (NOT mtp gradient). No leak here.
                    #   Verified by verl's implementation which has no isolator.
                    # Path 2: MTP loss → output_layer (lm_head) weights
                    #   MTP logits use the SHARED output_layer and output_weight.
                    #   MTP CE loss backpropagates through lm_head weights.
                    #   Fix: Detach output_weight in _postprocess MTP loop, and
                    #   use direct output_layer call.
                    #
                    # Path 3: MTP loss → embedding weights
                    #   MTP layers call embedding(input_ids, position_ids) using
                    #   the SHARED embedding layer. Gradient flows through
                    #   decoder_input back to embedding weights.
                    #   Fix: Patch _get_embeddings to detach decoder_input.
                    # -----------------------------------------------------------
                    _mtp_diag_mb_counter = [0]

                    if self.mtp_detach_heads:
                        _orig_postprocess = _unwrapped._postprocess.__func__

                        def _patched_postprocess(
                            self_model,
                            hidden_states,
                            input_ids,
                            position_ids,
                            labels,
                            rotary_pos_emb,
                            rotary_pos_cos,
                            rotary_pos_sin,
                            mtp_in_postprocess=None,
                            loss_mask=None,
                            decoder_input=None,
                            attention_mask=None,
                            inference_params=None,
                            packed_seq_params=None,
                            sequence_len_offset=None,
                            runtime_gather_output=None,
                            extra_block_kwargs=None,
                            inference_context=None,
                            _orig_fn=_orig_postprocess,
                            _logger=self.logger,
                        ):
                            """Patched _postprocess with comprehensive MTP
                            gradient isolation (Paths 2, 3). Path 1 removed
                            (MTPLossAutoScaler does not leak MTP grad to backbone).
                            """
                            from megatron.core.transformer.multi_token_prediction import (
                                MTPLossLoggingHelper,
                                roll_tensor,
                            )

                            in_inference_mode = (
                                inference_context is not None
                                and not self_model.training
                            )
                            if in_inference_mode:
                                assert runtime_gather_output, (
                                    "Inference must always gather TP logits"
                                )

                            output_weight = None
                            if self_model.share_embeddings_and_output_weights:
                                output_weight = (
                                    self_model.shared_embedding_or_output_weight()
                                )

                            if mtp_in_postprocess:
                                hidden_states = self_model.mtp(
                                    input_ids=input_ids,
                                    position_ids=position_ids,
                                    hidden_states=hidden_states,
                                    attention_mask=attention_mask,
                                    inference_params=inference_params,
                                    rotary_pos_emb=rotary_pos_emb,
                                    rotary_pos_cos=rotary_pos_cos,
                                    rotary_pos_sin=rotary_pos_sin,
                                    packed_seq_params=packed_seq_params,
                                    sequence_len_offset=sequence_len_offset,
                                    embedding=self_model.embedding,
                                    **(extra_block_kwargs or {}),
                                )

                            if not self_model.post_process:
                                return hidden_states

                            if self_model.config.mtp_num_layers is not None:
                                mtp_labels = labels.clone()
                                hidden_states_list = torch.chunk(
                                    hidden_states,
                                    1 + self_model.config.mtp_num_layers,
                                    dim=0,
                                )
                                hidden_states = hidden_states_list[0]
                                if loss_mask is None:
                                    loss_mask = torch.ones_like(mtp_labels)

                                for mtp_layer_number in range(
                                    self_model.config.mtp_num_layers
                                ):
                                    # Use direct output_layer call for MTP logits
                                    # Previous functional_call + detached params
                                    # broke the backward gradient chain, causing
                                    # mtp_grad_norm=0. The direct call allows
                                    # MTP loss gradient to also accumulate on
                                    # output_layer weights — this is acceptable
                                    # as MTP loss is small (scaled by
                                    # mtp_loss_scaling_factor) and matches
                                    # Megatron-Core's native implementation.
                                    _mtp_hs = hidden_states_list[mtp_layer_number + 1]
                                    # [v5-F1c] Gate MB#0 forward diag to first 3 steps + every 100.
                                    _gs_fwd = getattr(_engine_ref, '_global_step', 0)
                                    if (_mtp_diag_mb_counter[0] == 0
                                            and (_gs_fwd <= 3 or _gs_fwd % 100 == 0)):
                                        _mtp_hs_gfn = type(_mtp_hs.grad_fn).__name__ if _mtp_hs.grad_fn else "None"
                                        _logger.info(
                                            "[MTPFwdDiag] MB#0 Layer#%d step=%d: "
                                            "_mtp_hs.rg=%s, shape=%s, grad_fn=%s, "
                                            "hs.rg=%s",
                                            mtp_layer_number, _gs_fwd, _mtp_hs.requires_grad,
                                            list(_mtp_hs.shape), _mtp_hs_gfn,
                                            hidden_states.requires_grad)
                                    mtp_logits, _ = self_model.output_layer(
                                        _mtp_hs,
                                        weight=output_weight,
                                        runtime_gather_output=runtime_gather_output,
                                    )
                                    # Diagnostic: verify gradient chain is intact
                                    if self_model.training and _logger.isEnabledFor(10):
                                        _logger.debug(
                                            f"[MTPFwdDiag] _mtp_hs.requires_grad={_mtp_hs.requires_grad}, "
                                            f"_mtp_hs.grad_fn={type(_mtp_hs.grad_fn).__name__ if _mtp_hs.grad_fn else 'None'}, "
                                            f"mtp_logits.requires_grad={mtp_logits.requires_grad}, "
                                            f"mtp_logits.grad_fn={type(mtp_logits.grad_fn).__name__ if mtp_logits.grad_fn else 'None'}"
                                        )
                                    mtp_labels, _ = roll_tensor(
                                        mtp_labels,
                                        shifts=-1,
                                        dims=-1,
                                        cp_group=self_model.cp_group,
                                        packed_seq_params=packed_seq_params,
                                    )
                                    loss_mask, num_tokens = roll_tensor(
                                        loss_mask,
                                        shifts=-1,
                                        dims=-1,
                                        cp_group=self_model.cp_group,
                                        packed_seq_params=packed_seq_params,
                                    )
                                    mtp_loss = self_model.compute_language_model_loss(
                                        mtp_labels, mtp_logits
                                    )
                                    mtp_loss = loss_mask * mtp_loss
                                    # [v5-F1c] Gate MB#0 mtp_loss diag to first 3 steps + every 100.
                                    _gs_ml = getattr(_engine_ref, '_global_step', 0)
                                    if (_mtp_diag_mb_counter[0] == 0
                                            and (_gs_ml <= 3 or _gs_ml % 100 == 0)):
                                        _ml_gfn = type(mtp_loss.grad_fn).__name__ if mtp_loss.grad_fn else "None"
                                        _logger.info(
                                            "[MTPFwdDiag] MB#0 mtp_loss step=%d: "
                                            "rg=%s, grad_fn=%s, sum=%.6f, num_tokens=%s",
                                            _gs_ml, mtp_loss.requires_grad, _ml_gfn,
                                            mtp_loss.sum().item(), num_tokens)
                                    elif self_model.training and _logger.isEnabledFor(10):
                                        _logger.debug(
                                            "[MTPFwdDiag] mtp_loss.rg=%s, sum=%.6f",
                                            mtp_loss.requires_grad, mtp_loss.sum().item())
                                    if self_model.training:
                                        from megatron.core import (
                                            parallel_state,
                                        )

                                        MTPLossLoggingHelper.save_loss_to_tracker(
                                            torch.sum(mtp_loss) / num_tokens,
                                            mtp_layer_number,
                                            self_model.config.mtp_num_layers,
                                            avg_group=parallel_state.get_data_parallel_group(
                                                with_context_parallel=True
                                            ),
                                        )
                                    mtp_loss_scale = (
                                        self_model.config.mtp_loss_scaling_factor
                                        / self_model.config.mtp_num_layers
                                    )
                                    if self_model.config.calculate_per_token_loss:
                                        _mtp_loss_to_store = mtp_loss_scale * mtp_loss
                                    else:
                                        _mtp_loss_to_store = mtp_loss_scale * mtp_loss / num_tokens
                                    _engine_ref._mtp_loss_for_backward.append(_mtp_loss_to_store)
                                    # [v5-F4] Cap FIFO to avoid unbounded growth on producer/consumer drift.
                                    _fifo_len = len(_engine_ref._mtp_loss_for_backward)
                                    if _fifo_len > 32:
                                        _logger.warning(
                                            "[MTPFifoOverflow] MTP loss FIFO length=%d >32, "
                                            "dropping oldest entry (producer-consumer drift).",
                                            _fifo_len,
                                        )
                                        _engine_ref._mtp_loss_for_backward.pop(0)
                                    if self_model.training and _logger.isEnabledFor(10):
                                        _logger.debug(
                                            f"[MTPFix] Stored MTP loss for backward: "
                                            f"sum={_mtp_loss_to_store.sum().item():.6f}, "
                                            f"requires_grad={_mtp_loss_to_store.requires_grad}, "
                                            f"accumulator_len={_fifo_len}"
                                        )

                                # [v5-F1a] Gate per-step to first MB to avoid 1.4k lines/step spam.
                                if _mtp_diag_mb_counter[0] == 0:
                                    _logger.info(
                                        "[MTPDetach] MTP loss computed via direct output_layer call (first MB of step)")

                                # [v5-F1b] Gate backward hook registration to first 3 steps
                                # then every 100 steps; previously fired every step × every MB#0.
                                _gs_v5 = getattr(_engine_ref, '_global_step', 0)
                                _should_log_bwd = (_gs_v5 <= 3 or _gs_v5 % 100 == 0)
                                if (_mtp_diag_mb_counter[0] == 0
                                        and hidden_states.requires_grad
                                        and _should_log_bwd):
                                    def _mtp_backward_hook(grad, _lg=_logger, _gs=_gs_v5):
                                        # Inner hook fires once per backward; log only on gated steps.
                                        _lg.info(
                                            "[MTPBwdDiag] AutoScaler backward FIRED (step=%d): "
                                            "grad.shape=%s, grad.norm=%.8f, "
                                            "grad.abs_max=%.8f",
                                            _gs, list(grad.shape),
                                            grad.float().norm().item(),
                                            grad.float().abs().max().item())
                                    hidden_states.register_hook(_mtp_backward_hook)
                                    _logger.info(
                                        "[MTPFwdDiag] MB#0 Registered backward hook on "
                                        "hidden_states(post-AutoScaler) step=%d: shape=%s, rg=%s",
                                        _gs_v5, list(hidden_states.shape),
                                        hidden_states.requires_grad)

                                _mtp_diag_mb_counter[0] += 1

                            # Inference last-token optimization
                            sequence_parallel_override = False
                            if (
                                in_inference_mode
                                and inference_context.materialize_only_last_token_logits
                            ):
                                if inference_context.is_static_batching():
                                    hidden_states = hidden_states[-1:, :, :]
                                else:
                                    if self_model.output_layer.sequence_parallel:
                                        from megatron.core.tensor_parallel import (
                                            gather_from_sequence_parallel_region,
                                        )

                                        hidden_states = (
                                            gather_from_sequence_parallel_region(
                                                hidden_states,
                                                group=self_model.pg_collection.tp,
                                            )
                                        )
                                        self_model.output_layer.sequence_parallel = (
                                            False
                                        )
                                        sequence_parallel_override = True
                                    hidden_states = inference_context.last_token_logits(
                                        hidden_states.squeeze(1).unsqueeze(0)
                                    ).unsqueeze(1)

                            # Main logits: ORIGINAL output_weight (GRPO grad flows)
                            logits, _ = self_model.output_layer(
                                hidden_states,
                                weight=output_weight,
                                runtime_gather_output=runtime_gather_output,
                            )

                            if sequence_parallel_override:
                                assert (
                                    in_inference_mode
                                    and inference_context.is_dynamic_batching()
                                    and inference_context.materialize_only_last_token_logits
                                )
                                self_model.output_layer.sequence_parallel = True

                            if labels is None:
                                return logits.transpose(0, 1).contiguous()

                            loss = self_model.compute_language_model_loss(
                                labels, logits
                            )
                            return loss

                        import types

                        _unwrapped._postprocess = types.MethodType(
                            _patched_postprocess, _unwrapped
                        )
                        _postprocess_restore = (
                            _unwrapped,
                            _orig_postprocess,
                        )

                        # Path 3: patch _get_embeddings for embedding detach
                        _mtp_block = getattr(_unwrapped, "mtp", None)
                        if _mtp_block is not None and hasattr(_mtp_block, "layers"):
                            for _layer in _mtp_block.layers:
                                _orig_get_emb = _layer._get_embeddings

                                _emb_call_count = [0]  # Closure variable for call counting
                                def _patched_get_embeddings(
                                    input_ids,
                                    position_ids,
                                    embedding,
                                    hidden_states,
                                    packed_seq_params=None,
                                    _orig=_orig_get_emb,
                                ):
                                    """Detach decoder_input and hidden_states
                                    to prevent MTP gradient from flowing to
                                    shared embedding and backbone parameters.
                                    """
                                    result = _orig(
                                        input_ids=input_ids,
                                        position_ids=position_ids,
                                        embedding=embedding,
                                        hidden_states=hidden_states,
                                        packed_seq_params=packed_seq_params,
                                    )
                                    _ids, _pos, _dec_input, _hs = result

                                    _dec_input = _dec_input.detach().requires_grad_(True)
                                    _hs = _hs.detach().requires_grad_(True)

                                    _emb_call_count[0] += 1
                                    _call_n = _emb_call_count[0]

                                    # [v5-F1d] Relax throttle 500->2000 to cut MTPEmbDiag spam ~4x.
                                    if _call_n <= 4 or _call_n % 2000 == 0:
                                        _di_gfn = (
                                            type(_dec_input.grad_fn).__name__
                                            if _dec_input.grad_fn else "None(leaf)")
                                        _hs_gfn = (
                                            type(_hs.grad_fn).__name__
                                            if _hs.grad_fn else "None(leaf)")
                                        _engine_ref.logger.info(
                                            "[MTPEmbDiag] _patched_get_embeddings "
                                            "(call #%d, step=%d): "
                                            "_dec_input=[rg=%s, shape=%s, grad_fn=%s], "
                                            "_hs=[rg=%s, shape=%s, grad_fn=%s]",
                                            _call_n,
                                            getattr(_engine_ref, '_global_step', -1),
                                            _dec_input.requires_grad,
                                            list(_dec_input.shape),
                                            _di_gfn,
                                            _hs.requires_grad,
                                            list(_hs.shape),
                                            _hs_gfn,
                                        )

                                    if not _dec_input.requires_grad:
                                        _engine_ref.logger.error(
                                            "[MTPEmbDiag] CRITICAL: _dec_input.requires_grad "
                                            "is False! MTP gradients will be zero. "
                                            "call #%d", _call_n)
                                    if not _hs.requires_grad:
                                        _engine_ref.logger.error(
                                            "[MTPEmbDiag] CRITICAL: _hs.requires_grad "
                                            "is False! MTP gradients will be zero. "
                                            "call #%d", _call_n)

                                    return _ids, _pos, _dec_input, _hs

                                _layer._get_embeddings = _patched_get_embeddings
                                _mtp_get_emb_restore.append((_layer, _orig_get_emb))

                            self.logger.debug(
                                f"[MTPDetach] Patched _get_embeddings on "
                                f"{len(_mtp_get_emb_restore)} MTP layer(s) "
                                f"for embedding gradient isolation (Path 3)"
                            )

                        if random.random() < 0.001:
                            self.logger.info(
                                "[MTPDetach] MTP gradient isolation enabled "
                                f"(mtp_detach_heads={self.mtp_detach_heads}): "
                                "Path 2 (direct output_layer call for MTP logits, "
                                "matching verl/Megatron-Core approach), "
                                "Path 3 (detached decoder_input + hidden_states for embedding). "
                                "MTP CE loss gradients will update MTP params and "
                                "output_layer, but NOT backbone or embedding parameters."
                            )
                    else:
                        self.logger.info(
                            "[MTPDetach] Gradient isolation DISABLED "
                            "(mtp_detach_heads=False). MTP CE loss gradient "
                            "will flow through all model parameters. This is "
                            "intended for pre-training, NOT for RL training."
                        )

                    # Monkey-patch: make the LAST call to
                    # compute_language_model_loss (the main CE loss)
                    # return logits so AReaL gets logits, not loss.
                    _remaining = [self.mtp_num_layers]
                    _orig_clm = _unwrapped.compute_language_model_loss

                    def _mtp_loss_fn(
                        _labels,
                        _logits,
                        _rem=_remaining,
                        _orig=_orig_clm,
                        _lg=self.logger,
                    ):
                        # [v5-F1e] Gate LossFn diag to MB#0 of first 3 steps + every 100.
                        _gs_lfn = getattr(_engine_ref, '_global_step', 0)
                        if (_mtp_diag_mb_counter[0] == 0
                                and (_gs_lfn <= 3 or _gs_lfn % 100 == 0)):
                            _lg.info(
                                "[MTPLossFnDiag] _mtp_loss_fn called step=%d: "
                                "_rem=%d, _logits.rg=%s, shape=%s",
                                _gs_lfn, _rem[0], _logits.requires_grad,
                                list(_logits.shape))
                        if _rem[0] > 0:
                            _rem[0] -= 1
                            return _orig(_labels, _logits)
                        # Return logits in [b, s, v] matching the
                        # ``if labels is None`` path in _postprocess.
                        return _logits.transpose(0, 1).contiguous()

                    _unwrapped.compute_language_model_loss = _mtp_loss_fn
                    _clm_loss_restore = (_unwrapped, _orig_clm)

                    # -----------------------------------------------------------
                    # Megatron-Core 0.16.0 MTP _checkpointed_forward() does:
                    #   tensor_parallel.checkpoint(fn, ..., *args, *kwargs.values())
                    # This flattens ALL kwargs (including packed_seq_params which
                    # is a dataclass, not a tensor) into positional args that end
                    # up in CheckpointFunction.apply() → save_for_backward(),
                    # which only accepts tensors → TypeError.
                    #
                    # The main TransformerBlock avoids this by capturing
                    # packed_seq_params via closure (never passed as an arg).
                    # We apply the same pattern here by monkey-patching each
                    # MTP layer's _checkpointed_forward during training.
                    # -----------------------------------------------------------
                    _mtp_block = getattr(_unwrapped, "mtp", None)
                    if (
                        _mtp_block is not None
                        and hasattr(_mtp_block, "layers")
                        and _unwrapped.config.recompute_granularity == "full"
                    ):
                        for _layer in _mtp_block.layers:
                            _orig_ckpt_fwd = _layer._checkpointed_forward

                            def _patched_checkpointed_forward(
                                forward_func,
                                *args,
                                _layer_ref=_layer,
                                **kwargs,
                            ):
                                """Closure-based checkpoint that keeps
                                non-tensor args (packed_seq_params,
                                inference_params) out of save_for_backward.

                                Mirrors TransformerBlock._checkpointed_forward
                                from megatron-core 0.16.0: non-tensor kwargs
                                are captured in the closure of custom_forward,
                                only tensor values go through checkpoint().
                                """
                                # Separate tensor vs non-tensor kwargs.
                                _tensor_kw = {}
                                _non_tensor_kw = {}
                                for k, v in kwargs.items():
                                    if isinstance(v, torch.Tensor):
                                        _tensor_kw[k] = v
                                    else:
                                        _non_tensor_kw[k] = v

                                # Build a wrapper that re-injects non-tensor
                                # kwargs via closure (never saved by
                                # checkpoint).
                                def _ckpt_wrapper(*flat_args):
                                    # Reconstruct kwargs: first the tensor
                                    # ones from flat_args, then non-tensor
                                    # from closure.
                                    _tk_keys = list(_tensor_kw.keys())
                                    # flat_args = original *args + tensor kw
                                    # values in order.
                                    n_orig = len(args)
                                    _orig_args = flat_args[:n_orig]
                                    _tk_vals = flat_args[n_orig:]
                                    _rebuilt_kw = {
                                        k: v for k, v in zip(_tk_keys, _tk_vals)
                                    }
                                    _rebuilt_kw.update(_non_tensor_kw)
                                    return forward_func(*_orig_args, **_rebuilt_kw)

                                _cfg = _layer_ref.config
                                if _cfg.recompute_method == "uniform":
                                    assert _cfg.recompute_num_layers == 1, (
                                        "recompute_num_layers must be 1 "
                                        "for MTP recompute"
                                    )
                                    if _cfg.fp8:
                                        from megatron.core.extensions.transformer_engine import (
                                            te_checkpoint,
                                        )

                                        return te_checkpoint(
                                            _ckpt_wrapper,
                                            _cfg.distribute_saved_activations,
                                            tensor_parallel.random.get_cuda_rng_tracker,
                                            mpu.get_tensor_model_parallel_group(),
                                            *args,
                                            *_tensor_kw.values(),
                                        )
                                    else:
                                        return tensor_parallel.checkpoint(
                                            _ckpt_wrapper,
                                            _cfg.distribute_saved_activations,
                                            *args,
                                            *_tensor_kw.values(),
                                        )
                                elif _cfg.recompute_method == "block":
                                    import warnings

                                    warnings.warn(
                                        "recompute_method == 'block' is not "
                                        "supported for MTP yet. "
                                        "Skipping recompute."
                                    )
                                    return forward_func(*args, **kwargs)
                                else:
                                    raise ValueError(
                                        "Invalid activation recompute method."
                                    )

                            _layer._checkpointed_forward = _patched_checkpointed_forward
                            _mtp_ckpt_restore.append((_layer, _orig_ckpt_fwd))

                        self.logger.debug(
                            f"[MTPTrain] Patched _checkpointed_forward on "
                            f"{len(_mtp_ckpt_restore)} MTP layer(s) to fix "
                            f"gradient_checkpointing + PackedSeqParams crash "
                            f"(recompute_granularity="
                            f"{_unwrapped.config.recompute_granularity})"
                        )

                    self.logger.debug(
                        f"[MTPTrain] MTP enabled for training "
                        f"(mtp_num_layers={self.mtp_num_layers}, "
                        f"labels_shape={_mtp_labels.shape}, "
                        f"loss_mask={'yes' if _mtp_loss_mask is not None else 'no'})"
                    )

            try:
                output = packed_context_parallel_forward(
                    model,
                    mb_input.padded_mb,
                    extra_block_kwargs=extra_block_kwargs,
                )
            finally:
                if _postprocess_restore is not None:
                    import types as _types_mod

                    _uw, _orig_pp = _postprocess_restore
                    _uw._postprocess = _types_mod.MethodType(_orig_pp, _uw)
                for _layer, _orig_get_emb in _mtp_get_emb_restore:
                    _layer._get_embeddings = _orig_get_emb
                if _mtp_restore is not None:
                    _uw, _sm, _sp, _sl = _mtp_restore
                    _uw.mtp = _sm
                    _uw.mtp_process = _sp
                    _uw.config.mtp_num_layers = _sl
                    self.logger.debug("[MTPTrain] Restored MTP after inference forward")
                if _clm_loss_restore is not None:
                    _uw, _orig = _clm_loss_restore
                    _uw.compute_language_model_loss = _orig
                for _layer, _orig_ckpt in _mtp_ckpt_restore:
                    _layer._checkpointed_forward = _orig_ckpt

            # Release tree attention metadata after forward pass
            for key in tree_attn_keys:
                del mb_input.padded_mb[key]

            def _process_output(input_, output_):
                loss = process_output_fn(output_, input_)
                if loss is None:
                    loss = torch.tensor(1.0, device=output_.device)
                return loss, {}

            model_vp_stage = getattr(model, "vp_stage", 0)
            if mpu.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=model_vp_stage
            ):
                output = unpad_logits(
                    output,
                    padding_length=mb_input.padding_length,
                    cu_seqlens=cu_seqlens,
                    old_cu_seqlens=mb_input.old_cu_seqlens,
                )
            return output, functools.partial(_process_output, mb_input.orig_mb)

        forward_backward_func = get_forward_backward_func()
        with trace_scope("megatron_engine.forward_backward"):
            if len(self.model) > 1:
                data_iterator = [iter(mb_list) for _ in range(len(self.model))]
            else:
                data_iterator = iter(mb_list)
            forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=data_iterator,
                model=self.model if len(self.model) > 1 else self.model[0],
                num_microbatches=len(mb_list),
                seq_length=mb_list.max_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )

    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        self._ensure_ready()
        self.optimizer_zero_grad()
        DeviceRuntimeInfo.get_current().log("train_batch after zero_grad")

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)
        DeviceRuntimeInfo.get_current().log("train_batch after prepare_mb")

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, mpu.get_data_parallel_group()
        )

        # Expose num_microbatches to _compute_logprobs_and_loss so
        # the DoubleScale inversion can further divide the MTP contribution by
        # num_mb.
        self._current_num_microbatches = int(len(mb_list))
        # expose total token count for [MTPDataShapeDiag-v9] so
        # tokens_per_mb can be logged and correlated with accept_rate
        # regressions
        try:
            _tot = 0
            for _mb in mb_list:
                _ids = _mb.get("input_ids") if isinstance(_mb, dict) else None
                if _ids is not None and hasattr(_ids, "numel"):
                    _tot += int(_ids.numel())
            self._current_n_tokens = _tot
        except Exception:
            self._current_n_tokens = 0

        # Step 3: Forward-backward using Megatron's pipeline function
        loss_multiplier = (
            mpu.get_data_parallel_world_size() * self.optimizer.get_loss_scale().item()
        )

        def process_output(
            output: torch.Tensor, inputs: dict[str, Any]
        ) -> torch.Tensor:
            return self._compute_logprobs_and_loss(
                output,
                inputs,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier=loss_multiplier,
            )

        # Track global training step for diagnostic logging
        if not hasattr(self, '_global_step'):
            self._global_step = 0
        self._global_step += 1

        self.forward_backward_batch(mb_list, process_output, forward_only=False)
        DeviceRuntimeInfo.get_current().log("train_batch after forward_backward")

        # Step 4: Collect MTP loss after forward-backward
        mtp_loss_stats = {}
        if self.enable_mtp_training:
            mtp_loss_stats = self._collect_mtp_loss()

        # Step 5: Optimizer step
        train_stats = self.optimizer_step()
        DeviceRuntimeInfo.get_current().log("train_batch after optimizer_step")

        # Merge MTP stats into train stats
        if mtp_loss_stats:
            train_stats.update(mtp_loss_stats)

        return train_stats

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
            mb_list, loss_weight_fn, mpu.get_data_parallel_group()
        )

        # Step 3: Forward using Megatron's pipeline function, collecting losses
        losses: list[torch.Tensor] = []

        def process_output(
            output: torch.Tensor, inputs: dict[str, Any]
        ) -> torch.Tensor:
            loss = self._compute_logprobs_and_loss(
                output, inputs, loss_fn, loss_weight_fn, total_loss_weight
            )
            losses.append(loss.detach())
            return loss

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Collect MTP loss during eval if enabled
        if self.enable_mtp_training:
            mtp_loss_stats = self._collect_mtp_loss()
            if mtp_loss_stats:
                self.logger.info(
                    f"[MTPTrain] Eval MTP loss: {mtp_loss_stats.get('mtp_loss', 'N/A')}"
                )

        # Step 5: Aggregate losses
        if mpu.is_pipeline_last_stage():
            return aggregate_eval_losses(losses, mpu.get_data_parallel_group())
        return None

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
        DeviceRuntimeInfo.get_current().log("forward_batch after prepare_mb")

        # Step 3: Forward using Megatron's pipeline function, collecting results
        outputs: list[torch.Tensor] = []

        def process_output(output: torch.Tensor, inputs: dict[str, Any]) -> None:
            result = self._compute_forward_result(output, inputs)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)
        DeviceRuntimeInfo.get_current().log("forward_batch after forward_backward")

        # Step 4: Aggregate, reorder, and broadcast outputs
        res = None
        if mpu.is_pipeline_last_stage():
            if self.enable_tree_training:
                res = merge_packed_tree_results(outputs, batch_size)
            else:
                res = reorder_and_pad_outputs(
                    outputs, output_seqlens, mb_list, aggregate_fn
                )
        res = broadcast_tensor(
            res,
            src_rank=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        return res

    def export_stats(self) -> dict[str, float]:
        data = stats_tracker.export_all(reduce_group=self.data_parallel_group)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # Some log info only exist in last pipeline rank
            data_list = [data]
            dist.broadcast_object_list(
                data_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )
            data.update(data_list[0])
        return data

    def offload(self) -> None:
        """Offload model memory to CPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/actor.py
        """

        self.get_device_stats().log("before offload model")
        current_platform.clear_memory()
        torch_memory_saver.pause()

        # TODO: NCCL offload
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after offload model")

        self.is_offload = True

    def onload(self) -> None:
        """Onload model memory from CPU back to GPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/actor.py
        """

        torch_memory_saver.resume()
        current_platform.clear_memory()

        # TODO: NCCL onload
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after onload model")

        self.is_offload = False

    def clear_batches(self, *args):
        """Placeholder method of single-controller API."""

    def _normalize_adam_bf16_config(self) -> None:
        if self.optimizer_config is None or self.optimizer_config.type != "adam_bf16":
            return

        self.logger.info(
            "Detected 'adam_bf16' optimizer with Megatron Engine. "
            "Automatically converting to 'adam' with precision-aware optimizer "
            "and setting exp_avg_dtype/exp_avg_sq_dtype to 'bfloat16'."
        )

        self.optimizer_config.type = "adam"
        self.mcore_config.use_precision_aware_optimizer = True
        self.mcore_config.exp_avg_dtype = "bfloat16"
        self.mcore_config.exp_avg_sq_dtype = "bfloat16"

        if self.dtype != torch.bfloat16:
            self.logger.warning(
                "Overriding dtype from %s to bfloat16 for adam_bf16 optimizer.",
                self.config.dtype,
            )
            self.dtype = torch.bfloat16
            self.config.dtype = "bfloat16"

    def _check_and_apply_fp8_config(self):
        if not self.enable_fp8:
            return
        fp8_config = self.fp8_config
        special_mappings = {"mode": "fp8"}
        # Fields that use the same name in both configs (no prefix needed)
        same_fields = {
            "tp_only_amax_red",
            "first_last_layers_bf16",
            "num_layers_at_start_in_bf16",
            "num_layers_at_end_in_bf16",
        }
        # All other fields get the `fp8_` prefix
        for field in dataclasses.fields(fp8_config):
            fp8_field = field.name
            if fp8_field in special_mappings:
                tf_field = special_mappings[fp8_field]
            elif fp8_field in same_fields:
                tf_field = fp8_field
            else:
                tf_field = f"fp8_{fp8_field}"
            if hasattr(self.tf_config, tf_field):
                setattr(self.tf_config, tf_field, getattr(fp8_config, fp8_field))
            else:
                self.logger.warning(
                    f"Unknown FP8 field in TransformerConfig: {fp8_field}"
                )
        self.logger.info(
            f"FP8 training enabled: mode={fp8_config.mode}, "
            f"recipe={fp8_config.recipe}, "
            f"param={fp8_config.param}"
        )
        # fp8_param_gather is passed from make_mcore_model()

    def _validate_fp8_consistency(self):
        """Validate that FP8 configuration is consistent.

        If either training uses FP8, quantization_config must exist
        and quant_method must be "fp8" (weights must be FP8).
        """
        train_fp8 = self.enable_fp8
        weights_fp8 = (
            self.quantization_config is not None
            and self.quantization_config.get("quant_method", None) == "fp8"
        )

        if train_fp8 and not weights_fp8:
            raise RuntimeError(
                "FP8 configuration error: "
                "If training uses FP8, quantization_config must exist "
                "and quant_method must be 'fp8' (weights must be FP8). "
                f"Training fp8={train_fp8}, "
                f"weights fp8={weights_fp8}, "
                f"quantization_config={self.quantization_config}"
            )

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
    ) -> MegatronParallelStrategy:
        base_strategy = dataclasses.asdict(parallel_strategy)
        vpp_size = self.mcore_config.virtual_pipeline_parallel_size
        return MegatronParallelStrategy(
            use_sequence_parallel=parallel_strategy.tensor_parallel_size > 1,
            virtual_pipeline_parallel_size=vpp_size,
            **base_strategy,
        )

    def _init_context_and_model_parallel_group(self) -> None:
        # Initialize context and model parallel groups, which are only used in AReaL
        # for data distribution
        rank_generator = mpu.RankGenerator(
            tp=self.parallel_strategy.tensor_parallel_size,
            ep=1,
            dp=self.parallel_strategy.data_parallel_size,
            pp=self.parallel_strategy.pipeline_parallel_size,
            cp=self.parallel_strategy.context_parallel_size,
            order="tp-cp-ep-dp-pp",
            rank_offset=0,
        )
        context_and_model_parallel_ranks = rank_generator.get_ranks("tp-cp-pp")
        # create context and model_parallel_groups
        for dp_rank, ranks in enumerate(context_and_model_parallel_ranks):
            group = mpu.create_group(
                ranks,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                pg_options=mpu.get_nccl_options("tp-cp-pp", {}),
                group_desc="CONTEXT_AND_MODEL_PARALLEL_GROUP",
            )
            if dp_rank == mpu.get_data_parallel_rank():
                self._context_and_model_parallel_group = group

    def _create_optimizer(self, ft_spec: FinetuneSpec) -> None:
        if self.optimizer_config is None:
            return
        assert self.model is not None and len(self.model) > 0

        use_distributed_optimizer = (
            False
            if self.config.use_lora
            else self.mcore_config.ddp.use_distributed_optimizer
        )

        assert self.optimizer_config.type in [
            "adam",
            "sgd",
        ], "Only AdamW/sgd optimizer is supported in this engine."
        if self.optimizer_config.type == "sgd":
            self.logger.warning(
                "Using the 'sgd' optimizer with Megatron may be less stable. Consider using the 'adam' (AdamW) optimizer for improved stability."
            )

        # Make megatron optimizer config
        mcore_opt_config = MCoreOptimizerConfig(
            optimizer=self.optimizer_config.type,
            lr=self.optimizer_config.lr,
            min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
            bf16=self.dtype is torch.bfloat16,
            fp16=self.dtype is torch.float16,
            adam_beta1=self.optimizer_config.beta1,
            adam_beta2=self.optimizer_config.beta2,
            adam_eps=self.optimizer_config.eps,
            use_distributed_optimizer=use_distributed_optimizer,
            params_dtype=self.dtype,
            clip_grad=self.optimizer_config.gradient_clipping,
            fp8_recipe=(self.fp8_config.recipe if self.enable_fp8 else None),
        )
        mcore_opt_config.overlap_param_gather_with_optimizer_step = (
            self.mcore_config.overlap_param_gather_with_optimizer_step
        )
        mcore_opt_config.use_precision_aware_optimizer = (
            self.mcore_config.use_precision_aware_optimizer
        )
        mcore_opt_config.main_grads_dtype = getattr(
            torch, self.mcore_config.main_grads_dtype
        )
        mcore_opt_config.main_params_dtype = getattr(
            torch, self.mcore_config.main_params_dtype
        )
        mcore_opt_config.exp_avg_dtype = getattr(torch, self.mcore_config.exp_avg_dtype)
        mcore_opt_config.exp_avg_sq_dtype = getattr(
            torch, self.mcore_config.exp_avg_sq_dtype
        )


        # --- MTP independent learning rate  ---
        _mtp_lr_config_overrides = None
        _mtp_lr_scale = getattr(self.optimizer_config, 'mtp_lr_scale', 1.0)
        if self.enable_mtp_training and _mtp_lr_scale != 1.0:
            try:
                from megatron.core.optimizer.optimizer_config import ParamKey
            except ImportError:
                ParamKey = None
            if ParamKey is not None:
                _mtp_lr = self.optimizer_config.lr * _mtp_lr_scale
                _mtp_min_lr = (
                    self.optimizer_config.min_lr_ratio
                    * self.optimizer_config.lr
                    * _mtp_lr_scale
                )
                # Match all MTP parameters by name glob pattern
                _mtp_param_key = ParamKey(name=("*.mtp.*",))
                _mtp_lr_config_overrides = {
                    _mtp_param_key: {
                        "max_lr": _mtp_lr,
                        "min_lr": _mtp_min_lr,
                    }
                }
                self.logger.info(
                    "[MTPOptim] MTP parameters will use separate lr: "
                    "max_lr=%.2e (scale=%.1fx), min_lr=%.2e, base_lr=%.2e",
                    _mtp_lr, _mtp_lr_scale, _mtp_min_lr,
                    self.optimizer_config.lr,
                )
            else:
                self.logger.warning(
                    "[MTPOptim] ParamKey not available in this megatron-core "
                    "version. MTP parameters will use the global learning rate."
                )

        self.optimizer = get_megatron_optimizer(
            mcore_opt_config, self.model,
            config_overrides=_mtp_lr_config_overrides,
        )

        warmup_steps_proportion = self.optimizer_config.warmup_steps_proportion
        warmup_steps = int(warmup_steps_proportion * ft_spec.total_train_steps)
        lr_scheduler = OptimizerParamScheduler(
            self.optimizer,
            init_lr=0.0 if warmup_steps_proportion > 0 else self.optimizer_config.lr,
            max_lr=self.optimizer_config.lr,
            min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
            lr_warmup_steps=warmup_steps,
            lr_decay_steps=ft_spec.total_train_steps - warmup_steps,
            lr_decay_style=self.optimizer_config.lr_scheduler_type,
            start_wd=self.optimizer_config.weight_decay,
            end_wd=self.optimizer_config.weight_decay,
            wd_incr_steps=ft_spec.total_train_steps,
            wd_incr_style="constant",
        )
        self.lr_scheduler = lr_scheduler

        # MegatronCheckpointManager now only support distributed optimizer which lora does not support
        if not self.config.use_lora:
            self.checkpointer = MegatronCheckpointManager(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                use_distributed_optimizer=use_distributed_optimizer,
                use_checkpoint_opt_param_scheduler=self.mcore_config.use_checkpoint_opt_param_scheduler,
                async_save=self.mcore_config.async_save,
            )

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

        if self.model is None:
            raise RuntimeError("Model is not initialized.")

    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ) -> None:
        import time as _diag_time

        _diag_t0 = _diag_time.time()
        if not converted_named_tensors:
            return

        self.engine_lock.acquire()

        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in converted_named_tensors
        ]

        if self.config.use_lora:
            meta.peft_config = {
                "r": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": get_vllm_lora_target_modules(
                    list(self.config.target_modules or [])
                ),
                "bias": "none",
            }

        self.logger.info(
            f"[DiagBucket] _update_bucket_weights_from_distributed ENTERED: "
            f"n_tensors={len(converted_named_tensors)}, n_specs={len(param_specs)}, "
            f"names={[n for n, _ in converted_named_tensors[:5]]}..."
        )
        _t_post0 = _diag_time.time()
        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)
        self.logger.info(
            f"[DiagBucket] rollout_engine.update_weights_from_distributed POST sent "
            f"in {_diag_time.time() - _t_post0:.3f}s, fut={fut}"
        )

        _t_bc0 = _diag_time.time()
        handles = []
        for idx, (name, param) in enumerate(converted_named_tensors):
            handles.append(
                dist.broadcast(
                    param.data, 0, group=self.weight_update_group, async_op=True
                )
            )
        self.logger.info(
            f"[DiagBucket] Enqueued {len(handles)} async broadcasts "
            f"in {_diag_time.time() - _t_bc0:.3f}s, calling handle.wait()..."
        )
        _t_wait0 = _diag_time.time()
        for idx, handle in enumerate(handles):
            handle.wait()
            if idx % 10 == 0 or idx == len(handles) - 1:
                self.logger.info(
                    f"[DiagBucket] handle.wait() progress: {idx + 1}/{len(handles)} "
                    f"after {_diag_time.time() - _t_wait0:.3f}s"
                )
        self.logger.info(
            f"[DiagBucket] All handle.wait() completed in "
            f"{_diag_time.time() - _t_wait0:.3f}s"
        )

        try:
            fut.result(timeout=30)
        except TimeoutError:
            self.logger.warning(
                "Callback response timed out, but NCCL broadcast "
                "completed successfully. Continuing weight update."
            )
        except Exception as e:
            self.logger.warning(
                f"Callback response error: {e}. NCCL broadcast "
                "completed successfully. Continuing weight update."
            )

        converted_named_tensors.clear()

        self.engine_lock.release()
        self.logger.info(
            f"[DiagBucket] _update_bucket_weights_from_distributed COMPLETED "
            f"in {_diag_time.time() - _diag_t0:.3f}s"
        )

    @property
    def _duplicated_param_names(self) -> set[str]:
        """Parameter names whose parent module has parallel_mode='duplicated'.

        These params are replicated (not TP-sharded) but TE incorrectly marks
        them with tensor_model_parallel=True. Cached after first computation.
        """
        if not hasattr(self, "_cached_duplicated_param_names"):
            duplicated = set()
            if self.model is not None:
                for model in self.model:
                    for mod_name, module in model.named_modules():
                        if getattr(module, "parallel_mode", None) == "duplicated":
                            for p_name, _ in module.named_parameters(recurse=False):
                                full = f"{mod_name}.{p_name}" if mod_name else p_name
                                duplicated.add(full)
            self._cached_duplicated_param_names = duplicated
        return self._cached_duplicated_param_names

    def _collect_param(
        self,
        name: str,
        param: nn.Parameter | torch.Tensor,
    ) -> tuple[nn.Parameter | torch.Tensor, int]:
        """Collect and prepare a parameter for conversion.

        This method handles:
        - All-gathering the parameter across tensor parallel ranks
        - Removing padding for vocabulary-related parameters
        - Dequantizing FP8 parameters to bf16s
        - Calculating the parameter size in bytes

        Returns:
            Tuple of (prepared_param, param_size_in_bytes)
        """
        _has_tmp = hasattr(param, "tensor_model_parallel")
        _is_tmp = getattr(param, "tensor_model_parallel", False) if _has_tmp else False
        _is_dup = name in self._duplicated_param_names if self._duplicated_param_names else False
        # [v5-F1f] Downgrade per-param trace to DEBUG (was INFO, ~21k lines/run).
        self.logger.debug(
            f"[DiagImpl] Rank {dist.get_rank()} all_gather_param START "
            f"name={name}, has_tmp={_has_tmp}, is_tmp={_is_tmp}, is_dup={_is_dup}, "
            f"param_shape={tuple(param.shape)}, param_dtype={param.dtype}"
        )
        param = all_gather_param(
            name,
            param,
            self.fp8_direct_convert,
            quantization_config=self.quantization_config,
            duplicated_param_names=self._duplicated_param_names,
        )
        # [v5-F1f] Downgrade per-param trace to DEBUG.
        self.logger.debug(
            f"[DiagImpl] Rank {dist.get_rank()} all_gather_param DONE "
            f"name={name}, result_type={type(param).__name__}"
        )
        param = remove_padding(name, param, self.hf_config.vocab_size)

        if isinstance(param, FP8BlockwiseTensorHelper):
            # FP8 is stored as uint8, so element_size is 1 byte
            param_size = param.numel()
        else:
            param_size = param.numel() * param.element_size()

        return param, param_size

    def _impl_update_weight_from_distributed(
        self,
        meta: WeightUpdateMeta,
        name: str,
        param: nn.Parameter | torch.Tensor,
        converted_named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        buffer_size: int,
        weight_chunked_mem_size: int,
    ) -> int:
        import time as _diag_time

        _t0 = _diag_time.time()
        # [v5-F1f] Downgrade per-param trace to DEBUG.
        self.logger.debug(
            f"[DiagImpl] Rank {dist.get_rank()} _collect_param START "
            f"name={name}"
        )
        param, param_size = self._collect_param(name, param)
        # [v5-F1f] Downgrade per-param trace to DEBUG.
        self.logger.debug(
            f"[DiagImpl] Rank {dist.get_rank()} _collect_param DONE "
            f"name={name}, param_size={param_size / 1024 / 1024:.2f} MB, "
            f"took={_diag_time.time() - _t0:.3f}s"
        )

        if not self.is_pipeline_parallel_head():
            return buffer_size

        if buffer_size + param_size > weight_chunked_mem_size:
            self.logger.info(
                f"[DiagImpl] Buffer overflow ({buffer_size / 1024 / 1024:.2f} + "
                f"{param_size / 1024 / 1024:.2f} > {weight_chunked_mem_size / 1024 / 1024:.2f} MB), "
                f"flushing {len(converted_named_tensors)} tensors, name={name}"
            )
            self._update_bucket_weights_from_distributed(meta, converted_named_tensors)
            buffer_size = 0

        model_name = self.hf_config.model_type
        if self.config.use_lora:
            model_name = f"{model_name}_lora"

        converted_named_tensors.extend(
            convert_to_hf(
                self.tf_config,
                model_name,
                name,
                param,
                quantization_config=self.quantization_config,
                fp8_direct_convert=self.fp8_direct_convert,
            )
        )
        buffer_size += param_size
        return buffer_size

    def _update_bucket_expert_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ) -> None:
        """Gather a bucket of MoE expert weights and broadcast them.

        This function handles the distributed update for a bucket of Mixture-of-Experts
        (MoE) parameters. Since expert parameters are sharded across the expert
        parallel group, this function first performs an `all_gather` to collect all
        shards from all expert ranks.

        Once the full expert parameters are reconstructed on the pipeline parallel
        head, it converts them to the HuggingFace format and calls
        `_update_bucket_weights_from_distributed` to perform the actual broadcast
        to the inference engine.
        """

        # Early exit when chunk size is relatively small
        if not named_tensors:
            return

        group = mpu.get_expert_model_parallel_group()
        world_size = mpu.get_expert_model_parallel_world_size()

        names = [name for name, _ in named_tensors]
        all_names: list[list[str]] = [None] * world_size
        dist.all_gather_object(all_names, names, group=group)

        for rank_names in all_names:
            if len(named_tensors) != len(rank_names):
                raise RuntimeError(
                    "Named tensor count mismatch across expert parallel ranks: "
                    f"expected {len(rank_names)} but got {len(named_tensors)}"
                )

        gathered_params = [[] for _ in range(world_size)]
        handles = []
        for idx, (_, tensor) in enumerate(named_tensors):
            params = [
                torch.empty_like(tensor.data, device=current_platform.current_device())
                for _ in range(world_size)
            ]
            handle = dist.all_gather(params, tensor.data, group=group, async_op=True)
            handles.append(handle)
            for ep_rank, rank_names in enumerate(all_names):
                gathered_params[ep_rank].append((rank_names[idx], params[ep_rank]))

        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self.is_pipeline_parallel_head():
            return

        gathered_params = sum(gathered_params, [])

        converted_hf_tensors = []
        for name, param in gathered_params:
            converted_hf_tensors.extend(
                convert_to_hf(
                    self.tf_config,
                    self.hf_config.model_type,
                    name,
                    param,
                    quantization_config=self.quantization_config,
                    fp8_direct_convert=self.fp8_direct_convert,
                )
            )

        self._update_bucket_weights_from_distributed(meta, converted_hf_tensors)

    def _impl_update_expert_weight_from_distributed(
        self,
        meta: WeightUpdateMeta,
        name: str,
        param: nn.Parameter | torch.Tensor,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        buffer_size: int,
        weight_chunked_mem_size: int,
    ) -> int:
        param, param_size = self._collect_param(name, param)

        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > weight_chunked_mem_size:
            self._update_bucket_expert_weights_from_distributed(meta, named_tensors)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta) -> None:
        assert meta.type == "xccl"
        # Reset weight weight meta with local info
        meta.nccl_master_address = self.weight_update_master_addr = gethostip()
        meta.nccl_master_port = self.weight_update_master_port = find_free_ports(1)[0]
        meta.nccl_group_name = self.weight_update_group_name

        # NOTE: Processes launched with torchrun will set the following env var to True,
        # which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        if self.is_pipeline_parallel_head():
            assert meta.gen_allocation is not None

            self.engine_lock.acquire()

            fut = self.rollout_engine.init_weights_update_group(meta)

            gen_world_size = meta.gen_allocation.parallel.world_size
            init_method = f"tcp://{format_host_for_url(meta.nccl_master_address)}:{meta.nccl_master_port}"
            self.logger.info(
                f"Initializing weight update group: type={meta.type} "
                f"init_method={init_method} "
                f"group={self.weight_update_group_name}"
            )
            self.weight_update_group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=gen_world_size + 1,
                init_method=init_method,
                rank=0,
                group_name=self.weight_update_group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )

            fut.result()

            self.engine_lock.release()

    def _serialize_mtp_tensors_for_update(
        self,
        mtp_hf_tensors: list[tuple[str, torch.Tensor]],
        tp_size: int,
    ) -> dict:
        """Serialize MTP tensors for /update_weights_from_tensor transport.

        Pre-serializes tensor data using SGLang's MultiprocessingSerializer
        with CUDA IPC handles, then base64-encodes for JSON/HTTP transport.
        This is required for single-controller mode where the engine proxy
        (RolloutCallback) communicates via HTTP.

        Args:
            mtp_hf_tensors: List of (name, tensor) pairs in HF format.
            tp_size: Tensor parallel size of inference engine.

        Returns:
            Dict with 'serialized_named_tensors' and 'flush_cache' keys,
            ready for /update_weights_from_tensor endpoint.
        """
        import time as _time

        _t_total = _time.time()
        _total_bytes = sum(t.numel() * t.element_size() for _, t in mtp_hf_tensors)
        _tensor_names = [name for name, _ in mtp_hf_tensors]
        _tensor_shapes = [tuple(t.shape) for _, t in mtp_hf_tensors]
        _tensor_dtypes = [str(t.dtype) for _, t in mtp_hf_tensors]
        _tensor_sizes = [t.numel() * t.element_size() for _, t in mtp_hf_tensors]
        self.logger.info(
            f"[MTPSerialize] ENTERED: n_tensors={len(mtp_hf_tensors)}, "
            f"tp_size={tp_size}, total_raw_bytes={_total_bytes} "
            f"({_total_bytes / 1024 / 1024:.2f} MB), "
            f"tensor_names={_tensor_names}, "
            f"tensor_shapes={_tensor_shapes}, "
            f"tensor_dtypes={_tensor_dtypes}, "
            f"tensor_sizes_bytes={_tensor_sizes}"
        )

        # -------------------------------------------------------------------
        # GPU -> CPU copy on a *dedicated CUDA stream* that is insulated
        # from NCCL broadcast dependencies.
        #
        # Recorded _mtp_data_ready_event on the default stream
        # BEFORE any NCCL broadcasts started (in _update_weights_from_
        # distributed).  Here we create a fresh stream that waits ONLY on
        # that event, then do all .cpu() copies on the fresh stream.
        # This stream has no NCCL dependencies, so its synchronize() is
        # instantaneous once the MTP all_gather data is ready.
        # -------------------------------------------------------------------
        _t_sync = _time.time()

        # Create a dedicated serialization stream free of NCCL deps
        _ser_stream = torch.cuda.Stream()

        _has_event = hasattr(self, "_mtp_data_ready_event") and self._mtp_data_ready_event is not None
        if _has_event:
            _evt_query = self._mtp_data_ready_event.query()
            # Make ser_stream wait for MTP data (all_gather) but NOT NCCL broadcasts
            _ser_stream.wait_event(self._mtp_data_ready_event)
            self.logger.info(
                "[MTPSerialize] Created serialization stream and synced with "
                f"_mtp_data_ready_event (pre-NCCL). event_query={_evt_query}, "
                f"(device={torch.cuda.current_device()}, "
                f"default_stream={torch.cuda.current_stream()}, "
                f"ser_stream={_ser_stream})"
            )
        else:
            # Fallback: no event recorded (shouldn't happen, but be safe).
            # Wait on the default stream which may include NCCL deps.
            _ser_stream.wait_stream(torch.cuda.current_stream())
            self.logger.warning(
                "[MTPSerialize] _mtp_data_ready_event NOT found! "
                "Falling back to wait_stream(current_stream) -- "
                "this may block on NCCL. "
                f"(device={torch.cuda.current_device()})"
            )

        # Synchronize the serialization stream -- this should be fast
        # since it only waits on the pre-NCCL event, not NCCL broadcasts.
        self.logger.info(
            "[MTPSerialize] About to _ser_stream.synchronize() ..."
        )
        _sync_timeout = 60.0
        _sync_warn_interval = 1.0
        _sync_start = _time.time()
        _warned = False
        while True:
            _ser_stream.synchronize()
            break
        _sync_elapsed = _time.time() - _sync_start
        if _has_event:
            _evt_query_after = self._mtp_data_ready_event.query()
        else:
            _evt_query_after = "N/A"
        self.logger.info(
            f"[MTPSerialize] Serialization stream synced in "
            f"{_sync_elapsed:.3f}s, event_query={_evt_query_after}"
        )

        # Reclaim Python-side references before GPU->CPU copies.
        # We skip torch.cuda.empty_cache() here because it can trigger
        # an implicit device-wide sync (cudaDeviceSynchronize) which
        # would re-introduce the NCCL deadlock under near-OOM conditions.
        import gc
        _t_cache = _time.time()
        gc.collect()
        _mem_reserved = torch.cuda.memory_reserved()
        self.logger.info(
            f"[MTPSerialize] gc.collect() completed "
            f"(reserved={_mem_reserved / 1024 / 1024:.0f} MB, "
            f"no empty_cache to avoid device-wide sync), "
            f"took {_time.time() - _t_cache:.3f}s"
        )

        try:
            from sglang.srt.utils import MultiprocessingSerializer
        except ImportError:
            self.logger.error(
                "[MTPSerialize] Failed to import MultiprocessingSerializer from sglang"
            )
            raise ImportError(
                "SGLang >= 0.5.9 is required for tensor weight updates. "
                "Install sglang to use MTP draft weight sync."
            )
        self.logger.info(
            "[MTPSerialize] MultiprocessingSerializer imported successfully"
        )

        try:
            from sglang.srt.model_executor.model_runner import LocalSerializedTensor
        except ImportError:
            self.logger.error(
                "[MTPSerialize] Failed to import LocalSerializedTensor from sglang"
            )
            raise ImportError(
                "Cannot import LocalSerializedTensor from SGLang. "
                "Ensure sglang >= 0.5.9 is installed."
            )
        self.logger.info("[MTPSerialize] LocalSerializedTensor imported successfully")

        _t_ser0 = _time.time()
        import io as _io
        import pickle as _pickle

        serialized_pairs = []
        for name, tensor in mtp_hf_tensors:
            _t_ser_i = _time.time()
            # Perform GPU→CPU copy on the serialization stream which
            # is free of NCCL cross-stream dependencies.
            with torch.cuda.stream(_ser_stream):
                _cpu_tensor = tensor.detach().cpu().contiguous()
            # Standard pickle -- no shared-memory, no CUDA IPC handles.
            _buf = _io.BytesIO()
            _pickle.dump(_cpu_tensor, _buf, protocol=_pickle.HIGHEST_PROTOCOL)
            _ser_data = _buf.getvalue()
            del _buf  # release buffer immediately
            _ser_len = len(_ser_data)
            serialized_pairs.append((name, _ser_data))
            self.logger.info(
                f"[MTPSerialize] Serialized tensor '{name}': "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                f"device={tensor.device}, "
                f"raw_bytes={tensor.numel() * tensor.element_size()}, "
                f"serialized_bytes={_ser_len} ({_ser_len / 1024 / 1024:.2f} MB), "
                f"took {_time.time() - _t_ser_i:.3f}s"
            )
        self.logger.info(
            f"[MTPSerialize] All inner serializations completed in "
            f"{_time.time() - _t_ser0:.3f}s, "
            f"n_pairs={len(serialized_pairs)}, "
            f"total_serialized_bytes={sum(len(d) for _, d in serialized_pairs)} "
            f"({sum(len(d) for _, d in serialized_pairs) / 1024 / 1024:.2f} MB)"
        )

        _t_wrap0 = _time.time()
        per_rank_named_tensors = [
            (name, LocalSerializedTensor(values=[data] * tp_size))
            for name, data in serialized_pairs
        ]
        self.logger.info(
            f"[MTPSerialize] LocalSerializedTensor wrapping completed in "
            f"{_time.time() - _t_wrap0:.3f}s, "
            f"n_entries={len(per_rank_named_tensors)}, tp_size={tp_size}"
        )

        import base64

        _t_outer0 = _time.time()
        _outer_payload = MultiprocessingSerializer.serialize(per_rank_named_tensors)
        _outer_len = len(_outer_payload)
        self.logger.info(
            f"[MTPSerialize] Outer MultiprocessingSerializer.serialize completed: "
            f"payload_bytes={_outer_len} ({_outer_len / 1024 / 1024:.2f} MB), "
            f"took {_time.time() - _t_outer0:.3f}s"
        )

        _t_b64_0 = _time.time()
        _b64_str = base64.b64encode(_outer_payload).decode("utf-8")
        _b64_len = len(_b64_str)
        self.logger.info(
            f"[MTPSerialize] base64 encode completed: "
            f"b64_str_len={_b64_len} ({_b64_len / 1024 / 1024:.2f} MB), "
            f"overhead_ratio={_b64_len / _outer_len:.2f}x, "
            f"took {_time.time() - _t_b64_0:.3f}s"
        )

        serialized_named_tensors = [_b64_str for _ in range(tp_size)]
        self.logger.info(
            f"[MTPSerialize] Replicated b64 payload for {tp_size} TP ranks, "
            f"total_b64_bytes={_b64_len * tp_size} "
            f"({_b64_len * tp_size / 1024 / 1024:.2f} MB)"
        )

        _t_total_elapsed = _time.time() - _t_total
        self.logger.info(
            f"[MTPSerialize] COMPLETED: total_time={_t_total_elapsed:.3f}s, "
            f"n_tensors={len(mtp_hf_tensors)}, tp_size={tp_size}, "
            f"raw_bytes={_total_bytes} ({_total_bytes / 1024 / 1024:.2f} MB), "
            f"final_b64_per_rank={_b64_len} ({_b64_len / 1024 / 1024:.2f} MB), "
            f"final_total_b64={_b64_len * tp_size} "
            f"({_b64_len * tp_size / 1024 / 1024:.2f} MB)"
        )

        return {
            "serialized_named_tensors": serialized_named_tensors,
            "flush_cache": True,
        }

    @trace_perf("megatron_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta) -> None:
        import time as _diag_time

        _diag_t0 = _diag_time.time()
        DeviceRuntimeInfo.get_current().log("_update_weights_from_distributed start")
        self.logger.info(
            f"[DiagUW] _update_weights_from_distributed ENTERED "
            f"(rank={dist.get_rank()}, version={meta.version}, "
            f"mem_alloc={torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB, "
            f"mem_reserved={torch.cuda.memory_reserved() / 1024 / 1024:.0f} MB)"
        )
        meta.nccl_master_address = self.weight_update_master_addr
        meta.nccl_master_port = self.weight_update_master_port
        meta.nccl_group_name = self.weight_update_group_name

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()

        self.logger.info(
            f"[DiagUW] Rank {dist.get_rank()} about to enter first cpu_group barrier "
            f"at elapsed={_diag_time.time() - _diag_t0:.3f}s"
        )
        dist.barrier(group=self.cpu_group)
        self.logger.info(
            f"[DiagUW] Rank {dist.get_rank()} passed first cpu_group barrier "
            f"at elapsed={_diag_time.time() - _diag_t0:.3f}s"
        )

        num_moe_experts = self.tf_config.num_moe_experts
        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

        buffer_size = 0
        converted_named_tensors = []

        mtp_param_count = 0
        mtp_param_bytes = 0
        mtp_hf_tensors = []
        _collect_mtp_for_draft = (
            self.enable_mtp_training
            and getattr(self, "_engine_supports_tensor_update", False)
            and self.is_pipeline_parallel_head()
        )

        _param_idx = 0
        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." in name:
                continue
            if ".mtp." in name:
                mtp_param_count += 1
                mtp_param_bytes += param.numel() * param.element_size()
                if _collect_mtp_for_draft:
                    _mtp_param, _ = self._collect_param(name, param)
                    _mtp_model_name = self.hf_config.model_type
                    _prev_count = len(mtp_hf_tensors)
                    mtp_hf_tensors.extend(
                        convert_to_hf(
                            self.tf_config,
                            _mtp_model_name,
                            name,
                            _mtp_param,
                            quantization_config=self.quantization_config,
                            fp8_direct_convert=self.fp8_direct_convert,
                        )
                    )
                    # Diagnostic: log each converted MTP tensor with value
                    # statistics for post-mortem debugging of weight corruption.
                    for _hf_name, _hf_tensor in mtp_hf_tensors[_prev_count:]:
                        _abs = _hf_tensor.float().abs()
                        self.logger.info(
                            f"[MTPWeightDiag] convert_to_hf: "
                            f"megatron={name} -> hf={_hf_name}, "
                            f"shape={tuple(_hf_tensor.shape)}, "
                            f"dtype={_hf_tensor.dtype}, "
                            f"mean={_hf_tensor.float().mean().item():.6e}, "
                            f"abs_mean={_abs.mean().item():.6e}, "
                            f"abs_max={_abs.max().item():.6e}, "
                            f"norm={_hf_tensor.float().norm().item():.6e}"
                        )
                else:
                    self._collect_param(name, param)
                continue
            if self.config.use_lora and (
                ".adapter." not in name or not getattr(param, "requires_grad", False)
            ):
                continue
            if _param_idx < 5 or _param_idx % 50 == 0:
                self.logger.info(
                    f"[DiagUW] Rank {dist.get_rank()} main_loop param[{_param_idx}] "
                    f"name={name}, size={param.numel() * param.element_size() / 1024 / 1024:.2f} MB, "
                    f"buffer_size={buffer_size / 1024 / 1024:.2f} MB, "
                    f"elapsed={_diag_time.time() - _diag_t0:.3f}s"
                )
            buffer_size = self._impl_update_weight_from_distributed(
                meta,
                name,
                param,
                converted_named_tensors,
                buffer_size,
                weight_chunked_mem_size,
            )
            if _param_idx < 5 or _param_idx % 50 == 0:
                self.logger.info(
                    f"[DiagUW] Rank {dist.get_rank()} main_loop param[{_param_idx}] "
                    f"DONE, buffer_size={buffer_size / 1024 / 1024:.2f} MB"
                )
            _param_idx += 1

        self.logger.info(
            f"[DiagUW] Parameter loop completed in "
            f"{_diag_time.time() - _diag_t0:.3f}s. "
            f"mtp_hf_tensors={len(mtp_hf_tensors)}, "
            f"converted_named_tensors={len(converted_named_tensors)}, "
            f"mtp_param_count={mtp_param_count}, "
            f"buffer_size={buffer_size}"
        )

        if mtp_hf_tensors:
            # [v5-F3] Compute norms for ALL tensors (was: only first 5).
            # [v5-F5] Track prev norm per-tensor to surface drift direction
            # and detect stall (draft model not learning from RL data).
            if not hasattr(self, "_mtp_sync_prev_norms"):
                self._mtp_sync_prev_norms = {}
            _all_norms = []
            _deltas = []
            _stall_tensors = []
            for _tn, _tv in mtp_hf_tensors:
                _cur = _tv.float().norm().item()
                _prev = self._mtp_sync_prev_norms.get(_tn)
                if _prev is None:
                    _all_norms.append((_tn, _cur, None))
                else:
                    _d = _cur - _prev
                    _all_norms.append((_tn, _cur, _d))
                    _deltas.append(abs(_d))
                    # [v9] bf16-quantization-aware STALL threshold. v8 used
                    # 0.05 * lr * norm which, for a LayerNorm of dim=4096
                    # (norm~64, bf16_eps per-element ~7.6e-6), yielded
                    # ~9.6e-6 — same order as the bf16 stochastic-rounding
                    # noise floor. That still mis-fired STALL 10/14 times
                    # in the 0428 v7 log even though mtp_loss was
                    # converging 646->145 (training clearly healthy).
                    # v9 formula: use bf16 round-trip error as the true
                    # floor, and ONLY warn after N consecutive sub-floor
                    # versions to avoid any transient data-shape blip.
                    #   bf16 eps ~= 2^-7 (relative), so quantization error
                    #   on |w| ~ 1 is ~7.8e-3 per element; for a tensor of
                    #   numel elements the L2-norm of the quantization
                    #   delta is ~sqrt(numel) * 7.8e-3 / 2 (average).  But
                    #   our metric is the delta between two norms, not
                    #   the norm of the delta, and the norm itself is
                    #   already rounded each time — so the per-sync
                    #   observable floor is ~2^-17 * norm ~= 7.6e-6 * norm.
                    try:
                        _mtp_lr_cur = float(
                            getattr(self, "_last_logged_mtp_lr", 3e-6)
                        )
                    except Exception:
                        _mtp_lr_cur = 3e-6
                    _bf16_floor = 7.6e-6 * max(_cur, 1.0)
                    _expected_drift = max(
                        1e-9, _mtp_lr_cur * max(_cur, 1.0) * 0.1
                    )
                    _stall_thr = max(_bf16_floor, _expected_drift)
                    if _cur > 0 and abs(_d) < _stall_thr:
                        _stall_tensors.append(_tn)
                self._mtp_sync_prev_norms[_tn] = _cur
            # Compact per-tensor summary line (rank-0 only to avoid DP-spam).
            try:
                _rank_v5 = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            except Exception:
                _rank_v5 = 0
            if _rank_v5 == 0:
                _fmt_parts = []
                for _tn, _cur, _d in _all_norms:
                    if _d is None:
                        _fmt_parts.append(f"{_tn}:{_cur:.4f}")
                    else:
                        _fmt_parts.append(f"{_tn}:{_cur:.4f}(Δ{_d:+.3e})")
                _drift_summary = ""
                if _deltas:
                    _max_d = max(_deltas)
                    _sum_d = sum(_deltas)
                    _drift_summary = f" | max|Δ|={_max_d:.3e} sum|Δ|={_sum_d:.3e}"
                self.logger.info(
                    "[MTPSyncDiag] MTP weight norms at sync "
                    "(version=%d, %d tensors): %s%s",
                    meta.version,
                    len(mtp_hf_tensors),
                    ", ".join(_fmt_parts),
                    _drift_summary,
                )
                # Windowed STALL: only warn if ALL of the last 3
                # consecutive versions flagged >=90% tensors stalled AND
                # the *cumulative* drift over the window is below floor.
                # This eliminates bf16 round-trip false alarms while
                if not hasattr(self, "_mtp_stall_window"):
                    self._mtp_stall_window = []  # list of (version, pct, sum_d)
                _this_pct = (
                    len(_stall_tensors) / len(_deltas) if _deltas else 0.0
                )
                _this_sum_d = sum(_deltas) if _deltas else 0.0
                self._mtp_stall_window.append(
                    (meta.version, _this_pct, _this_sum_d)
                )
                if len(self._mtp_stall_window) > 3:
                    self._mtp_stall_window.pop(0)
                # Diagnostic: always log the window state to make
                # subsequent triage self-evident.
                _win_fmt = ",".join(
                    f"v{v}:{p*100:.0f}%/Σ={s:.1e}"
                    for v, p, s in self._mtp_stall_window
                )
                _bf16_floor_total = 7.6e-6 * len(_deltas) * 64  # ~per-tensor floor * 64
                self.logger.info(
                    "[MTPSyncHealth-v9] STALL window (last %d syncs): "
                    "[%s] | bf16_floor_est=%.2e",
                    len(self._mtp_stall_window), _win_fmt,
                    _bf16_floor_total,
                )
                # One-line version->step audit trail.  Makes
                try:
                    _gn_audit = float(
                        getattr(self, "_last_mtp_grad_norm", 0.0)
                    )
                except Exception:
                    _gn_audit = 0.0
                try:
                    _step_audit = int(
                        getattr(self, "_global_step", 0)
                    )
                except Exception:
                    _step_audit = 0
                try:
                    _ntok_audit = int(
                        getattr(self, "_current_n_tokens", 0)
                    )
                except Exception:
                    _ntok_audit = 0
                try:
                    _nmb_audit = int(
                        getattr(self, "_current_num_microbatches", 0)
                    )
                except Exception:
                    _nmb_audit = 0
                self.logger.info(
                    "[MTPVersionAudit-v11] version=%d step=%d "
                    "mtp_grad_norm=%.4e num_mb=%d n_tokens=%d "
                    "max|Δ|=%.3e sum|Δ|=%.3e stalled_frac=%d/%d",
                    meta.version, _step_audit, _gn_audit,
                    _nmb_audit, _ntok_audit,
                    max(_deltas) if _deltas else 0.0,
                    sum(_deltas) if _deltas else 0.0,
                    len(_stall_tensors), len(_deltas),
                )
                if (
                    len(self._mtp_stall_window) >= 3
                    and all(p >= 0.9 for _, p, _ in self._mtp_stall_window)
                    and sum(s for _, _, s in self._mtp_stall_window)
                        < _bf16_floor_total * 2
                ):
                    self.logger.warning(
                        "[MTPSyncHealth] MTP training STALL detected at "
                        "version=%d (3 consecutive sub-floor syncs): "
                        "%d/%d tensors drift<floor, cum_sumΔ=%.3e. "
                        "Likely causes: (1) mtp_lr_scale too small, "
                        "(2) mtp_loss_scaling_factor too small, "
                        "(3) MTP gradient is being zeroed by detach. "
                        "Stalled tensors (head): %s",
                        meta.version, len(_stall_tensors), len(_deltas),
                        sum(s for _, _, s in self._mtp_stall_window),
                        ", ".join(_stall_tensors[:3]),
                    )
                    # the draft IS training and the "stall" is a bf16
                    # quantization artefact at the broadcast boundary.
                    _last_gn = float(getattr(self, "_last_mtp_grad_norm", 0.0))
                    # Additional liveness escape hatch
                    if _last_gn > 1e-4:
                        try:
                            _step_sup = int(
                                getattr(self, "_global_step", 0)
                            )
                        except Exception:
                            _step_sup = 0
                        self.logger.info(
                            "[MTPSyncHealth-v10] STALL candidate at "
                            "version=%d step=%d SUPPRESSED: "
                            "last mtp_grad_norm=%.4e > 1e-4 (draft IS "
                            "learning; bf16 quantization at broadcast "
                            "absorbs sub-ULP weight updates). Window: "
                            "%d/%d tensors<floor, cum_sumΔ=%.3e, "
                            "bf16_floor_est=%.3e.",
                            meta.version, _step_sup, _last_gn,
                            len(_stall_tensors), len(_deltas),
                            sum(s for _, _, s in self._mtp_stall_window),
                            _bf16_floor_total,
                        )

        # Record a CUDA event on the default stream BEFORE any NCCL
        # broadcasts begin.  At this point, all MTP tensors from
        # _collect_param()'s synchronous dist.all_gather() are fully
        # materialised on the default stream.  We will use this event
        # in _serialize_mtp_tensors_for_update() to create a separate
        # CUDA stream that depends ONLY on work up to this point --
        # crucially, NOT on the NCCL broadcast operations that follow.
        if _collect_mtp_for_draft and mtp_hf_tensors:
            self._mtp_data_ready_event = torch.cuda.Event()
            self._mtp_data_ready_event.record(torch.cuda.current_stream())
            _mtp_bytes_total = sum(
                t.numel() * t.element_size() for _, t in mtp_hf_tensors
            )
            self.logger.info(
                f"[DiagUW] Recorded _mtp_data_ready_event on default stream "
                f"(device={torch.cuda.current_device()}) BEFORE first NCCL "
                f"broadcast. n_mtp_tensors={len(mtp_hf_tensors)}, "
                f"mtp_param_count={mtp_param_count}, "
                f"mtp_param_bytes={mtp_param_bytes / 1024 / 1024:.2f} MB"
            )

        if converted_named_tensors:
            self.logger.info(
                f"[DiagUW] Calling _update_bucket_weights_from_distributed with "
                f"{len(converted_named_tensors)} tensors at elapsed="
                f"{_diag_time.time() - _diag_t0:.3f}s"
            )
            self._update_bucket_weights_from_distributed(meta, converted_named_tensors)
            self.logger.info(
                f"[DiagUW] _update_bucket_weights_from_distributed completed at elapsed="
                f"{_diag_time.time() - _diag_t0:.3f}s"
            )
        elif self.is_pipeline_parallel_head() and not self.config.use_lora:
            self.logger.warning(
                "No tensors were collected for distributed update at version %s.",
                meta.version,
            )

        if mtp_param_count > 0:
            self.logger.info(
                f"[MTPTrain] Weight sync: {mtp_param_count} MTP parameters "
                f"({mtp_param_bytes / 1024 / 1024:.2f} MB) synced to inference engine "
                f"at version={meta.version}"
            )
        elif self.enable_mtp_training:
            self.logger.warning(
                f"[MTPTrain] enable_mtp_training=True but 0 MTP parameters found "
                f"during weight sync at version={meta.version}. "
                f"MTP draft model weights will NOT be updated!"
            )

        if _collect_mtp_for_draft and mtp_hf_tensors and dist.get_rank() == 0:
            try:
                tp_size = (
                    meta.gen_allocation.parallel.tp_size
                    if meta.gen_allocation is not None
                    else 1
                )
                _mtp_bytes = sum(
                    t.numel() * t.element_size() for _, t in mtp_hf_tensors
                )
                import time as _time

                self.logger.info(
                    f"[DiagUW] About to serialize and send {len(mtp_hf_tensors)} MTP tensors "
                    f"({_mtp_bytes / 1024 / 1024:.2f} MB) to EAGLE draft model "
                    f"via /update_weights_from_tensor "
                    f"(tp_size={tp_size}, version={meta.version}), "
                    f"elapsed={_diag_time.time() - _diag_t0:.3f}s, "
                    f"mem_alloc={torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB, "
                    f"mem_reserved={torch.cuda.memory_reserved() / 1024 / 1024:.0f} MB"
                )
                _t_ser0 = _time.time()
                self.logger.info(
                    f"[DiagUW] Starting _serialize_mtp_tensors_for_update "
                    f"(n_tensors={len(mtp_hf_tensors)}, tp_size={tp_size})..."
                )
                serialized_payload = self._serialize_mtp_tensors_for_update(
                    mtp_hf_tensors, tp_size
                )
                _t_ser1 = _time.time()
                _sp_keys = (
                    list(serialized_payload.keys())
                    if isinstance(serialized_payload, dict)
                    else "N/A"
                )
                _n_snt = (
                    len(serialized_payload.get("serialized_named_tensors", []))
                    if isinstance(serialized_payload, dict)
                    else 0
                )
                _snt_sizes = (
                    [
                        len(s)
                        for s in serialized_payload.get("serialized_named_tensors", [])
                    ]
                    if isinstance(serialized_payload, dict)
                    else []
                )
                self.logger.info(
                    f"[DiagUW] Serialization completed in {_t_ser1 - _t_ser0:.3f}s. "
                    f"payload_keys={_sp_keys}, n_serialized_tensors={_n_snt}, "
                    f"serialized_tensor_sizes_bytes={_snt_sizes}, "
                    f"rollout_engine_type={type(self.rollout_engine).__name__}"
                )
                _t_call0 = _time.time()
                self.logger.info(
                    f"[DiagUW] Calling rollout_engine.update_weights_from_tensor()... "
                    f"(engine_type={type(self.rollout_engine).__name__})"
                )
                self.rollout_engine.update_weights_from_tensor(
                    serialized_payload=serialized_payload,
                    flush_cache=True,
                )
                _t_call1 = _time.time()
                self.logger.info(
                    f"[DiagUW] Successfully updated EAGLE draft model "
                    f"MTP weights at version={meta.version} "
                    f"(serialize={_t_ser1 - _t_ser0:.3f}s, "
                    f"update_call={_t_call1 - _t_call0:.3f}s, "
                    f"total={_t_call1 - _t_ser0:.3f}s, "
                    f"overall_elapsed={_diag_time.time() - _diag_t0:.3f}s)"
                )
            except Exception as e:
                self.logger.error(
                    f"[MTPTrain] Failed to update EAGLE draft model "
                    f"MTP weights via tensor update: {e}. "
                    f"Draft model spec_accept_rate will degrade!",
                    exc_info=True,
                )
        elif (
            self.enable_mtp_training
            and not getattr(self, "_engine_supports_tensor_update", False)
            and not self._mtp_tensor_update_warned
        ):
            self._mtp_tensor_update_warned = True
            self.logger.warning(
                "[MTPTrain] Inference engine does not support "
                "update_weights_from_tensor. EAGLE draft model MTP weights "
                "will NOT be updated, causing spec_accept_rate degradation. "
                "Ensure SGLang backend is used with speculative decoding."
            )

        self.logger.info(
            f"[DiagUW] About to enter first dist.barrier(cpu_group) [after MTP update] "
            f"at elapsed={_diag_time.time() - _diag_t0:.3f}s"
        )
        dist.barrier(group=self.cpu_group)

        buffer_size = 0
        named_tensors = []

        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." not in name or self.config.use_lora:
                continue
            buffer_size = self._impl_update_expert_weight_from_distributed(
                meta,
                name,
                param,
                named_tensors,
                buffer_size,
                weight_chunked_mem_size,
            )

        if named_tensors:
            self.logger.info(
                f"[DiagUW] Calling _update_bucket_expert_weights_from_distributed "
                f"with {len(named_tensors)} expert tensors at elapsed="
                f"{_diag_time.time() - _diag_t0:.3f}s"
            )
            self._update_bucket_expert_weights_from_distributed(meta, named_tensors)

        self.logger.info(
            f"[DiagUW] About to enter second dist.barrier(cpu_group) [after expert update] "
            f"at elapsed={_diag_time.time() - _diag_t0:.3f}s"
        )
        dist.barrier(group=self.cpu_group)

        if dist.get_rank() == 0:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.logger.info(
            f"[DiagUW] _update_weights_from_distributed FULLY COMPLETED "
            f"in {_diag_time.time() - _diag_t0:.3f}s"
        )

    @trace_perf("megatron_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self, meta: WeightUpdateMeta) -> None:
        DeviceRuntimeInfo.get_current().log("_update_weights_from_disk start")
        fut = Future()

        if dist.get_rank() == 0:
            fut = self.rollout_engine.update_weights_from_disk(meta)

        self._save_model_to_hf(meta.path, self.tokenizer, None)
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
        tokenizer: Any | None = None,
        processor: Any | None = None,
        base_model_path: str | None = None,
    ) -> None:
        assert self.model is not None, "Model is not initialized."
        DeviceRuntimeInfo.get_current().log("_save_model_to_hf before gc/empty_cache")
        gc.collect()
        current_platform.empty_cache()
        DeviceRuntimeInfo.get_current().log("_save_model_to_hf after gc/empty_cache")
        os.makedirs(path, exist_ok=True)

        if self.bridge_cls == "megatron-bridge":
            if self.config.is_critic:
                raise ValueError(
                    "Saving critic model is not supported with megatron-bridge."
                )
            if self.config.use_lora:
                self.bridge.save_hf_adapter(
                    self.model,
                    path=path,
                    peft_config=self.bridge_lora,
                    base_model_name_or_path=base_model_path or self.config.path,
                )
            else:
                self.bridge.save_hf_pretrained(
                    self.model,
                    path,
                    source_path=base_model_path,
                )
        else:
            save_weights_to_hf_with_mbridge_fast(
                bridge=self.bridge,
                models=self.model,
                weights_path=path,
                base_model_path=base_model_path,
                max_shard_size_byte=int(3e9),
                max_workers=None,
                is_critic=self.config.is_critic,
                fp8_direct_convert=self.fp8_direct_convert,
            )

        DeviceRuntimeInfo.get_current().log("_save_model_to_hf after save_weights")

        if dist.get_rank() == 0:
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if processor is not None:
                processor.save_pretrained(path)

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def _load_model_from_hf(self, path: str) -> None:
        assert self.model is not None, "Model is not initialized."

        if self.bridge_cls == "megatron-bridge":
            if self.config.is_critic:
                raise ValueError(
                    "Loading critic model is not supported with megatron-bridge."
                )
            self.bridge.load_hf_weights(self.model, hf_path=path)
        else:
            load_weights_from_hf_with_mbridge_fast(
                bridge=self.bridge,
                models=self.model,
                weights_path=path,
                max_workers=None,
                is_critic=self.config.is_critic,
                fp8_direct_convert=self.fp8_direct_convert,
            )

    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        # Parallel sizes
        pp_size = self.parallel_strategy.pipeline_parallel_size
        cp_size = self.parallel_strategy.context_parallel_size
        tp_size = self.parallel_strategy.tensor_parallel_size
        if self.enable_tree_training:
            assert cp_size == 1, (
                "Context parallelism is not supported in tree training."
            )
            mb_list = build_packed_tree_batch(
                input_,
                mb_spec=self.config.mb_spec,
                pad_to_maximum=self.config.pad_to_maximum,
                dp_group=self.data_parallel_group,
                parallel_size=tp_size,
            )
            recommended_min_n_mbs = 2 * pp_size if pp_size > 1 else 1
            self.logger.info(
                f"Packed tree #microbatch: {len(mb_list)}, microbatch #tokens: {mb_list.group_lens}, "
                f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}."
            )
            if len(mb_list) < recommended_min_n_mbs:
                self.logger.warning(
                    f"Number of tree micro-batches ({len(mb_list)}) is less than recommended"
                    f" minimum ({recommended_min_n_mbs}) to avoid pipeline bubbles."
                )
            return mb_list
        # Amend position ids
        input_ = amend_position_ids(input_)
        # Split the input into micro-batches
        # NOTE: Here we use 2*pp_size in forward to align logprob precision
        # TODO: Performance check
        min_n_mbs = (
            2 * pp_size if pp_size > 1 else 1
        )  # avoid pipeline bubbles in training
        # NOTE: self.config.mb_spec.max_tokens_per_mb determines
        # the expected **total** number of tokens per micro-batch **in the forward pass**.
        # The micro batch list splitted here will be splitted to each
        # context parallel rank, so the total number of tokens per
        # GPU in a forward pass here will be `max_tokens_per_mb / cp_size`.
        mb_spec = MicroBatchSpec.new(
            self.config.mb_spec,
            n_mbs=max(min_n_mbs, self.config.mb_spec.n_mbs),
            n_mbs_divisor=pp_size,
        )
        mb_list = split_padded_tensor_dict_into_mb_list(
            input_,
            mb_spec,
            group=mpu.get_data_parallel_group(),
        )
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        # NOTE: Pad micro-batches to:
        # 1. Reduce GPU memory fragmentation, pad actual # tokens per mb to integer multiples
        #  of GPU page size or max_tokens_per_mb
        # 2. Align sequence lengths to integer multiples of `align_to_multiple_of=tp_size*cp_size*2`
        #    to satisfy the requirement of Megatron parallelism.
        align_to_multiple_of = tp_size * cp_size * 2 if cp_size > 1 else tp_size
        align_to_multiple_of = (
            math.lcm(align_to_multiple_of, DEFAULT_VECTORIZED_ALIGNMENT_BYTES)
            if self.enable_fp8
            else align_to_multiple_of
        )
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
            seq_align_to=align_to_multiple_of,
        )
        self.logger.info(
            f"#microbatch: {len(mb_list.group_lens)}, microbatch #tokens: {mb_list.group_lens}, "
            f"aligned to: {mb_list.align_to_lengths}, padded to: {mb_list.padded_to_lengths}, "
            f"padding lengths: {mb_list.padding_lengths}."
        )
        # Modern model implementations takes a dict as the input.
        # This eliminates a bug of Qwen2.5-VL for transformers<=4.53.1
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)
        for mb in mb_list.mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
        for mb in mb_list.padded_mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
        return mb_list

    def _compute_logprobs_and_loss(
        self,
        output: torch.Tensor,
        inputs: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        _mtp_loss_for_this_mb = None
        if (
            self.enable_mtp_training
            and hasattr(self, '_mtp_loss_for_backward')
            and self._mtp_loss_for_backward
        ):
            _mtp_loss_for_this_mb = self._mtp_loss_for_backward.pop(0)

        local_weight = loss_weight_fn(inputs)
        if local_weight == 0:
            return output.mean() * 0.0

        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )
        if not self.config.is_critic:
            if self.enable_tree_training:
                # Handle dummy trie (empty tree for DP synchronization)
                # When trie has no sequences, return zero loss with grad connection
                trie_node = inputs.get("trie_node")
                if trie_node is None or not trie_node.all_sequence_ids:
                    # Return zero loss that maintains gradient connection to output
                    # This ensures backward() works correctly for distributed synchronization
                    return output.mean() * 0.0

                # For tree training, use gather_packed_tree_vocab_stats to properly
                # unpack vocab stats from tree structure back to per-sequence format.
                # This is necessary because the logits are in packed tree format where
                # multiple sequences share prefix positions.
                vocab_min_logits, vocab_max_logits = gather_packed_tree_vocab_stats(
                    output, trie_node
                )
                logprobs, entropy = gather_packed_tree_logprobs_entropy(
                    output,
                    trie_node,
                    inputs["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=mpu.get_tensor_model_parallel_group()
                    if mpu.get_tensor_model_parallel_world_size() > 1
                    else None,
                )
            else:
                labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
                logprobs, entropy = gather_logprobs_entropy(
                    output,
                    labels,
                    temperature=self.config.temperature,
                    tp_group=mpu.get_tensor_model_parallel_group()
                    if mpu.get_tensor_model_parallel_world_size() > 1
                    else None,
                )
                vocab_min_logits = output.detach().min(-1).values.float()
                vocab_max_logits = output.detach().max(-1).values.float()
            loss = loss_fn(
                logprobs,
                entropy,
                inputs,
                vocab_min_logits=vocab_min_logits,
                vocab_max_logits=vocab_max_logits,
            )
        else:
            values = output.squeeze(-1)
            loss = loss_fn(values, inputs)

        loss_scale = local_weight / total_loss_weight * loss_multiplier

        if _mtp_loss_for_this_mb is not None:
            _mtp_contribution_raw = _mtp_loss_for_this_mb.sum()
            # --- MTP loss adaptive clipping (Fix: prevent loss spike feedback loop) ---
            # When mtp_detach_heads=True, MTP trains independently of backbone.
            # A sudden MTP loss spike (e.g., 5x normal) causes large gradient
            # updates that destabilize the draft model, crashing accept rate,
            # which in turn produces worse training data -> even higher loss.
            # Clipping breaks this positive feedback loop.
            _mtp_clip_threshold = 5.0  # Clip if loss > 5x EMA
            _mtp_ema_decay = 0.95
            _mtp_contribution = _mtp_contribution_raw
            _mtp_was_clipped = False
            self._mtp_loss_total_count += 1
            if self._mtp_loss_ema is None:
                # Initialize EMA with first observed value
                self._mtp_loss_ema = _mtp_contribution_raw.detach().item()
            else:
                _raw_val = _mtp_contribution_raw.detach().item()
                _ema_val = self._mtp_loss_ema
                if _ema_val > 0 and _raw_val > _mtp_clip_threshold * _ema_val:
                    # Clip: scale down to threshold * EMA
                    _clip_ratio = (_mtp_clip_threshold * _ema_val) / _raw_val
                    _mtp_contribution = _mtp_contribution_raw * _clip_ratio
                    _mtp_was_clipped = True
                    self._mtp_loss_clip_count += 1
                    self.logger.warning(
                        "[MTPLossClip] MTP loss clipped: raw=%.4f, ema=%.4f, "
                        "threshold=%.1fx, clip_ratio=%.4f, clipped=%.4f, "
                        "clip_count=%d/%d",
                        _raw_val, _ema_val, _mtp_clip_threshold,
                        _clip_ratio, _mtp_contribution.detach().item(),
                        self._mtp_loss_clip_count, self._mtp_loss_total_count,
                    )
                # Update EMA (use raw value for stable tracking, not clipped)
                self._mtp_loss_ema = (
                    _mtp_ema_decay * _ema_val
                    + (1 - _mtp_ema_decay) * _raw_val
                )
            loss = loss + _mtp_contribution
            _n = self._mtp_loss_total_count
            if _n <= 4 or _n % 100 == 0:
                self.logger.info(
                    "[MTPLossDiag] MTP loss added to RL loss (call #%d): "
                    "raw=%.6f, applied=%.6f, clipped=%s, "
                    "ema=%.6f, rl_before=%.6f, combined=%.6f, "
                    "loss_scale=%.6f",
                    _n,
                    _mtp_contribution_raw.detach().item(),
                    _mtp_contribution.detach().item(),
                    _mtp_was_clipped,
                    self._mtp_loss_ema if self._mtp_loss_ema else 0.0,
                    (loss - _mtp_contribution).detach().item(),
                    loss.detach().item(),
                    loss_scale,
                )

        if _mtp_loss_for_this_mb is not None and abs(loss_scale) > 0:
            # [v8] Refresh cached MTP LR from optimizer param_groups so the
            # DoubleScale log and SyncHealth STALL threshold can use the
            # realised LR (not a hardcoded default).
            try:
                for _pg in getattr(self.optimizer, "param_groups", []):
                    _nm = str(_pg.get("name", ""))
                    if "mtp" in _nm.lower():
                        self._last_logged_mtp_lr = float(_pg.get("lr", 3e-6))
                        break
            except Exception:
                pass
            # Match Megatron-native MTPLossAutoScaler:
            #   schedules.py sets main_loss_backward_scale = loss_scale
            #   / num_microbatches.
            _num_mb = max(1, int(getattr(self, "_current_num_microbatches", 1)))
            _inv = 1.0 / (loss_scale * _num_mb)
            # Subtract already-added mtp and re-add with corrected scaling so
            # `loss * loss_scale` contributes (mtp_loss_scale * mtp_loss) /
            # num_mb per microbatch
            loss = (loss - _mtp_contribution) + _mtp_contribution * _inv
            _n_ds = self._mtp_loss_total_count
            if _n_ds <= 4 or _n_ds % 100 == 0:
                _eff_per_mb = (
                    _mtp_contribution.detach().item() * _inv * loss_scale
                )
                # Also surface the realised per-step MTP weight update
                # magnitude estimate (= eff_contrib * mtp_lr). This directly
                # monitors whether the draft head is actually learning, and
                # its drift exposes data-shape driven instability
                try:
                    _mtp_lr_dbg = float(
                        getattr(self, "_last_logged_mtp_lr", 3e-6)
                    )
                except Exception:
                    _mtp_lr_dbg = 3e-6
                _eff_step_mag = _eff_per_mb * _mtp_lr_dbg
                self.logger.info(
                    "[MTPFix-DoubleScale-v6] Inverse-(loss_scale*num_mb) "
                    "applied: loss_scale=%.6f, num_mb=%d, inv=%.4f, "
                    "mtp_contribution=%.6f, effective_mtp_contrib_per_mb="
                    "%.6f, mtp_lr=%.3e, effective_per_step_update~=%.3e "
                    "(warn if <1e-8; accumulated over num_mb MBs = "
                    "mtp_loss_scale * mtp_loss; verl/megatron-native "
                    "equivalent).",
                    loss_scale, _num_mb, _inv,
                    _mtp_contribution.detach().item(),
                    _eff_per_mb, _mtp_lr_dbg, _eff_step_mag,
                )
                # Data-shape diagnostic.
                try:
                    _n_tokens = int(
                        getattr(self, "_current_n_tokens", 0)
                    )
                except Exception:
                    _n_tokens = 0
                _tok_per_mb = (
                    _n_tokens / max(1, _num_mb) if _n_tokens else 0
                )
                self.logger.info(
                    "[MTPDataShapeDiag-v9] num_mb=%d n_tokens=%d "
                    "tokens_per_mb=%.0f eff_per_step_update=%.3e "
                    "(accept_rate regressions in v7 log at num_mb "
                    "drop should show up here as correlated drops "
                    "in eff_per_step_update or tokens_per_mb).",
                    _num_mb, _n_tokens, _tok_per_mb, _eff_step_mag,
                )
                # Rolling 5-step token-count trend to surface
                # sequence-length collapse BEFORE it manifests as an
                # accept_rate drop.
                if not hasattr(self, "_mtp_tok_trend"):
                    self._mtp_tok_trend = []  # list[(step, n_tokens, num_mb)]
                try:
                    _gstep_v11 = int(getattr(self, "_global_step", 0))
                except Exception:
                    _gstep_v11 = 0
                self._mtp_tok_trend.append(
                    (_gstep_v11, int(_n_tokens), int(_num_mb))
                )
                if len(self._mtp_tok_trend) > 5:
                    self._mtp_tok_trend.pop(0)
                if (
                    len(self._mtp_tok_trend) >= 5
                    and self._mtp_tok_trend[0][1] > 0
                    and _n_tokens > 0
                ):
                    _prev_avg = sum(
                        t for _, t, _ in self._mtp_tok_trend[:-1]
                    ) / max(1, len(self._mtp_tok_trend) - 1)
                    _drop_pct = (
                        1.0 - _n_tokens / _prev_avg
                    ) if _prev_avg > 0 else 0.0
                    _tok_trend_msg = ",".join(
                        f"s{s}:{t//1000}k/{n}mb"
                        for s, t, n in self._mtp_tok_trend
                    )
                    if _drop_pct > 0.3:
                        self.logger.warning(
                            "[MTPDataTrend-v11] SEQUENCE-LENGTH "
                            "COLLAPSE: n_tokens dropped %.1f%% vs "
                            "5-step trailing avg (%.0f -> %d). "
                            "Trend: [%s]. Draft head will see "
                            "fewer tokens per update; accept_rate "
                            "regression is likely within 1-2 "
                            "versions. Mitigations: raise "
                            "mtp_loss_scaling_factor, enable reward "
                            "clipping, or widen rollout batch.",
                            _drop_pct * 100.0, _prev_avg, _n_tokens,
                            _tok_trend_msg,
                        )
                    elif _drop_pct > 0.15:
                        self.logger.info(
                            "[MTPDataTrend-v11] mild token drop "
                            "%.1f%% (%.0f -> %d) over last 5 "
                            "steps. Trend: [%s].",
                            _drop_pct * 100.0, _prev_avg, _n_tokens,
                            _tok_trend_msg,
                        )

        return loss * loss_scale

    def _compute_forward_result(
        self,
        output: torch.Tensor,
        inputs: dict[str, Any],
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )
        if not self.config.is_critic:
            if self.enable_tree_training:
                logprobs = _gather_packed_tree_logprobs(
                    output,
                    inputs["trie_node"],
                    inputs["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=mpu.get_tensor_model_parallel_group()
                    if mpu.get_tensor_model_parallel_world_size() > 1
                    else None,
                )
                return logprobs
            labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
            logprobs = gather_logprobs(
                output,
                labels,
                temperature=self.config.temperature,
                tp_group=mpu.get_tensor_model_parallel_group()
                if mpu.get_tensor_model_parallel_world_size() > 1
                else None,
            )
            return logprobs
        else:
            values = output.squeeze(-1)
            return values


# =============================================================================
# Algorithm-specific Megatron Engines
# =============================================================================


class MegatronPPOActor(MegatronEngine):
    """PPO Actor implementation using Megatron backend."""

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


class MegatronPPOCritic(MegatronEngine):
    """PPO Critic implementation using Megatron backend."""

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


class MegatronLMEngine(MegatronEngine):
    """Language model engine for SFT using Megatron backend."""

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


class MegatronRWEngine(MegatronEngine):
    """Reward model engine using Megatron backend."""

    def __init__(self, config: TrainEngineConfig):
        from copy import deepcopy

        from areal.trainer.rw.rw_engine import RWEngine

        super().__init__(config)
        self.rw_engine = RWEngine(self)
        if self.config.mb_spec.granularity != 2:
            rw_logger = logging.getLogger("RWEngine")
            rw_logger.warning("mb_spec.granularity must be 2 for reward modeling")
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
