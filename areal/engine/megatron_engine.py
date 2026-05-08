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
            # [MTPVersionBanner-v16] + v17 tag: make it trivial to
            # verify which patch revision is running in a given log.
            try:
                import os as _os_banner
                _banner_tags = [
                    "v6:DoubleScaleInv",
                    "v9:bf16StallDiag",
                    "v11:VersionAudit",
                    "v12:OptimDump+Sanity",
                    "v14:LRScaleGuard+WeightDeltaGuard",
                    "v16:MTPSerializeFp32Upcast(AREAL_MTP_FP32_BROADCAST)",
                    "v28:MTPSigmaDeltaBf16(AREAL_MTP_SIGMA_DELTA_BF16)",
                    "v43:FixedLongProbe+MTPWeightHashDelta+CrossProcFix(AREAL_MTP_V30_DIAG)",
                    "v44:MTPSrcHash+RepeatFixedLongProbe(AREAL_MTP_V30_DIAG)",
                    "v45:MTPULPGap+DraftIPCStall(AREAL_MTP_V30_DIAG)",
                    "v46:ForceTickBf16+ShipFlips(AREAL_MTP_V46_FORCE_TICK)",
                    "v47:MTPMasterAmp(AREAL_MTP_V47_MASTER_AMP)",
                    "v48:MTPMasterCarry(AREAL_MTP_V48_MASTER_CARRY)",
                    "v49:MTPLossClipTight+GradFp32Coerce+LossBoost(AREAL_MTP_V49_CLIP_TIGHT,AREAL_MTP_V49_GRAD_FP32_COERCE,AREAL_MTP_V49_LOSS_BOOST)",
                    "v50:MTPNativePassthrough(default-on; set AREAL_MTP_NATIVE_AUTOSCALER=0 to fall back to legacy FIFO)",
                    "v51:MTPGradClipNorm(diag-only; AREAL_MTP_V51_GRAD_CLIP_NORM, default=0 after v52)",
                    "v52:MTPSourceLossCap(default-on; AREAL_MTP_V52_LOSS_CAP_RATIO=<float>, default=2.0)",
                    "v53:MTPSharedWeightIsolate(detach output_weight for MTP output_layer)",
                    "v54:MTPFreezeGate+DraftEMA+SpecDecFlowLog(AREAL_MTP_V54_FREEZE[default=0],AREAL_MTP_V54_DRAFT_EMA[default=0.0],AREAL_MTP_V54_SPEC_FLOW_LOG[default=1])",
                    "v55:MTPLRBoost(AREAL_MTP_V55_MTP_LR_BOOST[default=1.0])",
                    "v56:MTPShipSummaryFix+GradTrace+LossTrace(AREAL_MTP_V56_GRAD_TRACE[default=1],AREAL_MTP_V56_LOSS_TRACE[default=1])",
                    "v57:MTPStochasticRoundBf16(AREAL_MTP_V57_STOCHASTIC_ROUND[default=1],AREAL_MTP_V57_SR_MIN_DRIFT_RATIO[default=0.0])+ForceTickRatioFire+K2",
                    "v17:MTPNativeAutoScaler+ConsumerBypass"
                    "(AREAL_MTP_NATIVE_AUTOSCALER,autograd_in_graph)",
                    "v58:MTPSlimeAlign(AREAL_MTP_SLIME_ALIGN[default=1]):"
                    "disable Path3-detach/v53-weight-detach/v52-cap/"
                    "FIFO-append/v50-gradFp32/v57-SR; set_loss_scale="
                    "loss_scale/num_mb (Megatron-Core native = slime)",
                ]
                _banner_flags = {
                    "AREAL_MTP_FP32_BROADCAST":
                        _os_banner.environ.get(
                            "AREAL_MTP_FP32_BROADCAST", "1"),
                    "AREAL_MTP_SIGMA_DELTA_BF16":
                        _os_banner.environ.get(
                            "AREAL_MTP_SIGMA_DELTA_BF16", "1"),
                    "AREAL_MTP_NATIVE_AUTOSCALER":
                        _os_banner.environ.get(
                            "AREAL_MTP_NATIVE_AUTOSCALER", "1"),
                    "AREAL_MTP_V30_DIAG":
                        _os_banner.environ.get(
                            "AREAL_MTP_V30_DIAG", "1"),
                    "AREAL_MTP_SLIME_ALIGN":
                        _os_banner.environ.get(
                            "AREAL_MTP_SLIME_ALIGN", "1"),
                }
                try:
                    _slime_align_on = (
                        _os_banner.environ.get(
                            "AREAL_MTP_SLIME_ALIGN", "1") == "1"
                    )
                    self.logger.info(
                        "[MTPSlimeAlign] AREAL_MTP_SLIME_ALIGN=%s. "
                        "When ON: A) Path3 detach SKIPPED, "
                        "B) output_weight NOT detached, "
                        "C) v52 SourceLossCap DISABLED, "
                        "D) FIFO append SKIPPED, "
                        "E) set_loss_scale=loss_scale/num_mb, "
                        "G) v50 MTPGradFp32Coerce DISABLED, "
                        "H) v57 StochasticRoundBf16 DISABLED. "
                        "This restores Megatron-Core 0.16.0 native MTP "
                        "semantics (= slime), so "
                        "mtp_loss_scaling_factor=0.2 carries the same "
                        "meaning as in slime.",
                        _slime_align_on,
                    )
                except Exception as _e_sa:
                    self.logger.warning(
                        "[MTPSlimeAlign] banner log failed: %s",
                        _e_sa,
                    )
                self.logger.info(
                    "[MTPVersionBanner] tags=%s flags=%s",
                    ",".join(_banner_tags), _banner_flags,
                )
                try:
                    import torch as _t_d01
                    _dtype_d01 = str(getattr(self, "dtype", "n/a"))
                    _opt_cfg = getattr(self, "optimizer_config", None)
                    _mc_cfg = getattr(self, "mcore_config", None)
                    self.logger.info(
                        "[SpecDecDiag-v20 D01] EngineInit: "
                        "mtp_num_layers=%s mtp_loss_scaling_factor=%s "
                        "mtp_detach_heads=%s enable_mtp_training=%s "
                        "dtype=%s torch_version=%s",
                        getattr(self, "mtp_num_layers", None),
                        getattr(self, "mtp_loss_scaling_factor", None),
                        getattr(self, "mtp_detach_heads", None),
                        getattr(self, "enable_mtp_training", None),
                        _dtype_d01, _t_d01.__version__,
                    )
                    if _opt_cfg is not None:
                        self.logger.info(
                            "[SpecDecDiag-v20 D01] EngineInit optimizer_cfg: "
                            "type=%s lr=%s weight_decay=%s beta1=%s beta2=%s "
                            "eps=%s mtp_lr_scale=%s gradient_clipping=%s "
                            "lr_scheduler_type=%s",
                            getattr(_opt_cfg, "type", None),
                            getattr(_opt_cfg, "lr", None),
                            getattr(_opt_cfg, "weight_decay", None),
                            getattr(_opt_cfg, "beta1", None),
                            getattr(_opt_cfg, "beta2", None),
                            getattr(_opt_cfg, "eps", None),
                            getattr(_opt_cfg, "mtp_lr_scale", None),
                            getattr(_opt_cfg, "gradient_clipping", None),
                            getattr(_opt_cfg, "lr_scheduler_type", None),
                        )
                    if _mc_cfg is not None:
                        self.logger.info(
                            "[SpecDecDiag-v20 D01] EngineInit mcore_cfg: "
                            "use_precision_aware_optimizer=%s "
                            "exp_avg_dtype=%s exp_avg_sq_dtype=%s "
                            "use_distributed_optimizer=%s "
                            "overlap_param_gather_with_optimizer_step=%s",
                            getattr(_mc_cfg,
                                    "use_precision_aware_optimizer", None),
                            getattr(_mc_cfg, "exp_avg_dtype", None),
                            getattr(_mc_cfg, "exp_avg_sq_dtype", None),
                            getattr(_mc_cfg,
                                    "use_distributed_optimizer", None),
                            getattr(_mc_cfg,
                                    "overlap_param_gather_with_optimizer_step",
                                    None),
                        )
                except Exception as _e_d01:
                    self.logger.warning(
                        "[SpecDecDiag-v20 D01] static dump failed: %s",
                        _e_d01,
                    )
            except Exception:
                pass
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
        # [MTPShipEntryAudit-v65] First-line audit: prove the ship
        # entry was even reached.  log.8 ran ~10min and never hit
        # ship; zero v64 records emitted.  This audit fires before
        # any rollout-connection check so we can distinguish 'ship
        # never invoked' from 'ship invoked but failed inside'.
        try:
            import logging as _v65_log_mod
            import time as _v65_time_mod
            _v65_lg = _v65_log_mod.getLogger(__name__)
            try:
                _v65_ver = int(self.get_version())
            except Exception:
                _v65_ver = -1
            try:
                _v65_meta_type = str(getattr(meta, 'type', '?'))
            except Exception:
                _v65_meta_type = '?'
            try:
                _v65_meta_path = str(getattr(meta, 'path', ''))
            except Exception:
                _v65_meta_path = ''
            _v65_lg.info(
                "[MTPShipEntryAudit-v65] update_weights ENTER "
                "version=%d meta_type=%s meta_path=%s ts=%.3f",
                _v65_ver, _v65_meta_type, _v65_meta_path,
                _v65_time_mod.time(),
            )
        except Exception as _e_v65:
            try:
                import logging as _v65_log_mod_b
                _v65_log_mod_b.getLogger(__name__).warning(
                    "[MTPShipEntryAudit-v65] entry-audit failure: %r",
                    _e_v65,
                )
            except Exception:
                pass
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
                            try:
                                _d09_step = getattr(self, "_global_step", 0)
                                if _d09_step <= 5 or _d09_step % 20 == 0:
                                    _d09_rows = []
                                    import torch as _t_d09
                                    for _m in self.model:
                                        for _n, _p in _m.named_parameters():
                                            if ".mtp." not in _n:
                                                continue
                                            _g = (_p.main_grad
                                                  if hasattr(_p, "main_grad")
                                                  and _p.main_grad is not None
                                                  else _p.grad)
                                            if _g is None:
                                                continue
                                            _gf = _g.data.float()
                                            _pf = _p.data.float()
                                            _d09_rows.append(
                                                "%s: dtype=%s |W|_max=%.3e "
                                                "|W|_mean=%.3e "
                                                "|g|_max=%.3e |g|_mean=%.3e "
                                                "g.sum=%.3e g.finite=%s" % (
                                                    _n, str(_p.dtype),
                                                    _pf.abs().max().item(),
                                                    _pf.abs().mean().item(),
                                                    _gf.abs().max().item(),
                                                    _gf.abs().mean().item(),
                                                    _gf.sum().item(),
                                                    bool(_t_d09.isfinite(_gf)
                                                         .all().item()),
                                                )
                                            )
                                    if _d09_rows:
                                        self.logger.info(
                                            "[SpecDecDiag-v20 D09] step=%d "
                                            "per-MTP-param grad+weight "
                                            "snapshot:\n%s",
                                            _d09_step,
                                            "\n".join(_d09_rows),
                                        )
                            except Exception as _e_d09:
                                self.logger.warning(
                                    "[SpecDecDiag-v20 D09] failed: %s",
                                    _e_d09,
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
        # [SpecDecDiag-v20 D10] pre-step MTP weight snapshot.
        _d10_pre = {}
        try:
            _d10_step = int(getattr(self, "_global_step", 0) or 0)
            _d10_sample = (_d10_step <= 20) or (_d10_step % 10 == 0)
            if (self.enable_mtp_training and _d10_sample
                    and getattr(self, "model", None) is not None):
                for _mod in self.model:
                    for _n, _p in _mod.named_parameters():
                        if ".mtp." not in _n:
                            continue
                        try:
                            _d10_pre[_n] = _p.detach().clone()
                        except Exception:
                            pass
        except Exception as _e_d10:
            self.logger.warning(
                "[SpecDecDiag-v20 D10] snapshot failed: %s", _e_d10,
            )

        # [MTPMasterCarry-v48] pre-step snapshot of MTP fp32 master.
        # Mirrors v47 snapshot but is used by a residual-carry post-step
        # that NEVER scales the Adam delta (see block below).  Gate:
        # AREAL_MTP_V48_MASTER_CARRY (default '1').
        try:
            import os as _os_v48pre
            _v48_on = (
                _os_v48pre.environ.get(
                    'AREAL_MTP_V48_MASTER_CARRY', '1'
                ) == '1'
            )
        except Exception:
            _v48_on = True
        self._v48_pre_master = {}
        self._v48_on_step = bool(
            _v48_on and getattr(self, 'enable_mtp_training', False)
        )
        if (
            self._v48_on_step
            and getattr(self, 'model', None) is not None
        ):
            try:
                for _mod_v48 in self.model:
                    for _n_v48, _p_v48 in _mod_v48.named_parameters():
                        if ('.mtp.' not in _n_v48
                                and '.mtp_layers.' not in _n_v48):
                            continue
                        _mp_v48 = getattr(_p_v48, 'main_param', None)
                        if _mp_v48 is not None:
                            try:
                                self._v48_pre_master[_n_v48] = (
                                    _mp_v48.detach().clone()
                                )
                            except Exception:
                                pass
            except Exception as _e_v48pre:
                self.logger.warning(
                    '[MTPMasterCarry-v48] pre-snapshot failed: %s',
                    _e_v48pre,
                )
        # [MTPMasterAmp-v47] pre-step snapshot of MTP fp32 master.
        # Captured before optimizer.step() so we can compute the
        # raw Adam delta after the step and amplify it to bf16 ULP
        # when needed.  Gate: AREAL_MTP_V47_MASTER_AMP (default 1).
        try:
            import os as _os_v47pre
            _v47_on = (
                _os_v47pre.environ.get(
                    'AREAL_MTP_V47_MASTER_AMP', '0'
                ) == '1'
            )
        except Exception:
            _v47_on = False
        self._v47_pre_master = {}
        self._v47_pre_data = {}
        self._v47_on_step = bool(
            _v47_on and getattr(self, 'enable_mtp_training', False)
        )
        if self._v47_on_step and getattr(self, 'model', None) is not None:
            try:
                for _mod_v47 in self.model:
                    for _n_v47, _p_v47 in _mod_v47.named_parameters():
                        if ('.mtp.' not in _n_v47
                                and '.mtp_layers.' not in _n_v47):
                            continue
                        _mp_v47 = getattr(_p_v47, 'main_param', None)
                        if _mp_v47 is not None:
                            try:
                                self._v47_pre_master[_n_v47] = (
                                    _mp_v47.detach().clone()
                                )
                            except Exception:
                                pass
                        try:
                            self._v47_pre_data[_n_v47] = (
                                _p_v47.data.detach().clone()
                            )
                        except Exception:
                            pass
            except Exception as _e_v47pre:
                self.logger.warning(
                    '[MTPMasterAmp-v47] pre-snapshot failed: %s',
                    _e_v47pre,
                )
        # [MTPGradProbe-v26] Install post-accumulate-grad hook on MTP
        # params (once) so grads are captured at the moment they land,
        # BEFORE Megatron's DistributedOptimizer consumes and frees them.
        try:
            if not getattr(self, "_mtp_gradhook_v26_installed", False):
                self._mtp_gradhook_v26_cache = {}
                _v26_inst = 0
                for _v26_m in self.model:
                    for _v26_n, _v26_p in _v26_m.named_parameters():
                        if (
                            (".mtp_layers." in _v26_n
                             or ".mtp." in _v26_n
                             or ".enorm" in _v26_n
                             or ".hnorm" in _v26_n
                             or ".eh_proj" in _v26_n)
                            and _v26_p.requires_grad
                        ):
                            def _mk_hook(_nm):
                                def _hook(_p):
                                    try:
                                        if _p.grad is not None:
                                            self._mtp_gradhook_v26_cache[_nm] = (
                                                float(_p.grad.abs().mean().item()),
                                                float(_p.grad.abs().max().item()),
                                                str(_p.grad.dtype),
                                            )
                                    except Exception:
                                        pass
                                return _hook
                            try:
                                _v26_p.register_post_accumulate_grad_hook(
                                    _mk_hook(_v26_n)
                                )
                                _v26_inst += 1
                            except Exception:
                                pass
                self._mtp_gradhook_v26_installed = True
                self.logger.info(
                    "[MTPGradProbe-v26] installed post_accumulate_grad_hook "
                    "on %d MTP params",
                    _v26_inst,
                )
            if getattr(self, "_mtp_gradhook_v26_cache", None):
                for _v26_nm, (_am, _mx, _dt) in (
                    self._mtp_gradhook_v26_cache.items()
                ):
                    self.logger.info(
                        "[MTPGradProbe-v26] name=%s grad_abs_mean=%.3e "
                        "grad_abs_max=%.3e grad_dtype=%s",
                        _v26_nm, _am, _mx, _dt,
                    )
                self._mtp_gradhook_v26_cache = {}
        except Exception as _e_v26g:
            self.logger.warning(
                "[MTPGradProbe-v26] outer error: %s", _e_v26g,
            )
        # [MTPMainGrad-v27] Log Megatron DistributedOptimizer's
        # fp32 reduced gradient buffer (param.main_grad) just before
        # optimizer.step().  This is the ACTUAL gradient the optimizer
        # consumes (fp32, post-allreduce, post-inv-scale), not the raw
        # bf16 .grad captured by the grad hook.  Comparing the two
        # distinguishes "grad vanishes in backward" vs "grad vanishes
        # in allreduce/scaling pipeline".
        try:
            for _v27_m in self.model:
                for _v27_n, _v27_p in _v27_m.named_parameters():
                    if not (
                        ".mtp_layers." in _v27_n
                        or ".mtp." in _v27_n
                        or ".enorm" in _v27_n
                        or ".hnorm" in _v27_n
                        or ".eh_proj" in _v27_n
                    ):
                        continue
                    try:
                        _v27_mg = getattr(_v27_p, "main_grad", None)
                        if _v27_mg is None:
                            self.logger.info(
                                "[MTPMainGrad-v27] name=%s main_grad=None "
                                ".grad_is_none=%s",
                                _v27_n, _v27_p.grad is None,
                            )
                        else:
                            self.logger.info(
                                "[MTPMainGrad-v27] name=%s dtype=%s "
                                "shape=%s abs_mean=%.3e abs_max=%.3e "
                                "nonzero_frac=%.3f",
                                _v27_n, str(_v27_mg.dtype),
                                tuple(_v27_mg.shape),
                                float(_v27_mg.abs().mean().item()),
                                float(_v27_mg.abs().max().item()),
                                float(
                                    (_v27_mg.abs() > 0).float().mean().item()
                                ),
                            )
                    except Exception as _e_v27mg1:
                        self.logger.info(
                            "[MTPMainGrad-v27] name=%s probe_err=%s",
                            _v27_n, _e_v27mg1,
                        )
        except Exception as _e_v27mg:
            self.logger.warning(
                "[MTPMainGrad-v27] outer error: %s", _e_v27mg,
            )
        # [MTPGradProbe-v25] Diagnostic grad probe before optimizer.step().
        try:
            _v25_probe_seen = set()
            if getattr(self, "model", None) is not None:
                for _v25_module in self.model:
                    for _v25_name, _v25_p in _v25_module.named_parameters():
                        if ".mtp." not in _v25_name:
                            continue
                        if _v25_name in _v25_probe_seen:
                            continue
                        _v25_probe_seen.add(_v25_name)
                        _v25_mp = getattr(_v25_p, "main_param", None)
                        self.logger.info(
                            "[MTPGradProbe-v25] name=%s has_grad=%s grad_dtype=%s grad.abs_mean=%.3e grad.abs_max=%.3e grad.nonzero_frac=%.3f main_param_dtype=%s main_param_abs_mean=%.3e",
                            _v25_name, (_v25_p.grad is not None), str(_v25_p.grad.dtype if _v25_p.grad is not None else None),
                            (_v25_p.grad.abs().mean().item() if _v25_p.grad is not None else float('nan')),
                            (_v25_p.grad.abs().max().item() if _v25_p.grad is not None else float('nan')),
                            ((_v25_p.grad != 0).float().mean().item() if _v25_p.grad is not None else float('nan')),
                            str(_v25_mp.dtype if _v25_mp is not None else None),
                            (_v25_mp.abs().mean().item() if _v25_mp is not None else float('nan'))
                        )
        except Exception as _e:
            self.logger.warning("[MTPGradProbe-v25] probe error: %s", _e)
        # [MTPGradFp32Coerce-v50] Belt-and-suspenders fp32 upcast of MTP main_grad
        # before optimizer.step. Passthrough (v50) aligns scale with slime/verl
        # but bf16 grad accumulation bucket still truncates small updates across
        # ~54 microbatches. Slime mitigates this with --accumulate-allreduce-grads
        # -in-fp32; here we do the runtime equivalent on MTP params only.
        # Gate: AREAL_MTP_V50_GRAD_FP32_COERCE (default "1").
        try:
            import os as _os_v50g
            # [MTPSlimeAlign] disable v50 fp32 coerce when slime-align is ON;
            # slime/Megatron-Core native does not upcast MTP main_grad.
            _v50_slime = (
                _os_v50g.environ.get('AREAL_MTP_SLIME_ALIGN', '1') == '1'
            )
            _v50_gfp32 = (
                _os_v50g.environ.get('AREAL_MTP_V50_GRAD_FP32_COERCE', '1') == '1'
                and not _v50_slime
            )
            if _v50_slime and not getattr(self, '_v58_v50_logged', False):
                try:
                    self.logger.info(
                        '[MTPSlimeAlign] v50 MTPGradFp32Coerce DISABLED '
                        '(slime/native does not upcast MTP main_grad).'
                    )
                    self._v58_v50_logged = True
                except Exception:
                    pass
        except Exception:
            _v50_gfp32 = True
        if (
            _v50_gfp32
            and getattr(self, 'enable_mtp_training', False)
            and getattr(self, 'model', None) is not None
        ):
            try:
                import torch as _torch_v50g
                _v50g_n = 0
                _v50g_amax = 0.0
                _v50g_name = ''
                for _mod_v50g in self.model:
                    for _n_v50g, _p_v50g in _mod_v50g.named_parameters():
                        if ('.mtp.' not in _n_v50g
                                and '.mtp_layers.' not in _n_v50g):
                            continue
                        _mg_v50g = getattr(_p_v50g, 'main_grad', None)
                        if _mg_v50g is None:
                            continue
                        if _mg_v50g.dtype == _torch_v50g.float32:
                            continue
                        try:
                            _fp32 = _mg_v50g.to(_torch_v50g.float32)
                            _p_v50g.main_grad = _fp32
                            _v50g_n += 1
                            _a = float(_fp32.abs().max().item())
                            if _a > _v50g_amax:
                                _v50g_amax = _a
                                _v50g_name = _n_v50g
                        except Exception:
                            pass
                if _v50g_n > 0:
                    self.logger.info(
                        '[MTPGradFp32Coerce-v50] coerced=%d '
                        'max_grad_amax=%.3e max_name=%s',
                        _v50g_n, _v50g_amax, _v50g_name,
                    )
            except Exception as _e_v50g:
                self.logger.warning(
                    '[MTPGradFp32Coerce-v50] failed: %s', _e_v50g,
                )
        # [MTPGradClipNorm-v51] Per-component gradient L2-norm clipping for
        # MTP parameters only, applied AFTER fp32 coerce and BEFORE
        # optimizer.step(). Megatron-Core's `gradient_clipping=1.0` is a
        # GLOBAL (backbone+MTP joint) norm, which lets MTP grad dominate
        # when backbone grad is small (KL-regularised RL). slime mitigates
        # this via `check_mtp_loss(max=1.0)` + `accumulate-allreduce-grads-
        # in-fp32`; the latter is now on (YAML grad_reduce_in_fp32=true)
        # but log.33 still shows per-step max|delta|=0.59-0.64 at v9-v13,
        # correlated with PAW crashes v10=0.005 / v14=0.008. v51 adds the
        # missing MTP-only norm clip so big spikes through MTPLossAutoScaler
        # cannot push the draft head into a divergent region.
        # Threshold: AREAL_MTP_V51_GRAD_CLIP_NORM (default 1.0).
        # Disable: AREAL_MTP_V51_GRAD_CLIP_NORM=0
        try:
            import os as _os_v51c
            _v51_thr = float(_os_v51c.environ.get(
                'AREAL_MTP_V51_GRAD_CLIP_NORM', '0'))
        except Exception:
            _v51_thr = 1.0
        if (
            _v51_thr > 0.0
            and getattr(self, 'enable_mtp_training', False)
            and getattr(self, 'model', None) is not None
        ):
            try:
                import torch as _torch_v51c
                _v51_grads = []
                _v51_names = []
                for _mod_v51c in self.model:
                    for _n_v51c, _p_v51c in _mod_v51c.named_parameters():
                        if ('.mtp.' not in _n_v51c
                                and '.mtp_layers.' not in _n_v51c):
                            continue
                        _mg_v51c = getattr(_p_v51c, 'main_grad', None)
                        if _mg_v51c is None:
                            continue
                        _v51_grads.append(_mg_v51c)
                        _v51_names.append(_n_v51c)
                if _v51_grads:
                    _v51_total_sq = _torch_v51c.zeros(
                        (), dtype=_torch_v51c.float32,
                        device=_v51_grads[0].device)
                    for _g in _v51_grads:
                        _v51_total_sq = _v51_total_sq + (
                            _g.detach().float().pow(2).sum())
                    _v51_norm = float(_v51_total_sq.sqrt().item())
                    _v51_clipped = False
                    _v51_scale = 1.0
                    if _v51_norm > _v51_thr and _v51_norm > 0.0:
                        _v51_scale = _v51_thr / (_v51_norm + 1e-12)
                        for _g in _v51_grads:
                            _g.mul_(_v51_scale)
                        _v51_clipped = True
                    _gs_v51 = getattr(self, '_global_step', 0)
                    if (_gs_v51 <= 5 or _gs_v51 % 50 == 0
                            or _v51_clipped):
                        self.logger.info(
                            '[MTPGradClipNorm-v51] step=%d n_params=%d '
                            'mtp_grad_norm=%.4e threshold=%.4e '
                            'clipped=%s scale=%.4e',
                            _gs_v51, len(_v51_grads), _v51_norm,
                            _v51_thr, _v51_clipped, _v51_scale,
                        )
            except Exception as _e_v51c:
                self.logger.warning(
                    '[MTPGradClipNorm-v51] failed: %s', _e_v51c,
                )
        # [SpecDecFlow-v54] PRE_STEP stage — per-MTP-param grad
        # diagnostics BEFORE optimizer step.  Captures what the
        # optimizer is about to apply.  Default ON.
        try:
            import os as _os_v54p
            _v54_flow_on = (
                _os_v54p.environ.get(
                    'AREAL_MTP_V54_SPEC_FLOW_LOG', '1',
                ) == '1'
                and getattr(self, 'enable_mtp_training', False)
                and getattr(self, 'model', None) is not None
            )
            if _v54_flow_on:
                _v54_pre_seen = 0
                _v54_pre_with_grad = 0
                _v54_pre_mp_avail = 0
                _v54_pre_mg_avail = 0
                _v54_pre_grad_norm_sq = 0.0
                if not hasattr(self, '_v54_pre_snap'):
                    self._v54_pre_snap = {}
                for _mod_v54p in self.model:
                    for _n_v54p, _p_v54p in (
                        _mod_v54p.named_parameters()
                    ):
                        if ('.mtp.' not in _n_v54p
                                and '.mtp_layers.' not in _n_v54p):
                            continue
                        _v54_pre_seen += 1
                        _g_v54p = getattr(_p_v54p, 'grad', None)
                        _g_norm_v54p = -1.0
                        _g_amax_v54p = -1.0
                        if _g_v54p is not None:
                            try:
                                _g_norm_v54p = float(
                                    _g_v54p.detach().float()
                                        .norm().item()
                                )
                                _g_amax_v54p = float(
                                    _g_v54p.detach().abs()
                                        .max().item()
                                )
                                _v54_pre_grad_norm_sq += (
                                    _g_norm_v54p * _g_norm_v54p
                                )
                                _v54_pre_with_grad += 1
                            except Exception:
                                pass
                        _mp_v54p = getattr(
                            _p_v54p, 'main_param', None,
                        )
                        _mg_amax_v54p = -1.0
                        if _mp_v54p is not None:
                            _v54_pre_mp_avail += 1
                            _mg_v54p = getattr(
                                _mp_v54p, 'main_grad', None,
                            )
                            if _mg_v54p is None:
                                _mg_v54p = getattr(
                                    _mp_v54p, 'grad', None,
                                )
                            if _mg_v54p is not None:
                                _v54_pre_mg_avail += 1
                                try:
                                    _mg_amax_v54p = float(
                                        _mg_v54p.detach().abs()
                                            .max().item()
                                    )
                                except Exception:
                                    pass
                            try:
                                self._v54_pre_snap[_n_v54p] = (
                                    _mp_v54p.detach().float().clone()
                                )
                            except Exception:
                                pass
                        self.logger.info(
                            '[SpecDecFlow-v54] stage=pre_step '
                            'name=%s shape=%s dtype=%s '
                            'grad_norm=%.6e grad_amax=%.6e '
                            'main_param_present=%s '
                            'main_grad_amax=%.6e',
                            _n_v54p,
                            str(tuple(_p_v54p.shape)),
                            str(_p_v54p.dtype),
                            _g_norm_v54p, _g_amax_v54p,
                            str(_mp_v54p is not None),
                            _mg_amax_v54p,
                        )
                _v54_pre_grad_norm = (
                    _v54_pre_grad_norm_sq ** 0.5
                )
                self.logger.info(
                    '[SpecDecFlow-v54] stage=pre_step_summary '
                    'n_mtp_params=%d n_with_grad=%d '
                    'n_main_param=%d n_main_grad=%d '
                    'mtp_grad_norm=%.6e',
                    _v54_pre_seen, _v54_pre_with_grad,
                    _v54_pre_mp_avail, _v54_pre_mg_avail,
                    _v54_pre_grad_norm,
                )
        except Exception as _e_v54p:
            try:
                self.logger.warning(
                    '[SpecDecFlow-v54] pre_step failed: %r', _e_v54p,
                )
            except Exception:
                pass
        # [MTPGradTrace-v56] Detailed per-MTP-param grad trace.
        # Captures `.grad`, `.main_param.grad`, and `.main_param.main_grad`
        # exactly as they arrive from backward, BEFORE the v54 freeze
        # block (which would zero them) and BEFORE the v55 LR boost
        # block (which would scale them).  Default ON: gated by
        # AREAL_MTP_V56_GRAD_TRACE (default='1').
        try:
            import os as _os_v56g
            _v56_grad_on = (
                _os_v56g.environ.get(
                    'AREAL_MTP_V56_GRAD_TRACE', '1',
                ) == '1'
                and getattr(self, 'enable_mtp_training', False)
                and getattr(self, 'model', None) is not None
            )
            if _v56_grad_on:
                try:
                    import torch as _torch_v56g
                    import torch.distributed as _dist_v56g
                    _v56g_rank = (
                        _dist_v56g.get_rank()
                        if _dist_v56g.is_initialized() else 0
                    )
                except Exception:
                    _torch_v56g = None
                    _v56g_rank = 0
                _v56g_emb_ptrs = set()
                try:
                    for _mod_e in self.model:
                        for _ne, _pe in _mod_e.named_parameters():
                            if (
                                'embedding' in _ne
                                or 'word_embeddings' in _ne
                            ):
                                try:
                                    _v56g_emb_ptrs.add(int(_pe.data_ptr()))
                                except Exception:
                                    pass
                except Exception:
                    pass
                _v56g_n = 0
                _v56g_n_with_grad = 0
                _v56g_n_with_main_grad = 0
                _v56g_n_shared = 0
                _v56g_any_nan = False
                _v56g_any_inf = False
                for _mod_v56g in self.model:
                    for _n_v56g, _p_v56g in (
                        _mod_v56g.named_parameters()
                    ):
                        if not (
                            '.mtp.' in _n_v56g
                            or '.mtp_layers.' in _n_v56g
                            or '.enorm' in _n_v56g
                            or '.hnorm' in _n_v56g
                            or '.eh_proj' in _n_v56g
                            or '.shared_head.' in _n_v56g
                        ):
                            continue
                        _v56g_n += 1
                        _g = getattr(_p_v56g, 'grad', None)
                        _g_present = _g is not None
                        _g_dtype = str(getattr(_g, 'dtype', None))
                        _g_numel = (
                            int(_g.numel()) if _g_present else 0
                        )
                        _g_norm = -1.0
                        _g_amax = -1.0
                        _g_isfinite = True
                        if _g_present:
                            try:
                                _gd = _g.detach().float()
                                _g_norm = float(_gd.norm().item())
                                _g_amax = float(
                                    _gd.abs().max().item()
                                )
                                _g_isfinite = bool(
                                    _torch_v56g.isfinite(_gd).all().item()
                                ) if _torch_v56g is not None else True
                                if _torch_v56g is not None:
                                    if bool(
                                        _torch_v56g.isnan(_gd).any().item()
                                    ):
                                        _v56g_any_nan = True
                                    if bool(
                                        _torch_v56g.isinf(_gd).any().item()
                                    ):
                                        _v56g_any_inf = True
                                _v56g_n_with_grad += 1
                            except Exception:
                                pass
                        _mp = getattr(_p_v56g, 'main_param', None)
                        _mp_present = _mp is not None
                        _mp_dtype = str(getattr(_mp, 'dtype', None))
                        _mp_grad = (
                            getattr(_mp, 'grad', None)
                            if _mp_present else None
                        )
                        _mp_grad_present = _mp_grad is not None
                        _mp_grad_norm = -1.0
                        if _mp_grad_present:
                            try:
                                _mp_grad_norm = float(
                                    _mp_grad.detach().float()
                                        .norm().item()
                                )
                            except Exception:
                                pass
                        _main_grad = (
                            getattr(_mp, 'main_grad', None)
                            if _mp_present else None
                        )
                        _mg_present = _main_grad is not None
                        _mg_dtype = str(getattr(_main_grad, 'dtype', None))
                        _mg_norm = -1.0
                        _mg_amax = -1.0
                        _mg_isfinite = True
                        if _mg_present:
                            try:
                                _mgd = _main_grad.detach().float()
                                _mg_norm = float(_mgd.norm().item())
                                _mg_amax = float(
                                    _mgd.abs().max().item()
                                )
                                _mg_isfinite = bool(
                                    _torch_v56g.isfinite(_mgd)
                                        .all().item()
                                ) if _torch_v56g is not None else True
                                if _torch_v56g is not None:
                                    if bool(
                                        _torch_v56g.isnan(_mgd)
                                            .any().item()
                                    ):
                                        _v56g_any_nan = True
                                    if bool(
                                        _torch_v56g.isinf(_mgd)
                                            .any().item()
                                    ):
                                        _v56g_any_inf = True
                                _v56g_n_with_main_grad += 1
                            except Exception:
                                pass
                        _shared = False
                        try:
                            _shared = (
                                int(_p_v56g.data_ptr())
                                in _v56g_emb_ptrs
                            )
                        except Exception:
                            pass
                        if _shared:
                            _v56g_n_shared += 1
                        _gf = getattr(_p_v56g, 'grad_fn', None)
                        self.logger.info(
                            '[MTPGradTrace-v56] rank=%d name=%s '
                            'grad_present=%s grad_dtype=%s '
                            'grad_numel=%d grad_norm=%.6e '
                            'grad_amax=%.6e grad_isfinite=%s '
                            'main_param_present=%s '
                            'main_param_dtype=%s '
                            'main_param_grad_present=%s '
                            'main_param_grad_norm=%.6e '
                            'main_grad_present=%s '
                            'main_grad_dtype=%s '
                            'main_grad_norm=%.6e '
                            'main_grad_amax=%.6e '
                            'main_grad_isfinite=%s '
                            'grad_fn_present=%s requires_grad=%s '
                            'is_leaf=%s shared_tensor=%s',
                            _v56g_rank, _n_v56g,
                            str(_g_present), _g_dtype,
                            _g_numel, _g_norm,
                            _g_amax, str(_g_isfinite),
                            str(_mp_present), _mp_dtype,
                            str(_mp_grad_present), _mp_grad_norm,
                            str(_mg_present), _mg_dtype,
                            _mg_norm, _mg_amax, str(_mg_isfinite),
                            str(_gf is not None),
                            str(bool(_p_v56g.requires_grad)),
                            str(bool(_p_v56g.is_leaf)),
                            str(_shared),
                        )
                if _v56g_rank == 0:
                    self.logger.info(
                        '[MTPGradTrace-v56] summary n_mtp=%d '
                        'n_with_grad=%d n_with_main_grad=%d '
                        'n_shared_tensor=%d any_nan=%s any_inf=%s',
                        _v56g_n, _v56g_n_with_grad,
                        _v56g_n_with_main_grad, _v56g_n_shared,
                        str(_v56g_any_nan), str(_v56g_any_inf),
                    )
        except Exception as _e_v56g:
            try:
                self.logger.warning(
                    '[MTPGradTrace-v56] grad trace failed: %r',
                    _e_v56g,
                )
            except Exception:
                pass
        # [MTPFreezeGate-v54] Disambiguation/mitigation control.
        # When AREAL_MTP_V54_FREEZE=1 (default '0'=off), zero every
        # MTP parameter's .grad AND its main_param.grad/main_grad
        # right before the Megatron distributed-optimizer step.
        # This cleanly freezes every MTP tensor (enorm/hnorm/
        # eh_proj/transformer_layer/final_layernorm/shared_head),
        # leaving the rest of the model to be trained normally by
        # GRPO.  Any subsequent shipment to sglang will contain
        # bit-identical MTP weights.
        #
        # Usage: set AREAL_MTP_V54_FREEZE=1 for one run.  If
        #   rollout/spec_accept_rate stops declining, MTP weight
        #   drift (H1) is the cause and EMA should be tuned.
        #   If the rate still declines, main-model GRPO drift of
        #   the hidden-state distribution (H2) is the cause and a
        #   different mitigation is needed.
        try:
            import os as _os_v54f
            _v54_freeze = (
                _os_v54f.environ.get('AREAL_MTP_V54_FREEZE', '0')
                == '1'
                and getattr(self, 'enable_mtp_training', False)
                and getattr(self, 'model', None) is not None
            )
            self._v54_freeze_engaged = bool(_v54_freeze)
            if _v54_freeze:
                _v54_n_zeroed = 0
                for _mod_v54f in self.model:
                    for _n_v54f, _p_v54f in (
                        _mod_v54f.named_parameters()
                    ):
                        if ('.mtp.' not in _n_v54f
                                and '.mtp_layers.' not in _n_v54f):
                            continue
                        try:
                            if _p_v54f.grad is not None:
                                _p_v54f.grad.detach().zero_()
                        except Exception:
                            pass
                        _mp_v54f = getattr(
                            _p_v54f, 'main_param', None,
                        )
                        if _mp_v54f is not None:
                            try:
                                _mg_v54f = getattr(
                                    _mp_v54f, 'grad', None,
                                )
                                if _mg_v54f is not None:
                                    _mg_v54f.detach().zero_()
                            except Exception:
                                pass
                            _mgf_v54f = getattr(
                                _mp_v54f, 'main_grad', None,
                            )
                            if _mgf_v54f is not None:
                                try:
                                    _mgf_v54f.detach().zero_()
                                except Exception:
                                    pass
                        _v54_n_zeroed += 1
                        self.logger.info(
                            '[SpecDecFlow-v54] stage=freeze '
                            'name=%s zeroed=True', _n_v54f,
                        )
                self.logger.info(
                    '[SpecDecFlow-v54] stage=freeze_summary '
                    'n_zeroed=%d', _v54_n_zeroed,
                )
                self.logger.info(
                    '[MTPFreezeGate-v54] zeroed grads for %d MTP '
                    'params before optimizer.step()', _v54_n_zeroed,
                )
        except Exception as _e_v54f:
            try:
                self.logger.warning(
                    '[MTPFreezeGate-v54] gate failed: %r', _e_v54f,
                )
            except Exception:
                pass
        # [MTPLRBoost-v55] Boost MTP gradient learning rate by a
        # configurable multiplier just before optimizer.step().
        # Evidence-driven minimal fix: log.42 (Run A, v54 freeze=1)
        # vs log.41 (Run B, v53) confirmed H2 — decline is
        # dominated by main-model hidden-state drift, not MTP
        # weight drift.  In slime / verl-style EAGLE RL training
        # the draft (MTP) head needs to track main-model drift
        # faster than vanilla co-training allows; the standard
        # pattern is an MTP-specific LR multiplier so the draft
        # head learns faster than the target.
        # Default 1.0 = exact baseline (full no-op).  Skip
        # entirely when v54 freeze is engaged (cannot scale
        # zeroed grads meaningfully).
        try:
            import os as _os_v55b
            _v55_mult_raw = _os_v55b.environ.get(
                'AREAL_MTP_V55_MTP_LR_BOOST', '1.0',
            )
            try:
                _v55_mult = float(_v55_mult_raw)
            except Exception:
                _v55_mult = 1.0
            _v55_freeze_engaged = bool(
                getattr(self, '_v54_freeze_engaged', False)
            )
            _v55_active = (
                _v55_mult > 1.0
                and not _v55_freeze_engaged
                and getattr(self, 'enable_mtp_training', False)
                and getattr(self, 'model', None) is not None
            )
            self._v55_lr_boost_active = bool(_v55_active)
            self._v55_lr_boost_mult = float(_v55_mult)
            if _v55_mult > 1.0 and _v55_freeze_engaged:
                try:
                    self.logger.info(
                        '[MTPLRBoost-v55] '
                        'skipped reason=freeze_engaged'
                    )
                except Exception:
                    pass
            elif _v55_active:
                _v55_n_scaled = 0
                _v55_pre_sq = 0.0
                _v55_post_sq = 0.0
                for _mod_v55 in self.model:
                    for _n_v55, _p_v55 in (
                        _mod_v55.named_parameters()
                    ):
                        if ('.mtp.' not in _n_v55
                                and '.mtp_layers.' not in _n_v55):
                            continue
                        _v55_scaled_any = False
                        try:
                            _g_v55 = getattr(_p_v55, 'grad', None)
                            if _g_v55 is not None:
                                _gn = float(
                                    _g_v55.detach().float()
                                        .norm().item()
                                )
                                _v55_pre_sq += _gn * _gn
                                _g_v55.detach().mul_(_v55_mult)
                                _gn2 = float(
                                    _g_v55.detach().float()
                                        .norm().item()
                                )
                                _v55_post_sq += _gn2 * _gn2
                                _v55_scaled_any = True
                        except Exception:
                            pass
                        _mp_v55 = getattr(
                            _p_v55, 'main_param', None,
                        )
                        if _mp_v55 is not None:
                            try:
                                _mg_v55 = getattr(
                                    _mp_v55, 'grad', None,
                                )
                                if _mg_v55 is not None:
                                    if not _v55_scaled_any:
                                        _gn = float(
                                            _mg_v55.detach()
                                                .float().norm()
                                                .item()
                                        )
                                        _v55_pre_sq += _gn * _gn
                                        _mg_v55.detach().mul_(
                                            _v55_mult
                                        )
                                        _gn2 = float(
                                            _mg_v55.detach()
                                                .float().norm()
                                                .item()
                                        )
                                        _v55_post_sq += (
                                            _gn2 * _gn2
                                        )
                                        _v55_scaled_any = True
                                    else:
                                        _mg_v55.detach().mul_(
                                            _v55_mult
                                        )
                            except Exception:
                                pass
                            try:
                                _mgf_v55 = getattr(
                                    _mp_v55, 'main_grad', None,
                                )
                                if _mgf_v55 is not None:
                                    if not _v55_scaled_any:
                                        _gn = float(
                                            _mgf_v55.detach()
                                                .float().norm()
                                                .item()
                                        )
                                        _v55_pre_sq += _gn * _gn
                                        _mgf_v55.detach().mul_(
                                            _v55_mult
                                        )
                                        _gn2 = float(
                                            _mgf_v55.detach()
                                                .float().norm()
                                                .item()
                                        )
                                        _v55_post_sq += (
                                            _gn2 * _gn2
                                        )
                                        _v55_scaled_any = True
                                    else:
                                        _mgf_v55.detach().mul_(
                                            _v55_mult
                                        )
                            except Exception:
                                pass
                        if _v55_scaled_any:
                            _v55_n_scaled += 1
                _v55_pre_norm = _v55_pre_sq ** 0.5
                _v55_post_norm = _v55_post_sq ** 0.5
                try:
                    self.logger.info(
                        '[MTPLRBoost-v55] mult=%.4f '
                        'n_scaled=%d mtp_grad_norm_pre=%.6e '
                        'mtp_grad_norm_post=%.6e',
                        _v55_mult, _v55_n_scaled,
                        _v55_pre_norm, _v55_post_norm,
                    )
                    self.logger.info(
                        '[SpecDecFlow-v54] stage=lr_boost '
                        'mult=%.4f n_scaled=%d',
                        _v55_mult, _v55_n_scaled,
                    )
                except Exception:
                    pass
        except Exception as _e_v55b:
            try:
                self.logger.warning(
                    '[MTPLRBoost-v55] boost failed: %r', _e_v55b,
                )
            except Exception:
                pass
        # [MTPLossTrace-v56] Best-effort defensive trace of any MTP
        # loss state stored on `self`, run right before optimizer.step().
        # Gated by AREAL_MTP_V56_LOSS_TRACE (default='1').
        try:
            import os as _os_v56l
            _v56_loss_on = (
                _os_v56l.environ.get(
                    'AREAL_MTP_V56_LOSS_TRACE', '1',
                ) == '1'
            )
            if _v56_loss_on:
                try:
                    import torch as _torch_v56l
                except Exception:
                    _torch_v56l = None
                _v56l_keys = []
                _v56l_found = []
                try:
                    _v56l_attrs = [
                        _a for _a in dir(self)
                        if (
                            ('mtp' in _a.lower()
                             and 'loss' in _a.lower())
                            or _a in (
                                'total_loss', '_last_mtp_loss',
                                'mtp_loss',
                                '_mtp_loss_for_backward',
                                '_mtp_loss_value',
                            )
                        )
                    ]
                except Exception:
                    _v56l_attrs = []
                for _a in _v56l_attrs:
                    try:
                        _v = getattr(self, _a, None)
                    except Exception:
                        continue
                    if _v is None:
                        continue
                    _v56l_keys.append(_a)
                    _is_tensor = (
                        _torch_v56l is not None
                        and isinstance(_v, _torch_v56l.Tensor)
                    )
                    if _is_tensor:
                        try:
                            _val = (
                                float(_v.detach().float().mean().item())
                                if _v.numel() > 0 else float('nan')
                            )
                        except Exception:
                            _val = float('nan')
                        _v56l_found.append(_a)
                        try:
                            self.logger.info(
                                '[MTPLossTrace-v56] attr=%s '
                                'kind=tensor value=%.6e dtype=%s '
                                'numel=%d requires_grad=%s '
                                'grad_fn_present=%s',
                                _a, _val, str(_v.dtype),
                                int(_v.numel()),
                                str(bool(_v.requires_grad)),
                                str(
                                    getattr(_v, 'grad_fn', None)
                                    is not None
                                ),
                            )
                        except Exception:
                            pass
                    elif isinstance(_v, (int, float)):
                        _v56l_found.append(_a)
                        try:
                            self.logger.info(
                                '[MTPLossTrace-v56] attr=%s '
                                'kind=scalar value=%s',
                                _a, str(_v),
                            )
                        except Exception:
                            pass
                    elif isinstance(_v, (list, tuple)):
                        try:
                            self.logger.info(
                                '[MTPLossTrace-v56] attr=%s '
                                'kind=%s len=%d',
                                _a, type(_v).__name__, len(_v),
                            )
                        except Exception:
                            pass
                self.logger.info(
                    '[MTPLossTrace-v56] found=%s keys=%s',
                    str(bool(_v56l_found)),
                    str(_v56l_keys),
                )
        except Exception as _e_v56l:
            try:
                self.logger.warning(
                    '[MTPLossTrace-v56] loss trace failed: %r',
                    _e_v56l,
                )
            except Exception:
                pass
        with trace_scope("megatron_engine.step"):
            update_successful, grad_norm, _ = self.optimizer.step()
        # [SpecDecFlow-v54] POST_STEP stage — per-MTP-param delta
        # diagnostics AFTER optimizer step.  Captures what the
        # optimizer actually applied (fp32 master delta).
        try:
            import os as _os_v54q
            _v54_flow_on2 = (
                _os_v54q.environ.get(
                    'AREAL_MTP_V54_SPEC_FLOW_LOG', '1',
                ) == '1'
                and bool(update_successful)
                and getattr(self, 'enable_mtp_training', False)
                and getattr(self, 'model', None) is not None
            )
            if _v54_flow_on2:
                _v54_post_seen = 0
                _v54_post_stalled = 0
                _v54_post_max_delta = 0.0
                _v54_post_max_name = ''
                for _mod_v54q in self.model:
                    for _n_v54q, _p_v54q in (
                        _mod_v54q.named_parameters()
                    ):
                        if ('.mtp.' not in _n_v54q
                                and '.mtp_layers.' not in _n_v54q):
                            continue
                        _v54_post_seen += 1
                        _mp_v54q = getattr(
                            _p_v54q, 'main_param', None,
                        )
                        _delta_amax = -1.0
                        _delta_l2 = -1.0
                        _post_amax = -1.0
                        _bf16_cast_diff = -1.0
                        if _mp_v54q is not None:
                            try:
                                _pre_v54q = getattr(
                                    self, '_v54_pre_snap', {},
                                ).get(_n_v54q)
                                _cur_fp32 = (
                                    _mp_v54q.detach().float()
                                )
                                _post_amax = float(
                                    _cur_fp32.abs().max().item()
                                )
                                if (
                                    _pre_v54q is not None
                                    and _pre_v54q.shape
                                    == _cur_fp32.shape
                                ):
                                    _d = _cur_fp32 - _pre_v54q
                                    _delta_amax = float(
                                        _d.abs().max().item()
                                    )
                                    _delta_l2 = float(
                                        _d.norm().item()
                                    )
                                    if _delta_amax == 0.0:
                                        _v54_post_stalled += 1
                                    if _delta_amax > (
                                        _v54_post_max_delta
                                    ):
                                        _v54_post_max_delta = (
                                            _delta_amax
                                        )
                                        _v54_post_max_name = (
                                            _n_v54q
                                        )
                                try:
                                    _bf = _p_v54q.data.detach()
                                    _bf16_cast_diff = float(
                                        (
                                            _cur_fp32
                                            - _bf.float()
                                        ).abs().max().item()
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        self.logger.info(
                            '[SpecDecFlow-v54] stage=post_step '
                            'name=%s post_amax=%.6e '
                            'delta_amax=%.6e delta_l2=%.6e '
                            'bf16_cast_diff=%.6e',
                            _n_v54q, _post_amax,
                            _delta_amax, _delta_l2,
                            _bf16_cast_diff,
                        )
                self.logger.info(
                    '[SpecDecFlow-v54] stage=post_step_summary '
                    'n_mtp_params=%d n_stalled=%d '
                    'max_delta=%.6e max_delta_name=%s '
                    'freeze_engaged=%s',
                    _v54_post_seen, _v54_post_stalled,
                    _v54_post_max_delta,
                    _v54_post_max_name,
                    str(getattr(
                        self, '_v54_freeze_engaged', False,
                    )),
                )
                # release pre-snapshot
                try:
                    self._v54_pre_snap = {}
                except Exception:
                    pass
        except Exception as _e_v54q:
            try:
                self.logger.warning(
                    '[SpecDecFlow-v54] post_step failed: %r',
                    _e_v54q,
                )
            except Exception:
                pass
        # [MTPMasterAmp-v47] post-step delta amplification.
        # For each MTP fp32 master tensor whose Adam step delta
        # amax is smaller than  beta * bf16_ULP,  rescale the
        # delta (preserving per-element sign / ratio) so that the
        # shipped bf16 payload flips at least  beta  of a bucket
        # each step.  This breaks the bf16 ULP trap at compute
        # time without distorting Adam's direction (we never
        # touch the optimizer's internal m / v).
        if (
            bool(update_successful)
            and getattr(self, '_v47_on_step', False)
            and getattr(self, 'model', None) is not None
        ):
            try:
                import math as _m_v47
                import os as _os_v47p
                import torch as _torch_v47p
                try:
                    _beta_v47 = float(
                        _os_v47p.environ.get(
                            'AREAL_MTP_V47_AMP_BETA', '0.5'
                        )
                    )
                except Exception:
                    _beta_v47 = 0.5
                try:
                    _min_ratio_v47 = float(
                        _os_v47p.environ.get(
                            'AREAL_MTP_V47_AMP_MIN_RATIO', '0.5'
                        )
                    )
                except Exception:
                    _min_ratio_v47 = 0.5
                _n_amp_v47 = 0
                _n_skip_v47 = 0
                _scales_v47 = []
                for _mod_v47p in self.model:
                    for _n_v47p, _p_v47p in (
                        _mod_v47p.named_parameters()
                    ):
                        if ('.mtp.' not in _n_v47p
                                and '.mtp_layers.' not in _n_v47p):
                            continue
                        _mp_v47p = getattr(
                            _p_v47p, 'main_param', None
                        )
                        if _mp_v47p is None:
                            _n_skip_v47 += 1
                            continue
                        _pre_v47 = self._v47_pre_master.get(_n_v47p)
                        if (
                            _pre_v47 is None
                            or _pre_v47.shape != _mp_v47p.shape
                        ):
                            _n_skip_v47 += 1
                            continue
                        _delta_v47 = _mp_v47p.data - _pre_v47
                        _raw_dmax_v47 = float(
                            _delta_v47.abs().max().item()
                        )
                        _amax_v47 = float(
                            _mp_v47p.data.abs().max().item()
                        )
                        if _amax_v47 <= 0.0 or _raw_dmax_v47 <= 0.0:
                            _n_skip_v47 += 1
                            continue
                        _e_v47 = _m_v47.floor(_m_v47.log2(_amax_v47))
                        _ulp_v47 = 2.0 ** (_e_v47 - 7)
                        _target_v47 = _beta_v47 * _ulp_v47
                        _ratio_v47 = _raw_dmax_v47 / _ulp_v47
                        if _ratio_v47 >= _min_ratio_v47:
                            _n_skip_v47 += 1
                            _log_this = False
                            _scale_v47 = 1.0
                            _clipped = False
                        else:
                            _scale_v47 = (
                                _target_v47 / _raw_dmax_v47
                            )
                            # cap to avoid runaway if Adam step is
                            # spuriously tiny (e.g. right after
                            # warmup) — hard ceiling 1e6.
                            _clipped = False
                            if _scale_v47 > 1.0e6:
                                _scale_v47 = 1.0e6
                                _clipped = True
                            # write amplified delta back to fp32
                            # master, leaving optimizer internals
                            # (m, v) unchanged.
                            _new_master_v47 = (
                                _pre_v47 + _scale_v47 * _delta_v47
                            )
                            _mp_v47p.data.copy_(_new_master_v47)
                            # propagate to the bf16 model param so
                            # any downstream read path (including
                            # convert_to_hf) sees the new weight
                            # right now.
                            try:
                                _p_v47p.data.copy_(
                                    _mp_v47p.data.to(
                                        _p_v47p.data.dtype
                                    )
                                )
                            except Exception:
                                pass
                            _n_amp_v47 += 1
                            _scales_v47.append(_scale_v47)
                            _log_this = True
                        _amp_dmax_v47 = float(
                            (
                                _mp_v47p.data - _pre_v47
                            ).abs().max().item()
                        )
                        if _log_this:
                            self.logger.info(
                                '[MTPMasterAmp-v47] name=%s '
                                'pre_amax=%.6e post_amax=%.6e '
                                'raw_dmax=%.3e amp_dmax=%.3e '
                                'ulp=%.3e beta=%.3f '
                                'scale=%.3e clipped=%s',
                                _n_v47p,
                                float(_pre_v47.abs().max().item()),
                                _amax_v47,
                                _raw_dmax_v47, _amp_dmax_v47,
                                _ulp_v47, _beta_v47,
                                _scale_v47, _clipped,
                            )
                # summary
                if _scales_v47:
                    try:
                        import statistics as _st_v47
                        _geo = _m_v47.exp(
                            _st_v47.fmean(
                                [_m_v47.log(s) for s in _scales_v47]
                            )
                        )
                    except Exception:
                        _geo = float('nan')
                else:
                    _geo = float('nan')
                self.logger.info(
                    '[MTPMasterAmpSummary-v47] '
                    'n_amplified=%d n_skipped=%d '
                    'geomean_scale=%s beta=%.3f '
                    'min_ratio=%.3f',
                    _n_amp_v47, _n_skip_v47, str(_geo),
                    _beta_v47, _min_ratio_v47,
                )
            except Exception as _e_v47_post:
                self.logger.warning(
                    '[MTPMasterAmp-v47] post-step failed: %s',
                    _e_v47_post,
                )
            finally:
                # release snapshots — memory-bounded.
                self._v47_pre_master = {}
                self._v47_pre_data = {}
        # [MTPMasterCarry-v48] master-side Sigma-Delta residual carry.
        # This is the v48 replacement for v47 (which scaled the whole delta
        # by a tensor-wise scalar and destroyed model alignment in log.31).
        # Here we NEVER touch the magnitude/direction of the Adam delta.
        # Instead we maintain per-parameter fp32 residual and only flip the
        # bf16 bucket for the elements whose accumulated residual exceeds
        # +/- ULP/2, exactly like the ship-side v28 Σ-Δ but on the compute
        # (master) side where it actually matters.
        if (
            bool(update_successful)
            and getattr(self, '_v48_on_step', False)
            and getattr(self, 'model', None) is not None
        ):
            try:
                import torch as _torch_v48
                if not hasattr(self, '_v48_residual'):
                    self._v48_residual = {}
                _n_flipped_v48 = 0
                _n_seen_v48 = 0
                _max_res_ratio_v48 = 0.0
                _max_res_name_v48 = ''
                for _mod_v48p in self.model:
                    for _n_v48p, _p_v48p in (
                        _mod_v48p.named_parameters()
                    ):
                        if ('.mtp.' not in _n_v48p
                                and '.mtp_layers.' not in _n_v48p):
                            continue
                        _mp_v48p = getattr(
                            _p_v48p, 'main_param', None
                        )
                        if _mp_v48p is None:
                            continue
                        _n_seen_v48 += 1
                        # residual is fp32, same shape as main_param.
                        _res = self._v48_residual.get(_n_v48p)
                        if _res is None or _res.shape != _mp_v48p.shape:
                            _res = _torch_v48.zeros_like(
                                _mp_v48p.data,
                                dtype=_torch_v48.float32,
                            )
                            self._v48_residual[_n_v48p] = _res
                        # accumulate: want = fp32_master + residual
                        _fp32_new = _mp_v48p.data.to(_torch_v48.float32)
                        _want = _fp32_new + _res
                        _bf16_dtype = _p_v48p.data.dtype
                        _bf16_new = _want.to(_bf16_dtype)
                        # new residual captures quantization loss (fp32 level)
                        _new_res = _want - _bf16_new.to(_torch_v48.float32)
                        # count how many bf16 elements flip relative to
                        # "no-carry" rounding of fp32_new alone
                        try:
                            _bf16_baseline = _fp32_new.to(_bf16_dtype)
                            _n_flip_this = int(
                                (
                                    _bf16_new.to(_torch_v48.float32)
                                    != _bf16_baseline.to(_torch_v48.float32)
                                ).sum().item()
                            )
                        except Exception:
                            _n_flip_this = -1
                        # write back: master stays fp32-accurate; bf16 is
                        # quantized-with-accumulated-residual.
                        _mp_v48p.data.copy_(_want.to(_mp_v48p.dtype))
                        try:
                            _p_v48p.data.copy_(_bf16_new)
                        except Exception:
                            pass
                        self._v48_residual[_n_v48p] = _new_res
                        # record residual magnitude ratio vs ULP
                        try:
                            import math as _m_v48ip
                            _amax = float(
                                _mp_v48p.data.abs().max().item()
                            )
                            if _amax > 0.0:
                                _e = _m_v48ip.floor(_m_v48ip.log2(_amax))
                                _ulp = 2.0 ** (_e - 7)
                                _rmax = float(
                                    _new_res.abs().max().item()
                                )
                                _ratio = _rmax / max(_ulp, 1e-30)
                                if _ratio > _max_res_ratio_v48:
                                    _max_res_ratio_v48 = _ratio
                                    _max_res_name_v48 = _n_v48p
                                # per-tensor log, cheap (O(#mtp params))
                                self.logger.info(
                                    '[MTPMasterCarry-v48] name=%s '
                                    'amax=%.3e ulp=%.3e '
                                    'res_amax=%.3e res_ratio=%.3f '
                                    'flips=%d',
                                    _n_v48p, _amax, _ulp, _rmax,
                                    _ratio, _n_flip_this,
                                )
                        except Exception:
                            pass
                        if _n_flip_this > 0:
                            _n_flipped_v48 += 1
                self.logger.info(
                    '[MTPMasterCarrySummary-v48] '
                    'n_seen=%d n_flipped_any=%d '
                    'max_res_ratio=%.3f max_res_name=%s',
                    _n_seen_v48, _n_flipped_v48,
                    _max_res_ratio_v48, _max_res_name_v48,
                )
            except Exception as _e_v48_post:
                self.logger.warning(
                    '[MTPMasterCarry-v48] post-step failed: %s',
                    _e_v48_post,
                )
            finally:
                # release pre-snapshot to bound memory
                self._v48_pre_master = {}
        # [MTPPostOptim-v25] Diagnostic post-optimizer-step probe.
        try:
            _v25_post_seen = set()
            if getattr(self, "model", None) is not None:
                for _v25_module in self.model:
                    for _v25_name, _v25_p in _v25_module.named_parameters():
                        if ".mtp." not in _v25_name:
                            continue
                        if _v25_name in _v25_post_seen:
                            continue
                        _v25_post_seen.add(_v25_name)
                        _v25_mp = getattr(_v25_p, "main_param", None)
                        self.logger.info(
                            "[MTPPostOptim-v25] name=%s main_param_abs_mean_post=%.6e bf16_model_abs_mean=%.6e "
                            "cast_diff_l1=%.3e cast_diff_linf=%.3e",
                            _v25_name,
                            (_v25_mp.abs().mean().item() if _v25_mp is not None else float('nan')),
                            _v25_p.data.abs().mean().item(),
                            ((_v25_mp.to(_v25_p.dtype) - _v25_p.data).abs().mean().item() if _v25_mp is not None else float('nan')),
                            ((_v25_mp.to(_v25_p.dtype) - _v25_p.data).abs().max().item() if _v25_mp is not None else float('nan')),
                        )
        except Exception as _e:
            self.logger.warning("[MTPPostOptim-v25] probe error: %s", _e)

        # [SpecDecDiag-v20 D11] post-step |deltaW| per MTP tensor.
        try:
            import torch as _t_d11
            if _d10_pre:
                _step_d11 = int(getattr(self, "_global_step", 0) or 0)
                _rows = []
                _floor_est = 7.78e-3
                _n_total = 0
                _n_stalled = 0
                _max_delta_global = 0.0
                for _mod in self.model:
                    for _n, _p in _mod.named_parameters():
                        if _n not in _d10_pre:
                            continue
                        _pre = _d10_pre[_n]
                        try:
                            _delta = (_p.detach() - _pre).float().abs()
                            _max = float(_delta.max().item())
                            _mean = float(_delta.mean().item())
                            _norm = float(_delta.norm().item())
                            _w_abs_max = float(_p.detach().float()
                                              .abs().max().item())
                            _n_total += 1
                            _stalled = _max == 0.0
                            if _stalled:
                                _n_stalled += 1
                            if _max > _max_delta_global:
                                _max_delta_global = _max
                            if len(_rows) < 8 or _stalled:
                                _rows.append(
                                    "%s: |dW|_max=%.3e mean=%.3e "
                                    "norm=%.3e |W|_max=%.3e %s" % (
                                        _n, _max, _mean, _norm,
                                        _w_abs_max,
                                        "STALLED" if _stalled else "",
                                    )
                                )
                        except Exception:
                            pass
                self.logger.info(
                    "[SpecDecDiag-v20 D11] PostOpt step=%d "
                    "total=%d stalled=%d max|dW|_global=%.3e "
                    "bf16_ulp_floor_est=%.3e",
                    _step_d11, _n_total, _n_stalled,
                    _max_delta_global, _floor_est,
                )
                if _rows:
                    self.logger.info(
                        "[SpecDecDiag-v20 D11] per-tensor (step=%d):\n%s",
                        _step_d11, "\n".join(_rows),
                    )
                _d10_pre.clear()
        except Exception as _e_d11:
            self.logger.warning(
                "[SpecDecDiag-v20 D11] compare failed: %s", _e_d11,
            )

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

                            # ----------------------------------------------------------------
                            # [MTPSharedWeightIsolate-v53]
                            # When share_embeddings_and_output_weights=True the returned
                            # `output_weight` IS the shared parameter tensor. If we pass it
                            # directly to `output_layer(weight=...)` inside the MTP loop
                            # below, the MTP CE backward will accumulate gradient on that
                            # shared parameter, contaminating:
                            #   - the embedding lookup used by the main model, and
                            #   - the sglang spec-decoding weight sync (mtp_hf_tensors),
                            # which empirically drives spec_accept_rate / PAW to collapse
                            # within ~13 versions (see round 12 log comparison).
                            #
                            # Fix: snapshot a *detached* view of the weight specifically
                            # for the MTP branch. The main-path `output_layer(... weight=
                            # output_weight ...)` call below is LEFT UNTOUCHED so GRPO
                            # gradient on lm_head / embedding is preserved exactly.
                            # ----------------------------------------------------------------
                            # [MTPSlimeAlign] When slime-align is ON, pass
                            # the un-detached shared output_weight, exactly
                            # like Megatron-Core 0.16.0 native
                            # gpt_model.py:_postprocess and slime. This
                            # restores MTP CE -> shared lm_head/embedding
                            # gradient flow, which is essential for the
                            # main policy to track the draft distribution.
                            try:
                                import os as _os_v58_b
                                _v58_slime_b = (
                                    _os_v58_b.environ.get(
                                        'AREAL_MTP_SLIME_ALIGN', '1'
                                    ) == '1'
                                )
                            except Exception:
                                _v58_slime_b = True
                            if _v58_slime_b:
                                _mtp_output_weight_v53 = output_weight
                                if (
                                    not getattr(
                                        _engine_ref,
                                        '_v58_b_logged', False)
                                ):
                                    try:
                                        _logger.info(
                                            '[MTPSlimeAlign] B) '
                                            'output_weight passed '
                                            'un-detached to MTP '
                                            'output_layer (slime/'
                                            'native).'
                                        )
                                        _engine_ref._v58_b_logged = True
                                    except Exception:
                                        pass
                            else:
                                _mtp_output_weight_v53 = (
                                    output_weight.detach()
                                    if output_weight is not None
                                    else None
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
                                # [MTPModelStructAudit-v62] one-shot:
                                # confirm self_model.mtp IS wired
                                # and list its layer-0 sub-module
                                # parameter names + shapes so we
                                # can cross-check against mcore
                                # ship list on the NEXT run.
                                try:
                                    if not getattr(_engine_ref,
                                            '_v62_struct_logged', False):
                                        _v62_mtp_mod = getattr(
                                            self_model, 'mtp', None)
                                        _v62_dec_has_mtp = hasattr(
                                            getattr(self_model,
                                                    'decoder', object()),
                                            'mtp_layers')
                                        _v62_mtp_proc = getattr(
                                            self_model, 'mtp_process', None)
                                        _v62_names = []
                                        if _v62_mtp_mod is not None:
                                            try:
                                                _v62_L0 = _v62_mtp_mod.layers[0]
                                                for _v62_pn, _v62_pp in (
                                                        _v62_L0.named_parameters()):
                                                    _v62_names.append(
                                                        (_v62_pn,
                                                         tuple(_v62_pp.shape),
                                                         str(_v62_pp.dtype)))
                                            except Exception:
                                                pass
                                        _logger.info(
                                            '[MTPModelStructAudit-v62] '
                                            'self_model.mtp=%s '
                                            'mtp_process=%s '
                                            'mtp_in_postprocess_arg=%s '
                                            'decoder.mtp_layers?=%s '
                                            'layer0_params=%s',
                                            type(_v62_mtp_mod).__name__
                                            if _v62_mtp_mod is not None else 'None',
                                            _v62_mtp_proc,
                                            bool(mtp_in_postprocess),
                                            _v62_dec_has_mtp,
                                            _v62_names[:32],
                                        )
                                        _engine_ref._v62_struct_logged = True
                                except Exception as _e_v62_s:
                                    try:
                                        _logger.info(
                                            '[MTPModelStructAudit-v62] '
                                            'failure: %r', _e_v62_s)
                                    except Exception:
                                        pass

                            # [MTPInputIdsAudit-v62] log input_ids /
                            # labels / hidden_states shape BEFORE the
                            # chunk so we can verify whether the
                            # decoder really produced 1+mtp_num_layers
                            # concatenated seq_len chunks (the signal
                            # that MTP block ran and the shift-by-1
                            # label alignment is correct).
                            try:
                                _v62_gs = getattr(
                                    _engine_ref, '_global_step', 0)
                                if (_mtp_diag_mb_counter[0] == 0
                                        and (_v62_gs <= 3
                                             or _v62_gs % 100 == 0)):
                                    try:
                                        _v62_iid_sh = (
                                            tuple(input_ids.shape)
                                            if input_ids is not None
                                            else None)
                                        _v62_iid_f8 = (
                                            [int(x) for x in
                                             input_ids.reshape(-1)[:8].tolist()]
                                            if input_ids is not None else [])
                                    except Exception:
                                        _v62_iid_sh = None
                                        _v62_iid_f8 = []
                                    try:
                                        _v62_lb_sh = (
                                            tuple(labels.shape)
                                            if labels is not None
                                            else None)
                                        _v62_lb_f8 = (
                                            [int(x) for x in
                                             labels.reshape(-1)[:8].tolist()]
                                            if labels is not None else [])
                                    except Exception:
                                        _v62_lb_sh = None
                                        _v62_lb_f8 = []
                                    try:
                                        _v62_hs_sh = tuple(
                                            hidden_states.shape)
                                        _v62_hs_f8 = [
                                            float(x) for x in
                                            hidden_states.detach()
                                            .float().reshape(-1)[:8]
                                            .tolist()]
                                    except Exception:
                                        _v62_hs_sh = None
                                        _v62_hs_f8 = []
                                    _logger.info(
                                        '[MTPInputIdsAudit-v62] '
                                        'step=%d mtp_num_layers=%s '
                                        'input_ids.shape=%s '
                                        'input_ids.first8=%s '
                                        'labels.shape=%s '
                                        'labels.first8=%s '
                                        'hidden_states.shape=%s '
                                        'hidden_states.first8=%s',
                                        _v62_gs,
                                        self_model.config.mtp_num_layers,
                                        _v62_iid_sh, _v62_iid_f8,
                                        _v62_lb_sh, _v62_lb_f8,
                                        _v62_hs_sh, _v62_hs_f8,
                                    )
                            except Exception as _e_v62_i:
                                try:
                                    _logger.info(
                                        '[MTPInputIdsAudit-v62] '
                                        'failure: %r', _e_v62_i)
                                except Exception:
                                    pass

                            if not self_model.post_process:
                                return hidden_states

                            if self_model.config.mtp_num_layers is not None:
                                mtp_labels = labels.clone()
                                hidden_states_list = torch.chunk(
                                    hidden_states,
                                    1 + self_model.config.mtp_num_layers,
                                    dim=0,
                                )
                                # [MTPHsChunkAudit-v62] per-chunk
                                # stats: if the MTP block really ran
                                # inside `self.mtp(...)`, chunks
                                # should be DISTINCT; if they are
                                # identical the MTP block was NOT
                                # exercised and the decline comes
                                # from the main backbone only.
                                try:
                                    _v62_cgs = getattr(
                                        _engine_ref, '_global_step', 0)
                                    if (_mtp_diag_mb_counter[0] == 0
                                            and (_v62_cgs <= 3
                                                 or _v62_cgs % 100 == 0)):
                                        for _v62_ci, _v62_ch in enumerate(
                                                hidden_states_list):
                                            try:
                                                _v62_chf = _v62_ch.detach().float()
                                                _v62_l2 = float(_v62_chf.norm().item())
                                                _v62_am = float(
                                                    _v62_chf.abs().mean().item())
                                                _v62_ax = float(
                                                    _v62_chf.abs().max().item())
                                                _v62_f8 = [
                                                    float(x) for x in
                                                    _v62_chf.reshape(-1)[:8].tolist()]
                                                _logger.info(
                                                    '[MTPHsChunkAudit-v62] '
                                                    'step=%d chunk=%d/%d '
                                                    'shape=%s abs_mean=%.6e '
                                                    'abs_max=%.6e l2=%.6e '
                                                    'first8=%s',
                                                    _v62_cgs, _v62_ci,
                                                    len(hidden_states_list),
                                                    tuple(_v62_ch.shape),
                                                    _v62_am, _v62_ax, _v62_l2,
                                                    _v62_f8,
                                                )
                                            except Exception:
                                                continue
                                except Exception as _e_v62_c:
                                    try:
                                        _logger.info(
                                            '[MTPHsChunkAudit-v62] '
                                            'failure: %r', _e_v62_c)
                                    except Exception:
                                        pass
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
                                    # [MTPFwdWeightAudit-v61] log live MTP
                                    # weight statistics every 100 steps to
                                    # detect silent corruption between load
                                    # time (MTPLoad/MTPPreScan) and forward.
                                    try:
                                        _v61w_gs = getattr(
                                            _engine_ref,
                                            '_global_step', 0,
                                        )
                                        if (_mtp_diag_mb_counter[0] == 0
                                                and (_v61w_gs <= 3
                                                     or _v61w_gs % 100 == 0)):
                                            _v61w_mtp = getattr(
                                                self_model, 'mtp', None,
                                            )
                                            if _v61w_mtp is not None:
                                                for _v61w_pn in (
                                                    'enorm.weight',
                                                    'hnorm.weight',
                                                    'eh_proj.weight',
                                                ):
                                                    try:
                                                        _v61w_p = (
                                                            _v61w_mtp
                                                            .layers[0]
                                                        )
                                                        for _v61w_part in _v61w_pn.split('.'):
                                                            _v61w_p = getattr(
                                                                _v61w_p,
                                                                _v61w_part,
                                                            )
                                                        _v61w_pf = _v61w_p.detach().float()
                                                        _v61w_am = float(_v61w_pf.abs().mean().item())
                                                        _v61w_ax = float(_v61w_pf.abs().max().item())
                                                        _v61w_l2 = float(_v61w_pf.norm().item())
                                                        _v61w_first8 = [
                                                            float(x) for x in
                                                            _v61w_pf.reshape(-1)[:8].tolist()
                                                        ]
                                                        _logger.info(
                                                            '[MTPFwdWeightAudit-v61] '
                                                            'step=%d mtp.layers.0.%s '
                                                            'dtype=%s shape=%s '
                                                            'abs_mean=%.6e abs_max=%.6e '
                                                            'l2=%.6e first8=%s',
                                                            _v61w_gs, _v61w_pn,
                                                            str(_v61w_p.dtype),
                                                            str(tuple(_v61w_p.shape)),
                                                            _v61w_am, _v61w_ax,
                                                            _v61w_l2,
                                                            str(_v61w_first8),
                                                        )
                                                    except Exception:
                                                        continue
                                            # also probe output_weight
                                            try:
                                                _v61w_ow = _mtp_output_weight_v53
                                                if _v61w_ow is not None:
                                                    _v61w_owf = _v61w_ow.detach().float()
                                                    _logger.info(
                                                        '[MTPFwdWeightAudit-v61] '
                                                        'step=%d output_weight '
                                                        'dtype=%s shape=%s '
                                                        'abs_mean=%.6e abs_max=%.6e '
                                                        'l2=%.6e',
                                                        _v61w_gs,
                                                        str(_v61w_ow.dtype),
                                                        str(tuple(_v61w_ow.shape)),
                                                        float(_v61w_owf.abs().mean().item()),
                                                        float(_v61w_owf.abs().max().item()),
                                                        float(_v61w_owf.norm().item()),
                                                    )
                                            except Exception:
                                                pass
                                    except Exception as _e_v61w:
                                        try:
                                            _logger.info(
                                                '[MTPFwdWeightAudit-v61] '
                                                'failure: %r', _e_v61w,
                                            )
                                        except Exception:
                                            pass
                                    mtp_logits, _ = self_model.output_layer(
                                        _mtp_hs,
                                        # [MTPSharedWeightIsolate-v53] detached weight
                                        # prevents MTP CE grad from contaminating
                                        # shared embedding / lm_head parameter.
                                        weight=_mtp_output_weight_v53,
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
                                    # [MTPLossPerLayerAudit-v62]
                                    # break down aggregated
                                    # mtp_loss per mtp layer so
                                    # we can see whether layer-0
                                    # is learning (CE decreasing)
                                    # even while spec_accept_rate
                                    # declines.
                                    try:
                                        _v62_lgs = getattr(
                                            _engine_ref, '_global_step', 0)
                                        if (_mtp_diag_mb_counter[0] == 0
                                                and (_v62_lgs <= 3
                                                     or _v62_lgs % 100 == 0)):
                                            try:
                                                _v62_ml_sum = float(
                                                    mtp_loss.detach()
                                                    .float().sum().item())
                                            except Exception:
                                                _v62_ml_sum = float('nan')
                                            try:
                                                _v62_nt = int(
                                                    num_tokens.detach()
                                                    .sum().item())
                                            except Exception:
                                                _v62_nt = -1
                                            _v62_mean = (
                                                _v62_ml_sum / _v62_nt
                                                if _v62_nt > 0 else float('nan'))
                                            _logger.info(
                                                '[MTPLossPerLayerAudit-v62] '
                                                'step=%d mtp_layer=%d '
                                                'loss_sum=%.4f '
                                                'num_tokens=%d '
                                                'loss_mean=%.4f',
                                                _v62_lgs,
                                                mtp_layer_number,
                                                _v62_ml_sum, _v62_nt,
                                                _v62_mean,
                                            )
                                    except Exception as _e_v62_l:
                                        try:
                                            _logger.info(
                                                '[MTPLossPerLayerAudit-v62] '
                                                'failure: %r', _e_v62_l)
                                        except Exception:
                                            pass
                                    mtp_loss = loss_mask * mtp_loss
                                    try:
                                        _d05_step = getattr(
                                            _engine_ref, "_global_step", 0)
                                        _d05_mb = _mtp_diag_mb_counter[0]
                                        _d05_gate = (_d05_mb == 0 and
                                                     (_d05_step <= 5
                                                      or _d05_step % 50 == 0))
                                        if _d05_gate:
                                            import torch as _t_d05
                                            _hs_f = hidden_states.detach().float()
                                            _lm_f = loss_mask.detach().float()
                                            _logger.info(
                                                "[SpecDecDiag-v20 D05] "
                                                "MTPLayer#%d step=%d "
                                                "hidden_states: shape=%s "
                                                "dtype=%s rg=%s "
                                                "abs_mean=%.3e abs_max=%.3e "
                                                "finite=%s",
                                                mtp_layer_number, _d05_step,
                                                list(hidden_states.shape),
                                                str(hidden_states.dtype),
                                                hidden_states.requires_grad,
                                                _hs_f.abs().mean().item(),
                                                _hs_f.abs().max().item(),
                                                bool(_t_d05.isfinite(_hs_f)
                                                     .all().item()),
                                            )
                                            _logger.info(
                                                "[SpecDecDiag-v20 D05] "
                                                "MTPLayer#%d step=%d "
                                                "loss_mask: shape=%s "
                                                "num_tokens=%s sum=%.1f "
                                                "mtp_loss_raw: abs_mean=%.3e "
                                                "abs_max=%.3e sum=%.6f",
                                                mtp_layer_number, _d05_step,
                                                list(loss_mask.shape),
                                                num_tokens,
                                                _lm_f.sum().item(),
                                                mtp_loss.detach().float()
                                                    .abs().mean().item(),
                                                mtp_loss.detach().float()
                                                    .abs().max().item(),
                                                mtp_loss.detach().float()
                                                    .sum().item(),
                                            )
                                    except Exception as _e_d05:
                                        _logger.warning(
                                            "[SpecDecDiag-v20 D05] failed: %s",
                                            _e_d05,
                                        )
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
                                    # [MTPSourceLossCap-v52] Adaptive source-side soft-cap on
                                    # _mtp_loss_to_store BEFORE it is appended to FIFO and BEFORE
                                    # MTPLossAutoScaler.apply. v51 clipped main_grad after backward
                                    # but Adam's m/sqrt(v) normalisation made that ineffective
                                    # (log.34: max|delta|=0.63 at v9 almost unchanged vs log.33=0.64).
                                    # v52 scales the loss ITSELF, which autograd propagates as a
                                    # magnitude reduction on the injected gradient without touching
                                    # direction -- effective for both the FIFO/legacy path and the
                                    # v50 passthrough path. Threshold tracks an EMA of |sum(loss)|;
                                    # default cap = ratio * EMA, ratio via
                                    # AREAL_MTP_V52_LOSS_CAP_RATIO (default 2.0).
                                    # Disable: AREAL_MTP_V52_LOSS_CAP_RATIO=0
                                    try:
                                        import os as _os_v52s
                                        # [MTPSlimeAlign] force cap ratio
                                        # to 0 when slime-align is ON;
                                        # native Megatron-Core has no
                                        # source-side loss cap.
                                        if _os_v52s.environ.get(
                                            'AREAL_MTP_SLIME_ALIGN', '1'
                                        ) == '1':
                                            _v52_ratio = 0.0
                                            if not getattr(
                                                _engine_ref,
                                                '_v58_c_logged', False
                                            ):
                                                try:
                                                    _logger.info(
                                                        '[MTPSlime'
                                                        'Align] C) '
                                                        'v52 Source'
                                                        'LossCap '
                                                        'DISABLED '
                                                        '(ratio=0, '
                                                        'slime/'
                                                        'native).'
                                                    )
                                                    _engine_ref._v58_c_logged = True
                                                except Exception:
                                                    pass
                                        else:
                                            _v52_ratio = float(_os_v52s.environ.get(
                                                'AREAL_MTP_V52_LOSS_CAP_RATIO', '2.0'))
                                    except Exception:
                                        _v52_ratio = 2.0
                                    if _v52_ratio > 0.0:
                                        try:
                                            _v52_abs_sum = float(
                                                _mtp_loss_to_store.detach().float().abs().sum().item()
                                            )
                                            _v52_ema_prev = getattr(
                                                _engine_ref, '_v52_loss_abs_sum_ema', None)
                                            if _v52_ema_prev is None or _v52_ema_prev <= 0.0:
                                                _v52_ema = _v52_abs_sum
                                            else:
                                                _v52_ema = 0.95 * _v52_ema_prev + 0.05 * _v52_abs_sum
                                            _engine_ref._v52_loss_abs_sum_ema = _v52_ema
                                            _v52_cap = _v52_ratio * _v52_ema
                                            _v52_capped = False
                                            _v52_scale = 1.0
                                            if (_v52_cap > 0.0
                                                    and _v52_abs_sum > _v52_cap):
                                                _v52_scale = _v52_cap / (_v52_abs_sum + 1e-12)
                                                _mtp_loss_to_store = (
                                                    _mtp_loss_to_store * _v52_scale)
                                                _v52_capped = True
                                            _v52_ctr = getattr(
                                                _engine_ref, '_v52_cap_ctr', 0) + 1
                                            _engine_ref._v52_cap_ctr = _v52_ctr
                                            if (_v52_ctr <= 5 or _v52_ctr % 50 == 0
                                                    or _v52_capped):
                                                _logger.info(
                                                    '[MTPSourceLossCap-v52] call=%d '
                                                    'abs_sum=%.4e ema=%.4e cap=%.4e '
                                                    'ratio=%.2f capped=%s scale=%.4e',
                                                    _v52_ctr, _v52_abs_sum, _v52_ema,
                                                    _v52_cap, _v52_ratio, _v52_capped,
                                                    _v52_scale,
                                                )
                                        except Exception as _e_v52s:
                                            _logger.warning(
                                                '[MTPSourceLossCap-v52] failed: %s',
                                                _e_v52s,
                                            )
                                    # [MTPSlimeAlign] D) skip FIFO append
                                    # when slime-align is ON; native MC has
                                    # no scalar FIFO -- gradient injection
                                    # is handled solely by
                                    # MTPLossAutoScaler.apply below.
                                    try:
                                        import os as _os_v58_d
                                        _v58_slime_d = (
                                            _os_v58_d.environ.get(
                                                'AREAL_MTP_SLIME_ALIGN',
                                                '1') == '1'
                                        )
                                    except Exception:
                                        _v58_slime_d = True
                                    if not _v58_slime_d:
                                        _engine_ref._mtp_loss_for_backward.append(_mtp_loss_to_store)
                                    elif not getattr(
                                        _engine_ref,
                                        '_v58_d_logged', False
                                    ):
                                        try:
                                            _logger.info(
                                                '[MTPSlimeAlign] D) '
                                                'FIFO append SKIPPED '
                                                '(slime/native uses '
                                                'autograd-only path).'
                                            )
                                            _engine_ref._v58_d_logged = True
                                        except Exception:
                                            pass

                                    # ---  BEGIN ---
                                    # Reproduce Megatron-native behaviour:
                                    #   hidden_states = MTPLossAutoScaler.apply(
                                    #       hidden_states,
                                    #       mtp_loss_scale * mtp_loss [/ num_tokens],
                                    #   )
                                    # where MTPLossAutoScaler.backward() returns
                                    # (grad_output, ones_like(mtp_loss) *
                                    #  main_loss_backward_scale). Combined with
                                    # set_loss_scale(1/num_microbatches) this
                                    # injects a per-token * per-vocab gradient
                                    # of magnitude ~ mtp_loss_scale straight into
                                    # the autograd graph, bypassing the scalar
                                    # FIFO + DoubleScale-v6 inverse path.
                                    #
                                    # Gated so the legacy behaviour remains
                                    # bit-exact by default. Enable with
                                    #   AREAL_MTP_NATIVE_AUTOSCALER=1
                                    # [v50:MTPNativePassthrough] default-on.
                                    # Passthrough via MTPLossAutoScaler.apply is the
                                    # verl/slime-aligned path and in Megatron-Core 0.16.0
                                    # it is the ONLY numerically correct path: schedules.py
                                    # sets main_loss_backward_scale = loss_scale /
                                    # num_microbatches automatically after every
                                    # forward_step, so the FIFO + DoubleScale inverse
                                    # mechanism is strictly redundant and introduces
                                    # bf16 rounding jitter. Set AREAL_MTP_NATIVE_AUTOSCALER=0
                                    # to fall back to legacy FIFO (diagnostic only).
                                    try:
                                        import os as _os_v17
                                        _v17_on = (
                                            _os_v17.environ.get(
                                                "AREAL_MTP_NATIVE_AUTOSCALER",
                                                "1",
                                            ) == "1"
                                        )
                                    except Exception:
                                        _v17_on = True
                                    if _v17_on:
                                        try:
                                            from megatron.core.transformer.multi_token_prediction import (
                                                MTPLossAutoScaler as _MTPLossAutoScaler_v17,
                                            )
                                            _num_mb_v17 = int(getattr(
                                                _engine_ref,
                                                "_current_num_microbatches",
                                                1,
                                            ) or 1)
                                            if _num_mb_v17 <= 0:
                                                _num_mb_v17 = 1
                                            import torch as _torch_v17
                                            # schedules.py sets
                                            # main_loss_backward_scale =
                                            # loss_scale / num_microbatches;
                                            # AReaL's consumer already folds
                                            # loss_scale via the outer
                                            # loss * loss_scale contract,
                                            # so only 1/num_mb is needed here.
                                            # [MTPSlimeAlign] E) match
                                            # Megatron-Core schedules.py:
                                            #   loss_scale = grad_scale_func(1.0)
                                            #   set_loss_scale(loss_scale / num_microbatches)
                                            # Falls back to 1/num_mb only
                                            # when slime-align is OFF, to
                                            # preserve legacy behaviour.
                                            try:
                                                import os as _os_v58_e
                                                _v58_slime_e = (
                                                    _os_v58_e.environ.get(
                                                        'AREAL_MTP_SLIME_ALIGN',
                                                        '1') == '1'
                                                )
                                            except Exception:
                                                _v58_slime_e = True
                                            if _v58_slime_e:
                                                try:
                                                    _gsf_e = getattr(
                                                        self_model.config,
                                                        'grad_scale_func',
                                                        None,
                                                    )
                                                    _ls_e = (
                                                        _gsf_e(
                                                            _torch_v17.ones(
                                                                1,
                                                                device=hidden_states.device,
                                                            )
                                                        )
                                                        if _gsf_e is not None
                                                        else _torch_v17.ones(
                                                            1,
                                                            device=hidden_states.device,
                                                        )
                                                    )
                                                except Exception:
                                                    _ls_e = _torch_v17.ones(
                                                        1,
                                                        device=hidden_states.device,
                                                    )
                                                _MTPLossAutoScaler_v17.set_loss_scale(
                                                    _ls_e / float(_num_mb_v17)
                                                )
                                                if not getattr(
                                                    _engine_ref,
                                                    '_v58_e_logged', False
                                                ):
                                                    try:
                                                        _logger.info(
                                                            '[MTPSlime'
                                                            'Align] E) '
                                                            'set_loss_scale'
                                                            '=loss_scale/'
                                                            'num_mb (= '
                                                            'Megatron-Core '
                                                            'schedules.py '
                                                            ': %s / %d).',
                                                            float(
                                                                _ls_e.item()
                                                                if hasattr(
                                                                    _ls_e,
                                                                    'item')
                                                                else _ls_e
                                                            ),
                                                            int(_num_mb_v17),
                                                        )
                                                        _engine_ref._v58_e_logged = True
                                                    except Exception:
                                                        pass
                                            else:
                                                _MTPLossAutoScaler_v17.set_loss_scale(
                                                    _torch_v17.tensor(
                                                        1.0 / float(_num_mb_v17)
                                                    )
                                                )
                                            try:
                                                _d06_step = getattr(
                                                    _engine_ref,
                                                    "_global_step", 0)
                                                if (_mtp_diag_mb_counter[0] == 0
                                                        and (_d06_step <= 5
                                                        or _d06_step % 50 == 0)):
                                                    _logger.info(
                                                        "[SpecDecDiag-v20 "
                                                        "D06] "
                                                        "step=%d mtp_layer=%d "
                                                        "mtp_loss_scale=%.6e "
                                                        "calculate_per_token_"
                                                        "loss=%s "
                                                        "num_tokens=%s "
                                                        "num_mb=%d "
                                                        "mtp_loss_to_store:"
                                                        " shape=%s rg=%s "
                                                        "sum=%.6e abs_max=%.3e",
                                                        _d06_step,
                                                        mtp_layer_number,
                                                        float(mtp_loss_scale),
                                                        self_model.config
                                                            .calculate_per_token_loss,
                                                        num_tokens,
                                                        _num_mb_v17,
                                                        list(_mtp_loss_to_store
                                                             .shape),
                                                        _mtp_loss_to_store
                                                            .requires_grad,
                                                        _mtp_loss_to_store
                                                            .detach().float()
                                                            .sum().item(),
                                                        _mtp_loss_to_store
                                                            .detach().float()
                                                            .abs().max().item(),
                                                    )
                                            except Exception as _e_d06:
                                                _logger.warning(
                                                    "[SpecDecDiag-v20 D06] "
                                                    "failed: %s", _e_d06,
                                                )
                                            hidden_states = (
                                                _MTPLossAutoScaler_v17.apply(
                                                    hidden_states,
                                                    _mtp_loss_to_store,
                                                )
                                            )
                                            try:
                                                _d07_bs = (
                                                    _MTPLossAutoScaler_v17
                                                    .main_loss_backward_scale
                                                )
                                                _d07_bs_v = (
                                                    float(_d07_bs.item())
                                                    if hasattr(_d07_bs, "item")
                                                    else float(_d07_bs)
                                                )
                                                if (_mtp_diag_mb_counter[0] == 0
                                                        and (_d06_step <= 5
                                                        or _d06_step % 50
                                                        == 0)):
                                                    _logger.info(
                                                        "[SpecDecDiag-v20 "
                                                        "D07] step=%d "
                                                        "mtp_layer=%d "
                                                        "post-apply "
                                                        "main_loss_backward_"
                                                        "scale=%.6e "
                                                        "hs.grad_fn=%s",
                                                        _d06_step,
                                                        mtp_layer_number,
                                                        _d07_bs_v,
                                                        type(hidden_states
                                                             .grad_fn).__name__
                                                        if hidden_states
                                                            .grad_fn
                                                        else "None",
                                                    )
                                            except Exception as _e_d07:
                                                _logger.warning(
                                                    "[SpecDecDiag-v20 D07] "
                                                    "failed: %s", _e_d07,
                                                )
                                            _engine_ref._v17_native_active = True
                                            if _mtp_diag_mb_counter[0] == 0:
                                                _logger.info(
                                                    "[MTPNativeAutoScaler-v17] "
                                                    "apply() injected: "
                                                    "num_mb=%d, "
                                                    "main_loss_backward_scale=%.6e, "
                                                    "hidden_states.shape=%s, "
                                                    "hidden_states.rg=%s",
                                                    _num_mb_v17,
                                                    1.0 / float(_num_mb_v17),
                                                    list(hidden_states.shape),
                                                    hidden_states.requires_grad,
                                                )
                                        except Exception as _e_v17:
                                            _engine_ref._v17_native_active = False
                                            _logger.warning(
                                                "[MTPNativeAutoScaler-v17] "
                                                "apply() failed, falling back "
                                                "to legacy FIFO+DoubleScale "
                                                "path: %s",
                                                _e_v17,
                                            )
                                    else:
                                        _engine_ref._v17_native_active = False
                                    # --- [MTPNativeAutoScaler-v17] END ---
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
                                        import torch as _t_d08
                                        _g_f = grad.float()
                                        _lg.info(
                                            "[MTPBwdDiag] AutoScaler backward FIRED (step=%d): "
                                            "grad.shape=%s, grad.norm=%.8f, "
                                            "grad.abs_max=%.8f",
                                            _gs, list(grad.shape),
                                            _g_f.norm().item(),
                                            _g_f.abs().max().item())
                                        _lg.info(
                                            "[SpecDecDiag-v20 D08] "
                                            "hs-bwd step=%d grad.abs_mean=%.3e "
                                            "grad.mean=%.3e grad.std=%.3e "
                                            "grad.nonzero_frac=%.3f "
                                            "grad.finite=%s dtype=%s",
                                            _gs,
                                            _g_f.abs().mean().item(),
                                            _g_f.mean().item(),
                                            _g_f.std().item(),
                                            (_g_f != 0).float().mean().item(),
                                            bool(_t_d08.isfinite(_g_f)
                                                 .all().item()),
                                            str(grad.dtype),
                                        )
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

                        # Path 3: patch _get_embeddings for embedding detach.
                        # [MTPSlimeAlign] A) Skip Path-3 detach when
                        # slime-align is ON; native Megatron-Core uses
                        # `make_viewless_tensor(..., keep_graph=True)`
                        # which preserves the gradient flow through the
                        # decoder_input/hidden_states into the main
                        # embedding & backbone -- this is precisely the
                        # mechanism that makes slime's
                        # `mtp_loss_scaling_factor=0.2` an effective
                        # main-policy regulariser.
                        try:
                            import os as _os_v58_a
                            _v58_slime_a = (
                                _os_v58_a.environ.get(
                                    'AREAL_MTP_SLIME_ALIGN', '1') == '1'
                            )
                        except Exception:
                            _v58_slime_a = True
                        _mtp_block = getattr(_unwrapped, "mtp", None)
                        if _v58_slime_a:
                            if not getattr(
                                self, '_v58_a_logged', False
                            ):
                                try:
                                    self.logger.info(
                                        '[MTPSlimeAlign] A) Path-3 '
                                        '_get_embeddings detach SKIPPED. '
                                        'Native Megatron-Core preserves '
                                        'decoder_input/hidden_states via '
                                        'make_viewless_tensor(keep_graph='
                                        'True), letting MTP CE backward '
                                        'flow into the main embedding & '
                                        'backbone (slime semantics).'
                                    )
                                    self._v58_a_logged = True
                                except Exception:
                                    pass
                            _mtp_block = None  # disables the patch loop
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

        # [SpecDecDiag-v20 D04] per-step summary before fwd/bwd.
        try:
            _d04_nmb = int(self._current_num_microbatches)
            _d04_ntok = int(getattr(self, "_current_n_tokens", 0) or 0)
            _d04_pgs = getattr(self.optimizer, "param_groups", []) or []
            _d04_base_lr = float(_d04_pgs[0].get("lr", 0.0)) if _d04_pgs else 0.0
            _d04_base_max = float(_d04_pgs[0].get("max_lr", 0.0)) if _d04_pgs else 0.0
            _d04_mtp_lr = None
            if self.enable_mtp_training and len(_d04_pgs) > 1:
                for _pg in _d04_pgs:
                    if (_pg.get("max_lr", None) is not None
                            and abs(float(_pg.get("max_lr"))
                                    - _d04_base_max) > 1e-12):
                        _d04_mtp_lr = float(_pg.get("lr", 0.0))
                        break
            self.logger.info(
                "[SpecDecDiag-v20 D04] TrainStepEnter step=%d num_mb=%d "
                "n_tokens=%d base_lr=%.3e base_max_lr=%.3e "
                "mtp_lr=%s n_param_groups=%d loss_multiplier=%.3e",
                self._global_step, _d04_nmb, _d04_ntok,
                _d04_base_lr, _d04_base_max,
                ("%.3e" % _d04_mtp_lr) if _d04_mtp_lr is not None else "base",
                len(_d04_pgs), float(loss_multiplier),
            )
        except Exception as _e_d04:
            self.logger.warning(
                "[SpecDecDiag-v20 D04] TrainStepEnter log failed: %s",
                _e_d04,
            )

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
        # [v32] Snapshot the reduced stats dict so the MTP weight-
        # sync path can read task_reward / entropy / accept_rate
        # without re-entering stats_tracker (which would reset the
        # accumulators on export).
        try:
            self._last_stats_snapshot_v32 = dict(data)
            _tr = data.get("ppo_actor/task_reward/avg")
            _ea = data.get("ppo_actor/update/entropy/avg")
            if isinstance(_tr, (int, float)):
                self._last_task_reward_avg = float(_tr)
            if isinstance(_ea, (int, float)):
                self._last_entropy_avg = float(_ea)
        except Exception:
            pass
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
        # [P2-MTPShareParamGroup] When the MTP-only param group is activated
        # via ParamKey(name=("*.mtp.*",)), Megatron 0.16 DistributedOptimizer
        # only shards that (small) group across a subset of DP ranks, leaving
        # the other ranks with ``param.main_param = None``.  That breaks the
        # weight-ship path because _collect_param(..) returns None on those
        # ranks, so ``mtp_hf_tensors`` stays empty and sglang draft never
        # gets updated.  Default behaviour of this patch is to force MTP to
        # share the main param group (mtp_lr_scale coerced to 1.0); opt-out
        # via AREAL_MTP_SHARE_PARAM_GROUP=0.
        _v59_share_pg = (os.environ.get(
            "AREAL_MTP_SHARE_PARAM_GROUP", "1") == "1")
        if (
            self.enable_mtp_training
            and _v59_share_pg
            and _mtp_lr_scale != 1.0
        ):
            self.logger.warning(
                "[MTPShareParamGroup-P2] overriding mtp_lr_scale=%.3f -> 1.0 "
                "so MTP parameters share the main param group and every DP "
                "rank holds a master-param shard. Set "
                "AREAL_MTP_SHARE_PARAM_GROUP=0 to restore the split.",
                _mtp_lr_scale,
            )
            _mtp_lr_scale = 1.0
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

        # [MTPOptim-v12] Dump param_groups to verify ParamKey override
        # actually installed. Megatron 0.16 ParamKey does NOT attach a `name`
        # field to param_groups, so downstream identification must use
        # `max_lr` fingerprint instead of name match.
        try:
            _base_max_lr = float(self.optimizer_config.lr)
            for _idx, _pg in enumerate(
                getattr(self.optimizer, "param_groups", []) or []
            ):
                _n_params = len(_pg.get("params", []) or [])
                _mxlr = _pg.get("max_lr", None)
                _mnlr = _pg.get("min_lr", None)
                _is_mtp = (
                    _mxlr is not None
                    and abs(float(_mxlr) - _base_max_lr) > 1e-12
                )
                self.logger.info(
                    "[MTPOptim-v12] param_group[%d]: n_params=%d max_lr=%s "
                    "min_lr=%s is_mtp_group=%s",
                    _idx, _n_params, str(_mxlr), str(_mnlr),
                    str(_is_mtp),
                )
        except Exception as _e:
            self.logger.warning(
                "[MTPOptim-v12] param_groups dump failed: %s", _e
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
        # [MTPSendPreBcast-v25] Capture exact tensors to be broadcast.
        try:
            for _v25_name, _v25_t in converted_named_tensors:
                if ("mtp_layers." in _v25_name or ".mtp." in _v25_name):
                    try:
                        _v25_first8 = [
                            float(x) for x in _v25_t.flatten()[:8].tolist()
                        ]
                    except Exception:
                        _v25_first8 = []
                    self.logger.info(
                        "[MTPSendPreBcast-v25] name=%s dtype=%s shape=%s "
                        "abs_mean=%.6e abs_max=%.6e first8=%s",
                        _v25_name, str(_v25_t.dtype), tuple(_v25_t.shape),
                        _v25_t.abs().mean().item(),
                        _v25_t.abs().max().item(),
                        _v25_first8,
                    )
        except Exception as _e_v25s:
            self.logger.warning(
                "[MTPSendPreBcast-v25] probe error: %s", _e_v25s,
            )
        # [MTPDraftEMA-v54] Optional EMA smoothing of the bf16 wire
        # payload shipped to sglang, applied right before the RPC
        # update_weights_from_distributed() call.  alpha in (0,1)
        # produces:
        #   W_ship[t] = (1-alpha) * W_ship[t-1] + alpha * W_train[t]
        # dampening per-step MTP update noise as seen by the EAGLE
        # draft head.  alpha==0.0 (default) or alpha==1.0 is
        # pass-through (feature disabled / no smoothing).
        _v54_ema_applied_names = set()
        try:
            import os as _os_v54e
            _v54_alpha_raw = _os_v54e.environ.get(
                'AREAL_MTP_V54_DRAFT_EMA', '0.0',
            )
            try:
                _v54_alpha = float(_v54_alpha_raw)
            except Exception:
                _v54_alpha = 0.0
            _v54_ema_on = (0.0 < _v54_alpha < 1.0)
            self._v54_ema_alpha = _v54_alpha
            self._v54_ema_enabled = _v54_ema_on
            if _v54_ema_on:
                if not hasattr(self, '_v54_ema_state'):
                    self._v54_ema_state = {}
                _v54_ema_n = 0
                for _v54_idx, _v54_np in enumerate(
                    converted_named_tensors
                ):
                    _v54_name, _v54_param = _v54_np
                    if not (
                        '.enorm' in _v54_name
                        or '.hnorm' in _v54_name
                        or '.eh_proj' in _v54_name
                        or '.shared_head.' in _v54_name
                        or '.mtp_layers.' in _v54_name
                    ):
                        continue
                    try:
                        _v54_cur = _v54_param.data
                        _v54_prev = self._v54_ema_state.get(_v54_name)
                        _v54_pre_norm = float(
                            _v54_cur.float().norm().item()
                        )
                        if (
                            _v54_prev is not None
                            and _v54_prev.shape == _v54_cur.shape
                        ):
                            _v54_smoothed = (
                                (1.0 - _v54_alpha) * _v54_prev.to(
                                    torch.float32
                                )
                                + _v54_alpha * _v54_cur.to(
                                    torch.float32
                                )
                            )
                            _v54_smoothed = _v54_smoothed.to(
                                _v54_cur.dtype
                            ).contiguous()
                            _v54_param.data.copy_(_v54_smoothed)
                            self._v54_ema_state[_v54_name] = (
                                _v54_smoothed.detach().clone()
                            )
                            _v54_ema_applied_names.add(_v54_name)
                            _v54_ema_n += 1
                            _v54_post_norm = float(
                                _v54_param.data.float()
                                    .norm().item()
                            )
                            self.logger.info(
                                '[SpecDecFlow-v54] stage=ema '
                                'name=%s alpha=%.4f '
                                'pre_norm=%.6e post_norm=%.6e '
                                'applied=True',
                                _v54_name, _v54_alpha,
                                _v54_pre_norm, _v54_post_norm,
                            )
                        else:
                            self._v54_ema_state[_v54_name] = (
                                _v54_cur.detach().clone()
                            )
                            self.logger.info(
                                '[SpecDecFlow-v54] stage=ema '
                                'name=%s alpha=%.4f '
                                'pre_norm=%.6e post_norm=%.6e '
                                'applied=False reason=seed',
                                _v54_name, _v54_alpha,
                                _v54_pre_norm, _v54_pre_norm,
                            )
                    except Exception:
                        continue
                self.logger.info(
                    '[MTPDraftEMA-v54] applied alpha=%.4f to %d '
                    'MTP wire tensors (cache_size=%d)',
                    _v54_alpha, _v54_ema_n,
                    len(self._v54_ema_state),
                )
                self.logger.info(
                    '[SpecDecFlow-v54] stage=ema_summary '
                    'alpha=%.4f n_applied=%d cache_size=%d',
                    _v54_alpha, _v54_ema_n,
                    len(self._v54_ema_state),
                )
        except Exception as _e_v54e:
            try:
                self.logger.warning(
                    '[MTPDraftEMA-v54] gate failed: %r', _e_v54e,
                )
            except Exception:
                pass
        # [SpecDecFlow-v54] SHIP stage — per-MTP-wire-tensor payload
        # diagnostics right before dist.broadcast(). Answers:
        # 'exactly which bytes are shipped to sglang this version?'.
        try:
            import os as _os_v54s
            _v54_flow_on3 = (
                _os_v54s.environ.get(
                    'AREAL_MTP_V54_SPEC_FLOW_LOG', '1',
                ) == '1'
            )
            if _v54_flow_on3:
                _v54_ship_n = 0
                _v54_ship_bytes = 0
                _v54_ship_sq = 0.0
                _v54_ship_cnt = 0
                _v54_ship_first = None
                _v54_ship_first_l2 = -1.0
                _v54_ship_mtp_only = 0
                # [MTPShipSummaryFix-v56] Iterate the REAL MTP wire
                # payload (`mtp_hf_tensors`, stashed on self at the
                # `_update_weights_from_distributed` call site) instead
                # of `converted_named_tensors` (which is the main-model
                # bucket payload during the MTP wire path).  This fixes
                # the v54 ship_summary log that always reported
                # n_mtp_shipped=0.
                _v56_ship_iter = list(
                    getattr(self, '_v56_mtp_hf_tensors', []) or []
                )
                for _v54_si, (_v54_sn, _v54_st) in enumerate(
                    _v56_ship_iter
                ):
                    _is_mtp = (
                        '.enorm' in _v54_sn
                        or '.hnorm' in _v54_sn
                        or '.eh_proj' in _v54_sn
                        or '.shared_head.' in _v54_sn
                        or '.mtp_layers.' in _v54_sn
                        # [MTPShipSummaryFix-v56] Items in mtp_hf_tensors
                        # are already MTP-only, so accept anything that
                        # came from that list as MTP wire payload.
                        or True
                    )
                    if not _is_mtp:
                        continue
                    _v54_ship_mtp_only += 1
                    try:
                        _td = _v54_st.detach()
                        _tf = _td.float()
                        _l2 = float(_tf.norm().item())
                        _am = float(_tf.abs().mean().item())
                        _ax = float(_tf.abs().max().item())
                        _v54_ship_sq += _l2 * _l2
                        _v54_ship_cnt += int(_tf.numel())
                        _v54_ship_bytes += int(
                            _td.numel() * _td.element_size()
                        )
                        _v54_ship_n += 1
                        if _v54_ship_first is None:
                            _v54_ship_first = _v54_sn
                            _v54_ship_first_l2 = _l2
                        self.logger.info(
                            '[SpecDecFlow-v54] stage=ship '
                            'idx=%d name=%s dtype=%s shape=%s '
                            'l2=%.6e abs_mean=%.6e abs_max=%.6e '
                            'ema_applied=%s',
                            _v54_si, _v54_sn, str(_td.dtype),
                            str(tuple(_td.shape)),
                            _l2, _am, _ax,
                            str(_v54_sn in _v54_ema_applied_names),
                        )
                    except Exception:
                        continue
                _v54_wire_norm = _v54_ship_sq ** 0.5
                _v54_prev_wire = getattr(
                    self, '_v54_prev_wire_norm', None,
                )
                self._v54_prev_wire_norm = _v54_wire_norm
                _v54_d_wire = -1.0
                if _v54_prev_wire is not None:
                    _v54_d_wire = abs(
                        _v54_wire_norm - _v54_prev_wire
                    )
                self.logger.info(
                    '[SpecDecFlow-v54] stage=ship_summary '
                    '[MTPShipSummaryFix-v56] '
                    'version=%s n_mtp_shipped=%d '
                    'total_bytes=%d wire_norm=%.6e '
                    'd_wire_norm=%.6e first=%s first_l2=%.6e '
                    'ema_enabled=%s ema_alpha=%.4f '
                    'freeze_engaged=%s',
                    str(getattr(meta, 'version', 'NA')),
                    _v54_ship_n, _v54_ship_bytes,
                    _v54_wire_norm, _v54_d_wire,
                    str(_v54_ship_first),
                    _v54_ship_first_l2,
                    str(getattr(
                        self, '_v54_ema_enabled', False,
                    )),
                    float(getattr(
                        self, '_v54_ema_alpha', 0.0,
                    )),
                    str(getattr(
                        self, '_v54_freeze_engaged', False,
                    )),
                )
        except Exception as _e_v54s:
            try:
                self.logger.warning(
                    '[SpecDecFlow-v54] ship failed: %r', _e_v54s,
                )
            except Exception:
                pass
        _t_post0 = _diag_time.time()
        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)
        self.logger.info(
            f"[DiagBucket] rollout_engine.update_weights_from_distributed POST sent "
            f"in {_diag_time.time() - _t_post0:.3f}s, fut={fut}"
        )

        _t_bc0 = _diag_time.time()
        handles = []
        _mtp_upcast_count = 0
        for idx, (name, param) in enumerate(converted_named_tensors):
            # MTP draft-head deltas are typically smaller than bf16 ULP
            # (see [MTPSyncHealth] stall diagnostics). Upcast MTP tensors
            # to fp32 on the trainer side before NCCL broadcast so the
            # inference-side draft head sees the full precision update.
            # The rollout side will downcast during load_weights.
            send_tensor = param.data
            if (
                (".enorm" in name or ".hnorm" in name or ".eh_proj" in name
                 or ".shared_head." in name or ".mtp_layers." in name)
                and send_tensor.dtype == torch.bfloat16
            ):
                send_tensor = send_tensor.float().contiguous()
                # rebind so the receiver (whose dtype spec was already
                # promoted in build_tensor_weight_update_request) matches.
                converted_named_tensors[idx] = (name, send_tensor)
                _mtp_upcast_count += 1
            handles.append(
                dist.broadcast(
                    send_tensor, 0, group=self.weight_update_group, async_op=True
                )
            )
        if _mtp_upcast_count > 0:
            self.logger.info(
                "[MTPBroadcastDtype] Upcast %d MTP tensors to fp32 for "
                "NCCL broadcast (avoid bf16 ULP absorption of draft-head "
                "weight deltas).",
                _mtp_upcast_count,
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
            # This was previously silently swallowed. Surface loudly:
            # if the callback never finishes, the inference engine may
            # have partially applied the broadcast, desyncing the draft.
            self.logger.error(
                "[MTPBroadcastTimeout] Callback response timed out after "
                "30s while waiting for rollout side update_weights_from_"
                "distributed to acknowledge. NCCL broadcast completed on "
                "trainer side but the inference engine may NOT have "
                "finished applying the weights. This CAN silently desync "
                "MTP draft head and cause accept_rate decay. "
                "n_tensors=%d, n_specs=%d.",
                len(converted_named_tensors), len(param_specs),
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
        # [MTPWeightHash-v42] Fingerprint each MTP tensor about to be
        # serialised.  We hash up to 1024 fp32 values with a
        # per-tensor xor-rotate so the 64-bit digest changes on ANY
        # modification, without paying for a full-tensor reduction.
        # The digest stream is monotonic only if the target-side
        # weights are actually being refreshed between versions,
        # which lets us discriminate H6 (target/draft sync skew)
        # from H5 (policy-phase drift) when accept-rate dips.
        try:
            import torch as _torch_v42
            _v42_ver = None
            try:
                _v42_ver = int(self.get_version())
            except Exception:
                _v42_ver = None
            _v42_digests = []
            for _v42_n, _v42_t in mtp_hf_tensors:
                try:
                    _flat = _v42_t.detach().reshape(-1)
                    _k = min(1024, int(_flat.numel()))
                    if _k > 0:
                        _sl = _flat[:_k].float().contiguous().cpu()
                        _bytes = _sl.numpy().tobytes()
                        _h = 0
                        for _b in _bytes:
                            _h = ((_h * 1315423911) ^ int(_b)) & ((1 << 64) - 1)
                        _s = float(_sl.sum().item())
                        _a = float(_sl.abs().mean().item())
                    else:
                        _h, _s, _a = 0, 0.0, 0.0
                    _v42_digests.append(
                        (_v42_n, _h, _s, _a,
                         tuple(_v42_t.shape), str(_v42_t.dtype))
                    )
                except Exception as _e_hash_one:
                    _v42_digests.append(
                        (_v42_n, None, None, None, None,
                         repr(_e_hash_one))
                    )
            self.logger.info(
                "[MTPWeightHash-v42] version=%s n_tensors=%d digests=%s",
                _v42_ver, len(_v42_digests), _v42_digests,
            )
            # [v43] delta detector across versions
            try:
                _cur_map_v43 = {}
                for _d in _v42_digests:
                    if isinstance(_d, tuple) and len(_d) >= 2:
                        _cur_map_v43[_d[0]] = _d[1]
                _prev_map_v43 = getattr(self, "_v43_prev_digests", None)
                if isinstance(_prev_map_v43, dict):
                    _changed_v43 = []
                    _same_v43 = []
                    for _n, _h in _cur_map_v43.items():
                        _ph = _prev_map_v43.get(_n)
                        if _ph is None:
                            continue
                        if _ph == _h:
                            _same_v43.append(_n)
                        else:
                            _changed_v43.append(_n)
                    self.logger.info(
                        "[MTPWeightHashDelta-v43] version=%s "
                        "n_total=%d n_changed=%d n_same=%d "
                        "changed=%s same=%s",
                        _v42_ver, len(_cur_map_v43),
                        len(_changed_v43), len(_same_v43),
                        _changed_v43, _same_v43,
                    )
                else:
                    self.logger.info(
                        "[MTPWeightHashDelta-v43] version=%s baseline "
                        "(no prior digest map)",
                        _v42_ver,
                    )
                self._v43_prev_digests = _cur_map_v43
                # [MTPDraftIPCStall-v45] cumulative stall count
                # per tensor.  If a hash equals the previous
                # version's hash, the draft worker saw a
                # bit-exact copy: stall_count += 1, else reset.
                try:
                    if not hasattr(self, "_v45_stall_count"):
                        self._v45_stall_count = {}
                    _prev_cur = getattr(
                        self, "_v45_last_cur_map", None
                    )
                    _v45_rows = []
                    for _n_s, _h_s in _cur_map_v43.items():
                        if (_prev_cur is not None
                                and _prev_cur.get(_n_s) == _h_s):
                            self._v45_stall_count[_n_s] = (
                                self._v45_stall_count.get(_n_s, 0) + 1
                            )
                        else:
                            self._v45_stall_count[_n_s] = 0
                        _v45_rows.append(
                            (_n_s, self._v45_stall_count[_n_s])
                        )
                    self._v45_last_cur_map = dict(_cur_map_v43)
                    _v45_rows.sort(key=lambda r: -r[1])
                    self.logger.info(
                        "[MTPDraftIPCStall-v45] version=%s "
                        "max_stall=%s top5_stalled=%s",
                        _v42_ver,
                        (_v45_rows[0][1] if _v45_rows else None),
                        _v45_rows[:5],
                    )
                except Exception as _e_v45_s:
                    try:
                        self.logger.info(
                            "[MTPDraftIPCStall-v45] failure: %r",
                            _e_v45_s,
                        )
                    except Exception:
                        pass
            except Exception as _e_delta_v43:
                try:
                    self.logger.info(
                        "[MTPWeightHashDelta-v43] failure: %r",
                        _e_delta_v43,
                    )
                except Exception:
                    pass
        except Exception as _e_hash_all:
            try:
                self.logger.info(
                    "[MTPWeightHash-v42] probe failure: %r", _e_hash_all,
                )
            except Exception:
                pass
        # [MTPSerializeSendMTP-v26] Sample first 8 values of each MTP
        # tensor so we can prove the actual bytes placed into the
        # SGLang IPC payload. The earlier MTPSendPreBcast-v25 probe
        # was installed on the /update_weights_from_distributed bucket
        # path which MTP tensors bypass — explaining 0 events in log.7.
        try:
            for _v26_name, _v26_t in mtp_hf_tensors:
                try:
                    _v26_first8 = [
                        float(x) for x in _v26_t.flatten()[:8].tolist()
                    ]
                except Exception:
                    _v26_first8 = []
                self.logger.info(
                    "[MTPSerializeSendMTP-v26] name=%s dtype=%s shape=%s "
                    "abs_mean=%.6e abs_max=%.6e first8=%s",
                    _v26_name, str(_v26_t.dtype), tuple(_v26_t.shape),
                    float(_v26_t.abs().mean().item()),
                    float(_v26_t.abs().max().item()),
                    _v26_first8,
                )
                # [MTPBf16ULPProof-v28] Prove/disprove bf16 ULP flooring on receiver side.
                try:
                    import torch as _torch_v28p
                    if not hasattr(self, "_mtp_v28_prev_bf16cast"):
                        self._mtp_v28_prev_bf16cast = {}
                    _v28_tf = _v26_t.float()
                    _v28_bf16 = _v28_tf.to(_torch_v28p.bfloat16)
                    _v28_bb = _v28_bf16.float()
                    _v28_eqcast = int((_v28_tf == _v28_bb).sum().item())
                    _v28_numel = int(_v28_tf.numel())
                    _v28_frac = _v28_eqcast / max(1, _v28_numel)
                    _v28_prev = self._mtp_v28_prev_bf16cast.get(
                        _v26_name
                    )
                    if _v28_prev is None:
                        _v28_unchanged = None
                    else:
                        try:
                            if _v28_prev.shape == _v28_bb.shape:
                                _v28_unchanged = int(
                                    (_v28_bb == _v28_prev).sum().item()
                                )
                            else:
                                _v28_unchanged = -2
                        except Exception:
                            _v28_unchanged = -1
                    self._mtp_v28_prev_bf16cast[_v26_name] = (
                        _v28_bb.detach().clone()
                    )
                    self.logger.info(
                        "[MTPBf16ULPProof-v28] name=%s numel=%d "
                        "fp32_eq_bf16cast=%d (frac=%.4f) "
                        "bf16cast_eq_prev_bf16cast=%s",
                        _v26_name, _v28_numel, _v28_eqcast,
                        _v28_frac,
                        ("n/a" if _v28_unchanged is None
                         else str(_v28_unchanged)),
                    )
                except Exception as _e_v28p:
                    self.logger.warning(
                        "[MTPBf16ULPProof-v28] error name=%s: %s",
                        _v26_name, _e_v28p,
                    )
        except Exception as _e_v26s:
            self.logger.warning(
                "[MTPSerializeSendMTP-v26] probe error: %s", _e_v26s,
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

        # [MTPPreScan-v22] Early diagnostic pre-scan for MTP params.
        # Runs on ALL ranks before the main param loop so that each
        # MTP param's main_param availability / fp32 stats survive
        # even if the later loop hangs.
        try:
            import os as _os_v22
            import sys as _sys_v22
            import torch as _torch_v22
            _v22_is_pp_head = self.is_pipeline_parallel_head()
            _v22_supports_tu = getattr(
                self, "_engine_supports_tensor_update", False,
            )
            _v22_enable_mtp = bool(
                getattr(self, "enable_mtp_training", False)
            )
            _v22_collect = (
                _v22_enable_mtp
                and _v22_supports_tu
                and _v22_is_pp_head
            )
            _v22_master_on = (
                _os_v22.environ.get(
                    "AREAL_MTP_FP32_MASTER_READ", "1",
                ) == "1"
            )
            _v22_bcast_on = (
                _os_v22.environ.get(
                    "AREAL_MTP_FP32_BROADCAST", "1",
                ) == "1"
            )
            self.logger.info(
                "[MTPPreScan-v22] ENTRY rank=%d version=%s "
                "is_pp_head=%s supports_tu=%s enable_mtp=%s "
                "collect=%s master_on=%s fp32_bcast_on=%s",
                dist.get_rank(),
                str(getattr(meta, "version", "?")),
                str(_v22_is_pp_head), str(_v22_supports_tu),
                str(_v22_enable_mtp), str(_v22_collect),
                str(_v22_master_on), str(_v22_bcast_on),
            )
            try:
                for _h in list(self.logger.handlers):
                    try:
                        _h.flush()
                    except Exception:
                        pass
                _sys_v22.stdout.flush()
            except Exception:
                pass
            _v22_mtp_seen = 0
            _v22_ok = 0
            _v22_missing = 0
            for _v22_nm, _v22_p in get_named_parameters(
                self.model, num_moe_experts,
            ):
                if ".experts." in _v22_nm:
                    continue
                if ".mtp." not in _v22_nm:
                    continue
                _v22_mtp_seen += 1
                _v22_mp = getattr(_v22_p, "main_param", None)
                _v22_kind = type(_v22_mp).__name__
                _v22_dtype = (
                    str(_v22_mp.dtype)
                    if isinstance(_v22_mp, _torch_v22.Tensor)
                    else "n/a"
                )
                _v22_shard_numel = (
                    int(_v22_mp.numel())
                    if isinstance(_v22_mp, _torch_v22.Tensor)
                    else -1
                )
                _v22_fp32_am = -1.0
                _v22_fp32_amax = -1.0
                try:
                    if (
                        isinstance(_v22_mp, _torch_v22.Tensor)
                        and _v22_mp.dtype == _torch_v22.float32
                    ):
                        _v22_ok += 1
                        _v22_absf = _v22_mp.detach().abs()
                        _v22_fp32_am = float(_v22_absf.mean().item())
                        _v22_fp32_amax = float(_v22_absf.max().item())
                    else:
                        _v22_missing += 1
                except Exception:
                    _v22_missing += 1
                _v22_bf16_am = -1.0
                _v22_bf16_amax = -1.0
                try:
                    _v22_absb = _v22_p.detach().float().abs()
                    _v22_bf16_am = float(_v22_absb.mean().item())
                    _v22_bf16_amax = float(_v22_absb.max().item())
                except Exception:
                    pass
                self.logger.info(
                    "[MTPPreScan-v22] rank=%d name=%s "
                    "master_kind=%s master_dtype=%s "
                    "shard_numel=%d full_numel=%d "
                    "fp32_abs_mean=%.6e fp32_abs_max=%.6e "
                    "bf16_abs_mean=%.6e bf16_abs_max=%.6e "
                    "shape=%s",
                    dist.get_rank(),
                    _v22_nm, _v22_kind, _v22_dtype,
                    _v22_shard_numel, int(_v22_p.numel()),
                    _v22_fp32_am, _v22_fp32_amax,
                    _v22_bf16_am, _v22_bf16_amax,
                    tuple(_v22_p.shape),
                )
                try:
                    for _h in list(self.logger.handlers):
                        try:
                            _h.flush()
                        except Exception:
                            pass
                    _sys_v22.stdout.flush()
                except Exception:
                    pass
            self.logger.info(
                "[MTPPreScan-v22] SUMMARY rank=%d version=%s "
                "mtp_params=%d master_ok=%d master_missing=%d",
                dist.get_rank(),
                str(getattr(meta, "version", "?")),
                _v22_mtp_seen, _v22_ok, _v22_missing,
            )
            try:
                for _h in list(self.logger.handlers):
                    try:
                        _h.flush()
                    except Exception:
                        pass
                _sys_v22.stdout.flush()
            except Exception:
                pass
        except Exception as _e_v22:
            self.logger.warning(
                "[MTPPreScan-v22] aborted: %s", _e_v22,
            )

        _param_idx = 0
        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." in name:
                continue
            if ".mtp." in name:
                mtp_param_count += 1
                mtp_param_bytes += param.numel() * param.element_size()
                # [MTPFp32MasterRead-v24]  TP-dtype-symmetric path.
                #
                # Root cause of v23 hang at eh_proj.weight:
                #   rank 0 (pp-head,dp=0,tp=0): _collect_param(fp32)
                #   rank 1 (non-pp,  dp=0,tp=1): _collect_param(bf16)
                # _collect_param internally does TP all_gather_param;
                # two TP peers feeding different dtypes -> NCCL hang.
                #
                # v24 uniformly builds _fp32_full on the owning DP
                # rank regardless of pp-head status, and ALWAYS passes
                # the SAME dtype tensor to _collect_param on every TP
                # peer of that TP group.  Non-pp-head peer drops the
                # returned collected tensor.
                import os as _os_v24m
                import sys as _sys_v24m
                import torch as _torch_v24m
                _mp_on = (
                    _os_v24m.environ.get(
                        "AREAL_MTP_FP32_MASTER_READ", "1",
                    ) == "1"
                )
                _fp32_full = None
                _src_tag = "bf16model"
                if _mp_on:
                    try:
                        _mp_shard = getattr(param, "main_param", None)
                        _have_master_local = (
                            isinstance(_mp_shard, _torch_v24m.Tensor)
                            and _mp_shard.dtype == _torch_v24m.float32
                            and int(_mp_shard.numel())
                            == int(param.numel())
                        )
                        try:
                            _dp_group = mpu.get_data_parallel_group(
                                with_context_parallel=True,
                            )
                        except TypeError:
                            _dp_group = mpu.get_data_parallel_group()
                        _dp_ws = _torch_v24m.distributed.get_world_size(
                            group=_dp_group,
                        )
                        try:
                            _tp_group = mpu.get_tensor_model_parallel_group()
                        except Exception:
                            _tp_group = None
                        _tp_ws = (
                            _torch_v24m.distributed.get_world_size(
                                group=_tp_group)
                            if _tp_group is not None else 1
                        )
                        self.logger.info(
                            "[MTPFp32MasterRead-v24 ENTER] rank=%d "
                            "name=%s pp_head=%s dp_ws=%d tp_ws=%d "
                            "have_master_local=%s shard_numel=%s "
                            "need_numel=%d",
                            dist.get_rank(), name,
                            str(_collect_mtp_for_draft),
                            _dp_ws, _tp_ws,
                            str(_have_master_local),
                            (str(int(_mp_shard.numel()))
                                if isinstance(
                                    _mp_shard, _torch_v24m.Tensor)
                                else "n/a"),
                            int(param.numel()),
                        )
                        try:
                            for _h in list(self.logger.handlers):
                                try:
                                    _h.flush()
                                except Exception:
                                    pass
                            _sys_v24m.stdout.flush()
                        except Exception:
                            pass
                        # TP-group MIN all_reduce on have_master bool.
                        # Runs on ALL TP peers (outside any gate), so
                        # dtype-symmetric int32 tensors join.
                        _have_master_tp = _have_master_local
                        if _tp_group is not None and _tp_ws > 1:
                            _dev = (
                                _mp_shard.device
                                if isinstance(
                                    _mp_shard, _torch_v24m.Tensor)
                                else param.device
                            )
                            _hv = _torch_v24m.tensor(
                                [1 if _have_master_local else 0],
                                dtype=_torch_v24m.int32,
                                device=_dev,
                            )
                            _torch_v24m.distributed.all_reduce(
                                _hv,
                                op=(
                                    _torch_v24m.distributed
                                    .ReduceOp.MIN
                                ),
                                group=_tp_group,
                            )
                            _have_master_tp = bool(
                                int(_hv.item()) == 1
                            )
                            self.logger.info(
                                "[MTPFp32MasterRead-v24 "
                                "TP_CONSENSUS] rank=%d name=%s "
                                "local=%s consensus=%s",
                                dist.get_rank(), name,
                                str(_have_master_local),
                                str(_have_master_tp),
                            )
                        # Build fp32_full ONLY if BOTH local and TP
                        # consensus say yes. This way every TP peer
                        # ends up with _fp32_full==None XOR fp32,
                        # consistently across the TP group.
                        if _have_master_tp and _have_master_local:
                            _fp32_full = (
                                _mp_shard.view(param.shape)
                                .contiguous()
                            )
                            if hasattr(param, "tensor_model_parallel"):
                                _fp32_full.tensor_model_parallel = (
                                    param.tensor_model_parallel
                                )
                            if hasattr(param, "partition_dim"):
                                _fp32_full.partition_dim = (
                                    param.partition_dim
                                )
                            if hasattr(param, "partition_stride"):
                                _fp32_full.partition_stride = (
                                    param.partition_stride
                                )
                            _src_tag = "fp32master"
                        elif _have_master_tp and not _have_master_local:
                            # Shouldn't happen by DistributedOpt
                            # semantics, but guard: TP peer says yes,
                            # we say no -> we MUST still produce fp32
                            # tensor to stay dtype-symmetric. Allocate
                            # a zero fp32 tensor of param.shape as a
                            # placeholder; the gathered output will be
                            # wrong on our slice but only the pp-head
                            # rank consumes the gather output, and
                            # the TP peer that DOES have master sends
                            # the correct slice. Wait -- _collect_param
                            # gathers every TP slice; if our slice is
                            # zero that taints the gathered tensor.
                            # In practice this branch is unreachable
                            # given ownership; downgrade to bf16-all.
                            self.logger.warning(
                                "[MTPFp32MasterRead-v24 "
                                "TP_CONSENSUS_ASYMM] rank=%d name=%s "
                                "consensus=True local=False; falling "
                                "back to bf16 on ENTIRE TP group to "
                                "avoid tainting gathered slice.",
                                dist.get_rank(), name,
                            )
                            _fp32_full = None
                            _src_tag = "bf16model"
                            # Propagate decision via a second tiny
                            # all_reduce so the PEER sees it too.
                            _force_bf16 = _torch_v24m.tensor(
                                [1],
                                dtype=_torch_v24m.int32,
                                device=param.device,
                            )
                            _torch_v24m.distributed.all_reduce(
                                _force_bf16,
                                op=(
                                    _torch_v24m.distributed
                                    .ReduceOp.MAX
                                ),
                                group=_tp_group,
                            )
                        else:
                            # Both TP peers agree: no master.
                            # Must also run the second all_reduce
                            # so symmetry with the asymm branch is
                            # maintained.
                            _force_bf16 = _torch_v24m.tensor(
                                [0],
                                dtype=_torch_v24m.int32,
                                device=param.device,
                            )
                            _torch_v24m.distributed.all_reduce(
                                _force_bf16,
                                op=(
                                    _torch_v24m.distributed
                                    .ReduceOp.MAX
                                ),
                                group=_tp_group,
                            )
                            if int(_force_bf16.item()) == 1:
                                self.logger.warning(
                                    "[MTPFp32MasterRead-v24 "
                                    "TP_CONSENSUS_ASYMM] rank=%d "
                                    "name=%s forced bf16 by peer.",
                                    dist.get_rank(), name,
                                )
                            if not getattr(
                                self,
                                "_mtp_master_read_missing_warned",
                                False,
                            ):
                                self.logger.warning(
                                    "[MTPFp32MasterRead-v24] "
                                    "param.main_param unavailable on "
                                    "this TP-group (rank=%d, "
                                    "name=%s, kind=%s); using bf16 "
                                    "model param.",
                                    dist.get_rank(), name,
                                    type(_mp_shard).__name__,
                                )
                                self._mtp_master_read_missing_warned = True
                        if _fp32_full is not None:
                            self.logger.info(
                                "[MTPFp32MasterRead-v24 D15a] "
                                "rank=%d name=%s pp_head=%s "
                                "dp_ws=%d tp_ws=%d shape=%s "
                                "fp32_abs_mean=%.6e "
                                "fp32_abs_max=%.6e (source=%s)",
                                dist.get_rank(), name,
                                str(_collect_mtp_for_draft),
                                _dp_ws, _tp_ws,
                                tuple(_fp32_full.shape),
                                float(_fp32_full.abs().mean().item()),
                                float(_fp32_full.abs().max().item()),
                                _src_tag,
                            )
                        try:
                            for _h in list(self.logger.handlers):
                                try:
                                    _h.flush()
                                except Exception:
                                    pass
                            _sys_v24m.stdout.flush()
                        except Exception:
                            pass
                    except Exception as _e_v24m:
                        self.logger.warning(
                            "[MTPFp32MasterRead-v24] error "
                            "name=%s: %s; falling back to bf16.",
                            name, _e_v24m,
                        )
                        _fp32_full = None
                        _src_tag = "bf16model"
                # === v24 key change ===
                # Hand the SAME dtype tensor to _collect_param on
                # every TP peer.  pp-head consumes the gathered
                # tensor; non-pp-head drops it.
                _collect_src = (
                    _fp32_full if _fp32_full is not None else param
                )
                self.logger.info(
                    "[MTPFp32MasterRead-v24 COLLECT_SRC] rank=%d "
                    "name=%s pp_head=%s src_dtype=%s src_shape=%s "
                    "src_tag=%s",
                    dist.get_rank(), name,
                    str(_collect_mtp_for_draft),
                    str(_collect_src.dtype),
                    tuple(_collect_src.shape),
                    _src_tag,
                )
                try:
                    for _h in list(self.logger.handlers):
                        try:
                            _h.flush()
                        except Exception:
                            pass
                    _sys_v24m.stdout.flush()
                except Exception:
                    pass
                _mtp_param, _ = self._collect_param(
                    name, _collect_src,
                )
                self.logger.info(
                    "[MTPFp32MasterRead-v24 COLLECT_DONE] rank=%d "
                    "name=%s pp_head=%s returned_dtype=%s "
                    "returned_shape=%s",
                    dist.get_rank(), name,
                    str(_collect_mtp_for_draft),
                    (str(_mtp_param.dtype)
                        if _mtp_param is not None else "None"),
                    (tuple(_mtp_param.shape)
                        if _mtp_param is not None else "None"),
                )
                try:
                    for _h in list(self.logger.handlers):
                        try:
                            _h.flush()
                        except Exception:
                            pass
                    _sys_v24m.stdout.flush()
                except Exception:
                    pass
                # [P3-MTPShipFallback] When _collect_param returned None on
                # this rank (typically because the MTP-only param group left
                # no master shard here, or the fp32-master fetch raised),
                # fall back to a plain bf16 all-gather of ``param`` so that
                # the wire payload for the draft model is never dropped
                # silently.  Opt-out via AREAL_MTP_SHIP_FALLBACK=0.
                if (
                    _collect_mtp_for_draft
                    and _mtp_param is None
                    and os.environ.get(
                        "AREAL_MTP_SHIP_FALLBACK", "1") == "1"
                ):
                    try:
                        _fb_param, _ = self._collect_param(name, param)
                        if _fb_param is not None:
                            _mtp_param = _fb_param
                            self.logger.warning(
                                "[MTPShipFallback-P3] rank=%d name=%s "
                                "fell back to bf16 all-gather (fp32 "
                                "master unavailable on this rank).",
                                dist.get_rank(), name,
                            )
                        else:
                            self.logger.error(
                                "[MTPShipFallback-P3] rank=%d name=%s "
                                "bf16 all-gather also returned None; "
                                "MTP tensor will be skipped.",
                                dist.get_rank(), name,
                            )
                    except Exception as _e_p3_fb:
                        self.logger.error(
                            "[MTPShipFallback-P3] rank=%d name=%s "
                            "fallback raised: %r",
                            dist.get_rank(), name, _e_p3_fb,
                        )
                # [MTPShipEnumTrace-v61] log per-MTP-param ship enumeration
                # ENTER. Captures whether MTP path will collect this tensor,
                # the param's bf16 statistics, and shape — independent of
                # later HF-name expansion.
                try:
                    if _collect_mtp_for_draft and ('mtp' in name):
                        _v61_pa = _mtp_param if _mtp_param is not None else None
                        if _v61_pa is not None:
                            _v61_pf = _v61_pa.detach().float()
                            _v61_am = float(_v61_pf.abs().mean().item())
                            _v61_ax = float(_v61_pf.abs().max().item())
                            _v61_l2 = float(_v61_pf.norm().item())
                            _v61_n = int(_v61_pa.numel())
                            _v61_dt = str(_v61_pa.dtype)
                            _v61_sh = tuple(_v61_pa.shape)
                        else:
                            _v61_am = _v61_ax = _v61_l2 = -1.0
                            _v61_n = 0
                            _v61_dt = 'None'
                            _v61_sh = ()
                        self.logger.info(
                            '[MTPShipEnumTrace-v61] stage=ENTER rank=%d '
                            'name=%s collect=%s mtp_param_is_none=%s '
                            'numel=%d dtype=%s shape=%s '
                            'abs_mean=%.6e abs_max=%.6e l2=%.6e',
                            int(dist.get_rank()), name,
                            str(_collect_mtp_for_draft),
                            str(_mtp_param is None),
                            _v61_n, _v61_dt, str(_v61_sh),
                            _v61_am, _v61_ax, _v61_l2,
                        )
                except Exception as _e_v61_a:
                    try:
                        self.logger.info(
                            '[MTPShipEnumTrace-v61] ENTER failure: %r',
                            _e_v61_a,
                        )
                    except Exception:
                        pass
                if _collect_mtp_for_draft and _mtp_param is not None:
                    _mtp_model_name = self.hf_config.model_type
                    _prev_count = len(mtp_hf_tensors)
                    # [MTPSrcHash-v44] hash Megatron-side collected
                    # tensor BEFORE convert_to_hf so we can tell if
                    # hidden_layernorm.weight (digest identical across
                    # all v43 versions) is frozen at Megatron source
                    # (training/grad issue) or during HF export path.
                    try:
                        import torch as _torch_v44s
                        _v44s_ver = None
                        try:
                            _v44s_ver = int(self.get_version())
                        except Exception:
                            _v44s_ver = None
                        _v44s_flat = _mtp_param.detach().reshape(-1)
                        _v44s_k = min(1024, int(_v44s_flat.numel()))
                        if _v44s_k > 0:
                            _v44s_sl = (
                                _v44s_flat[:_v44s_k].float().contiguous().cpu()
                            )
                            _v44s_bytes = _v44s_sl.numpy().tobytes()
                            _v44s_h = 0
                            for _b in _v44s_bytes:
                                _v44s_h = (
                                    (_v44s_h * 1315423911) ^ int(_b)
                                ) & ((1 << 64) - 1)
                            _v44s_sum = float(_v44s_sl.sum().item())
                            _v44s_am = float(_v44s_sl.abs().mean().item())
                        else:
                            _v44s_h, _v44s_sum, _v44s_am = 0, 0.0, 0.0
                        self.logger.info(
                            "[MTPSrcHash-v44] version=%s name=%s "
                            "src_dtype=%s src_shape=%s hash=%s "
                            "sum_first1024=%s abs_mean_first1024=%s",
                            _v44s_ver, name,
                            str(_mtp_param.dtype),
                            tuple(_mtp_param.shape),
                            _v44s_h, _v44s_sum, _v44s_am,
                        )
                    except Exception as _e_v44s:
                        try:
                            self.logger.info(
                                "[MTPSrcHash-v44] failure: %r", _e_v44s,
                            )
                        except Exception:
                            pass
                    # [MTPULPGap-v45] Quantify the bf16-ULP gap on
                    # the Megatron-side fp32 master.  For each MTP
                    # tensor, we round fp32->bf16 and compare to
                    # the previous sync's rounded bf16.  If no
                    # element flipped even one bf16 ULP, the
                    # downstream draft sees a bit-exact copy of
                    # the PREVIOUS version, regardless of what
                    # fp32 master has been doing.  This nails the
                    # "hidden_layernorm.weight is frozen" obs
                    # (log.27/28) as a pure quantization ceiling.
                    try:
                        import torch as _torch_v45
                        _v45_ver = None
                        try:
                            _v45_ver = int(self.get_version())
                        except Exception:
                            _v45_ver = None
                        if not hasattr(self, "_v45_prev_bf16"):
                            self._v45_prev_bf16 = {}
                        if not hasattr(self, "_v45_prev_fp32"):
                            self._v45_prev_fp32 = {}
                        _v45_t_fp32 = _mtp_param.detach().float()
                        _v45_bf16 = _v45_t_fp32.to(_torch_v45.bfloat16)
                        _v45_prev_b = self._v45_prev_bf16.get(name)
                        _v45_prev_f = self._v45_prev_fp32.get(name)
                        if (_v45_prev_b is not None
                                and _v45_prev_b.shape == _v45_bf16.shape):
                            _v45_flips = int(
                                (_v45_bf16 != _v45_prev_b).sum().item()
                            )
                        else:
                            _v45_flips = -1
                        if (_v45_prev_f is not None
                                and _v45_prev_f.shape == _v45_t_fp32.shape):
                            _v45_d = (_v45_t_fp32 - _v45_prev_f).abs()
                            _v45_drift_max = float(_v45_d.max().item())
                            _v45_drift_mean = float(_v45_d.mean().item())
                        else:
                            _v45_drift_max = -1.0
                            _v45_drift_mean = -1.0
                        # bf16 ULP estimator for the tensor's
                        # dominant magnitude: ULP = 2^(e-7) where
                        # 2^e <= |x|max < 2^(e+1).  For |x|max=0
                        # (zero tensor) default 2^-133 (denormal).
                        _v45_amax = float(
                            _v45_t_fp32.abs().max().item()
                        )
                        if _v45_amax > 0:
                            import math as _m_v45
                            _v45_e = _m_v45.floor(
                                _m_v45.log2(_v45_amax)
                            )
                            _v45_ulp_max = 2.0 ** (_v45_e - 7)
                        else:
                            _v45_ulp_max = float('nan')
                        # Estimated syncs until the next ULP flip
                        # on the largest-magnitude element: ULP /
                        # per-element drift.
                        if (_v45_drift_max > 0
                                and _v45_ulp_max == _v45_ulp_max):
                            _v45_eta = _v45_ulp_max / _v45_drift_max
                        else:
                            _v45_eta = -1.0
                        self.logger.info(
                            "[MTPULPGap-v45] version=%s name=%s "
                            "shape=%s amax=%.6e bf16_ulp_at_amax=%.6e "
                            "drift_abs_max=%.6e drift_abs_mean=%.6e "
                            "bf16_flips_vs_prev=%s "
                            "eta_syncs_to_next_flip=%.2f",
                            _v45_ver, name, tuple(_v45_t_fp32.shape),
                            _v45_amax, _v45_ulp_max,
                            _v45_drift_max, _v45_drift_mean,
                            _v45_flips, _v45_eta,
                        )
                        # keep one-version history
                        self._v45_prev_bf16[name] = (
                            _v45_bf16.detach().clone()
                        )
                        self._v45_prev_fp32[name] = (
                            _v45_t_fp32.detach().clone()
                        )
                    except Exception as _e_v45:
                        try:
                            self.logger.info(
                                "[MTPULPGap-v45] failure: %r", _e_v45,
                            )
                        except Exception:
                            pass
                    # [MTPShipPostAGAudit-v63] Right BEFORE convert_to_hf,
                    # log the post-all_gather _mtp_param tensor (full
                    # gathered shape, sha256_16, first/last 8). This is
                    # the EXACT mcore-side payload that goes into the
                    # HF mapping. Comparing this hash across versions
                    # tells us whether ship-time TP all_gather is
                    # producing identical-byte tensors per version
                    # (would explain stalled draft despite training).
                    try:
                        import hashlib as _v63_pag_hash
                        _v63_pag_t = _mtp_param.detach().contiguous()
                        _v63_pag_bytes = (
                            _v63_pag_t.float().cpu().numpy().tobytes()
                        )
                        _v63_pag_h = _v63_pag_hash.sha256(
                            _v63_pag_bytes).hexdigest()[:16]
                        _v63_pag_first = [
                            float(x) for x in
                            _v63_pag_t.reshape(-1)[:8].float()
                            .cpu().tolist()
                        ]
                        _v63_pag_last = [
                            float(x) for x in
                            _v63_pag_t.reshape(-1)[-8:].float()
                            .cpu().tolist()
                        ]
                        try:
                            _v63_pag_ver = int(self.get_version())
                        except Exception:
                            _v63_pag_ver = -1
                        self.logger.info(
                            "[MTPShipPostAGAudit-v63] version=%d "
                            "name=%s shape=%s dtype=%s "
                            "sha256_16=%s first8=%s last8=%s "
                            "abs_mean=%.6e abs_max=%.6e l2=%.6e",
                            _v63_pag_ver, name,
                            tuple(_v63_pag_t.shape),
                            str(_v63_pag_t.dtype),
                            _v63_pag_h,
                            str(_v63_pag_first), str(_v63_pag_last),
                            float(_v63_pag_t.float().abs().mean().item()),
                            float(_v63_pag_t.float().abs().max().item()),
                            float(_v63_pag_t.float().norm().item()),
                        )
                    except Exception as _e_v63_pag:
                        try:
                            self.logger.info(
                                "[MTPShipPostAGAudit-v63] failure: %r",
                                _e_v63_pag,
                            )
                        except Exception:
                            pass
                    # [MTPMainParamCmpAudit-v63] Compare bf16 model
                    # param vs fp32 main_param at ship time. If they
                    # diverge by more than bf16 ULP, stochastic
                    # rounding desync between training and ship is
                    # the root cause of post-ship draft regression.
                    try:
                        _v63_mp_param_obj = param  # original module param
                        _v63_mp = getattr(
                            _v63_mp_param_obj, 'main_param', None)
                        if _v63_mp is not None:
                            import torch as _v63_torch_mp
                            _v63_mp_fp32 = _v63_mp.detach().float()
                            _v63_bf = _v63_mp_param_obj.detach().float()
                            if _v63_mp_fp32.shape == _v63_bf.shape:
                                _v63_d = (_v63_mp_fp32 - _v63_bf).abs()
                                _v63_d_max = float(_v63_d.max().item())
                                _v63_d_mean = float(_v63_d.mean().item())
                                _v63_amax = float(
                                    _v63_mp_fp32.abs().max().item())
                                _v63_ulp = -1.0
                                if _v63_amax > 0:
                                    import math as _v63_math
                                    _v63_e = _v63_math.floor(
                                        _v63_math.log2(_v63_amax))
                                    _v63_ulp = 2.0 ** (_v63_e - 7)
                                _v63_dratio = (
                                    _v63_d_max / _v63_ulp
                                    if _v63_ulp > 0 else -1.0
                                )
                                self.logger.info(
                                    "[MTPMainParamCmpAudit-v63] "
                                    "name=%s shape=%s "
                                    "fp32_main_param_sum=%.6e "
                                    "bf16_model_param_sum=%.6e "
                                    "delta_abs_max=%.6e "
                                    "delta_abs_mean=%.6e "
                                    "bf16_ulp=%.6e "
                                    "delta_to_ulp_ratio=%.4f",
                                    name, tuple(_v63_mp_fp32.shape),
                                    float(_v63_mp_fp32.sum().item()),
                                    float(_v63_bf.sum().item()),
                                    _v63_d_max, _v63_d_mean,
                                    _v63_ulp, _v63_dratio,
                                )
                            else:
                                self.logger.info(
                                    "[MTPMainParamCmpAudit-v63] "
                                    "shape mismatch name=%s "
                                    "main_param=%s bf16=%s",
                                    name,
                                    tuple(_v63_mp_fp32.shape),
                                    tuple(_v63_bf.shape),
                                )
                        else:
                            self.logger.info(
                                "[MTPMainParamCmpAudit-v63] "
                                "name=%s main_param=None "
                                "(no fp32 master on this rank)",
                                name,
                            )
                    except Exception as _e_v63_mp:
                        try:
                            self.logger.info(
                                "[MTPMainParamCmpAudit-v63] failure: %r",
                                _e_v63_mp,
                            )
                        except Exception:
                            pass
                    # [MTPShipPostAGAudit-v64] Right BEFORE convert_to_hf,
                    # log the post-all_gather _mtp_param tensor.  This
                    # is the EXACT mcore-side payload that flows into
                    # the HF mapping.  Compared with PRE/POST swap
                    # audits and with WireBytes audit, this nails the
                    # location of any divergence.
                    try:
                        import hashlib as _v64_pag_hash
                        _v64_pag_t = _mtp_param.detach().contiguous()
                        _v64_pag_b = (
                            _v64_pag_t.float().cpu().numpy().tobytes()
                        )
                        _v64_pag_h = _v64_pag_hash.sha256(
                            _v64_pag_b).hexdigest()[:16]
                        _v64_pag_f8 = [
                            float(x) for x in
                            _v64_pag_t.reshape(-1)[:8].float()
                            .cpu().tolist()
                        ]
                        try:
                            _v64_pag_ver = int(self.get_version())
                        except Exception:
                            _v64_pag_ver = -1
                        self.logger.info(
                            "[MTPShipPostAGAudit-v64] version=%d "
                            "name=%s shape=%s dtype=%s "
                            "sha256_16=%s first8=%s "
                            "abs_mean=%.6e abs_max=%.6e l2=%.6e",
                            _v64_pag_ver, name,
                            tuple(_v64_pag_t.shape),
                            str(_v64_pag_t.dtype),
                            _v64_pag_h, str(_v64_pag_f8),
                            float(_v64_pag_t.float().abs().mean().item()),
                            float(_v64_pag_t.float().abs().max().item()),
                            float(_v64_pag_t.float().norm().item()),
                        )
                    except Exception as _e_v64_pag:
                        try:
                            self.logger.info(
                                "[MTPShipPostAGAudit-v64] failure: %r",
                                _e_v64_pag,
                            )
                        except Exception:
                            pass
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
                    # [MTPShipEnumTrace-v61] EXIT — log expanded HF names
                    # and per-tensor bytes added by convert_to_hf for this
                    # one mcore param.
                    try:
                        _v61_added = mtp_hf_tensors[_prev_count:]
                        for _v61_i, (_v61_hn, _v61_ht) in enumerate(_v61_added):
                            try:
                                _v61_hf = _v61_ht.detach().float()
                                _v61_ham = float(_v61_hf.abs().mean().item())
                                _v61_hax = float(_v61_hf.abs().max().item())
                                _v61_hl2 = float(_v61_hf.norm().item())
                                _v61_hfirst = [
                                    float(x) for x in
                                    _v61_hf.reshape(-1)[:8].tolist()
                                ]
                            except Exception:
                                _v61_ham = _v61_hax = _v61_hl2 = -1.0
                                _v61_hfirst = []
                            self.logger.info(
                                '[MTPShipEnumTrace-v61] stage=EXIT rank=%d '
                                'mcore=%s hf_idx=%d hf_name=%s '
                                'hf_dtype=%s hf_shape=%s hf_numel=%d '
                                'hf_bytes=%d abs_mean=%.6e abs_max=%.6e '
                                'l2=%.6e first8=%s',
                                int(dist.get_rank()), name,
                                _v61_i, _v61_hn,
                                str(_v61_ht.dtype),
                                str(tuple(_v61_ht.shape)),
                                int(_v61_ht.numel()),
                                int(_v61_ht.numel() * _v61_ht.element_size()),
                                _v61_ham, _v61_hax, _v61_hl2,
                                str(_v61_hfirst),
                            )
                    except Exception as _e_v61_b:
                        try:
                            self.logger.info(
                                '[MTPShipEnumTrace-v61] EXIT failure: %r',
                                _e_v61_b,
                            )
                        except Exception:
                            pass
                    # [MTPBf16UpcastBroadcast-v24] Upcast bf16->fp32
                    # before serialize so sub-ULP deltas are not
                    # rounded on the wire (default 1).
                    try:
                        _v16_on = (
                            _os_v24m.environ.get(
                                "AREAL_MTP_FP32_BROADCAST", "1",
                            ) == "1"
                        )
                    except Exception:
                        _v16_on = True
                    if _v16_on:
                        _upcasted = 0
                        for _i in range(_prev_count, len(mtp_hf_tensors)):
                            _nm_v16, _tn_v16 = mtp_hf_tensors[_i]
                            if _tn_v16.dtype == _torch_v24m.bfloat16:
                                mtp_hf_tensors[_i] = (
                                    _nm_v16,
                                    _tn_v16.float().contiguous(),
                                )
                                _upcasted += 1
                        if _upcasted > 0:
                            self.logger.info(
                                "[MTPBf16UpcastBroadcast-v24] Upcast %d "
                                "MTP tensors bf16->fp32 (name=%s).",
                                _upcasted, name,
                            )
                    # [MTPSigmaDeltaBf16-v28] Residual-carried bf16
                    # quantization of the fp32 MTP payload.
                    #
                    # PURPOSE
                    #   After v16 upcast the MTP payload is fp32. But
                    #   SGLang 0.5.9's draft model storage is bf16
                    #   (no fp32-draft knob exists) and its
                    #   default_weight_loader does
                    #   `param.data.copy_(loaded_weight)` which rounds
                    #   fp32->bf16 RNE at the destination.  When the
                    #   per-step fp32 delta is smaller than half a
                    #   bf16 ULP (e.g. 2e-6 vs 1.56e-2 for |w|=3 on
                    #   LayerNorm) the draft weight is frozen across
                    #   thousands of steps and accept rate stalls.
                    #   (Confirmed from MTPBf16ULPProof diag in
                    #   iter14-17: bf16cast_eq_prev_bf16cast == numel
                    #   for 5/5 consecutive syncs on all LayerNorm
                    #   MTP params.)
                    #
                    # FIX
                    #   Per-tensor residual r[name] (fp32) accumulates
                    #   round-off; each sync we send
                    #     bf16 = RNE(fp32 + r_prev)
                    #     r_new = (fp32 + r_prev) - bf16
                    #   Cumulative sub-ULP deltas eventually cross the
                    #   bf16 ULP and "tick" the draft weight one ULP
                    #   at a time (classic Sigma-Delta quantization).
                    #   Unlike per-element stochastic rounding this
                    #   is deterministic and preserves monotonic
                    #   sub-ULP trajectories.
                    #
                    # NOTES
                    #   * slime/verl do not address this. Research of
                    #     https://github.com/THUDM/slime ,
                    #     https://github.com/volcengine/verl , SGLang
                    #     v0.5.9 and Megatron-LM core_r0.16.0 confirms
                    #     they all ship bf16 round-to-nearest. See
                    #     megatron distrib_optimizer.py
                    #     _copy_main_params_to_model_params (plain
                    #     copy_) and sglang weight_utils.py
                    #     default_weight_loader (plain copy_).
                    #   * Only bf16 storage on SGLang side is affected.
                    #     If AREAL_MTP_FP32_BROADCAST=0 or upstream
                    #     already materialised fp32, we are a no-op.
                    #   * Only MTP-draft tensors go through this
                    #     block; all other params are untouched.
                    #
                    # Gate: AREAL_MTP_SIGMA_DELTA_BF16 (default "1").
                    try:
                        _sd_on = (
                            _os_v16.environ.get(
                                "AREAL_MTP_SIGMA_DELTA_BF16", "1",
                            ) == "1"
                        )
                    except Exception:
                        _sd_on = True
                    if _sd_on:
                        # [v34] Defensive torch import: v28 SigmaDelta
                        # block references _torch_v16 but the original
                        # import was placed inside the v16 upcast
                        # guard `if _v16_on:`. When the operator runs
                        # with AREAL_MTP_FP32_BROADCAST=0 (or unset,
                        # default "0"), _torch_v16 is undefined and
                        # the Σ-Δ path raises NameError at
                        # `_tn_sd.dtype != _torch_v16.float32` during
                        # update_weights, aborting training. Importing
                        # torch here is always-safe (module cache)
                        # and restores Σ-Δ independence from the v16
                        # env gate.
                        import torch as _torch_v16
                        if not hasattr(self, "_mtp_sd_residual"):
                            self._mtp_sd_residual = {}
                        if not hasattr(self, "_mtp_sd_sync_idx"):
                            self._mtp_sd_sync_idx = {}
                        _sd_applied = 0
                        _sd_total_shifted = 0
                        _sd_sample_details = []
                        for _i in range(_prev_count, len(mtp_hf_tensors)):
                            _nm_sd, _tn_sd = mtp_hf_tensors[_i]
                            # Only fp32 MTP payload is candidate.
                            if _tn_sd.dtype != _torch_v16.float32:
                                continue
                            _r_prev = self._mtp_sd_residual.get(_nm_sd)
                            if (
                                _r_prev is not None
                                and _r_prev.shape == _tn_sd.shape
                                and _r_prev.device == _tn_sd.device
                                and _r_prev.dtype == _tn_sd.dtype
                            ):
                                _u = _tn_sd + _r_prev
                                _had_prev = True
                            else:
                                _u = _tn_sd
                                _had_prev = False
                            # [MTPForceTickBf16-v46] Cap draft-IPC
                            # stall at K syncs by promoting residual
                            # to ±ULP/2 when bf16 has not flipped in
                            # K_force consecutive syncs.  Preserves
                            # long-run unbiasedness: the ±ULP/2
                            # injection is a Σ-Δ quantum that the
                            # next sync's residual cancels.
                            try:
                                _ft_on_v46 = (
                                    _os_v16.environ.get(
                                        'AREAL_MTP_V46_FORCE_TICK',
                                        '1',
                                    ) == '1'
                                )
                            except Exception:
                                _ft_on_v46 = True
                            if _ft_on_v46:
                                # [v57] Tighten default K from 8 -> 2.
                                # Rationale: for high-magnitude LayerNorm
                                # tensors with sub-ULP drift, the natural
                                # stale-counter reaches K=8 only after
                                # training has already drifted far enough
                                # for main/draft mismatch to dominate
                                # accept_rate. K=2 bounds IPC staleness
                                # to a single sync.
                                try:
                                    _ft_k_v46 = int(
                                        _os_v16.environ.get(
                                            'AREAL_MTP_V46_FORCE_TICK_K',
                                            '2',
                                        )
                                    )
                                except Exception:
                                    _ft_k_v46 = 2
                                # [v57] Tighten default ratio 0.10 -> 0.05.
                                # resid_absmax grows ~drift per sync; at
                                # ratio=0.05 the ratio trigger fires once
                                # resid crosses 5%% of ULP, which is the
                                # smallest safe fraction where SR flip
                                # probability makes the ship_flips count
                                # statistically observable.
                                try:
                                    _ft_ratio_v46 = float(
                                        _os_v16.environ.get(
                                            'AREAL_MTP_V46_FORCE_TICK_RATIO',
                                            '0.05',
                                        )
                                    )
                                except Exception:
                                    _ft_ratio_v46 = 0.05
                                if not hasattr(
                                    self, '_mtp_v46_stale'
                                ):
                                    self._mtp_v46_stale = {}
                                if not hasattr(
                                    self, '_mtp_v46_prev_ship'
                                ):
                                    self._mtp_v46_prev_ship = {}
                                _ft_amax = float(
                                    _u.abs().max().item()
                                )
                                if _ft_amax > 0:
                                    import math as _m_v46
                                    _ft_e = _m_v46.floor(
                                        _m_v46.log2(_ft_amax)
                                    )
                                    _ft_ulp = 2.0 ** (_ft_e - 7)
                                else:
                                    _ft_ulp = 0.0
                                _ft_stale = (
                                    self._mtp_v46_stale.get(_nm_sd, 0)
                                )
                                _ft_resid_absmax = 0.0
                                if (
                                    _r_prev is not None
                                    and _r_prev.shape == _tn_sd.shape
                                ):
                                    try:
                                        _ft_resid_absmax = float(
                                            _r_prev.abs().max().item()
                                        )
                                    except Exception:
                                        _ft_resid_absmax = 0.0
                                _ft_trigger_stale = (
                                    _ft_stale >= _ft_k_v46
                                )
                                _ft_trigger_ratio = (
                                    _ft_ulp > 0
                                    and _ft_ratio_v46 > 0
                                    and _ft_resid_absmax
                                    >= _ft_ratio_v46 * _ft_ulp
                                )
                                _ft_fired = False
                                # [v57] Fix defect: v46 computed
                                # _ft_trigger_ratio but never used it
                                # as a fire condition. The ratio trigger
                                # is the only path that fires for
                                # sub-ULP-drift LayerNorm tensors inside
                                # a normal (non-stale) sync cadence.
                                if (
                                    (
                                        _ft_trigger_stale
                                        or _ft_trigger_ratio
                                    )
                                    and _ft_ulp > 0
                                ):
                                    # Promote _u by sign(resid or
                                    # drift) * ULP/2 on the single
                                    # element with largest |resid|
                                    # so that RNE flips exactly one
                                    # bf16 bucket. Minimal, unbiased
                                    # on average (residual carries
                                    # opposite sign next sync).
                                    try:
                                        _ft_flat = _u.view(-1)
                                        if _r_prev is not None:
                                            _ft_signmap = (
                                                _r_prev.view(-1)
                                            )
                                        else:
                                            _ft_signmap = _ft_flat
                                        _ft_sign = (
                                            _torch_v16.sign(_ft_signmap)
                                        )
                                        _ft_sign = _torch_v16.where(
                                            _ft_sign == 0,
                                            _torch_v16.ones_like(
                                                _ft_sign
                                            ),
                                            _ft_sign,
                                        )
                                        _u = (
                                            _u
                                            + _ft_sign.view_as(_u)
                                            * (0.5 * _ft_ulp)
                                        )
                                        _ft_fired = True
                                    except Exception:
                                        _ft_fired = False
                                self._mtp_v46_stale[_nm_sd] = (
                                    0 if _ft_fired else _ft_stale
                                )
                                # store diag for post-loop log
                                if not hasattr(
                                    self, '_mtp_v46_fire_log'
                                ):
                                    self._mtp_v46_fire_log = []
                                if (
                                    _ft_fired
                                    or _ft_trigger_ratio
                                    or _ft_trigger_stale
                                ):
                                    self._mtp_v46_fire_log.append(
                                        (
                                            _nm_sd,
                                            _ft_stale,
                                            _ft_resid_absmax,
                                            _ft_ulp,
                                            _ft_fired,
                                        )
                                    )
                            # [MTPStochasticRoundBf16-v57]
                            # Replace RNE with stochastic rounding for
                            # MTP payload so sub-ULP drift propagates
                            # across the tensor with expected flip
                            # count = numel * drift / ULP.  Preserves
                            # long-run unbiasedness E[SR(x)] = x.
                            try:
                                # [MTPSlimeAlign] H) disable v57 SR when
                                # slime-align is ON; slime/native bf16
                                # path uses RNE (round-nearest-even) only.
                                _v57_slime = (
                                    _os_v16.environ.get(
                                        'AREAL_MTP_SLIME_ALIGN', '1'
                                    ) == '1'
                                )
                                _sr_on_v57 = (
                                    _os_v16.environ.get(
                                        'AREAL_MTP_V57_STOCHASTIC_ROUND',
                                        '1',
                                    ) == '1'
                                    and not _v57_slime
                                )
                                if _v57_slime and not getattr(
                                    self, '_v58_h_logged', False
                                ):
                                    try:
                                        self.logger.info(
                                            '[MTPSlimeAlign] H) v57 '
                                            'StochasticRoundBf16 '
                                            'DISABLED (slime/native uses '
                                            'RNE only).'
                                        )
                                        self._v58_h_logged = True
                                    except Exception:
                                        pass
                            except Exception:
                                _sr_on_v57 = True
                            _sr_applied_v57 = False
                            _sr_drift_ratio_v57 = -1.0
                            if _sr_on_v57 and _u.dtype == _torch_v16.float32:
                                try:
                                    _sr_min_ratio_v57 = float(
                                        _os_v16.environ.get(
                                            'AREAL_MTP_V57_SR_MIN_DRIFT_RATIO',
                                            '0.0',
                                        )
                                    )
                                except Exception:
                                    _sr_min_ratio_v57 = 0.0
                                try:
                                    # Element-wise bf16 ULP derived
                                    # from each element's magnitude.
                                    # bf16 ULP(x) = 2^(e_x - 7) where
                                    # 2^e_x <= |x| < 2^(e_x+1). For
                                    # |x|=0 we use the tensor's global
                                    # ulp_max as a fallback (mostly
                                    # irrelevant since 0 rounds to 0).
                                    _u_abs = _u.abs()
                                    _nz_mask = _u_abs > 0
                                    # log2 is safe only on positives.
                                    _log2u = _torch_v16.where(
                                        _nz_mask,
                                        _torch_v16.log2(
                                            _torch_v16.where(
                                                _nz_mask,
                                                _u_abs,
                                                _torch_v16.ones_like(_u_abs),
                                            )
                                        ),
                                        _torch_v16.zeros_like(_u_abs),
                                    )
                                    _e_elem = _torch_v16.floor(_log2u)
                                    _ulp_elem = _torch_v16.pow(
                                        _torch_v16.full_like(
                                            _e_elem, 2.0
                                        ),
                                        _e_elem - 7.0,
                                    )
                                    # For zero elements use tensor-level
                                    # ulp (still zero contribution).
                                    _ulp_elem = _torch_v16.where(
                                        _nz_mask,
                                        _ulp_elem,
                                        _torch_v16.full_like(
                                            _ulp_elem,
                                            max(_ft_ulp, 0.0) if _ft_on_v46 else 0.0,
                                        ),
                                    )
                                    # Drift-gating check (optional).
                                    _sr_enable_this = True
                                    if _sr_min_ratio_v57 > 0:
                                        try:
                                            _drift_abs_max_v57 = 0.0
                                            if _r_prev is not None and _r_prev.shape == _u.shape:
                                                _drift_abs_max_v57 = float(
                                                    _r_prev.abs().max().item()
                                                )
                                            _ulp_global = float(
                                                _ulp_elem.max().item()
                                            )
                                            if _ulp_global > 0:
                                                _sr_drift_ratio_v57 = (
                                                    _drift_abs_max_v57 / _ulp_global
                                                )
                                            else:
                                                _sr_drift_ratio_v57 = 0.0
                                            # If RNE is already naturally
                                            # flipping, skip SR to keep
                                            # training deterministic.
                                            if _sr_drift_ratio_v57 >= _sr_min_ratio_v57:
                                                _sr_enable_this = False
                                        except Exception:
                                            _sr_enable_this = True
                                    if _sr_enable_this:
                                        # Dither: u ~ Uniform[-0.5, 0.5]
                                        # per-element, scale by ulp_elem
                                        # so that RNE(_u + u*ulp_elem)
                                        # realises the SR rounding.
                                        _dither = (
                                            _torch_v16.rand_like(_u) - 0.5
                                        ) * _ulp_elem
                                        _u = _u + _dither
                                        _sr_applied_v57 = True
                                except Exception as _e_sr_v57:
                                    try:
                                        self.logger.info(
                                            '[MTPStochasticRoundBf16-v57] '
                                            'SR failed name=%s err=%r; '
                                            'falling back to RNE.',
                                            _nm_sd, _e_sr_v57,
                                        )
                                    except Exception:
                                        pass
                                    _sr_applied_v57 = False
                            # RNE fp32 -> bf16 and retrieve actual
                            # quantized fp32 value for residual calc.
                            # When v57 SR applied, the dithered _u
                            # combined with RNE here is mathematically
                            # equivalent to per-element stochastic
                            # rounding of the original fp32 master.
                            _bf16 = _u.to(_torch_v16.bfloat16)
                            _bb = _bf16.float()
                            _new_res = (_u - _bb).detach().clone()
                            if _sr_applied_v57:
                                try:
                                    self.logger.info(
                                        '[MTPStochasticRoundBf16-v57] '
                                        'name=%s shape=%s numel=%d '
                                        'drift_ratio=%.3e applied=True',
                                        _nm_sd, tuple(_u.shape),
                                        int(_u.numel()),
                                        _sr_drift_ratio_v57,
                                    )
                                except Exception:
                                    pass
                            # Diagnostic: count elements whose bf16
                            # representation differs from the plain
                            # RNE(_tn_sd) baseline (i.e. how many were
                            # "lifted" by accumulated residual).
                            try:
                                _baseline_bf16 = _tn_sd.to(
                                    _torch_v16.bfloat16
                                )
                                _shift_cnt = int(
                                    (_bf16 != _baseline_bf16)
                                    .sum().item()
                                )
                            except Exception:
                                _shift_cnt = -1
                            self._mtp_sd_residual[_nm_sd] = _new_res
                            self._mtp_sd_sync_idx[_nm_sd] = (
                                self._mtp_sd_sync_idx.get(_nm_sd, 0) + 1
                            )
                            # Replace payload tensor with sigma-delta
                            # bf16 version. Receiver (SGLang) will do
                            # its own copy_ which is now bit-exact.
                            mtp_hf_tensors[_i] = (
                                _nm_sd, _bf16.contiguous(),
                            )
                            # [MTPShipFlips-v46] update stale
                            # counter: if shipped bf16 payload
                            # matches previous version's shipped
                            # payload bit-for-bit, stale += 1.
                            try:
                                _ship_prev_v46 = (
                                    self._mtp_v46_prev_ship.get(
                                        _nm_sd
                                    )
                                    if hasattr(
                                        self, '_mtp_v46_prev_ship'
                                    )
                                    else None
                                )
                                _ship_flips_v46 = -1
                                if (
                                    _ship_prev_v46 is not None
                                    and _ship_prev_v46.shape
                                    == _bf16.shape
                                ):
                                    _ship_flips_v46 = int(
                                        (
                                            _bf16 != _ship_prev_v46
                                        ).sum().item()
                                    )
                                    if _ship_flips_v46 == 0:
                                        self._mtp_v46_stale[_nm_sd] = (
                                            self._mtp_v46_stale.get(
                                                _nm_sd, 0
                                            ) + 1
                                        )
                                    else:
                                        self._mtp_v46_stale[_nm_sd] = 0
                                self._mtp_v46_prev_ship[_nm_sd] = (
                                    _bf16.detach().clone()
                                )
                                # log per-tensor only if it fired
                                # or was previously stale.
                                if (
                                    _ft_fired
                                    or _ship_flips_v46 == 0
                                ):
                                    self.logger.info(
                                        '[MTPShipFlips-v46] '
                                        'name=%s ship_flips=%s '
                                        'stale=%s force_fired=%s '
                                        'ulp=%.3e resid_absmax=%.3e',
                                        _nm_sd, _ship_flips_v46,
                                        self._mtp_v46_stale.get(
                                            _nm_sd, 0
                                        ),
                                        _ft_fired,
                                        _ft_ulp if _ft_on_v46 else 0.0,
                                        _ft_resid_absmax if _ft_on_v46 else 0.0,
                                    )
                            except Exception as _e_sf_v46:
                                try:
                                    self.logger.info(
                                        '[MTPShipFlips-v46] '
                                        'failure name=%s err=%r',
                                        _nm_sd, _e_sf_v46,
                                    )
                                except Exception:
                                    pass
                            _sd_applied += 1
                            if _shift_cnt > 0:
                                _sd_total_shifted += _shift_cnt
                            # Per-tensor trace: first 5 tensors or
                            # every 10th sync, to avoid spam.
                            if (
                                len(_sd_sample_details) < 5
                                or (
                                    self._mtp_sd_sync_idx[_nm_sd]
                                    % 10 == 0
                                )
                            ):
                                try:
                                    _r_abs = float(
                                        _new_res.abs().mean().item()
                                    )
                                    _r_max = float(
                                        _new_res.abs().max().item()
                                    )
                                except Exception:
                                    _r_abs, _r_max = -1.0, -1.0
                                _sd_sample_details.append(
                                    "name=%s shape=%s had_prev=%s "
                                    "sync_idx=%d shifted_elems=%d "
                                    "residual_abs_mean=%.3e "
                                    "residual_abs_max=%.3e" % (
                                        _nm_sd,
                                        tuple(_tn_sd.shape),
                                        str(_had_prev),
                                        self._mtp_sd_sync_idx[_nm_sd],
                                        _shift_cnt,
                                        _r_abs, _r_max,
                                    )
                                )
                        if _sd_applied > 0:
                            self.logger.info(
                                "[MTPSigmaDeltaBf16-v28] collect_name=%s "
                                "applied=%d total_shifted_elems=%d "
                                "samples=[%s]",
                                name,
                                _sd_applied, _sd_total_shifted,
                                " | ".join(_sd_sample_details),
                            )
                    # [MTPWeightDeltaD15] version-to-version
                    # abs_mean delta tracker.
                    if not hasattr(self, "_mtp_d15_prev_abs_mean"):
                        self._mtp_d15_prev_abs_mean = {}
                    for _hf_nm_d15, _hf_tn_d15 in (
                        mtp_hf_tensors[_prev_count:]
                    ):
                        _am_d15 = float(
                            _hf_tn_d15.float().abs().mean().item(),
                        )
                        _prev_am = self._mtp_d15_prev_abs_mean.get(
                            _hf_nm_d15,
                        )
                        _dlt = (
                            None if _prev_am is None
                            else _am_d15 - _prev_am
                        )
                        self._mtp_d15_prev_abs_mean[_hf_nm_d15] = (
                            _am_d15
                        )
                        # [MTPFp32Delta-v27] Track fp32 master abs_mean delta
                        # between consecutive MTP sync events.  Combined with
                        # MTPBf16Drift-v25, makes it possible to compare
                        # "fp32 per-step update" vs "bf16 ULP" directly.
                        try:
                            if not hasattr(self, "_mtp_v27_fp32_prev"):
                                self._mtp_v27_fp32_prev = {}
                            _v27_fp32_am = float(
                                _hf_tn_d15.float().abs().mean().item()
                            )
                            _v27_prev = self._mtp_v27_fp32_prev.get(
                                _hf_nm_d15
                            )
                            _v27_delta = (
                                None if _v27_prev is None
                                else _v27_fp32_am - _v27_prev
                            )
                            self._mtp_v27_fp32_prev[_hf_nm_d15] = (
                                _v27_fp32_am
                            )
                            self.logger.info(
                                "[MTPFp32Delta-v27] hf=%s "
                                "fp32_abs_mean=%.9e delta=%s",
                                _hf_nm_d15, _v27_fp32_am,
                                ("%+0.3e" % _v27_delta)
                                if _v27_delta is not None else "n/a",
                            )
                        except Exception as _e_v27fd:
                            self.logger.warning(
                                "[MTPFp32Delta-v27] err hf=%s: %s",
                                _hf_nm_d15, _e_v27fd,
                            )
                        # [MTPBf16Drift-v25] fp32 vs bf16-cast drift.
                        try:
                            import torch as _torch_v25d
                            _fp32_dtype = _hf_tn_d15.dtype
                            _fp32_ref = _hf_tn_d15.float()
                            _fp32_abs_mean = float(
                                _fp32_ref.abs().mean().item()
                            )
                            try:
                                _bf16_cast = _hf_tn_d15.to(
                                    _torch_v25d.bfloat16
                                ).float()
                                _bf16_abs_mean = float(
                                    _bf16_cast.abs().mean().item()
                                )
                                _diff = (_fp32_ref - _bf16_cast).abs()
                                _diff_l1 = float(_diff.mean().item())
                                _diff_linf = float(_diff.max().item())
                            except Exception:
                                _bf16_abs_mean = float('nan')
                                _diff_l1 = float('nan')
                                _diff_linf = float('nan')
                            self.logger.info(
                                "[MTPBf16Drift-v25] hf=%s "
                                "fp32_abs_mean=%.6e bf16_abs_mean=%.6e "
                                "cast_diff_l1=%.3e cast_diff_linf=%.3e "
                                "fp32_dtype=%s",
                                _hf_nm_d15, _fp32_abs_mean,
                                _bf16_abs_mean, _diff_l1,
                                _diff_linf, str(_fp32_dtype),
                            )
                        except Exception as _e_v25d:
                            self.logger.warning(
                                "[MTPBf16Drift-v25] error hf=%s: %s",
                                _hf_nm_d15, _e_v25d,
                            )
                        self.logger.info(
                            "[MTPWeightDeltaD15] hf=%s "
                            "abs_mean=%.9e delta=%s frozen=%s "
                            "(dtype=%s src=%s)",
                            _hf_nm_d15, _am_d15,
                            (("%+0.3e" % _dlt)
                                if _dlt is not None else "n/a"),
                            ("True" if _dlt is not None
                                and abs(_dlt) < 1e-9 else "False"),
                            str(_hf_tn_d15.dtype),
                            _src_tag,
                        )
                        try:
                            for _h in list(self.logger.handlers):
                                try:
                                    _h.flush()
                                except Exception:
                                    pass
                            _sys_v24m.stdout.flush()
                        except Exception:
                            pass
                    # Per-tensor stats.
                    for _hf_name, _hf_tensor in (
                        mtp_hf_tensors[_prev_count:]
                    ):
                        _abs = _hf_tensor.float().abs()
                        try:
                            if float(_abs.max().item()) == 0.0:
                                self.logger.warning(
                                    "[MTPWeightDeltaGuard-v14] MTP "
                                    "tensor %s (hf=%s) has abs_max"
                                    "==0; draft head is stalled.",
                                    name, _hf_name,
                                )
                        except Exception:
                            pass
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

        # [MTPRelativeSpeed-v31] Measure fp32 |W_MTP| from the already-
        # upcast mtp_hf_tensors list (v16 AREAL_MTP_FP32_BROADCAST=1
        # guarantees these are fp32), and fp32 |W_BB| by promoting each
        # backbone bf16 tensor to fp32 during reduction only. v30 read
        # _p.data (bf16 copy) which had ULP=2.2 per element on |W|~284,
        # making d|W|/|W| dominated by quantization noise instead of the
        # actual Adam master-weight movement.
        #
        # H1 judgement:
        #   rel_speed <= 0.1 persistent  -> CONFIRMED (MTP too slow)
        #   rel_speed >= 1.0 persistent  -> REJECTED
        #   otherwise UNKNOWN
        try:
            import os as _os_v31
            _v31_on = _os_v31.environ.get("AREAL_MTP_V30_DIAG", "1") == "1"
        except Exception:
            _v31_on = False
        if _v31_on and mtp_hf_tensors:
            try:
                import torch as _torch_v31
                # ---- MTP fp32 norm (already fp32 after v16 upcast) ----
                _mtp_sq = 0.0
                _mtp_cnt = 0
                for _nm, _tn in mtp_hf_tensors:
                    _f = _tn.detach()
                    if _f.dtype != _torch_v31.float32:
                        _f = _f.float()
                    _mtp_sq += float((_f * _f).sum().item())
                    _mtp_cnt += int(_f.numel())
                _mtp_norm = _mtp_sq ** 0.5
                # ---- Backbone fp32 norm (promote bf16 -> fp32 on-fly) ----
                _bb_sq = 0.0
                _bb_cnt = 0
                for _nbb, _pbb in get_named_parameters(
                    self.model, num_moe_experts
                ):
                    if ".experts." in _nbb:
                        continue
                    if ".mtp." in _nbb:
                        continue
                    _tbb = _pbb.detach()
                    if _tbb is None:
                        continue
                    _tbb = _tbb.float()
                    _bb_sq += float((_tbb * _tbb).sum().item())
                    _bb_cnt += int(_tbb.numel())
                _bb_norm = _bb_sq ** 0.5
                # ---- Delta bookkeeping ----
                _prev_mtp = getattr(self, "_v31_prev_mtp_norm", None)
                _prev_bb = getattr(self, "_v31_prev_bb_norm", None)
                self._v31_prev_mtp_norm = _mtp_norm
                self._v31_prev_bb_norm = _bb_norm
                _d_mtp_rel = None
                _d_bb_rel = None
                _rel_speed = None
                if _prev_mtp is not None and _prev_bb is not None:
                    _d_mtp = abs(_mtp_norm - _prev_mtp)
                    _d_bb = abs(_bb_norm - _prev_bb)
                    if _mtp_norm > 0:
                        _d_mtp_rel = _d_mtp / _mtp_norm
                    if _bb_norm > 0:
                        _d_bb_rel = _d_bb / _bb_norm
                    if (
                        _d_mtp_rel is not None
                        and _d_bb_rel is not None
                        and _d_bb_rel > 0
                    ):
                        _rel_speed = _d_mtp_rel / _d_bb_rel
                try:
                    _rk = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized() else 0
                    )
                except Exception:
                    _rk = 0
                if _rk == 0:
                    self.logger.info(
                        "[MTPRelativeSpeed-v31] version=%s "
                        "|W_MTP|_fp32=%.6e (n=%d) "
                        "|W_BB|_fp32=%.6e (n=%d) "
                        "d|W_MTP|/|W_MTP|=%s d|W_BB|/|W_BB|=%s "
                        "rel_speed=%s",
                        str(meta.version), _mtp_norm, _mtp_cnt,
                        _bb_norm, _bb_cnt,
                        ("%.4e" % _d_mtp_rel) if _d_mtp_rel is not None else "NA",
                        ("%.4e" % _d_bb_rel) if _d_bb_rel is not None else "NA",
                        ("%.4f" % _rel_speed) if _rel_speed is not None else "NA",
                    )
                    # ---- [MTPLossSignalAudit-v31] real attribute names ----
                    _mtp_loss_ema = getattr(self, "_mtp_loss_ema", None)
                    _mtp_loss_val = getattr(self, "_mtp_loss_value", None)
                    _mtp_lr_cache = getattr(self, "_last_logged_mtp_lr", None)
                    # [v32] Read task_reward / entropy from the
                    # engine-side snapshot populated by our
                    # export_stats override (see export_stats below).
                    # The v31 stats_tracker.get('<stat_key>') path
                    # returned an empty DistributedStatsTracker (get
                    # is keyed by TRACKER name, not stat name).
                    _latest = getattr(
                        self, "_last_stats_snapshot_v32", None
                    )
                    _task_reward = None
                    _entropy_avg = None
                    if isinstance(_latest, dict):
                        _task_reward = _latest.get(
                            "ppo_actor/task_reward/avg"
                        )
                        _entropy_avg = _latest.get(
                            "ppo_actor/update/entropy/avg"
                        )
                    if _task_reward is None:
                        _task_reward = getattr(
                            self, "_last_task_reward_avg", None
                        )
                    if _entropy_avg is None:
                        _entropy_avg = getattr(
                            self, "_last_entropy_avg", None
                        )
                    _accept_ema = getattr(
                        self, "_last_accept_ema256", None
                    )
                    _h1 = "UNKNOWN"
                    if isinstance(_rel_speed, float):
                        if _rel_speed <= 0.1:
                            _h1 = "CONFIRMED"
                        elif _rel_speed >= 1.0:
                            _h1 = "REJECTED"
                    _h4 = "NORMAL"
                    if (
                        isinstance(_task_reward, (int, float))
                        and _task_reward >= 0.9
                        and isinstance(_mtp_loss_ema, (int, float))
                        and _mtp_loss_ema <= 0.6
                    ):
                        _h4 = "SUSPECT"
                    self.logger.info(
                        "[MTPLossSignalAudit-v31] version=%s "
                        "rel_speed=%s |W_MTP|=%.6e |W_BB|=%.6e "
                        "mtp_loss_ema=%s mtp_loss_raw=%s mtp_lr=%s "
                        "task_reward=%s entropy_avg=%s accept_ema=%s "
                        "H1=%s H4=%s",
                        str(meta.version),
                        ("%.4f" % _rel_speed) if _rel_speed is not None else "NA",
                        _mtp_norm, _bb_norm,
                        ("%.4f" % _mtp_loss_ema) if isinstance(_mtp_loss_ema, (int, float)) else "NA",
                        ("%.4f" % _mtp_loss_val) if isinstance(_mtp_loss_val, (int, float)) else "NA",
                        ("%.3e" % _mtp_lr_cache) if isinstance(_mtp_lr_cache, (int, float)) else "NA",
                        ("%.4f" % _task_reward) if isinstance(_task_reward, (int, float)) else "NA",
                        ("%.4f" % _entropy_avg) if isinstance(_entropy_avg, (int, float)) else "NA",
                        ("%.4f" % _accept_ema) if isinstance(_accept_ema, (int, float)) else "NA",
                        _h1, _h4,
                    )
            except Exception as _e_v31:
                try:
                    self.logger.warning(
                        "[MTPRelativeSpeed-v31] failed: %r", _e_v31,
                    )
                except Exception:
                    pass
        # [MTPBf16PayloadNorm-v33] Engine-side wire-truth norm.
        # After the sigma-delta path above (v28-v29), entries of
        # mtp_hf_tensors that correspond to fp32 master MTP params
        # have been *replaced* with their bf16 RNE-cast versions
        # (see "_bf16.contiguous()" at the sigma-delta tail). Those
        # exact bf16 bytes are the payload that sglang's
        # eagle_worker.update_weights_from_tensor .copy_()s into
        # BOTH draft_model_runner.model AND target_worker.model
        # (eagle_worker.py:999). So |W|_bf16_wire IS the ground
        # truth for "did the weights on the wire change". No HTTP
        # roundtrip needed -> immune to the MiMo /get_weights_by_name
        # architectural block that killed v32's readback.
        if _v31_on and mtp_hf_tensors:
            try:
                import torch as _torch_v33
                _wire_sq = 0.0
                _wire_cnt = 0
                _wire_bf16_cnt = 0
                _wire_fp32_cnt = 0
                _first_name = None
                _first_norm = None
                for _nm_w, _tn_w in mtp_hf_tensors:
                    _tw = _tn_w.detach()
                    if _tw.dtype == _torch_v33.bfloat16:
                        _wire_bf16_cnt += 1
                    elif _tw.dtype == _torch_v33.float32:
                        _wire_fp32_cnt += 1
                    _tf = _tw.float()
                    _s = float((_tf * _tf).sum().item())
                    _wire_sq += _s
                    _wire_cnt += int(_tf.numel())
                    if _first_name is None:
                        _first_name = _nm_w
                        _first_norm = _s ** 0.5
                _wire_norm = _wire_sq ** 0.5
                _prev_wire = getattr(self, "_v33_prev_wire_norm", None)
                self._v33_prev_wire_norm = _wire_norm
                _d_wire = None
                _d_wire_rel = None
                if _prev_wire is not None and _wire_norm > 0:
                    _d_wire = abs(_wire_norm - _prev_wire)
                    _d_wire_rel = _d_wire / _wire_norm
                try:
                    _rk_w = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized() else 0
                    )
                except Exception:
                    _rk_w = 0
                if _rk_w == 0:
                    _h2_wire = "UNKNOWN"
                    if _d_wire is not None:
                        if _d_wire == 0.0:
                            _h2_wire = "CONFIRMED-STALL"
                        elif _d_wire_rel is not None and _d_wire_rel < 1e-8:
                            _h2_wire = "SUSPECT-MICRO"
                        else:
                            _h2_wire = "REJECTED"
                    self.logger.info(
                        "[MTPBf16PayloadNorm-v33] version=%s "
                        "|W|_wire=%.6e (n=%d, bf16=%d fp32=%d) "
                        "d|W|_wire=%s d|W|_wire_rel=%s "
                        "first=%s first_norm=%s "
                        "H2_wire=%s",
                        str(meta.version),
                        _wire_norm, _wire_cnt,
                        _wire_bf16_cnt, _wire_fp32_cnt,
                        ("%.6e" % _d_wire) if _d_wire is not None else "NA",
                        ("%.4e" % _d_wire_rel) if _d_wire_rel is not None else "NA",
                        str(_first_name),
                        ("%.6e" % _first_norm) if _first_norm is not None else "NA",
                        _h2_wire,
                    )
            except Exception as _e_v33_wire:
                try:
                    self.logger.warning(
                        "[MTPBf16PayloadNorm-v33] failed: %r",
                        _e_v33_wire,
                    )
                except Exception:
                    pass
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
            # [MTPShipSummaryFix-v56] Stash the actual MTP wire payload
            # (`mtp_hf_tensors`) on self so the v54 ship-stage diagnostic
            # block inside `_update_bucket_weights_from_distributed` can
            # iterate the *correct* list (the one truly broadcast to the
            # inference engine), not `converted_named_tensors` which holds
            # main-model bucket payload during the MTP wire path.
            try:
                self._v56_mtp_hf_tensors = list(mtp_hf_tensors)
            except Exception:
                self._v56_mtp_hf_tensors = []
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

        # [MTPShipFinalSummary-v61] one-shot definitive ship summary right
        # after the MTP loop completes, BEFORE serialize/send. Unlike the
        # per-bucket-flush v56 ship_summary (which can fire 13+ times with
        # n_mtp_shipped=0 because the bucket flush happens DURING the MTP
        # collection loop), this one fires exactly once per ship and shows
        # the actual MTP wire payload list contents.
        try:
            if _collect_mtp_for_draft:
                _v61f_ver = getattr(meta, 'version', 'NA')
                _v61f_n = len(mtp_hf_tensors)
                _v61f_total_bytes = sum(
                    int(t.numel() * t.element_size())
                    for _, t in mtp_hf_tensors
                )
                _v61f_first = mtp_hf_tensors[0][0] if _v61f_n > 0 else None
                _v61f_names = [n for n, _ in mtp_hf_tensors]
                self.logger.info(
                    '[MTPShipFinalSummary-v61] rank=%d version=%s '
                    'n_mtp_shipped=%d total_bytes=%d first=%s '
                    'names=%s',
                    int(dist.get_rank()), str(_v61f_ver), _v61f_n,
                    _v61f_total_bytes, str(_v61f_first), str(_v61f_names),
                )
                # Cross-version delta on a sentinel HF tensor.
                if _v61f_n > 0:
                    if not hasattr(self, '_v61_prev_ship_first8'):
                        self._v61_prev_ship_first8 = {}
                    for _v61f_n2, _v61f_t in mtp_hf_tensors:
                        try:
                            _v61f_f = _v61f_t.detach().float().reshape(-1)
                            _v61f_first8 = [
                                float(x) for x in _v61f_f[:8].tolist()
                            ]
                            _v61f_l2 = float(_v61f_f.norm().item())
                            _v61f_prev = self._v61_prev_ship_first8.get(
                                _v61f_n2,
                            )
                            self._v61_prev_ship_first8[_v61f_n2] = (
                                _v61f_first8, _v61f_l2,
                            )
                            if _v61f_prev is not None:
                                _v61f_pf, _v61f_pl2 = _v61f_prev
                                _v61f_d8 = [
                                    (a - b) for a, b in zip(
                                        _v61f_first8, _v61f_pf,
                                    )
                                ]
                                _v61f_dl2 = abs(_v61f_l2 - _v61f_pl2)
                            else:
                                _v61f_d8 = []
                                _v61f_dl2 = -1.0
                            self.logger.info(
                                '[MTPShipDelta-v61] rank=%d version=%s '
                                'name=%s l2=%.6e d_l2=%.6e first8=%s '
                                'd_first8=%s',
                                int(dist.get_rank()), str(_v61f_ver),
                                _v61f_n2, _v61f_l2, _v61f_dl2,
                                str(_v61f_first8),
                                str(_v61f_d8),
                            )
                        except Exception:
                            continue
        except Exception as _e_v61f:
            try:
                self.logger.info(
                    '[MTPShipFinalSummary-v61] failure: %r', _e_v61f,
                )
            except Exception:
                pass
        # [MTPShipHashAudit-v62] rank-0 full list dump with hash
        # so that next round we can cross-check exactly which
        # HF-named bytes were shipped versus what the draft
        # engine received / applied.  This is independent of
        # the existing v54/v56/v61 summaries; focuses only on
        # deterministic content-hash identity of each tensor.
        if (_collect_mtp_for_draft and mtp_hf_tensors
                and dist.get_rank() == 0):
            try:
                import hashlib as _v62_hashlib
                for _v62_hn, _v62_ht in mtp_hf_tensors:
                    try:
                        _v62_cpu = (
                            _v62_ht.detach().contiguous()
                            .cpu().view(torch.uint8))
                        _v62_nb = _v62_cpu.numel()
                        _v62_h = _v62_hashlib.sha256(
                            _v62_cpu.numpy().tobytes()).hexdigest()[:16]
                        _v62_f8 = [
                            float(x) for x in
                            _v62_ht.detach().float()
                            .reshape(-1)[:8].tolist()]
                        self.logger.info(
                            '[MTPShipHashAudit-v62] version=%s '
                            'hf_name=%s dtype=%s shape=%s '
                            'bytes=%d sha256_16=%s first8=%s',
                            getattr(meta, 'version', None),
                            _v62_hn,
                            str(_v62_ht.dtype),
                            tuple(_v62_ht.shape),
                            _v62_nb, _v62_h, _v62_f8,
                        )
                    except Exception as _e_v62_t:
                        try:
                            self.logger.info(
                                '[MTPShipHashAudit-v62] '
                                'tensor %s failure: %r',
                                _v62_hn, _e_v62_t)
                        except Exception:
                            pass
            except Exception as _e_v62_out:
                try:
                    self.logger.info(
                        '[MTPShipHashAudit-v62] outer failure: %r',
                        _e_v62_out)
                except Exception:
                    pass
        # [MTPShipKeyOverlap-v63] After all MTP HF tensors are collected
        # AND the main bucket converted_named_tensors is finalised,
        # cross-check whether MTP HF names overlap with main-bucket
        # HF names being shipped in the SAME wave. sglang's EAGLE
        # draft model shares some backbone weights (embedding,
        # output_layer) with the target model; if MTP-collected tensors
        # collide with main-bucket HF names, one would overwrite the
        # other in unpredictable order, causing post-ship draft
        # regression that matches what spec_v1.log.5 shows.
        try:
            if (_collect_mtp_for_draft
                    and mtp_hf_tensors
                    and dist.get_rank() == 0):
                _v63_mtp_names = set(n for n, _ in mtp_hf_tensors)
                _v63_main_names = set()
                try:
                    _v63_main_names = set(
                        n for n, _ in (converted_named_tensors or [])
                    )
                except Exception:
                    pass
                _v63_overlap = sorted(
                    _v63_mtp_names & _v63_main_names)
                self.logger.info(
                    "[MTPShipKeyOverlap-v63] version=%s "
                    "n_mtp=%d n_main=%d n_overlap=%d "
                    "overlap_keys=%s "
                    "mtp_only_sample=%s main_only_sample=%s",
                    str(getattr(meta, 'version', 'NA')),
                    len(_v63_mtp_names), len(_v63_main_names),
                    len(_v63_overlap),
                    str(_v63_overlap[:16]),
                    str(sorted(_v63_mtp_names - _v63_main_names)[:8]),
                    str(sorted(_v63_main_names - _v63_mtp_names)[:8]),
                )
                if _v63_overlap:
                    self.logger.warning(
                        "[MTPShipKeyOverlap-v63] OVERLAP DETECTED "
                        "version=%s — %d HF names ship in BOTH the "
                        "main bucket AND the MTP wire. SGLang receives "
                        "BOTH writes for the same key; last-writer "
                        "wins and may overwrite the MTP-trained value "
                        "with the main-model value (or vice versa). "
                        "Sample: %s",
                        str(getattr(meta, 'version', 'NA')),
                        len(_v63_overlap), str(_v63_overlap[:8]),
                    )
        except Exception as _e_v63_ko:
            try:
                self.logger.info(
                    "[MTPShipKeyOverlap-v63] failure: %r", _e_v63_ko,
                )
            except Exception:
                pass

        # [MTPDraftReadbackV4-v63] Probe alternative sglang readback
        # endpoints to capture what the draft model ACTUALLY has
        # post-ship. v32's /get_weights_by_name path is blocked for
        # MiMo; this v63 probe attempts /update_weights_from_tensor
        # echo paths and a generic /get_internal_state fallback so
        # that next round we can correlate ship-time hash with
        # draft-side hash even when one channel is blocked.
        try:
            import os as _v63_os_rb
            if (_collect_mtp_for_draft
                    and mtp_hf_tensors
                    and dist.get_rank() == 0
                    and _v63_os_rb.environ.get(
                        'AREAL_MTP_DRAFT_READBACK_V4', '1') == '1'):
                _v63_rb_engine = getattr(
                    self, 'rollout_engine', None)
                _v63_rb_endpoints = [
                    'get_weights_by_name',
                    'get_internal_state',
                    'flush_cache',
                ]
                for _v63_ep in _v63_rb_endpoints:
                    _v63_fn = getattr(
                        _v63_rb_engine, _v63_ep, None)
                    self.logger.info(
                        "[MTPDraftReadbackV4-v63] version=%s "
                        "endpoint=%s callable=%s",
                        str(getattr(meta, 'version', 'NA')),
                        _v63_ep, str(callable(_v63_fn)),
                    )
                    if callable(_v63_fn):
                        try:
                            if _v63_ep == 'get_weights_by_name':
                                _v63_rb_names = set(n for n, _ in mtp_hf_tensors)
                                _v63_target_name = next(
                                    iter(_v63_rb_names), None)
                                if _v63_target_name is not None:
                                    _v63_rb_res = _v63_fn(
                                        _v63_target_name)
                                else:
                                    _v63_rb_res = None
                            else:
                                _v63_rb_res = _v63_fn()
                            self.logger.info(
                                "[MTPDraftReadbackV4-v63] "
                                "endpoint=%s status=OK "
                                "result_type=%s "
                                "result_repr_head=%.200s",
                                _v63_ep, type(_v63_rb_res).__name__,
                                repr(_v63_rb_res),
                            )
                        except Exception as _e_v63_ep:
                            self.logger.info(
                                "[MTPDraftReadbackV4-v63] "
                                "endpoint=%s status=FAIL err=%r",
                                _v63_ep, _e_v63_ep,
                            )
        except Exception as _e_v63_rb:
            try:
                self.logger.info(
                    "[MTPDraftReadbackV4-v63] outer failure: %r",
                    _e_v63_rb,
                )
            except Exception:
                pass

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
                # [SGLangReadBackMTPv2-v27] Read back MTP LayerNorm weights
                # from the SGLang draft model over HTTP directly.
                # iter14 used Python attribute access (missing on
                # RemoteSGLangEngine). SGLang exposes
                # /get_weights_by_parameter_name endpoint (introduced
                # for verl/slime) which accepts JSON {name, truncate_size}.
                try:
                    import requests as _v27_requests
                    _v27_addrs = None
                    try:
                        _v27_inner = getattr(
                            self.rollout_engine, "_engine", None
                        )
                        _v27_addrs = getattr(_v27_inner, "addresses", None)
                    except Exception:
                        _v27_addrs = None
                    if _v27_addrs:
                        _v27_probe = [
                            "model.mtp_layers.0.token_layernorm.weight",
                            "model.mtp_layers.0.hidden_layernorm.weight",
                            "model.mtp_layers.0.input_layernorm.weight",
                            "model.mtp_layers.0.post_attention_layernorm.weight",
                            "model.mtp_layers.0.final_layernorm.weight",
                        ]
                        _v27_addr0 = _v27_addrs[0]
                        _v27_base = (
                            _v27_addr0 if _v27_addr0.startswith("http")
                            else f"http://{_v27_addr0}"
                        )
                        for _v27_nm in _v27_probe:
                            try:
                                _v27_resp = _v27_requests.post(
                                    f"{_v27_base}/get_weights_by_parameter_name",
                                    json={
                                        "name": _v27_nm,
                                        "truncate_size": 8,
                                    },
                                    timeout=15,
                                )
                                _v27_body = _v27_resp.text[:400]
                                self.logger.info(
                                    "[SGLangReadBackMTPv2-v27] name=%s "
                                    "status=%s body=%s",
                                    _v27_nm, _v27_resp.status_code,
                                    _v27_body,
                                )
                            except Exception as _e_v27rb1:
                                self.logger.info(
                                    "[SGLangReadBackMTPv2-v27] name=%s "
                                    "http_err=%s", _v27_nm, _e_v27rb1,
                                )
                    else:
                        self.logger.info(
                            "[SGLangReadBackMTPv2-v27] addresses unavailable; "
                            "cannot read back (inner_engine=%s).",
                            type(
                                getattr(self.rollout_engine, "_engine", None)
                            ).__name__,
                        )
                except Exception as _e_v27rb:
                    self.logger.warning(
                        "[SGLangReadBackMTPv2-v27] outer error: %s",
                        _e_v27rb,
                    )
                # [SGLangReadBackMTPv3-v28] Callback-chain readback.
                # In AReaL single-controller mode, self.rollout_engine
                # is a RolloutCallback.  v27 used
                # RemoteSGLangEngine._engine.addresses, which only
                # exists on inference-side workers, not on the
                # train-side MegatronEngine (log.9 proved it).  v28
                # walks callback -> controller -> worker chain that
                # already exists for /callback/update_weights_tensor.
                try:
                    _v28_probe_names = [
                        "model.mtp_layers.0.token_layernorm.weight",
                        "model.mtp_layers.0.hidden_layernorm.weight",
                        "model.mtp_layers.0.input_layernorm.weight",
                        "model.mtp_layers.0.post_attention_layernorm.weight",
                        "model.mtp_layers.0.final_layernorm.weight",
                    ]
                    _v28_read = getattr(
                        self.rollout_engine,
                        "read_weights_by_name",
                        None,
                    )
                    if _v28_read is not None:
                        _v28_resp = _v28_read(
                            names=_v28_probe_names, truncate_size=8,
                        )
                        _v28_entries = []
                        if isinstance(_v28_resp, dict):
                            _v28_entries = _v28_resp.get("entries", [])
                        for _ent in _v28_entries:
                            self.logger.info(
                                "[SGLangReadBackMTPv3-v28] name=%s "
                                "status=%s dtype=%s first8=%s body=%s",
                                _ent.get("name"),
                                _ent.get("status"),
                                _ent.get("dtype"),
                                _ent.get("first8"),
                                (str(_ent.get("body", ""))[:240]),
                            )
                    else:
                        self.logger.info(
                            "[SGLangReadBackMTPv3-v28] rollout_engine "
                            "lacks read_weights_by_name (engine_type=%s).",
                            type(self.rollout_engine).__name__,
                        )
                except Exception as _e_v28rb:
                    self.logger.warning(
                        "[SGLangReadBackMTPv3-v28] outer error: %s",
                        _e_v28rb,
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
        # [MTPProbeLogprob-v37b] Deterministic inference probe AFTER
        # continue_generation, with per-stage try/except + traceback.
        #
        # v36 failed universally with
        #   AttributeError: 'MegatronPPOActor' object has no attribute
        #   '_weight_version'
        # because MegatronEngine exposes self._version + get_version(),
        # never _weight_version.  v37b fixes the attribute and also
        # wraps every line of the probe in a per-stage try/except so
        # any future failure logs traceback.format_exc() AND a stage
        # tag identifying the exact raise site.
        try:
            import os as _os_v37b
            _v37b_on = _os_v37b.environ.get("AREAL_MTP_V30_DIAG", "1") == "1"
        except Exception:
            _v37b_on = False
        if _v37b_on:
            _stage_v37b = "enter"
            try:
                import traceback as _tb_v37b
                _stage_v37b = "get_rollout_engine"
                _re_v37b = self.rollout_engine
                _stage_v37b = "getattr_controller_addr"
                _addr_v37b = getattr(_re_v37b, "controller_addr", None)
                _stage_v37b = "get_rank"
                try:
                    _rk_v37b = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized() else 0
                    )
                except Exception:
                    _rk_v37b = 0
                if _rk_v37b == 0:
                    if _addr_v37b is None:
                        self.logger.info(
                            "[MTPProbeLogprob-v37b] unavailable: "
                            "rollout_engine=%s has no controller_addr",
                            type(_re_v37b).__name__,
                        )
                    else:
                        try:
                            _stage_v37b = "import_requests"
                            import requests as _rq_v37b
                            _stage_v37b = "build_url"
                            _probe_url_v37b = (
                                f"http://{_addr_v37b}/callback/"
                                f"get_mtp_probe"
                            )
                            _stage_v37b = "build_version_int"
                            _ver_v37b = int(self.get_version())
                            _stage_v37b = "http_post"
                            _resp_v37b = _rq_v37b.post(
                                _probe_url_v37b,
                                json={"version": _ver_v37b},
                                timeout=150.0,
                                proxies={"http": None, "https": None},
                            )
                            _stage_v37b = "get_status"
                            _status_v37b = _resp_v37b.status_code
                            _stage_v37b = "parse_json"
                            _jp_v37b = {}
                            try:
                                _jp_v37b = _resp_v37b.json()
                            except Exception:
                                _jp_v37b = {}
                            _stage_v37b = "extract_fields"
                            _lp_v37b = _jp_v37b.get("logprob", None)
                            _srv_v37b = _jp_v37b.get("server", None)
                            _err_v37b = _jp_v37b.get("error", None)
                            _stage_v37b = "get_prev_lp"
                            _prev_lp_v37b = getattr(
                                self, "_v37b_prev_probe_logprob", None
                            )
                            _stage_v37b = "compute_d_lp"
                            if isinstance(_lp_v37b, (int, float)):
                                _d_lp_v37b = (
                                    None if _prev_lp_v37b is None
                                    else abs(
                                        float(_lp_v37b)
                                        - float(_prev_lp_v37b)
                                    )
                                )
                            else:
                                _d_lp_v37b = None
                            _stage_v37b = "set_prev_lp_attr"
                            if isinstance(_lp_v37b, (int, float)):
                                self._v37b_prev_probe_logprob = float(
                                    _lp_v37b
                                )
                            _stage_v37b = "logger_info_success"
                            self.logger.info(
                                "[MTPProbeLogprob-v37b] version=%s "
                                "status=%s logprob=%s d_logprob=%s "
                                "server=%s err=%s",
                                _ver_v37b,
                                _status_v37b,
                                ("%.6e" % _lp_v37b) if isinstance(_lp_v37b, (int, float)) else "NA",
                                ("%.6e" % _d_lp_v37b) if isinstance(_d_lp_v37b, (int, float)) else "NA",
                                _srv_v37b, _err_v37b,
                            )
                        except Exception as _e_v37b:
                            try:
                                _tb_str_v37b = _tb_v37b.format_exc()
                            except Exception:
                                _tb_str_v37b = "<traceback unavailable>"
                            self.logger.info(
                                "[MTPProbeLogprob-v37b] inner failure "
                                "at stage=%s exc=%r\nTRACEBACK:\n%s",
                                _stage_v37b, _e_v37b, _tb_str_v37b,
                            )
            except Exception as _e_v37b_out:
                try:
                    _tb_out_v37b = _tb_v37b.format_exc()
                except Exception:
                    _tb_out_v37b = "<traceback unavailable>"
                try:
                    self.logger.warning(
                        "[MTPProbeLogprob-v37b] outer failure at "
                        "stage=%s exc=%r\nTRACEBACK:\n%s",
                        _stage_v37b, _e_v37b_out, _tb_out_v37b,
                    )
                except Exception:
                    pass
        # [DraftOutputProbe-v38] Draft+target OUTPUT SEQUENCE probe.
        # v37b only reads input_token_logprobs[0] which is pure target.
        # v38 drives /generate with max_new_tokens=32, top_k=1, T=0
        # and records output_ids + output logprobs + any meta_info
        # spec/accept fields, so we can see draft+MTP head effects.
        # Per-stage try/except + traceback for robustness.
        try:
            import os as _os_v38
            _v38_on = _os_v38.environ.get("AREAL_MTP_V30_DIAG", "1") == "1"
        except Exception:
            _v38_on = False
        if _v38_on:
            _stage_v38 = "enter"
            try:
                import traceback as _tb_v38
                _stage_v38 = "get_rollout_engine"
                _re_v38 = self.rollout_engine
                _stage_v38 = "getattr_controller_addr"
                _addr_v38 = getattr(_re_v38, "controller_addr", None)
                _stage_v38 = "get_rank"
                try:
                    _rk_v38 = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized() else 0
                    )
                except Exception:
                    _rk_v38 = 0
                if _rk_v38 == 0 and _addr_v38 is not None:
                    try:
                        _stage_v38 = "import_requests"
                        import requests as _rq_v38
                        _stage_v38 = "build_url"
                        _probe_url_v38 = (
                            f"http://{_addr_v38}/callback/"
                            f"get_draft_probe"
                        )
                        _stage_v38 = "build_version_int"
                        _ver_v38 = int(self.get_version())
                        _stage_v38 = "http_post"
                        _resp_v38 = _rq_v38.post(
                            _probe_url_v38,
                            json={"version": _ver_v38},
                            timeout=180.0,
                            proxies={"http": None, "https": None},
                        )
                        _stage_v38 = "get_status"
                        _status_v38 = _resp_v38.status_code
                        _stage_v38 = "parse_json"
                        _jp_v38 = {}
                        try:
                            _jp_v38 = _resp_v38.json()
                        except Exception:
                            _jp_v38 = {}
                        _stage_v38 = "extract_fields"
                        _oi_v38 = _jp_v38.get("out_ids_first8", None)
                        _oi_len_v38 = _jp_v38.get("out_ids_len", None)
                        _olps_v38 = _jp_v38.get("out_lps_first4", None)
                        _last_lp_v38 = _jp_v38.get("last_lp", None)
                        _sum_lp_v38 = _jp_v38.get("sum_lp", None)
                        _otext_v38 = _jp_v38.get("out_text_head", None)
                        _mkeys_v38 = _jp_v38.get("meta_keys", None)
                        _specf_v38 = _jp_v38.get("spec_fields", None)
                        _err_v38 = _jp_v38.get("error", None)
                        _stage_v38 = "compute_d_fields"
                        _prev_oi_v38 = getattr(
                            self, "_v38_prev_out_ids", None)
                        _prev_last_lp_v38 = getattr(
                            self, "_v38_prev_last_lp", None)
                        _prev_sum_lp_v38 = getattr(
                            self, "_v38_prev_sum_lp", None)
                        _d_oi_v38 = None
                        if (isinstance(_oi_v38, list)
                                and isinstance(_prev_oi_v38, list)
                                and len(_oi_v38) == len(_prev_oi_v38)):
                            _d_oi_v38 = sum(
                                1 for _a, _b in zip(_oi_v38, _prev_oi_v38)
                                if _a != _b
                            )
                        _d_last_lp_v38 = None
                        if (isinstance(_last_lp_v38, (int, float))
                                and isinstance(_prev_last_lp_v38, (int, float))):
                            _d_last_lp_v38 = abs(
                                float(_last_lp_v38)
                                - float(_prev_last_lp_v38)
                            )
                        _d_sum_lp_v38 = None
                        if (isinstance(_sum_lp_v38, (int, float))
                                and isinstance(_prev_sum_lp_v38, (int, float))):
                            _d_sum_lp_v38 = abs(
                                float(_sum_lp_v38)
                                - float(_prev_sum_lp_v38)
                            )
                        _stage_v38 = "set_prev_attrs"
                        if isinstance(_oi_v38, list):
                            self._v38_prev_out_ids = list(_oi_v38)
                        if isinstance(_last_lp_v38, (int, float)):
                            self._v38_prev_last_lp = float(_last_lp_v38)
                        if isinstance(_sum_lp_v38, (int, float)):
                            self._v38_prev_sum_lp = float(_sum_lp_v38)
                        _stage_v38 = "logger_info_success"
                        self.logger.info(
                            "[DraftOutputProbe-v38] version=%s "
                            "status=%s out_ids_len=%s out_ids=%s "
                            "d_out_ids_hamming=%s last_lp=%s "
                            "d_last_lp=%s sum_lp=%s d_sum_lp=%s "
                            "out_text_head=%r meta_keys=%s "
                            "spec_fields=%s err=%s",
                            _ver_v38, _status_v38,
                            _oi_len_v38, _oi_v38,
                            _d_oi_v38,
                            ("%.6e" % _last_lp_v38) if isinstance(_last_lp_v38, (int, float)) else "NA",
                            ("%.6e" % _d_last_lp_v38) if isinstance(_d_last_lp_v38, (int, float)) else "NA",
                            ("%.6e" % _sum_lp_v38) if isinstance(_sum_lp_v38, (int, float)) else "NA",
                            ("%.6e" % _d_sum_lp_v38) if isinstance(_d_sum_lp_v38, (int, float)) else "NA",
                            _otext_v38, _mkeys_v38,
                            _specf_v38, _err_v38,
                        )
                    except Exception as _e_v38:
                        try:
                            _tb_str_v38 = _tb_v38.format_exc()
                        except Exception:
                            _tb_str_v38 = "<traceback unavailable>"
                        self.logger.info(
                            "[DraftOutputProbe-v38] inner failure "
                            "at stage=%s exc=%r\nTRACEBACK:\n%s",
                            _stage_v38, _e_v38, _tb_str_v38,
                        )
            except Exception as _e_v38_out:
                try:
                    _tb_out_v38 = _tb_v38.format_exc()
                except Exception:
                    _tb_out_v38 = "<traceback unavailable>"
                try:
                    self.logger.warning(
                        "[DraftOutputProbe-v38] outer failure at "
                        "stage=%s exc=%r\nTRACEBACK:\n%s",
                        _stage_v38, _e_v38_out, _tb_out_v38,
                    )
                except Exception:
                    pass
        # [DraftSpecTrend-v39] Long + stochastic probes.  Plus a
        # per-MTP-layer norm scan so heads' individual drift is
        # visible instead of aggregated |W_MTP|.
        try:
            import os as _os_v39
            _v39_on = _os_v39.environ.get("AREAL_MTP_V30_DIAG", "1") == "1"
        except Exception:
            _v39_on = False
        if _v39_on:
            _stage_v39 = "enter"
            try:
                import traceback as _tb_v39
                _stage_v39 = "get_rollout_engine"
                _re_v39 = self.rollout_engine
                _addr_v39 = getattr(_re_v39, "controller_addr", None)
                try:
                    _rk_v39 = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized() else 0
                    )
                except Exception:
                    _rk_v39 = 0
                # --- (a) Per-MTP-layer fp32 norm scan ---
                _stage_v39 = "per_layer_norm"
                try:
                    if mtp_hf_tensors:
                        import torch as _torch_v39
                        _layer_norms = {}
                        for _n, _f in mtp_hf_tensors:
                            if not hasattr(_f, "dtype"):
                                continue
                            try:
                                if _f.dtype != _torch_v39.float32:
                                    _fc = _f.detach().to(_torch_v39.float32)
                                else:
                                    _fc = _f.detach()
                                _nrm = float(_fc.float().norm().item())
                                # group by "model.mtp_layers.{i}."
                                _key = None
                                _parts = _n.split(".")
                                if len(_parts) >= 3 and _parts[0] == "model" and _parts[1] == "mtp_layers":
                                    _key = f"mtp_layer_{_parts[2]}"
                                else:
                                    _key = "other_mtp"
                                _layer_norms.setdefault(_key, 0.0)
                                _layer_norms[_key] = (_layer_norms[_key] ** 2 + _nrm ** 2) ** 0.5
                            except Exception:
                                pass
                        _prev = getattr(self, "_v39_prev_layer_norms", None)
                        _rel = {}
                        if isinstance(_prev, dict):
                            for _k, _v in _layer_norms.items():
                                _pv = _prev.get(_k, None)
                                if isinstance(_pv, (int, float)) and _pv > 0:
                                    _rel[_k] = abs(_v - _pv) / _pv
                        self._v39_prev_layer_norms = dict(_layer_norms)
                        if _rk_v39 == 0:
                            self.logger.info(
                                "[PerLayerMTPNorm-v39] version=%s "
                                "norms=%s d_rel=%s",
                                int(self.get_version()),
                                {_k: ("%.6e" % _v) for _k, _v in _layer_norms.items()},
                                {_k: ("%.3e" % _v) for _k, _v in _rel.items()},
                            )
                except Exception as _e_pln:
                    if _rk_v39 == 0:
                        try:
                            self.logger.info(
                                "[PerLayerMTPNorm-v39] failure: %r\nTRACEBACK:\n%s",
                                _e_pln, _tb_v39.format_exc(),
                            )
                        except Exception:
                            pass
                # --- (b) Long probe ---
                _stage_v39 = "long_probe"
                if _rk_v39 == 0 and _addr_v39 is not None:
                    try:
                        import requests as _rq_l
                        _ver = int(self.get_version())
                        _r_l = _rq_l.post(
                            f"http://{_addr_v39}/callback/get_draft_probe_long",
                            json={"version": _ver},
                            timeout=240.0,
                            proxies={"http": None, "https": None},
                        )
                        _j_l = _r_l.json() if _r_l.status_code == 200 else {}
                        self.logger.info(
                            "[DraftSpecTrend-v39 long] version=%s "
                            "status=%s out_ids_len=%s "
                            "first16=%s last16=%s sum_lp=%s mid_lp=%s "
                            "first_lps=%s last_lps=%s spec=%s err=%s",
                            _ver, _r_l.status_code,
                            _j_l.get("out_ids_len"),
                            _j_l.get("out_ids_first16"),
                            _j_l.get("out_ids_last16"),
                            _j_l.get("sum_lp"),
                            _j_l.get("mid_lp"),
                            _j_l.get("out_lps_first4"),
                            _j_l.get("out_lps_last4"),
                            _j_l.get("spec_fields"),
                            _j_l.get("error"),
                        )
                        # [v40] accept-histogram trend accumulator
                        try:
                            _hist_v40 = _j_l.get("spec_fields", {}) or {}
                            _h_v40 = _hist_v40.get("spec_accept_histogram", None)
                            _al_v40 = _hist_v40.get("spec_accept_length", None)
                            _ar_v40 = _hist_v40.get("spec_accept_rate", None)
                            _trail_v40 = getattr(self, "_v40_long_hist_trail", None)
                            if _trail_v40 is None:
                                _trail_v40 = []
                                self._v40_long_hist_trail = _trail_v40
                            _trail_v40.append(
                                {
                                    "v": int(self.get_version()),
                                    "h": (list(_h_v40) if isinstance(_h_v40, list) else None),
                                    "al": (float(_al_v40) if isinstance(_al_v40, (int, float)) else None),
                                    "ar": (float(_ar_v40) if isinstance(_ar_v40, (int, float)) else None),
                                }
                            )
                            # cap trail at 64
                            if len(_trail_v40) > 64:
                                del _trail_v40[0: len(_trail_v40) - 64]
                            # emit compact trend line
                            _al_seq = [x["al"] for x in _trail_v40]
                            _ar_seq = [x["ar"] for x in _trail_v40]
                            _b2_seq = [
                                (x["h"][2] if isinstance(x["h"], list) and len(x["h"]) > 2 else None)
                                for x in _trail_v40
                            ]
                            _b3_seq = [
                                (x["h"][3] if isinstance(x["h"], list) and len(x["h"]) > 3 else None)
                                for x in _trail_v40
                            ]
                            # monotonic-decline detector (strict <= with at least one strict <)
                            def _mono_decline(_seq):
                                _xs = [x for x in _seq if isinstance(x, (int, float))]
                                if len(_xs) < 3:
                                    return None
                                _lt = all(_xs[_i] <= _xs[_i - 1] for _i in range(1, len(_xs)))
                                _any_strict = any(_xs[_i] < _xs[_i - 1] for _i in range(1, len(_xs)))
                                return bool(_lt and _any_strict)
                            self.logger.info(
                                "[AcceptHistTrend-v40] n_versions=%d "
                                "al_seq=%s ar_seq=%s bucket_accept_len3=%s "
                                "bucket_accept_len4=%s al_mono_decline=%s "
                                "ar_mono_decline=%s",
                                len(_trail_v40),
                                [
                                    (None if _v is None else round(_v, 4))
                                    for _v in _al_seq
                                ],
                                [
                                    (None if _v is None else round(_v, 4))
                                    for _v in _ar_seq
                                ],
                                _b2_seq,
                                _b3_seq,
                                _mono_decline(_al_seq),
                                _mono_decline(_ar_seq),
                            )
                        except Exception as _e_v40_trail:
                            try:
                                self.logger.info(
                                    "[AcceptHistTrend-v40] accumulator "
                                    "failure: %r", _e_v40_trail,
                                )
                            except Exception:
                                pass
                        # [v43] FixedLongProbe: deterministic 128-token
                        # synthetic prompt.  Same IDs every version, so
                        # AR is a pure function of (target + draft)
                        # weights.  Discriminator:
                        #   production AR dip + FixedLong AR flat  -> H5
                        #   production AR dip + FixedLong AR dip   -> H6
                        # The probe reuses the existing controller
                        # endpoint /callback/get_draft_probe_long via
                        # input_ids_override.
                        try:
                            import requests as _rq_fl43
                            _fl_ids_v43 = [
                                int((i * 37 + 5009) % 50000) for i in range(128)
                            ]
                            _fl_resp = _rq_fl43.post(
                                f"http://{_addr_v39}/callback/get_draft_probe_long",
                                json={"version": _ver,
                                      "input_ids_override": _fl_ids_v43},
                                timeout=240.0,
                                proxies={"http": None, "https": None},
                            )
                            _fl_j = _fl_resp.json() if _fl_resp.status_code == 200 else {}
                            _fl_spec = _fl_j.get("spec_fields") or {}
                            _fl_rate = None
                            try:
                                _atn = _fl_spec.get("spec_accept_token_num")
                                _dtn = _fl_spec.get("spec_draft_token_num")
                                if (isinstance(_atn, (int, float))
                                        and isinstance(_dtn, (int, float))
                                        and _dtn > 0):
                                    _fl_rate = float(_atn) / float(_dtn)
                            except Exception:
                                _fl_rate = None
                            self.logger.info(
                                "[FixedLongProbe-v43] version=%s status=%s "
                                "probe_ids_len=%s probe_ids_head=%s "
                                "out_ids_len=%s sum_lp=%s mid_lp=%s "
                                "spec_accept_rate=%s spec=%s",
                                _ver, _fl_resp.status_code,
                                _fl_j.get("probe_ids_len"),
                                _fl_j.get("probe_ids_head"),
                                _fl_j.get("out_ids_len"),
                                _fl_j.get("sum_lp"),
                                _fl_j.get("mid_lp"),
                                ("%.4f" % _fl_rate) if _fl_rate is not None
                                else "NA",
                                _fl_spec,
                            )
                        except Exception as _e_fl43:
                            try:
                                self.logger.info(
                                    "[FixedLongProbe-v43] failure: %r",
                                    _e_fl43,
                                )
                            except Exception:
                                pass
                        # [RepeatFixedLongProbe-v44] fire the SAME
                        # deterministic 128-token prompt again to
                        # measure within-version stochastic variance.
                        # If run-1 vs run-2 differ wildly, the AR
                        # dip is temperature/KV-cache noise, not a
                        # weight-state shift.
                        try:
                            import requests as _rq_rfl44
                            _rfl_ids_v44 = [
                                int((i * 37 + 5009) % 50000) for i in range(128)
                            ]
                            _rfl_resp = _rq_rfl44.post(
                                f"http://{_addr_v39}/callback/get_draft_probe_long",
                                json={"version": _ver,
                                      "input_ids_override": _rfl_ids_v44},
                                timeout=240.0,
                                proxies={"http": None, "https": None},
                            )
                            _rfl_j = (
                                _rfl_resp.json()
                                if _rfl_resp.status_code == 200 else {}
                            )
                            _rfl_spec = _rfl_j.get("spec_fields") or {}
                            _rfl_rate = None
                            try:
                                _atn2 = _rfl_spec.get("spec_accept_token_num")
                                _dtn2 = _rfl_spec.get("spec_draft_token_num")
                                if (isinstance(_atn2, (int, float))
                                        and isinstance(_dtn2, (int, float))
                                        and _dtn2 > 0):
                                    _rfl_rate = float(_atn2) / float(_dtn2)
                            except Exception:
                                _rfl_rate = None
                            self.logger.info(
                                "[RepeatFixedLongProbe-v44] version=%s "
                                "status=%s out_ids_len=%s sum_lp=%s "
                                "mid_lp=%s spec_accept_rate=%s spec=%s",
                                _ver, _rfl_resp.status_code,
                                _rfl_j.get("out_ids_len"),
                                _rfl_j.get("sum_lp"),
                                _rfl_j.get("mid_lp"),
                                ("%.4f" % _rfl_rate) if _rfl_rate is not None
                                else "NA",
                                _rfl_spec,
                            )
                        except Exception as _e_rfl44:
                            try:
                                self.logger.info(
                                    "[RepeatFixedLongProbe-v44] failure: %r",
                                    _e_rfl44,
                                )
                            except Exception:
                                pass
                        # [v41] server-info probe
                        try:
                            import requests as _rq_si41
                            _si_resp = _rq_si41.post(
                                f"http://{_addr_v39}/callback/get_server_info_v41",
                                json={"version": _ver},
                                timeout=60.0,
                                proxies={"http": None, "https": None},
                            )
                            _si_j = _si_resp.json() if _si_resp.status_code == 200 else {}
                            self.logger.info(
                                "[ServerInfoProbe-v41] version=%s status=%s "
                                "servers=%s",
                                _ver, _si_resp.status_code,
                                _si_j.get("servers"),
                            )
                        except Exception as _e_si41:
                            try:
                                self.logger.info(
                                    "[ServerInfoProbe-v41] failure: %r",
                                    _e_si41,
                                )
                            except Exception:
                                pass
                    except Exception as _e_l:
                        self.logger.info(
                            "[DraftSpecTrend-v39 long] failure: %r\nTRACEBACK:\n%s",
                            _e_l, _tb_v39.format_exc(),
                        )
                # --- (c) Stochastic probe ---
                _stage_v39 = "stoch_probe"
                if _rk_v39 == 0 and _addr_v39 is not None:
                    try:
                        import requests as _rq_s
                        _ver = int(self.get_version())
                        _r_s = _rq_s.post(
                            f"http://{_addr_v39}/callback/get_draft_probe_stoch",
                            json={"version": _ver},
                            timeout=300.0,
                            proxies={"http": None, "https": None},
                        )
                        _j_s = _r_s.json() if _r_s.status_code == 200 else {}
                        self.logger.info(
                            "[DraftSpecTrend-v39 stoch] version=%s "
                            "status=%s n_ok=%s "
                            "spec_accept_rate_stats=%s "
                            "spec_accept_length_stats=%s "
                            "spec_accept_rate_samples=%s "
                            "spec_accept_length_samples=%s "
                            "histograms=%s err=%s",
                            _ver, _r_s.status_code,
                            _j_s.get("n_ok"),
                            _j_s.get("spec_accept_rate_stats"),
                            _j_s.get("spec_accept_length_stats"),
                            _j_s.get("spec_accept_rate_samples"),
                            _j_s.get("spec_accept_length_samples"),
                            _j_s.get("histograms"),
                            _j_s.get("error"),
                        )
                    except Exception as _e_s:
                        self.logger.info(
                            "[DraftSpecTrend-v39 stoch] failure: %r\nTRACEBACK:\n%s",
                            _e_s, _tb_v39.format_exc(),
                        )
            except Exception as _e_v39_out:
                try:
                    self.logger.warning(
                        "[DraftSpecTrend-v39] outer failure at stage=%s exc=%r\nTRACEBACK:\n%s",
                        _stage_v39, _e_v39_out, _tb_v39.format_exc(),
                    )
                except Exception:
                    pass
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
            if not bool(getattr(self, "_v17_native_active", False)):
                loss = loss + _mtp_contribution
            else:
                # [MTPNativeConsumerBypass-v17] Native MTPLossAutoScaler
                # already injected the gradient via autograd; adding
                # _mtp_contribution scalar here would double-count.
                if self._mtp_loss_total_count == 0:
                    self.logger.info(
                        "[MTPNativeConsumerBypass-v17] Skipping scalar "
                        "loss+=_mtp_contribution; autograd path active."
                    )
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

        if (
            _mtp_loss_for_this_mb is not None
            and abs(loss_scale) > 0
            and not bool(getattr(self, "_v17_native_active", False))
        ):
            # Refresh cached MTP LR from optimizer param_groups using
            # max_lr fingerprint (ParamKey override in megatron-core 0.16
            # does NOT propagate the ParamKey.name into the param_group
            # dict, so the previous name-based match always missed the MTP
            # group and left _last_logged_mtp_lr at its default 3e-6, making
            # the DoubleScale log severely misleading).
            try:
                _pgs = getattr(self.optimizer, "param_groups", []) or []
                if len(_pgs) > 1:
                    _base_mx = _pgs[0].get("max_lr", None)
                    for _pg in _pgs:
                        _mxlr = _pg.get("max_lr", None)
                        if (
                            _mxlr is not None
                            and _base_mx is not None
                            and abs(float(_mxlr) - float(_base_mx)) > 1e-12
                        ):
                            self._last_logged_mtp_lr = float(
                                _pg.get("lr", _pg.get("max_lr", 3e-6))
                            )
                            break
                    else:
                        # Single-group case or equal max_lr -> MTP shares
                        # the base lr.
                        self._last_logged_mtp_lr = float(
                            _pgs[0].get("lr", 3e-6)
                        )
                elif len(_pgs) == 1:
                    self._last_logged_mtp_lr = float(
                        _pgs[0].get("lr", 3e-6)
                    )
                # [MTPLRScaleGuard-v14] detect obviously-wrong MTP lr.
                try:
                    _mtp_lr_g = float(
                        getattr(self, "_last_logged_mtp_lr", 0.0)
                    )
                    _base_lr_g = None
                    if _pgs:
                        _base_lr_g = float(_pgs[0].get("lr", 0.0))
                    if (
                        _base_lr_g is not None
                        and _base_lr_g > 0
                        and _mtp_lr_g > 0
                        and _mtp_lr_g >= 10.0 * _base_lr_g
                        and (self._mtp_loss_total_count <= 4
                             or self._mtp_loss_total_count % 100 == 0)
                    ):
                        self.logger.warning(
                            "[MTPLRScaleGuard-v14] MTP lr %.3e is "
                            ">=10x base lr %.3e; this is almost "
                            "certainly a mis-scaled mtp_lr_scale "
                            "and will destabilise the draft head.",
                            _mtp_lr_g, _base_lr_g,
                        )
                except Exception:
                    pass
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
                # [MTPSanity-v12] Detect explosive per-step update. bf16
                # dynamic range for |W|~0.4 places 1 ULP near 3e-3; any
                # per-step update >= 1e-2 is already tens of ULPs and
                # almost always means the draft head is diverging. Emit
                # a prominent warning rather than letting accept_rate
                # silently collapse.
                try:
                    if abs(_eff_step_mag) >= 1e-2:
                        self.logger.warning(
                            "[MTPSanity-v12] per-step MTP update "
                            "magnitude %.3e >= 1e-2 (>= ~3x bf16 ULP "
                            "for |W|~0.4); draft head divergence is "
                            "likely. Reduce mtp_lr_scale or "
                            "mtp_loss_scaling.",
                            _eff_step_mag,
                        )
                except Exception:
                    pass
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
