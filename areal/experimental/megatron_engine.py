import dataclasses
import functools
import gc
import os
from concurrent.futures import Future
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import mbridge
import torch
import torch.distributed as dist
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
from transformers import PretrainedConfig

from areal.api.alloc_mode import MegatronParallelStrategy, ParallelStrategy
from areal.api.cli_args import MicroBatchSpec
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import FinetuneSpec, ParamSpec, SaveLoadMeta, WeightUpdateMeta
from areal.experimental.api.cli_args import (
    ExperimentalTrainEngineConfig as TrainEngineConfig,
)
from areal.experimental.model.hf_load import load_weights_from_hf_with_mbridge_fast
from areal.experimental.model.hf_save import save_weights_to_hf_with_mbridge_fast
from areal.experimental.model.registry import make_hf_and_mcore_config, make_mcore_model
from areal.experimental.utils.mcore.determinisitc import set_deterministic_algorithms
from areal.experimental.utils.mcore.packed_context_parallel import (
    packed_context_parallel_forward,
)
from areal.experimental.utils.megatron import (
    all_gather_param,
    convert_to_hf,
    get_named_parameters,
    remove_padding,
)
from areal.experimental.utils.megatron_checkpointer import MegatronCheckpointManager
from areal.platforms import current_platform
from areal.utils import logging, name_resolve, names
from areal.utils.data import (
    MicroBatchList,
    amend_position_ids,
    broadcast_tensor,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    pad_mb_list,
    reorder_list,
    split_padded_tensor_dict_into_mb_list,
    unpack_sequence,
    unpad_logits,
)
from areal.utils.distributed import init_custom_process_group
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.lock import DistributedLock
from areal.utils.model import disable_dropout_in_model
from areal.utils.nccl import NCCL_DEFAULT_TIMEOUT


class MegatronEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.hf_config: PretrainedConfig
        self.tf_config: TransformerConfig
        self.model = None
        self.dtype = getattr(torch, self.config.dtype)
        self.device = None
        self.optimizer_config = config.optimizer
        self.mcore_config = config.megatron
        self.parallel_strategy = None
        self.optimizer = None
        self.lr_scheduler = None
        self.bridge = None
        self.process_group_initialized = False
        self.rollout_engine: InferenceEngine | None = None
        self.weight_update_group_initialized: bool = False
        self.weight_update_group_name: str
        self._version: int = 0
        self.rank = None
        self.is_pp_head: bool
        self.world_size = None
        self.rank_generator = None
        self.checkpointer = None
        self.seed = 0

    def initialize(
        self,
        addr: str | None,
        ft_spec: FinetuneSpec,
        parallel_strategy: ParallelStrategy,
        seed: int = 0,
    ):
        # TODO: add parallel_strategy & seed in engine api when moving out of experimental
        if self.parallel_strategy is None:
            self.parallel_strategy = self._make_parallel_strategy(parallel_strategy)
        self.seed = seed

        assert addr is None, "FSDPEngine does not support remote initialization."
        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
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

        self.tokenizer = load_hf_tokenizer(self.config.path)
        self.bridge = mbridge.AutoBridge.from_pretrained(self.config.path)
        self.bridge.dtype = self.dtype
        # Set gradient checkpointing options
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

        self.hf_config, self.tf_config = make_hf_and_mcore_config(
            self.config.path, dtype=self.dtype, bridge=self.bridge
        )
        # initialize mcore (DDP Wrapped) GPTModel
        with self.device:
            self.model = make_mcore_model(
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                mcore_config=self.mcore_config,
                bridge=self.bridge,
            )
            self._load_model_from_hf(self.config.path)

        if self.config.disable_dropout:
            disable_dropout_in_model(self.model)

        model_config = get_model_config(self.model)
        # NOTE: It is recommended to set this option to True for RL training on MoE models for stability.
        if self.mcore_config.use_deterministic_algorithms:
            set_deterministic_algorithms(model_config)

        if isinstance(self.model, DDP) and self.mcore_config.ddp.overlap_grad_reduce:
            model_config.no_sync_func = self.model.no_sync
        if (
            self.mcore_config.ddp.overlap_param_gather
            and self.mcore_config.ddp.align_param_gather
        ):
            model_config.param_sync_func = self.model.start_param_sync
        model_config.finalize_model_grads_func = finalize_model_grads
        self.create_optimizer(ft_spec)

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> MegatronParallelStrategy:
        return MegatronParallelStrategy(
            use_sequence_parallel=parallel_strategy.tensor_parallel_size > 1,
            **dataclasses.asdict(parallel_strategy),
        )

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()
        assert not dist.is_initialized()
        # TODO: Change engine_api.py and FSDPEngine API to seperate create_process_group
        # from engine initialize when moving out of experimental.
        self.parallel_strategy = self._make_parallel_strategy(parallel_strategy)
        # Required by NCCL weight update group for SGLang
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        # TODO: Handle the condition when WORLD_SIZE and RANK is not set in launcher
        # NOTE: device_id **SHOULD NOT** be passed into init_process_group,
        # otherwise initializing the NCCL weight update group will be wrong!
        dist.init_process_group(
            backend="nccl",
            timeout=NCCL_DEFAULT_TIMEOUT,
        )
        # Initialize Megatron parallel states
        # NOTE: we assume all MegatronEngine has the same parallel strategy.
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.parallel_strategy.tensor_parallel_size,
            pipeline_model_parallel_size=self.parallel_strategy.pipeline_parallel_size,
            virtual_pipeline_model_parallel_size=self.parallel_strategy.virtual_pipeline_parallel_size,
            use_sharp=False,
            order="tp-cp-ep-dp-pp",
            context_parallel_size=self.parallel_strategy.context_parallel_size,
            expert_model_parallel_size=self.parallel_strategy.expert_parallel_size,
            expert_tensor_parallel_size=self.parallel_strategy.expert_tensor_parallel_size,
            distributed_timeout_minutes=int(NCCL_DEFAULT_TIMEOUT.seconds / 60),
        )
        # Set megatron model parallel seed
        tensor_parallel.model_parallel_cuda_manual_seed(self.seed)

        self.logger = logging.getLogger(f"[Megatron Engine Rank {dist.get_rank()}]")
        self._parallelism_group = dist.new_group()
        self._context_and_model_parallel_group = None
        self._init_context_and_model_parallel_group()
        self.process_group_initialized = True

    def _init_context_and_model_parallel_group(self):
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
                timeout=NCCL_DEFAULT_TIMEOUT,
                pg_options=mpu.get_nccl_options("tp-cp-pp", {}),
                group_desc="CONTEXT_AND_MODEL_PARALLEL_GROUP",
            )
            if dp_rank == mpu.get_data_parallel_rank():
                self._context_and_model_parallel_group = group

    def create_optimizer(self, ft_spec: FinetuneSpec):
        if self.optimizer_config is None:
            return
        assert self.model is not None

        assert self.optimizer_config.type in [
            "adam",
            "sgd",
        ], "Only AdamW/sgd optimizer is supported in this engine."
        if self.optimizer_config.type == "sgd":
            self.logger.warning(
                f"Using the 'sgd' optimizer with Megatron may be less stable. Consider using the 'adam' (AdamW) optimizer for improved stability."
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
            use_distributed_optimizer=self.mcore_config.ddp.use_distributed_optimizer,
            params_dtype=self.dtype,
            clip_grad=self.optimizer_config.gradient_clipping,
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

        self.optimizer = get_megatron_optimizer(
            mcore_opt_config,
            [self.model],
            no_weight_decay_cond=lambda n, p: any(
                k in n for k in ["bias", "ln.weight", "ln_f.weight"]
            ),
            scale_lr_cond=None,
            lr_mult=1.0,
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

        self.checkpointer = MegatronCheckpointManager(
            model=[self.model],
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.mcore_config.ddp.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.mcore_config.use_checkpoint_opt_param_scheduler,
            async_save=self.mcore_config.async_save,
        )

    @property
    def parallelism_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return self._parallelism_group

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return self._context_and_model_parallel_group

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

    def destroy(self):
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        current_platform.empty_cache()
        gc.collect()
        dist.destroy_process_group(self.parallelism_group)
        dist.destroy_process_group(self.context_and_model_parallel_group)

    def destroy_process_groups(self):
        # Should be explicitly called after experiments.
        assert dist.is_initialized()
        mpu.destroy_model_parallel()
        dist.destroy_process_group()
        self.process_group_initialized = False

    def train(self, mode: bool = True):
        self.model.train(mode=mode)
        return self

    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        converted_named_tensors: List[Tuple[str, nn.Parameter | torch.Tensor]],
    ):
        # Early exit when chunk size is relatively small
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

        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        for _, param in converted_named_tensors:
            handles.append(
                dist.broadcast(
                    param.data, 0, group=self.weight_update_group, async_op=True
                )
            )
        for handle in handles:
            handle.wait()

        fut.result()

        converted_named_tensors.clear()

        self.engine_lock.release()

    def _impl_update_weight_from_distributed(
        self,
        meta: WeightUpdateMeta,
        name: str,
        param: nn.Parameter | torch.Tensor,
        converted_named_tensors: List[Tuple[str, nn.Parameter | torch.Tensor]],
        buffer_size: int,
        weight_chunked_mem_size: int,
    ) -> int:
        param = all_gather_param(name, param)
        param = remove_padding(name, param, self.hf_config.vocab_size)
        if not self.is_pipeline_parallel_head():
            return buffer_size

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > weight_chunked_mem_size:
            self._update_bucket_weights_from_distributed(meta, converted_named_tensors)
            buffer_size = 0

        converted_named_tensors.extend(
            convert_to_hf(self.tf_config, self.hf_config.model_type, name, param)
        )
        buffer_size += param_size
        return buffer_size

    def _update_bucket_expert_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: List[Tuple[str, nn.Parameter | torch.Tensor]],
    ):
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
        all_names: List[List[str]] = [None] * world_size
        dist.all_gather_object(all_names, names, group=group)

        for rank_names in all_names:
            assert len(named_tensors) == len(
                rank_names
            ), f"mismatch names length: {len(named_tensors)} != {len(rank_names)}"

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
                convert_to_hf(self.tf_config, self.hf_config.model_type, name, param)
            )
        return self._update_bucket_weights_from_distributed(meta, converted_hf_tensors)

    def _impl_update_expert_weight_from_distributed(
        self,
        meta: WeightUpdateMeta,
        name: str,
        param: nn.Parameter | torch.Tensor,
        named_tensors: List[Tuple[str, nn.Parameter | torch.Tensor]],
        buffer_size: int,
        weight_chunked_mem_size: int,
    ) -> int:
        param = all_gather_param(name, param)
        param = remove_padding(name, param, self.hf_config.vocab_size)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > weight_chunked_mem_size:
            self._update_bucket_expert_weights_from_distributed(meta, named_tensors)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta):
        assert meta.type == current_platform.communication_backend

        # NOTE: Processes launched with torchrun will set the following env var to True,
        # which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        if self.is_pipeline_parallel_head():
            assert meta.alloc_mode is not None

            fut = self.rollout_engine.init_weights_update_group(meta)

            self.logger.info(
                f"Initializing weight update group: type={meta.type} "
                f"init_method=tcp://{meta.nccl_master_address}:{meta.nccl_master_port} "
                f"group={self.weight_update_group_name}"
            )
            self.weight_update_group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=meta.alloc_mode.gen.world_size + 1,
                init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
                rank=0,
                group_name=self.weight_update_group_name,
                timeout=NCCL_DEFAULT_TIMEOUT,
            )

            fut.result()

    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()

        dist.barrier(device_ids=[self.device.index])

        num_moe_experts = self.tf_config.num_moe_experts
        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

        buffer_size = 0
        converted_named_tensors = []

        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." in name:
                continue
            buffer_size = self._impl_update_weight_from_distributed(
                meta,
                name,
                param,
                converted_named_tensors,
                buffer_size,
                weight_chunked_mem_size,
            )

        # Only pipeline parallel heads CAN contain named tensors here
        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(meta, converted_named_tensors)

        dist.barrier(device_ids=[self.device.index])

        buffer_size = 0
        named_tensors = []

        for name, param in get_named_parameters(self.model, num_moe_experts):
            if ".experts." not in name:
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
            # This function will early return if not pipeline parallel head
            self._update_bucket_expert_weights_from_distributed(meta, named_tensors)

        dist.barrier(device_ids=[self.device.index])

        if dist.get_rank() == 0:
            self.rollout_engine.continue_generation()

        dist.barrier(device_ids=[self.device.index])
        current_platform.synchronize()

    def _update_weights_from_disk(self, meta: WeightUpdateMeta):
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

        dist.barrier(device_ids=[self.device.index])
        current_platform.synchronize()

    def update_weights(self, meta: WeightUpdateMeta):
        if meta.type == current_platform.communication_backend:
            assert self.weight_update_group_initialized
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

        if (
            meta.type == current_platform.communication_backend
            and not self.weight_update_group_initialized
        ):
            self._init_weight_update_from_distributed(meta)
            self.weight_update_group_initialized = True

        dist.barrier(device_ids=[self.device.index])
        current_platform.synchronize()

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            assert (
                not meta.with_optim
            ), "HF format does not support optimizer state saving, please use DCP format instead."
            self._save_model_to_hf(
                meta.path,
                tokenizer=meta.tokenizer,
                processor=meta.processor,
                base_model_path=meta.base_model_path,
            )
        elif meta.weight_format == "dcp":
            self.checkpointer.save_checkpoint(meta.path, with_optimizer=meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: Any | None = None,
        processor: Any | None = None,
        base_model_path: str | None = None,
    ):
        assert self.model is not None, "Model is not initialized."
        os.makedirs(path, exist_ok=True)

        save_weights_to_hf_with_mbridge_fast(
            bridge=self.bridge,
            models=[self.model],
            weights_path=path,
            base_model_path=base_model_path,
            max_shard_size_byte=int(3e9),
            max_workers=None,
        )

        if dist.get_rank() == 0:
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if processor is not None:
                processor.save_pretrained(path)
        dist.barrier(device_ids=[self.device.index])

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            assert (
                not meta.with_optim
            ), "HF format does not support optimizer state loading, please use DCP format instead."
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            self.checkpointer.load_checkpoint(meta.path, with_optimizer=meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def _load_model_from_hf(self, path: str):
        assert self.model is not None, "Model is not initialized."
        load_weights_from_hf_with_mbridge_fast(
            bridge=self.bridge,
            models=[self.model],
            weights_path=path,
            max_workers=None,
        )

    def prepare_mb_list(self, input_: Dict[str, Any]) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        input_ = amend_position_ids(input_)
        # Parallel sizes
        pp_size = self.parallel_strategy.pipeline_parallel_size
        cp_size = self.parallel_strategy.context_parallel_size
        tp_size = self.parallel_strategy.tensor_parallel_size
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
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
            align_sequences=True,
            align_to_multiple_of=align_to_multiple_of,
        )
        self.logger.info(
            f"Microbatch #tokens (rank {dist.get_rank()}): {mb_list.group_lens}, aligned to: {mb_list.align_to_lengths}, "
            f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}."
        )
        # FIXME: the resulting max_seqlen is a tensor rather than an integer
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

    def step_lr_scheduler(self):
        assert self.lr_scheduler is not None, "LR Scheduler is not initialized."
        self.lr_scheduler.step(1)

    def train_batch(
        self,
        input_: Dict[str, Any],
        loss_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[Dict[str, Any]], torch.Tensor],
    ) -> Dict[str, float]:
        assert self.model is not None, "Model is not initialized."
        assert self.optimizer is not None, "Optimizer is not initialized."
        self.optimizer.zero_grad()
        self.model.zero_grad_buffer()
        # Assume input_ is identical across context and model parallel group
        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)

        total_loss_weight = (
            torch.stack([loss_weight_fn(mb) for mb in mb_list.padded_mbs])
            .sum()
            .detach()
            .clone()
            .to(dtype=torch.float32)
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight, group=mpu.get_data_parallel_group())
        max_total_len = max(m["cu_seqlens"][-1].item() for m in mb_list.padded_mbs)
        micro_batch_generator = iter(mb_list.padded_mbs)
        forward_step_count = 0

        def forward_step(batch_iter, model):
            nonlocal forward_step_count
            batch = next(batch_iter)
            padding_length = mb_list.padding_lengths[forward_step_count]
            orig_input = mb_list.mbs[forward_step_count]
            cu_seqlens = batch["cu_seqlens"]
            old_cu_seqlens = mb_list.old_cu_seqlens_list[forward_step_count]

            forward_step_count += 1
            output = packed_context_parallel_forward(model, batch)

            if mpu.is_pipeline_last_stage():
                output = unpad_logits(
                    output,
                    padding_length=padding_length,
                    cu_seqlens=cu_seqlens,
                    old_cu_seqlens=old_cu_seqlens,
                )

            def _scaled_loss_fn(input_, output):
                loss = loss_fn(output, input_)
                loss_scale = loss_weight_fn(input_) / total_loss_weight
                # Megatron will average gradients across DP ranks.
                # If we normalize loss across micro batches of all DP ranks,
                # we should revert the effect of gradient averaging in megatron
                # to make sure loss from each token is scaled properly.
                loss_scale *= mpu.get_data_parallel_world_size()
                loss_scale *= self.optimizer.get_loss_scale().item()
                loss *= loss_scale
                return loss, {}

            return output, functools.partial(_scaled_loss_fn, orig_input)

        forward_backward_func = get_forward_backward_func()
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=micro_batch_generator,
            model=self.model,
            num_microbatches=len(mb_list.padded_mbs),
            seq_length=max_total_len,  # no use when input_shapes was set
            micro_batch_size=1,  # no use when input_shapes was set
            forward_only=False,
        )
        update_successful, grad_norm, _ = self.optimizer.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict[str, Any],
        loss_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[Dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        assert self.model is not None, "Model is not initialized."
        # Assume input_ is identical across context and model parallel group
        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)

        total_loss_weight = (
            torch.stack([loss_weight_fn(mb) for mb in mb_list.padded_mbs])
            .sum()
            .detach()
            .clone()
            .to(dtype=torch.float32)
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight, group=mpu.get_data_parallel_group())
        max_total_len = max(m["cu_seqlens"][-1].item() for m in mb_list.padded_mbs)
        micro_batch_generator = iter(mb_list.padded_mbs)
        forward_step_count = 0

        def forward_step(batch_iter, model):
            nonlocal forward_step_count
            batch = next(batch_iter)
            padding_length = mb_list.padding_lengths[forward_step_count]
            orig_input = mb_list.mbs[forward_step_count]
            cu_seqlens = batch["cu_seqlens"]
            old_cu_seqlens = mb_list.old_cu_seqlens_list[forward_step_count]

            forward_step_count += 1
            output = packed_context_parallel_forward(model, batch)

            if mpu.is_pipeline_last_stage():
                output = unpad_logits(
                    output,
                    padding_length=padding_length,
                    cu_seqlens=cu_seqlens,
                    old_cu_seqlens=old_cu_seqlens,
                )

            def _scaled_loss_fn(input_, output):
                # NOTE: Do not need to record loss here, will be
                # automatically recorded by stats_tracker
                loss = loss_fn(output, input_)
                loss_scale = loss_weight_fn(input_) / total_loss_weight
                # eval_batch does not run backward, the grad will not be averaged over DP group
                # so we shouldn't multiple dp_size in loss_scale
                loss *= loss_scale
                return loss, {}

            return output, functools.partial(_scaled_loss_fn, orig_input)

        forward_backward_func = get_forward_backward_func()
        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=micro_batch_generator,
            model=self.model,
            num_microbatches=len(mb_list.padded_mbs),
            seq_length=max_total_len,  # no use when input_shapes was set
            micro_batch_size=1,  # no use when input_shapes was set
            forward_only=True,
        )

        return None

    @torch.no_grad()
    def forward(
        self,
        input_: Dict[str, Any],
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, Dict[str, Any]], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        assert self.model is not None, "Model is not initialized."
        # Assume input_ is identical across context and model parallel group
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)

        # NOTE: Move tensors to correct device, since dist.broadcast_object_list does not
        # ensure the device of tensors in the object list
        cu_seqlens: torch.Tensor = cu_seqlens.to(self.device)
        mb_list: MicroBatchList = mb_list.to(self.device)

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()

        max_total_len = max(m["max_seqlen"] for m in mb_list.padded_mbs)
        micro_batch_generator = iter(mb_list.padded_mbs)
        forward_step_count = 0

        def forward_step(batch_iter, model):
            nonlocal forward_step_count
            batch = next(batch_iter)
            padding_length = mb_list.padding_lengths[forward_step_count]
            orig_input = mb_list.mbs[forward_step_count]
            cu_seqlens = batch["cu_seqlens"]
            old_cu_seqlens = mb_list.old_cu_seqlens_list[forward_step_count]

            forward_step_count += 1
            output = packed_context_parallel_forward(model, batch)

            if mpu.is_pipeline_last_stage():
                output = unpad_logits(
                    output,
                    padding_length=padding_length,
                    cu_seqlens=cu_seqlens,
                    old_cu_seqlens=old_cu_seqlens,
                )

            def _post_process_fn(input_, output):
                loss = torch.tensor(1.0, device=output.device)
                if post_hook is not None:
                    output = post_hook(output, input_)
                return loss, {"output": output}

            return output, functools.partial(_post_process_fn, orig_input)

        forward_backward_func = get_forward_backward_func()
        output_list = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=micro_batch_generator,
            model=self.model,
            num_microbatches=len(mb_list.padded_mbs),
            seq_length=max_total_len,  # max # tokens across all micro-batches
            micro_batch_size=1,  # should be 1 when using packed input
            forward_only=True,
        )

        result = None
        if mpu.is_pipeline_last_stage():
            res = aggregate_fn([o["output"] for o in output_list])
            output_seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
            unpacked = unpack_sequence(res, lens=output_seqlens, dim=0)
            reordered = reorder_list(unpacked, mb_list.backward_indices)
            result = pad_and_stack_tensors_along_first_dim(reordered)

        # Broadcast the shape of the result tensor
        result = broadcast_tensor(
            result,
            src_rank=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        return result
