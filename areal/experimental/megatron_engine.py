import os
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from tensordict import TensorDict

from areal.api.engine_api import FinetuneSpec, TrainEngine
from areal.api.io_struct import ParamSpec, SaveLoadMeta, WeightUpdateMeta
from areal.experimental.api.cli_args import (
    ExperimentalTrainEngineConfig as TrainEngineConfig,
)
from areal.experimental.model.registry import (
    load_from_hf,
    make_hf_and_mcore_config,
    make_mcore_model,
    save_to_hf,
)
from areal.utils.data import amend_position_ids
from areal.utils.model import disable_dropout_in_model
from realhf.base import constants, logging, pkg_version

USE_MBRIDGE = False
if pkg_version.is_available("mbridge"):
    import mbridge

    USE_MBRIDGE = True
else:
    USE_MBRIDGE = False


logger = logging.getLogger("MegatronEngine")


class MegatronEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.hf_config = None
        self.tf_config = None
        self.model = None
        self.dtype = getattr(torch, self.config.dtype)
        self.device = None
        self.optimizer_config = config.optimizer
        self.mcore_config = config.megatron
        self.bridge = None

    def create_optimizer(self, ft_spec: FinetuneSpec):
        if self.optimizer_config is None:
            return
        assert self.model is not None

        assert (
            self.optimizer_config.type == "adam"
        ), "Only AdamW optimizer is supported in this engine."

        # Make megatron optimizer config, from legacy MegatronEngine
        # TODO: add DDP options
        # TODO: check if there is more options in mcore v0.13.1
        mcore_opt_config = MCoreOptimizerConfig(
            optimizer="adam",
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
            bf16=self.dtype is torch.bfloat16,
            fp16=self.dtype is torch.float16,
            adam_beta1=self.optimizer_config.beta1,
            adam_beta2=self.optimizer_config.beta2,
            adam_eps=self.optimizer_config.eps,
            use_distributed_optimizer=True,
            params_dtype=self.dtype,
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

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        # Required by NCCL weight update group for SGLang
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        if not dist.is_initialized():
            # TODO: Handle the condition when WORLD_SIZE and RANK is not set in launcher
            # NOTE: device_id **SHOULD NOT** be passed into init_process_group,
            # otherwise initializing the NCCL weight update group will be wrong!
            dist.init_process_group(
                backend="nccl",
                timeout=constants.NCCL_DEFAULT_TIMEOUT,
            )
            self.own_global_group = True
        self._parallelism_group = dist.new_group()

        # TODO: initialize parallelism
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=1,
            hierarchical_context_parallel_sizes=None,
            expert_model_parallel_size=1,
            num_distributed_optimizer_instances=1,
            expert_tensor_parallel_size=1,
            nccl_communicator_config_path=None,
            distributed_timeout_minutes=30,  # ignored
            order="tp-cp-ep-dp-pp",
            encoder_tensor_model_parallel_size=0,
            encoder_pipeline_model_parallel_size=0,
            get_embedding_ranks=None,  # use megatron default embedding ranks
            get_position_embedding_ranks=None,  # use megatron default position embedding ranks
        )
        # TODO: Fix rng seed
        tensor_parallel.model_parallel_cuda_manual_seed(0)

        if USE_MBRIDGE:
            self.bridge = mbridge.AutoBridge.from_pretrained(self.config.path)
            logger.info(
                "Using mbridge to create models and hf model save/load in MegatronEngine."
            )

        self.hf_config, self.tf_config = make_hf_and_mcore_config(
            self.config.path, dtype=self.dtype, bridge=self.bridge
        )
        # initialize mcore (DDP Wrapped) GPTModel
        with torch.device("cuda"):
            self.model = make_mcore_model(
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                mcore_config=self.mcore_config,
                bridge=self.bridge
            )
        if self.config.disable_dropout:
            disable_dropout_in_model(self.model)
        self.create_optimizer(ft_spec)

    @property
    def parallelism_group(self) -> dist.ProcessGroup:
        raise NotImplementedError()

    def destroy(self):
        raise NotImplementedError()

    def train(self, mode: bool = True):
        self.model.train(mode=mode)
        return self

    def upload_weights(self, meta: WeightUpdateMeta):
        raise NotImplementedError()

    def get_param_specs(
        self, weight_chunked_mem_mb: int = 1024
    ) -> List[List[ParamSpec]]:
        raise NotImplementedError()

    def set_version(self, version: int):
        raise NotImplementedError()

    def get_version(self) -> int:
        raise NotImplementedError()

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            assert (
                not meta.with_optim
            ), "HF format does not support optimizer state saving, please use DCP format instead."
            self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
        elif meta.weight_format == "dcp":
            # TODO: implement DCP save/load for FSDP
            raise NotImplementedError("DCP format saving is not implemented yet. ")
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def _save_model_to_hf(
        self, path: str, tokenizer: Any | None = None, processor: Any | None = None
    ):
        assert self.model is not None, "Model is not initialized."
        os.makedirs(path, exist_ok=True)

        # Save model weights
        save_to_hf(
            hf_config=self.hf_config,
            save_path=path,
            model=self.model,
            bridge=self.bridge
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
            # TODO: implement DCP save/load for FSDP
            raise NotImplementedError("DCP format loading is not implemented yet. ")
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

    def _load_model_from_hf(self, path: str):
        assert self.model is not None, "Model is not initialized."
        load_from_hf(
            hf_config=self.hf_config,
            load_path=path,
            model=self.model,
            bridge=self.bridge
        )

    def step_lr_scheduler(self):
        assert self.lr_scheduler is not None, "LR Scheduler is not initialized."
        self.lr_scheduler.step(1)

    def train_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> Dict[str, float]:
        # TODO: simple training for testing, no parallelism and packing
        assert self.model is not None, "Model is not initialized."
        assert self.optimizer is not None, "Optimizer is not initialized."
        input_ = amend_position_ids(input_)
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(**input_)
        loss = loss_fn(output, input_)
        loss.backward()

        # Update optimizer
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "update_successful": update_successful,
        }

    @torch.no_grad()
    def eval_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> torch.Tensor | None:
        raise NotImplementedError()

    @torch.no_grad()
    def forward(
        self,
        input_: TensorDict,
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, TensorDict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        # TODO: simple forward for testing, no parallelism and packing
        assert self.model is not None, "Model is not initialized."
        input_ = amend_position_ids(input_)
        return self.model(**input_)
