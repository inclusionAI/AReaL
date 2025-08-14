import os
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from tensordict import TensorDict
from transformers import AutoConfig

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import FinetuneSpec, TrainEngine
from areal.api.io_struct import ParamSpec, SaveLoadMeta, WeightUpdateMeta
from areal.experimental.model.registry import hf_to_mcore_config, make_mcore_model
from realhf.base import constants, logging

logger = logging.getLogger("MegatronEngine")


class MegatronEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.hf_config = None
        self.tf_config = None
        self.model = None
        self.dtype = getattr(torch, self.config.dtype)
        self.device = None

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

        self.hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.tf_config = hf_to_mcore_config(hf_config=self.hf_config, dtype=self.dtype)
        # initialize mcore GPTModel
        self.model = make_mcore_model(
            hf_config=self.hf_config,
            tf_config=self.tf_config,
        )
        self.model.to(self.device)

    @property
    def parallelism_group(self) -> dist.ProcessGroup:
        raise NotImplementedError()

    def destroy(self):
        raise NotImplementedError()

    def train(self, mode: bool = True):
        raise NotImplementedError()

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
        raise NotImplementedError()

    def load(self, meta: SaveLoadMeta):
        raise NotImplementedError()

    def step_lr_scheduler(self):
        raise NotImplementedError()

    def train_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> Dict[str, float]:
        raise NotImplementedError()

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
        raise NotImplementedError()
