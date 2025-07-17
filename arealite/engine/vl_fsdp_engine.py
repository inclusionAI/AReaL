import os
import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import transformers
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from transformers import (
    AutoModelForImageTextToText,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from arealite.api.cli_args import TrainEngineConfig
from arealite.api.engine_api import (
    FinetuneSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from arealite.utils.data import (
    MicroBatchList,
    amend_position_ids,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    pad_mb_list,
    reorder_list,
    split_padded_tensor_dict_into_mb_list,
    unpack_sequence,
    unsqueeze_mb_list,
)
from arealite.utils.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    create_fsdp_device_mesh,
    get_cosine_schedule_with_warmup,
)
from realhf.base import  logging, name_resolve, names, pkg_version
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
from arealite.engine.fsdp_engine import FSDPEngine


logger = logging.getLogger("FSDPEngine")


class VL_FSDPEngine(FSDPEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)
        self.processor=None
        self.tokenizer = None
        


    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
        # Initialize distributed enviroments and load model.
        assert addr is None, "FSDPEngine does not support remote initialization."

        assert pkg_version.is_version_greater_or_equal(
            "torch", "2.4.0"
        ), f"arealite only supports FSDP2, which requires torch>=2.4.0"

        """Initialize distributed communication and model."""
        if not dist.is_initialized():
            # TODO: Handle the condition when WORLD_SIZE and RANK is not set in launcher
            dist.init_process_group(backend="nccl")

        # TODO: Handle the condition when LOCAL_RANK is not set in launcher
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        dtype = torch.bfloat16 
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(self.config.path)
        with torch.device("cuda"):
            # initialize scratch model from config
            model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=self.config.path,
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation=self.config.attn_impl,
            )

        # Simple auto wrap policy
        self.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        self.device_mesh = create_fsdp_device_mesh(self.world_size, self.world_size)
        # sharding_strategy = ShardingStrategy.FULL_SHARD
        self.cpu_offload = (
            CPUOffloadPolicy() if self.config.fsdp.offload_params else None
        )

        fsdp_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": self.mixed_precision_policy,
            "offload_policy": self.cpu_offload,
            "reshard_after_forward": True,
        }

        # Wrap with FSDP2
        apply_fsdp2(model, fsdp_kwargs, self.config.fsdp.wrap_policy)
        self.model = model
        
        # Set up optimizer
        if self.optimizer_config is not None:
            assert (
                self.optimizer_config.type == "adam"
            ), "Only AdamW optimizer is supported in this engine."
            lr = self.optimizer_config.lr
            weight_decay = self.optimizer_config.weight_decay
            beta1 = self.optimizer_config.beta1
            beta2 = self.optimizer_config.beta2
            eps = self.optimizer_config.eps

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
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

        self.initialized = True
    
    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._save_model_to_hf(meta.path, meta.tokenizer,meta.processor)
        elif meta.weight_format == "dcp":
            # TODO: implement DCP save/load for FSDP
            raise NotImplementedError("DCP format saving is not implemented yet. ")
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim:
            self._save_optimizer_state(meta.path)

    def _save_model_to_hf(
        self, path: str, tokenizer: Optional[transformers.PreTrainedTokenizerFast], processor: Optional[transformers.AutoProcessor]
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
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if processor is not None:
                processor.save_pretrained(path)
        dist.barrier()

    def upload_weights(self, meta: WeightUpdateMeta):
        if meta.type == "nccl":
            if not self.weight_update_group_initialized:
                self._init_distributed_weight_update(meta)
            self._update_weights_from_distributed()
        elif meta.type == "disk":
            self._save_model_to_hf(meta.path, self.tokenizer, self.processor)
            # dist.barrier() are called when _save_model_to_hf finished
            if dist.get_rank() == 0:
                update_name = names.update_weights_from_disk(
                    self.config.experiment_name,
                    self.config.trial_name,
                    meta.model_version,
                )
                name_resolve.add(update_name, str(time.time_ns()), keepalive_ttl=120)
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def _prepare_mb_list(self, input_: TensorDict) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_ and "pixel_values" in input_ and "image_grid_thw" in input_, \
            "Input TensorDict must contain 'attention_mask', 'input_ids', 'pixel_values', and 'image_grid_thw' keys."
        if isinstance(input_, dict):
            input_ = TensorDict(input_, batch_size=[input_["input_ids"].shape[0]])
        input_ = amend_position_ids(input_)
        packed_input = pack_tensor_dict(input_)
        mb_list = split_padded_tensor_dict_into_mb_list(
            packed_input,
            self.config.mb_spec,
        )
        mb_list = pad_mb_list(mb_list, pad_value=0.0)
        # NOTE: We unsqueeze here because huggingface transformer models requires
        # packed input to be of shape [1, total_seqlen].
        mb_list = unsqueeze_mb_list(mb_list)
        return mb_list

        