import gc
import os
import time
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from tensordict import TensorDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import FinetuneSpec, TrainEngine
from areal.utils.data import (
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
from areal.utils.fsdp import get_cosine_schedule_with_warmup
from areal.utils.model import VALID_VISION_MODELS, disable_dropout_in_model
from realhf.api.core.data_api import load_hf_processor_and_tokenizer, load_hf_tokenizer
from realhf.base import constants, logging

logger = logging.getLogger("Base HF Engine")


class BaseHFEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.optimizer_config = config.optimizer

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.tokenizer: PreTrainedTokenizerFast
        self.processor: AutoProcessor | None = None
        # huggingface model config
        self.model_config: PretrainedConfig
        self._version: int = 0

        # initialization
        self.initialized = False
        self.own_global_group = False
        self._parallelism_group: dist.ProcessGroup
        self.weight_update_group_initialized = False

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.is_vision_model = self.model_config.model_type in VALID_VISION_MODELS

        self.world_size = int(os.environ["WORLD_SIZE"])

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    def train(self, mode: bool = True):
        assert self.model is not None
        self.model.train(mode=mode)
        return self

    @property
    def parallelism_group(self) -> dist.ProcessGroup:
        assert self.initialized
        return self._parallelism_group

    def create_process_group(self):
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

    def create_device_model(self):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
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
            with torch.device("cuda"):
                model = AutoModelForImageTextToText.from_pretrained(
                    pretrained_model_name_or_path=self.config.path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    attn_implementation=self.config.attn_impl,
                )
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)
        else:
            self.tokenizer = load_hf_tokenizer(self.config.path)
            tik = time.perf_counter()
            with torch.device("cuda"):
                if self.config.init_from_scratch:
                    # initialize scratch model from config
                    # NOTE: VLM cannot directly load state dict using this
                    # random initialized model, so otherwise we call
                    # from_pretrained rather than loading weights into this random model.
                    model = AutoModelForCausalLM.from_config(
                        self.model_config,
                        torch_dtype=dtype,
                        attn_implementation=self.config.attn_impl,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=self.config.path,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        attn_implementation=self.config.attn_impl,
                    )
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        logger.info(f"Model creation and loading time: {time.perf_counter() - tik}")
        self.model = model

    def create_optimizer(self, ft_spec: FinetuneSpec):
        if self.optimizer_config is None:
            return
        assert self.model is not None
        # Set up optimizer
        tik = time.perf_counter()
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
        logger.info(f"Create optimizer time: {time.perf_counter() - tik}")

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        del self.optimizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        dist.destroy_process_group(self.parallelism_group)
        if self.own_global_group:
            dist.destroy_process_group()
        self.initialized = False

    def save_optimizer_state(self, path: str):
        # Save FSDP sharded state dict on each rank
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        state_dict = self.optimizer.state_dict()
        torch.save(state_dict, shard_path)
        dist.barrier(device_ids=[self.device.index])

    def load_optimizer_state(self, path: str):
        # Load FSDP sharded state dict
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        optimizer_state_dict = torch.load(shard_path, weights_only=False)
        self.optimizer.load_state_dict(optimizer_state_dict)
        dist.barrier(device_ids=[self.device.index])

    def step_lr_scheduler(self):
        assert self.lr_scheduler is not None
        self.lr_scheduler.step()

    def prepare_mb_list(self, input_: TensorDict) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        if self.is_vision_model:
            assert (
                "pixel_values" in input_ and "image_grid_thw" in input_
            ), "For vision-language models, pixel_values and image_grid_thw must be present in input_"

        if isinstance(input_, dict):
            input_ = TensorDict(input_, batch_size=[input_["input_ids"].shape[0]])
        input_ = amend_position_ids(input_)

        mb_list = split_padded_tensor_dict_into_mb_list(input_, self.config.mb_spec)
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
        )
        logger.info(
            f"Microbatch #tokens (rank {dist.get_rank()}): {mb_list.group_lens}, "
            f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}"
        )
        # NOTE: We unsqueeze here because huggingface transformer models requires
        # packed input to be of shape [1, total_seqlen].
        mb_list = unsqueeze_mb_list(mb_list)

        # FIXME: the resulting max_seqlen is a tensor rather than an integer
        for mb in mb_list.mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
            mb["use_cache"] = False
        for mb in mb_list.padded_mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
            mb["use_cache"] = False

        return mb_list

    def train_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> Dict[str, float]:
        """Train on a batch using gradient accumulation."""
        input_ = input_.to(self.device)
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        self.optimizer.zero_grad()
        mb_list = self.prepare_mb_list(input_)

        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_list.mbs]), dtype=torch.float32
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight)

        # Process microbatches with gradient accumulation
        for i, (pad_length, padded_mb_input, mb_input) in enumerate(
            zip(mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs)
        ):

            outputs = self.model(**padded_mb_input)

            logits = outputs.logits.squeeze(0)
            logits = logits[:-pad_length] if pad_length > 0 else logits
            loss = loss_fn(logits, mb_input)

            loss_scale = loss_weight_fn(mb_input) / total_loss_weight

            # Scale loss for accumulation
            # Revert gradient averaging across dp ranks
            # FIXME: should be DP size
            loss_scale *= self.world_size

            loss *= loss_scale
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.optimizer_config.gradient_clipping,
            norm_type=2.0,
            error_if_nonfinite=False,
            foreach=None,
        )
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad()
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

    @torch.no_grad()
    def eval_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> torch.Tensor | None:
        """Evaluate on a batch."""
        input_ = input_.to(self.device)
        mb_list = self.prepare_mb_list(input_)
        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_list.mbs]), dtype=torch.float32
        )
        assert total_loss_weight != 0

        total_loss = 0.0
        total_weight = 0.0

        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):
            outputs = self.model(**padded_mb_input)
            logits = outputs.logits.squeeze(0)
            logits = logits[:-pad_length] if pad_length > 0 else logits
            loss = loss_fn(logits, mb_input)

            # Simple weight calculation (could be improved)
            loss_scale = loss_weight_fn(mb_input) / total_loss_weight
            total_loss += loss.item() * loss_scale
            total_weight += loss_scale

        return torch.tensor(total_loss / total_weight)

    @torch.no_grad()
    def forward(
        self,
        input_: TensorDict,
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, TensorDict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Forward pass with optional post-processing."""
        input_ = input_.to(self.device)
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        mb_list = self.prepare_mb_list(input_)

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()

        results = []
        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):
            outputs = self.model(**padded_mb_input)
            logits = outputs.logits.squeeze(0)
            logits = logits[:-pad_length] if pad_length > 0 else logits

            if post_hook:
                result = post_hook(logits, mb_input)
                results.append(result)
            else:
                results.append(logits)

        res = aggregate_fn(results)
        output_seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
        unpacked = unpack_sequence(res, lens=output_seqlens, dim=0)
        reordered = reorder_list(unpacked, mb_list.backward_indices)
        return pad_and_stack_tensors_along_first_dim(reordered)
