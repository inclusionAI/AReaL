import gc
import os
import time
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    ProcessorMixin,
)

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.platforms import current_platform
from areal.utils import logging
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
from areal.utils.hf_utils import load_hf_processor_and_tokenizer, load_hf_tokenizer
from areal.utils.model import (
    disable_dropout_in_model,
    is_gemma3_model,
    is_qwen2_vl_model,
    is_qwen3_moe_model,
    is_valid_vision_model,
)
from areal.utils.nccl import NCCL_DEFAULT_TIMEOUT


class BaseHFEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.optimizer_config = config.optimizer

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.tokenizer: PreTrainedTokenizerFast
        self.processor: ProcessorMixin | None = None
        # huggingface model config
        self.model_config: PretrainedConfig
        self._version: int = 0

        # initialization
        self.initialized = False
        self.own_global_group = False
        self._parallelism_group: dist.ProcessGroup
        self.mp_group: dist.ProcessGroup
        self.weight_update_group_initialized = False

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.is_vision_model = is_valid_vision_model(self.model_config.model_type)

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
    def data_parallel_group(self) -> dist.ProcessGroup:
        assert self.initialized
        return _get_default_group()

    @property
    def data_parallel_rank(self) -> int:
        return dist.get_rank()

    @property
    def data_parallel_world_size(self) -> int:
        return self.world_size

    def current_data_parallel_head(self) -> int:
        return dist.get_rank()

    def is_data_parallel_head(self) -> bool:
        return True

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        assert self.initialized
        return self.mp_group

    @property
    def parallelism_group(self) -> dist.ProcessGroup:
        assert self.initialized
        return _get_default_group()

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        backend = current_platform.communication_backend
        if not dist.is_initialized():
            # TODO: Handle the condition when WORLD_SIZE and RANK is not set in launcher
            # NOTE: device_id **SHOULD NOT** be passed into init_process_group,
            # otherwise initializing the NCCL weight update group will be wrong!
            dist.init_process_group(
                backend=backend,
                timeout=NCCL_DEFAULT_TIMEOUT,
            )
            self.own_global_group = True
        # Each process is its own model parallel group.
        mp_group = dist.new_group([dist.get_rank()])
        assert mp_group is not None
        self.mp_group = mp_group

        self.logger = logging.getLogger(f"[HF Engine Rank {dist.get_rank()}]")

    def create_device_model(self):
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
                    torch_dtype=dtype,
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

    def _create_llm_actor_or_critic(self):
        dtype = getattr(torch, self.config.dtype)
        if not self.config.is_critic:
            if self.config.init_from_scratch:
                # initialize model from config
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
        else:
            if self.config.init_from_scratch:
                model = AutoModelForTokenClassification.from_config(
                    self.model_config,
                    torch_dtype=dtype,
                    num_labels=1,
                    attn_implementation=self.config.attn_impl,
                )
            else:
                model = AutoModelForTokenClassification.from_pretrained(
                    pretrained_model_name_or_path=self.config.path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    num_labels=1,
                    attn_implementation=self.config.attn_impl,
                )
        return model

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        current_platform.empty_cache()
        gc.collect()
        non_trivial_world = dist.get_world_size() > 1
        if non_trivial_world:
            dist.destroy_process_group(self.context_and_model_parallel_group)
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

    def prepare_mb_list(self, input_: Dict[str, Any]) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        input_ = input_.copy()

        if is_qwen2_vl_model(self.model_config.model_type):
            # Create the special t,h,w position IDs for qwen 2.5 VL
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
            # [3, bs, seqlen] -> [bs, seqlen, 3]
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
        # NOTE: We unsqueeze here because huggingface transformer models requires
        # packed input to be of shape [1, total_seqlen].
        mb_list = unsqueeze_mb_list(mb_list)
        if is_qwen2_vl_model(self.model_config.model_type):
            assert mb_list.padded_mbs is not None
            for mb in mb_list.padded_mbs:
                # [1, total_seqlen, 3] -> [3, 1, total_seqlen]
                mb["position_ids"] = torch.einsum("ijk->kij", mb["position_ids"])

        # FIXME: the resulting max_seqlen is a tensor rather than an integer

        # Modern model implementations takes a dict as the input.
        # This eliminates a bug of Qwen2.5-VL for transformers<=4.53.1
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
            if is_qwen3_moe_model(self.model_config.model_type):
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
                    # Individual pixel_values shape: (#patches_per_sample, patch_size)
                    # Concatenate along dim=0 to get shape: (#total_patches, patch_size)
                    # For Qwen:
                    # - image_grid_thw shape: (#total_images, 3) where 3 -> [grid_t, grid_h, grid_w]
                    # - pixel_values shape: (#total_patches, patch_size)
                    # - total_patches = torch.sum(torch.prod(image_grid_thw, dim=1))
                    # - total_image_pad_tokens = total_patches // (merge_size**2)
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

    def get_model_name_parameters(self):
        name_params_iterator = self.model.named_parameters()
        if self.is_vision_model and is_qwen2_vl_model(self.model_config.model_type):
            # Qwen2_5_VLForConditionalGeneration has a different naming convention in SGLang
            def name_remapping_generator():
                for name, value in name_params_iterator:
                    new_name = name.replace("model.", "", 1).replace(
                        "language_model", "model"
                    )
                    yield new_name, value

            yield from name_remapping_generator()
        elif self.is_vision_model and is_gemma3_model(self.model_config.model_type):
            # Gemma3ForConditionalGeneration has a different naming convention in SGLang
            def name_remapping_generator():
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

            yield from name_remapping_generator()
        else:
            yield from name_params_iterator

    def train_batch(
        self,
        input_: Dict[str, Any],
        loss_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[Dict[str, Any]], torch.Tensor],
    ) -> Dict[str, float]:
        """Train on a batch using gradient accumulation."""
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        self.optimizer.zero_grad()
        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)

        total_loss_weight = (
            torch.stack([loss_weight_fn(mb) for mb in mb_list.mbs])
            .sum()
            .detach()
            .clone()
            .to(dtype=torch.float32)
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight)

        # Process microbatches with gradient accumulation
        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
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
        input_: Dict[str, Any],
        loss_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[Dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate on a batch."""
        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)

        total_loss_weight = (
            torch.stack([loss_weight_fn(mb) for mb in mb_list.mbs])
            .sum()
            .detach()
            .clone()
            .to(dtype=torch.float32)
        )
        assert total_loss_weight != 0

        total_loss = torch.zeros(1, device=self.device, dtype=torch.float32)
        total_weight = torch.zeros(1, device=self.device, dtype=torch.float32)

        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):
            outputs = self.model(**padded_mb_input)
            logits = outputs.logits.squeeze(0)
            logits = logits[:-pad_length] if pad_length > 0 else logits
            loss = loss_fn(logits, mb_input)

            # Simple weight calculation (could be improved)
            loss_scale = loss_weight_fn(mb_input) / total_loss_weight
            total_loss += loss * loss_scale
            total_weight += loss_scale

        return total_loss / total_weight

    @torch.no_grad()
    def forward(
        self,
        input_: Dict[str, Any],
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, Dict[str, Any]], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Forward pass with optional post-processing."""
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        mb_list = self.prepare_mb_list(input_)
        mb_list = mb_list.to(self.device)

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None

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
