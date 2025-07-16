import gc
import os
import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import transformers
from safetensors.torch import save_file
from tensordict import TensorDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from arealite.api.cli_args import MicroBatchSpec, TrainEngineConfig
from arealite.api.engine_api import (
    FinetuneSpec,
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
)
from arealite.utils.data import (
    MicroBatchList,
    amend_position_ids,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    pad_mb_list,
    reorder_list,
    split_packed_tensor_dict_into_mb_list,
    unpack_sequence,
    unsqueeze_mb_list,
)
from arealite.utils.fsdp import get_cosine_schedule_with_warmup
from arealite.utils.save_load import (
    get_state_dict_from_repo_id_or_path,
    is_existing_local_path,
)
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, name_resolve, names

logger = logging.getLogger("HFEngine")


class HFEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.optimizer_config = config.optimizer

        self.model = None
        self.optimizer = None
        self.tokenizer = None
        # huggingface model config
        self.model_config = None
        # initialization
        self.initialized = False
        self.weight_update_group_initialized = False

    def train(self, mode: bool = True):
        assert self.model is not None
        self.model.train(mode=mode)
        return self

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
        """Initialize distributed communication and model."""
        assert addr is None, "HFEngine does not support remote initialization."

        world_size = int(os.environ.get("WORLD_SIZE", 0))
        if not dist.is_initialized() and world_size > 1:
            try:
                import deepspeed
            except ImportError:
                print(
                    "Warning: deepspeed is not installed. Some functionality may be disabled."
                )
            deepspeed.init_distributed(dist_backend="nccl", world_size=world_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")

        dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.tokenizer = load_hf_tokenizer(self.config.path)

        self.model = AutoModelForCausalLM.from_config(
            self.model_config,
            torch_dtype=dtype,
            attn_implementation=self.config.attn_impl,
        ).to(f"cuda:{local_rank}")

        if not self.config.init_from_scratch:
            # Load model from a initial checkpoint path,
            # which should only be a huggingface checkpoint.
            load_meta = SaveLoadMeta(
                path=self.config.path,
                weight_format="hf",
                with_optim=False,
                tokenizer=None,
                base_model_path=self.config.path,
                naive_distributed=False,
            )

            self.load(load_meta)

        if world_size > 1:
            if self._check_autotp():
                self.model = deepspeed.tp_model_init(
                    self.model, tp_size=self.config.hf.autotp_size, dtype=dtype
                )
            else:
                raise RuntimeError("DeepSpeed AutoTP configuration error in HFEngine. ")

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

    def _check_autotp(self):
        tp_size = self.config.hf.autotp_size
        config = self.model_config
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        return (
            num_attention_heads % tp_size == 0
            and num_key_value_heads % tp_size == 0
            and hidden_size % tp_size == 0
            and intermediate_size % tp_size == 0
        )

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        self.model = None
        self.optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        self.initialized = False

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._save_model_to_hf(meta.path, meta.tokenizer, meta.naive_distributed)
        elif meta.weight_format == "dcp":
            # TODO: implement DCP save/load for HF
            raise NotImplementedError("DCP format saving is not implemented yet. ")
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim:
            self._save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._load_model_from_hf(meta.path, meta.naive_distributed)
        elif meta.weight_format == "dcp":
            # TODO: implement DCP save/load for HF
            raise NotImplementedError("DCP format loading is not implemented yet. ")
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim:
            self._load_optimizer_state(meta.path)

    def _save_optimizer_state(self, path: str):
        assert self.optimizer is not None
        os.makedirs(path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optim.pt"))

    def _load_optimizer_state(self, path: str):
        assert self.optimizer is not None
        path = os.path.join(path, "optim.pt")
        optimizer_state_dict = torch.load(path, weights_only=False)
        self.optimizer.load_state_dict(optimizer_state_dict)

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: Optional[transformers.PreTrainedTokenizerFast],
        naive_distributed: bool,
    ):
        """Save model in HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            os.makedirs(path, exist_ok=True)
            self.model_config.save_pretrained(path)
            if tokenizer is not None:
                tokenizer.save_pretrained(path)

        if world_size > 1:
            dist.barrier()

        state_dict = self.model.state_dict()

        if hasattr(self.model, "module"):
            state_dict = {
                k.replace("module.", "", 1) if k.startswith("module.") else k: v.cpu()
                for k, v in state_dict.items()
            }
        else:
            state_dict = {k: v.cpu() for k, v in state_dict.items()}

        if world_size > 1 and naive_distributed:
            # Only support store parameters from model partitions respectively
            gathered_state_dicts = None
            if rank == 0:
                gathered_state_dicts = [None for _ in range(world_size)]

            dist.gather_object(
                obj=state_dict, object_gather_list=gathered_state_dicts, dst=0
            )

            if rank == 0:
                for i, state_dict in enumerate(gathered_state_dicts):
                    save_file(state_dict, f"{path}/rank_{i:02d}_model.safetensors")
        else:
            self.model.save_pretrained(path, state_dict=state_dict)

        if world_size > 1:
            dist.barrier()

    def _load_model_from_hf(self, path: str, naive_distributed: bool):
        """Load model from HuggingFace format."""

        rank = dist.get_rank()
        # Only support load full model parameters from huggingface
        # and load model partition locally
        if rank == 0 or is_existing_local_path(path):
            if naive_distributed:
                path = f"{path}/rank_{rank:02d}_model.safetensors"
            full_state = get_state_dict_from_repo_id_or_path(path)

            if hasattr(self.model, "module") and not hasattr(full_state):
                full_state = {
                    f"module.{k}" if not k.startswith("module.") else k: v
                    for k, v in full_state.items()
                }
            self.model.load_state_dict(
                full_state, strict=not self.model_config.tie_word_embeddings
            )

            if self.model_config.tie_word_embeddings:
                self.model.tie_weights()

    def upload_weights(self, meta: WeightUpdateMeta):
        if meta.type == "nccl":
            if not self.weight_update_group_initialized:
                self._init_distributed_weight_update(meta)
            self._update_weights_from_distributed()
        elif meta.type == "disk":
            self._save_model_to_hf(meta.path, self.tokenizer, meta.naive_distributed)
            update_name = names.update_weights_from_disk(
                self.config.experiment_name,
                self.config.trial_name,
                meta.model_version,
            )
            name_resolve.add(update_name, str(time.time_ns()), keepalive_ttl=120)
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def _init_distributed_weight_update(self, meta: WeightUpdateMeta):
        raise NotImplementedError(
            "Distributed weight update is not implemented for HFEngine yet. "
        )

    def _update_weights_from_distributed(self):
        raise NotImplementedError(
            "Distributed weight update is not implemented for HFEngine yet. "
        )

    def step_lr_scheduler(self):
        assert self.lr_scheduler is not None
        return self.lr_scheduler.step()

    def _prepare_mb_list(self, input_: TensorDict) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        if isinstance(input_, dict):
            input_ = TensorDict(input_, batch_size=[input_["input_ids"].shape[0]])
        input_ = amend_position_ids(input_)
        packed_input = pack_tensor_dict(input_)
        mb_list = split_packed_tensor_dict_into_mb_list(
            packed_input,
            self.config.mb_spec,
        )
        mb_list = pad_mb_list(mb_list, pad_value=0.0)
        # NOTE: We unsqueeze here because huggingface transformer models requires
        # packed input to be of shape [1, total_seqlen].
        mb_list = unsqueeze_mb_list(mb_list)
        return mb_list

    def train_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> Dict[str, float]:
        """Train on a batch using gradient accumulation."""
        input_ = input_.to(self.device)
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        self.optimizer.zero_grad()
        mb_list = self._prepare_mb_list(input_)

        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_list.mbs]), dtype=torch.float32
        )
        assert total_loss_weight != 0

        # Process microbatches with gradient accumulation
        for pad_length, padded_mb_input, mb_input in zip(
            mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs
        ):
            outputs = self.model(**padded_mb_input)

            logits = outputs.logits.squeeze(0)
            logits = logits[:-pad_length] if pad_length > 0 else logits
            loss = loss_fn(logits, mb_input)
            loss_scale = loss_weight_fn(mb_input) / total_loss_weight

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
        # Optimizer step
        self.optimizer.step()
        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    @torch.no_grad()
    def eval_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> torch.Tensor | None:
        """Evaluate on a batch."""
        mb_list = self._prepare_mb_list(input_)
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
        post_hook: Callable[[torch.Tensor, Dict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Forward pass with optional post-processing."""
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        mb_list = self._prepare_mb_list(input_)

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
