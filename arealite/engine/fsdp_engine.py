import gc
import os
import time
from datetime import datetime
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
    AutoConfig,
    AutoModelForCausalLM,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from arealite.api.cli_args import TrainEngineConfig
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
    split_padded_tensor_dict_into_mb_list,
    unpack_sequence,
    unsqueeze_mb_list,
)
from arealite.utils.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    create_fsdp_device_mesh,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_cosine_schedule_with_warmup,
)
from arealite.utils.model import disable_dropout_in_model
from arealite.utils.save_load import get_state_dict_from_repo_id_or_path
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import constants, logging, name_resolve, names, pkg_version

logger = logging.getLogger("FSDPEngine")


class FSDPEngine(TrainEngine):
    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.optimizer_config = config.optimizer

        self.model = None
        self.optimizer = None
        self.tokenizer = None
        # huggingface model config
        self.model_config = None
        # FSDP options
        self.mixed_precision_policy = None
        self.device_mesh = None
        self.cpu_offload = None
        # initialization
        self.initialized = False
        self.own_global_group = False
        self._parallelism_group = None
        self.weight_update_group_initialized = False

        # TODO: Handle the case when WORLD_SIZE is not set in launcher
        self.world_size = int(os.environ["WORLD_SIZE"])

    def train(self, mode: bool = True):
        assert self.model is not None
        self.model.train(mode=mode)
        return self

    @property
    def parallelism_group(self) -> dist.ProcessGroup:
        assert self.initialized
        return self._parallelism_group

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
        # Initialize distributed enviroments and load model.
        assert addr is None, "FSDPEngine does not support remote initialization."

        assert pkg_version.is_version_greater_or_equal(
            "torch", "2.4.0"
        ), f"arealite only supports FSDP2, which requires torch>=2.4.0"

        """Initialize distributed communication and model."""
        if not dist.is_initialized():
            # TODO: Handle the condition when WORLD_SIZE and RANK is not set in launcher
            dist.init_process_group(
                backend="nccl",
                timeout=constants.NCCL_DEFAULT_TIMEOUT,
                device_id=torch.device(int(os.environ["LOCAL_RANK"])),
            )
            self.own_global_group = True
        self._parallelism_group = dist.new_group()

        # TODO: Handle the condition when LOCAL_RANK is not set in launcher
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        dtype = getattr(torch, self.config.dtype)
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
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

        # Simple auto wrap policy
        self.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=dtype,
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
        tik = time.perf_counter()
        apply_fsdp2(model, fsdp_kwargs, self.config.fsdp.wrap_policy)
        logger.info(f"Applying FSDP2 time: {time.perf_counter() - tik}")
        self.model = model

        # Set up optimizer
        if self.optimizer_config is not None:
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

        self.initialized = True

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        self.model = None
        self.optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        dist.destroy_process_group(self.parallelism_group)
        if self.own_global_group:
            dist.destroy_process_group()
        self.initialized = False

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._save_model_to_hf(meta.path, meta.tokenizer)
        elif meta.weight_format == "dcp":
            # TODO: implement DCP save/load for FSDP
            raise NotImplementedError("DCP format saving is not implemented yet. ")
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim:
            self._save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            # TODO: implement DCP save/load for FSDP
            raise NotImplementedError("DCP format loading is not implemented yet. ")
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")

        if meta.with_optim:
            self._load_optimizer_state(meta.path)

    def _save_optimizer_state(self, path: str):
        # Save FSDP sharded state dict on each rank
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        state_dict = self.optimizer.state_dict()
        torch.save(state_dict, shard_path)
        dist.barrier()

    def _load_optimizer_state(self, path: str):
        # Load FSDP sharded state dict
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        optimizer_state_dict = torch.load(shard_path, weights_only=False)
        self.optimizer.load_state_dict(optimizer_state_dict)
        dist.barrier()

    def _save_model_to_hf(
        self, path: str, tokenizer: Optional[transformers.PreTrainedTokenizerFast]
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
            self.model_config.save_pretrained(path)
            if tokenizer is not None:
                tokenizer.save_pretrained(path)

        dist.barrier()

    def _load_model_from_hf(self, path: str):
        """Load model from HuggingFace format."""
        if dist.get_rank() == 0:
            full_state = get_state_dict_from_repo_id_or_path(path)
        else:
            full_state = {}

        fsdp2_load_full_state_dict(
            self.model,
            full_state,
            self.cpu_offload,
            tie_word_embeddings=self.model_config.tie_word_embeddings,
        )

    def upload_weights(self, meta: WeightUpdateMeta):
        if meta.type == "nccl":
            if not self.weight_update_group_initialized:
                self._init_distributed_weight_update(meta)
            self._update_weights_from_distributed()
        elif meta.type == "disk":
            self._save_model_to_hf(meta.path, self.tokenizer)
            # dist.barrier() are called when _save_model_to_hf finished
            if dist.get_rank() == 0:
                update_name = names.update_weights_from_disk(
                    self.config.experiment_name,
                    self.config.trial_name,
                    meta.model_version,
                )
                name_resolve.add(
                    update_name, str(datetime.now().timestamp()), keepalive_ttl=120
                )
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def _init_distributed_weight_update(self, meta: WeightUpdateMeta):
        raise NotImplementedError(
            "Distributed weight update is not implemented for FSDPEngine yet. "
        )

    def _update_weights_from_distributed(self):
        raise NotImplementedError(
            "Distributed weight update is not implemented for FSDPEngine yet. "
        )

    def step_lr_scheduler(self):
        assert self.lr_scheduler is not None
        self.lr_scheduler.step()

    def _prepare_mb_list(self, input_: TensorDict) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        if isinstance(input_, dict):
            input_ = TensorDict(input_, batch_size=[input_["input_ids"].shape[0]])
        input_ = amend_position_ids(input_)
        mb_list = split_padded_tensor_dict_into_mb_list(input_, self.config.mb_spec)
        logger.info(
            f"Microbatch #tokens (rank {dist.get_rank()}): {mb_list.group_lens}"
        )
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        mb_list = pad_mb_list(mb_list, pad_value=0.0)
        # NOTE: We unsqueeze here because huggingface transformer models requires
        # packed input to be of shape [1, total_seqlen].
        mb_list = unsqueeze_mb_list(mb_list)
        # FIXME: the resulting max_seqlen is a tensor rather than an integer
        for mb in mb_list.mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
            mb["use_cache"] = False
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
        dist.all_reduce(total_loss_weight)

        # Process microbatches with gradient accumulation
        for i, (pad_length, padded_mb_input, mb_input) in enumerate(
            zip(mb_list.padding_lengths, mb_list.padded_mbs, mb_list.mbs)
        ):
            self.model.set_is_last_backward(i == len(mb_list.mbs) - 1)
            outputs = self.model(**padded_mb_input)

            logits = outputs.logits.squeeze(0)
            logits = logits[:-pad_length] if pad_length > 0 else logits
            loss = loss_fn(logits, mb_input)
            loss_scale = loss_weight_fn(mb_input) / total_loss_weight

            # Scale loss for accumulation
            # Revert gradient averaging across dp ranks
            loss_scale *= self.world_size

            loss *= loss_scale
            loss.backward()

        grad_norm = fsdp2_clip_grad_norm_(
            self.model.parameters(), max_norm=self.optimizer_config.gradient_clipping
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
        input_ = input_.to(self.device)
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
        input_ = input_.to(self.device)
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
