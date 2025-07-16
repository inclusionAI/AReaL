import dis
import gc
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import transformers
from tensordict import TensorDict
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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
    split_packed_tensor_dict_into_mb_list,
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
from arealite.utils.save_load import get_state_dict_from_repo_id_or_path
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, name_resolve, names, pkg_version
from sglang.srt.utils import init_custom_process_group

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
        self.weight_update_group_initialized = False

        # TODO: Handle the case when WORLD_SIZE is not set in launcher
        self.world_size = int(os.environ["WORLD_SIZE"])

    def train(self, mode: bool = True):
        assert self.model is not None
        self.model.train(mode=mode)
        return self

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

        dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.tokenizer = load_hf_tokenizer(self.config.path)
        with torch.device("cuda"):
            # initialize scratch model from config
            model = AutoModelForCausalLM.from_config(
                self.model_config,
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

        if not self.config.init_from_scratch:
            # Load model from a initial checkpoint path,
            # which should only be a huggingface checkpoint.
            load_meta = SaveLoadMeta(
                path=self.config.path,
                weight_format="hf",
                with_optim=False,
                tokenizer=None,
                base_model_path=self.config.path,
            )
            self.load(load_meta)

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
            print("Initialized distributed weight update group in training engine", flush=True)
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
                name_resolve.add(update_name, str(time.time_ns()), keepalive_ttl=120)
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def _init_distributed_weight_update(self, meta: WeightUpdateMeta):
        if dist.get_rank() == 0:
            print(f"[FSDP Engine]World size: {meta.world_size}, Master address: {meta.master_address}, Master port: {meta.master_port}", flush=True)
            self.weight_update_group = init_custom_process_group(
                backend="nccl",
                world_size=meta.world_size,
                init_method=f"tcp://{meta.master_address}:{meta.master_port}",
                rank=0,
                group_name=meta.group_name,
            )
            self.weight_update_group_initialized = True
            dist.barrier(group=self.weight_update_group,device_ids=[0])

    def _update_weights_from_distributed(self):
        """Broadcast parameters from rank 0 (FSDP2 compatible)."""
        if dist.get_rank() == 0:
            for name, param in self.model.named_parameters():
                if isinstance(param.data, DTensor):
                    tensor = param.data.full_tensor()
                else:
                    tensor = param.data
                print(f"Broadcasting {name} with shape {tensor.shape}", flush=True)
                dist.broadcast(tensor, src=0, group=self.weight_update_group)
                del tensor  # optional, for memory hygiene
            torch.cuda.empty_cache()
            # dist.barrier(group=self.weight_update_group)

    def get_param_meta_for_distributed_update(self) -> Dict[str, Tuple[int]]:
        """Return a dict mapping param name to its shape (expanded if DTensor)."""
        param_shapes = {}

        for name, param in self.model.named_parameters():
            try:
                if isinstance(param.data, DTensor):
                    tensor = param.data.full_tensor()
                else:
                    tensor = param.data

                param_shapes[name] = tuple(tensor.shape)
                del tensor  # free memory if full_tensor was created
            except Exception as e:
                logger.warning(f"Failed to get shape for param {name}: {e}")

        return param_shapes

    def step_lr_scheduler(self):
        assert self.lr_scheduler is not None
        self.lr_scheduler.step()

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
