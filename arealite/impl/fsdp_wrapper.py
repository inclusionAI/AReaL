# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import functools
import math
import os
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from arealite.api.cli_args import EngineConfig, FSDPConfig, MicroBatchSpec, TrainingArgs
from arealite.api.engine_api import SPMDWrapper
from arealite.api.io_struct import FinetuneSpec
from arealite.utils import (
    find_free_port,
    recorder_list,
    split_dict_tensor_with_cu_seqlens,
    unpack_sequence,
)
from realhf.api.cli_args import ParallelismConfig
from realhf.base.pkg_version import is_version_greater_or_equal

if is_version_greater_or_equal("torch", "2.6.0"):
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
elif is_version_greater_or_equal("torch", "2.4.0"):
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy = (
        None,
        None,
        None,
        None,
    )

from torch.distributed.device_mesh import init_device_mesh


def fsdp2_clip_grad_norm_(
    parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None
):
    """torch.nn.utils.clip_grad_norm_ cann't run on cpu parameter DTensor"""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(torch.cuda.current_device(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def create_fsdp_device_mesh(shard_size, world_size):
    if shard_size < 0 or shard_size >= world_size:
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",)
        )
    else:
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // shard_size, shard_size),
            mesh_dim_names=("ddp", "fsdp"),
        )
    return device_mesh


def apply_fsdp2(model, fsdp_kwargs, wrap_policy):
    """model: AutoModelForCausalLM"""
    assert (
        CPUOffloadPolicy is not None
    ), "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", list())
    fsdp_transformer_layer_cls_to_wrap = (
        wrap_policy.transformer_layer_cls_to_wrap if wrap_policy is not None else list()
    )
    if not fsdp_transformer_layer_cls_to_wrap:
        fsdp_transformer_layer_cls_to_wrap = default_transformer_cls_names_to_wrap

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert (
        len(fsdp_transformer_layer_cls_to_wrap) > 0
        and fsdp_transformer_layer_cls_to_wrap[0] is not None
    )

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        fully_shard(module, **fsdp_kwargs)
    fully_shard(
        model, **fsdp_kwargs
    )  # fsdp2 will not reshard_after_forward for root module


def fsdp2_load_full_state_dict(
    model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None
):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    device = torch.cuda.current_device()

    # To broadcast, it needs to be instantiated in the GPU.
    if dist.get_rank() == 0:
        model = model.to(device=device, non_blocking=True)
    else:
        model = model.to_empty(device=device)

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True
    )
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(device)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    min_lr_ratio = 0.0 if min_lr_ratio is None else min_lr_ratio
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * (
                float(current_step) / float(max(1, num_warmup_steps))
            )
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(min_lr_ratio, x * coef + intercept)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class FSDPEngine(SPMDWrapper):
    """Simplified FSDP engine for transformer models."""

    def __init__(self, args: TrainingArgs, engine_config: EngineConfig):
        super().__init__(args, engine_config)
        assert is_version_greater_or_equal(
            "torch", "2.4.0"
        ), f"arealite only supports FSDP2, which requires torch>=2.4.0"

        self.fsdp_config = engine_config.backend.fsdp
        if self.fsdp_config is None:
            self.fsdp_config = FSDPConfig()
        self.optimizer_config = engine_config.optimizer

        self.model = None
        self.optimizer = None
        self.model_config = None
        self.device_mesh = None
        self.cpu_offload = None

        self.world_size = args.n_gpus_per_node * args.n_nodes

    def train(self, mode: bool = True):
        """Set the module in training mode."""
        assert self.model is not None
        self.model.train(mode=mode)
        return self

    def init_distributed(self, config: ParallelismConfig, ft_spec: FinetuneSpec):
        """Initialize distributed communication and model."""
        if self.args.n_gpus_per_node == 1 and self.args.n_nodes == 1:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    rank=0,
                    world_size=1,
                    init_method=f"tcp://localhost:{find_free_port()}",
                )
            torch.cuda.set_device("cuda:0")
        else:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            # print(f"LOCAL_RANK = {os.environ["LOCAL_RANK"]}")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            # print(f"current rank = {dist.get_rank()}, current device = {torch.cuda.current_device()}")

        # Load model
        dtype = torch.bfloat16 if self.engine_config.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.engine_config.path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.engine_config.path,
            trust_remote_code=True,
        )

        # Simple auto wrap policy
        # TODO: fix wrap policy
        mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        device_mesh = create_fsdp_device_mesh(self.world_size, self.world_size)
        self.device_mesh = device_mesh
        # sharding_strategy = ShardingStrategy.FULL_SHARD
        self.cpu_offload = (
            CPUOffloadPolicy() if self.fsdp_config.offload_params else None
        )

        fsdp_kwargs = {
            "mesh": device_mesh,
            "mp_policy": mixed_precision_policy,
            "offload_policy": self.cpu_offload,
            "reshard_after_forward": True,
        }

        # Wrap with FSDP2
        full_state = model.state_dict()
        apply_fsdp2(model, fsdp_kwargs, self.fsdp_config.wrap_policy)
        fsdp2_load_full_state_dict(model, full_state, device_mesh, self.cpu_offload)

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

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def train_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> Dict:
        """Train on a batch using gradient accumulation."""
        # self._initialize_fsdp_train()
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        self.optimizer.zero_grad()

        mb_inputs = split_dict_tensor_with_cu_seqlens(input_, mb_spec).mbs

        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_inputs]), dtype=torch.float32
        )
        assert total_loss_weight != 0
        dist.all_reduce(total_loss_weight)

        # Process microbatches with gradient accumulation
        # TODO: step lr scheduler if required
        for i, mb_input in enumerate(mb_inputs):
            outputs = self.model(**mb_input)

            loss = loss_fn(outputs.logits, mb_input)
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

    def step_lr_scheduler(self):
        assert self.lr_scheduler is not None
        self.lr_scheduler.step()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> torch.Tensor | None:
        """Evaluate on a batch."""
        mb_splits = split_dict_tensor_with_cu_seqlens(input_, mb_spec)
        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_splits.mbs]), dtype=torch.float32
        )
        assert total_loss_weight != 0

        total_loss = 0.0
        total_weight = 0.0

        for mb_input in mb_splits.mbs:
            outputs = self.model(**mb_input)
            loss = loss_fn(outputs.logits, mb_input)

            # Simple weight calculation (could be improved)
            loss_scale = loss_weight_fn(mb_input) / total_loss_weight
            total_loss += loss.item() * loss_scale
            total_weight += loss_scale

        return torch.tensor(total_loss / total_weight)

    @torch.no_grad()
    def forward(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, Dict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = functools.partial(torch.cat, dim=1),
    ) -> Any | None:
        """Forward pass with optional post-processing."""
        mb_splits = split_dict_tensor_with_cu_seqlens(input_, mb_spec)
        if output_seqlens is None:
            cu_seqlens = input_["cu_seqlens"]
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()

        results = []
        for mb_input in mb_splits.mbs:
            outputs = self.model(**mb_input)
            if post_hook:
                result = post_hook(outputs.logits, mb_input)
                results.append(result)
            else:
                results.append(outputs.logits)

        res = aggregate_fn(results)
        unpacked = unpack_sequence(res, lens=output_seqlens, dim=1)
        return aggregate_fn(recorder_list(unpacked, mb_splits.backward_indices))

    def get_hf_model_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dict for saving."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            return self.model.state_dict()

    def save_model_to_hf(
        self,
        path: str,
        tokenizer: transformers.PreTrainedTokenizerFast,
        base_model_path: Optional[str] = None,
    ):
        """Save model in HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        os.makedirs(path, exist_ok=True)

        # FSDP2 checkpoint saving
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        # Get full state dict with FSDP2
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(self.model, options=options)

        # save huggingface model
        if dist.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.model_config.save_pretrained(path)
            tokenizer.save_pretrained(path)

        dist.barrier()

    def load_model_from_hf(self, path: str):
        """Load model from HuggingFace format."""
        # if self.model is None:
        #     raise RuntimeError("Model not initialized")

        # state_dict = torch.load(
        #     os.path.join(path, "pytorch_model.bin"), map_location="cpu"
        # )

        # with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
        #     self.model.load_state_dict(state_dictt
        dtype = torch.bfloat16 if self.engine_config.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        full_state = model.state_dict()

        fsdp2_load_full_state_dict(
            self.model, full_state, self.device_mesh, self.cpu_offload
        )

    def save_optimizer_state(self, path: str):
        """Save optimizer state."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized")

        os.makedirs(path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

    def load_optimizer_state(self, path: str):
        """Load optimizer state."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized")

        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu")
            )
        else:
            raise RuntimeError(f"Optimizer state file not found: {optimizer_path}")
