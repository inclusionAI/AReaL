import math
import os
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from transformers import AutoConfig, AutoModelForCausalLM

from arealite.api.cli_args import EngineConfig, MicroBatchSpec, TrainingArgs
from arealite.api.engine_api import SPMDWrapper
from arealite.api.io_struct import FinetuneSpec
from arealite.utils import split_dict_tensor_with_cu_seqlens
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


def apply_fsdp2(model, fsdp_kwargs):
    """model: AutoModelForCausalLM"""
    assert (
        CPUOffloadPolicy is not None
    ), "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    # fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get("transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap)
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
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(0.0, x * coef + intercept)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class FSDPEngine(SPMDWrapper):
    """Simplified FSDP engine for transformer models."""

    def __init__(self, args: TrainingArgs, engine_config: EngineConfig):
        super().__init__(args, engine_config)
        assert is_version_greater_or_equal(
            "torch", "2.4.0"
        ), f"arealite only supports FSDP2, which requires torch>=2.4.0"

        self.config = engine_config.backend.fsdp

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
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

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
        cpu_offload = None
        self.cpu_offload = cpu_offload

        fsdp_kwargs = {
            "mesh": device_mesh,
            "mp_policy": mixed_precision_policy,
            "offload_policy": cpu_offload,
            "reshard_after_forward": True,
        }

        # Wrap with FSDP2
        # self.model = FSDP(
        #     model,
        #     auto_wrap_policy=auto_wrap_policy,
        #     sharding_strategy=ShardingStrategy.FULL_SHARD,
        #     device_id=torch.cuda.current_device(),
        #     sync_module_states=self.config.sync_module_states,
        #     use_orig_params=self.config.use_orig_params,
        # )

        full_state = model.state_dict()
        apply_fsdp2(model, fsdp_kwargs)
        fsdp2_load_full_state_dict(model, full_state, device_mesh, cpu_offload)

        self.model = model

        # Set up optimizer
        optimizer_config = self.engine_config.optimizer
        if optimizer_config is not None:
            assert (
                optimizer_config.type == "adam"
            ), "Only AdamW optimizer is supported in this engine."
            lr = optimizer_config.lr
            weight_decay = optimizer_config.weight_decay
            beta1 = optimizer_config.beta1
            beta2 = optimizer_config.beta2
            eps = optimizer_config.eps

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
            )
            total_train_steps = ft_spec.total_train_steps
            num_warmup_steps = int(
                optimizer_config.warmup_steps_proportion * total_train_steps
            )

            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                total_train_steps,
                min_lr_ratio=optimizer_config.min_lr_ratio,
            )

    # def _split_microbatches(self, input_: Dict, mb_spec: MicroBatchSpec) -> List[Dict]:
    #     """Split input into microbatches."""
    #     batch_size = len(input_["input_ids"])
    #     n_mbs = min(mb_spec.n_mbs, batch_size)
    #     mb_size = batch_size // n_mbs

    #     microbatches = []
    #     for i in range(n_mbs):
    #         start = i * mb_size
    #         end = start + mb_size if i < n_mbs - 1 else batch_size

    #         mb = {}
    #         for key, value in input_.items():
    #             if isinstance(value, (list, torch.Tensor)):
    #                 mb[key] = value[start:end]
    #             else:
    #                 mb[key] = value
    #         microbatches.append(mb)

    #     return microbatches

    # def _initialize_fsdp_train(self):
    #     if not self.train_initialized:
    #         for fsdp_state in traversal_utils._get_fsdp_states(self.module):
    #             fsdp_state._is_root = None
    #         self.train_initialized = True

    def train_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
        version_steps: int,
        token_normalize_scope: Literal["global", "dp"] = "global",
    ) -> Dict:
        """Train on a batch using gradient accumulation."""
        # self._initialize_fsdp_train()
        assert self.optimizer is not None
        assert self.lr_scheduler is not None

        self.optimizer.zero_grad()

        mb_inputs = split_dict_tensor_with_cu_seqlens(input_, mb_spec)

        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_inputs]), dtype=torch.float32
        )
        assert total_loss_weight != 0
        if token_normalize_scope == "global":
            dist.all_reduce(total_loss_weight)

        # total_loss = 0.0
        # total_n_tokens = 0
        # Process microbatches with gradient accumulation
        # TODO: step lr scheduler if required
        for mb_input in mb_inputs:
            outputs = self.model(**mb_input)
            loss = loss_fn(outputs.logits, mb_input)
            loss_scale = loss_weight_fn(mb_input) / total_loss_weight

            # Scale loss for accumulation
            # TODO: check if this is required for FSDP
            if token_normalize_scope == "global":
                loss_scale *= self.world_size

            loss *= loss_scale
            loss.backward()

        # Optimizer step
        self.optimizer.step()

        return {}

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate on a batch."""
        self.eval()

        assert "cu_seqlens" in input_
        input_["cu_seqlens"]
        mb_inputs = split_dict_tensor_with_cu_seqlens(input_, mb_spec)

        total_loss = 0.0
        total_weight = 0.0

        for mb_input in mb_inputs:
            outputs = self.model(**mb_input)
            loss = loss_fn(outputs.logits, mb_input)

            # Simple weight calculation (could be improved)
            weight = mb_input["input_ids"].numel()

            total_loss += loss.item() * weight
            total_weight += weight

        return torch.tensor(total_loss / max(total_weight, 1e-8))

    @torch.no_grad()
    def forward(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, Dict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Forward pass with optional post-processing."""
        self.model.eval()
        mb_inputs = split_dict_tensor_with_cu_seqlens(input_, mb_spec)

        results = []

        for mb_input in mb_inputs:
            outputs = self.model(**mb_input)

            if post_hook:
                result = post_hook(outputs.logits, mb_input)
                results.append(result)
            else:
                results.append(outputs.logits)

        return aggregate_fn(results) if results else None

    def get_hf_model_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dict for saving."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            return self.model.state_dict()

    def save_model_to_hf(
        self,
        tokenizer: transformers.PreTrainedTokenizerFast,
        path: str,
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
