import asyncio
import functools
import math
import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from arealite.api.cli_args import (
    EngineConfig,
    MicroBatchSpec,
    ParallelismConfig,
    TrainingArgs,
)
from arealite.api.engine_api import SPMDWrapper
from arealite.api.io_struct import FinetuneSpec
from arealite.api.llm_client_api import LLMClient
from arealite.utils import (
    find_free_port,
    recorder_list,
    split_dict_tensor_with_cu_seqlens,
    unpack_sequence,
)
from realhf.base import constants


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


class HFEngine(SPMDWrapper):
    """Simplified HF engine for transformer models."""

    def __init__(self, args: TrainingArgs, engine_config: EngineConfig):
        super().__init__(args, engine_config)

        self.model = None
        self.optimizer = None
        self.model_config = None

        self.weight_update_group_initialized = False

    def init_distributed(self, config: ParallelismConfig, ft_spec: FinetuneSpec):
        """Initialize model in single node."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=0,
                world_size=1,
                init_method=f"tcp://localhost:{find_free_port()}",
            )
        if dist.get_world_size() > 1:
            raise RuntimeError(
                "Distributed training is not supported in this engine. "
                "Please use FSDP for distributed training."
            )
        torch.cuda.set_device("cuda:0")

        # Load model
        dtype = torch.bfloat16 if self.engine_config.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.engine_config.path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="cuda:0",
        )
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.engine_config.path,
            trust_remote_code=True,
        )

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

    def train(self, mode: bool = True):
        """Set the module in training mode."""
        return self.model.train(mode)

    def train_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> Dict:
        """Train on a batch using gradient accumulation."""
        assert self.optimizer is not None
        assert self.lr_scheduler is not None

        self.optimizer.zero_grad()
        mb_splits = split_dict_tensor_with_cu_seqlens(input_, mb_spec)
        total_loss_weight = torch.tensor(
            sum([loss_weight_fn(mb) for mb in mb_splits.mbs]), dtype=torch.float32
        )
        assert total_loss_weight != 0

        for mb_input in mb_splits.mbs:
            outputs = self.model(**mb_input)
            loss = loss_fn(outputs.logits, mb_input)
            loss_scale = loss_weight_fn(mb_input) / total_loss_weight
            loss *= loss_scale
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.engine_config.optimizer.gradient_clipping,
            norm_type=2.0,
            error_if_nonfinite=False,
            foreach=None,
        )
        current_lr = self.lr_scheduler.get_last_lr()[0]
        # Optimizer step
        self.optimizer.step()

        return {
            "grad_norm": grad_norm,
            "lr": current_lr,
        }

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

    def step_lr_scheduler(self):
        """Step the learning rate scheduler."""
        return self.lr_scheduler.step()

    def save_model_to_hf(
        self,
        path: str,
        tokenizer: Optional[transformers.PreTrainedTokenizerFast] = None,
        base_model_path: Optional[str] = None,
    ):
        """Save model in HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        os.makedirs(path, exist_ok=True)

        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        self.model.save_pretrained(path, state_dict=state_dict)
        self.model_config.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

    def load_model_from_hf(self, path: str):
        """Load model from HuggingFace format."""
        dtype = torch.bfloat16 if self.engine_config.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="cuda:0",
        )
        full_state = model.state_dict()
        self.model.load_state_dict(full_state)

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

    async def aupdate_weights_to(self, llm_client: LLMClient):
        path = constants.get_param_realloc_path(self.args)
        self.save_model_to_hf(path)
        tasks = [
            llm_client.aupdate_weights_from_disk(server_info=server_info, path=path)
            for server_info in llm_client.get_healthy_servers()
        ]
        await asyncio.gather(*tasks)

    def update_weights_to(self, llm_client: LLMClient):
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.aupdate_weights_to(llm_client))
        finally:
            loop.close()
