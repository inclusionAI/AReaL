import math
import os
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM

from arealite.api.cli_args import EngineConfig, MicroBatchSpec, TrainingArgs
from arealite.api.engine_api import SPMDWrapper
from realhf.api.cli_args import ParallelismConfig


def get_transformer_layer_cls(model):
    """Get transformer layer classes for wrapping policy."""
    # Common transformer layer class names
    common_layer_names = ["Block", "DecoderLayer"]

    layer_classes = set()
    for name, module in model.named_modules():
        module_name = type(module).__name__
        if any(layer_name in module_name for layer_name in common_layer_names):
            layer_classes.add(type(module))

    # Fallback to standard PyTorch layers if none found
    if not layer_classes:
        layer_classes = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}

    return layer_classes


class FSDPEngine(SPMDWrapper):
    """Simplified FSDP engine for transformer models."""

    def __init__(self, args: TrainingArgs, engine_config: EngineConfig):
        super().__init__(args, engine_config)

        self.config = engine_config.fsdp

        self.model = None
        self.optimizer = None

    def init_distributed(self, config: ParallelismConfig):
        """Initialize distributed communication and model."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # Load model
        dtype = torch.bfloat16 if self.engine_config.bf16 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self.engine_config.path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        # Simple auto wrap policy
        transformer_layer_cls = get_transformer_layer_cls(model)
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls=transformer_layer_cls
        )

        # Wrap with FSDP
        self.model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            sync_module_states=self.config.sync_module_states,
            use_orig_params=self.config.use_orig_params,
        )

        # Set up optimizer
        optimizer_config = self.engine_config.optimizer
        assert (
            optimizer_config.type == "adamw"
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

    def _split_microbatches(self, input_: Dict, mb_spec: MicroBatchSpec) -> List[Dict]:
        """Split input into microbatches."""
        batch_size = len(input_["input_ids"])
        n_mbs = min(mb_spec.n_mbs, batch_size)
        mb_size = batch_size // n_mbs

        microbatches = []
        for i in range(n_mbs):
            start = i * mb_size
            end = start + mb_size if i < n_mbs - 1 else batch_size

            mb = {}
            for key, value in input_.items():
                if isinstance(value, (list, torch.Tensor)):
                    mb[key] = value[start:end]
                else:
                    mb[key] = value
            microbatches.append(mb)

        return microbatches

    def train_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[torch.Tensor, Dict], float],
        version_steps: int,
        token_normalize_scope: Literal["global", "dp"] = "global",
    ) -> Dict:
        """Train on a batch using gradient accumulation."""
        self.model.train()
        self.optimizer.zero_grad()

        microbatches = self._split_microbatches(input_, mb_spec)
        total_loss = 0.0
        total_weight = 0.0

        # Process microbatches with gradient accumulation
        for mb in microbatches:
            outputs = self.model(**mb)
            loss = loss_fn(outputs.logits, mb)
            weight = loss_weight_fn(outputs.logits, mb)

            # Scale loss for accumulation
            scaled_loss = loss / len(microbatches)
            scaled_loss.backward()

            total_loss += loss.item() * weight
            total_weight += weight

        # Normalize across ranks if needed
        if token_normalize_scope == "global" and dist.is_initialized():
            metrics = torch.tensor(
                [total_loss, total_weight], device=torch.cuda.current_device()
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, total_weight = metrics.tolist()

        avg_loss = total_loss / max(total_weight, 1e-8)

        # Optimizer step
        self.optimizer.step()

        return {
            "loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "total_tokens": total_weight,
        }

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate on a batch."""
        self.model.eval()

        microbatches = self._split_microbatches(input_, mb_spec)
        total_loss = 0.0
        total_weight = 0.0

        for mb in microbatches:
            outputs = self.model(**mb)
            loss = loss_fn(outputs.logits, mb)

            # Simple weight calculation (could be improved)
            weight = mb["input_ids"].numel()

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

        microbatches = self._split_microbatches(input_, mb_spec)
        results = []

        for mb in microbatches:
            outputs = self.model(**mb)

            if post_hook:
                result = post_hook(outputs.logits, mb)
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

        # Save model state
        state_dict = self.get_hf_model_state_dict()
        torch.save(state_dict, os.path.join(path, "pytorch_model.bin"))

        # Save tokenizer and config
        tokenizer.save_pretrained(path)

        config_path = base_model_path or self.config.path
        config = transformers.AutoConfig.from_pretrained(config_path)
        config.save_pretrained(path)

    def load_model_from_hf(self, path: str):
        """Load model from HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        state_dict = torch.load(
            os.path.join(path, "pytorch_model.bin"), map_location="cpu"
        )

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            self.model.load_state_dict(state_dict)

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
