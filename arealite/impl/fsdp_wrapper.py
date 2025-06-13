import functools
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


class FSDPEngine(SPMDWrapper):
    """FSDP-based engine for training and inference of transformer models."""

    def __init__(self, args: TrainingArgs, config: EngineConfig):
        super().__init__(args, config)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self._distributed_initialized = False

    def init_distributed(self, config: ParallelismConfig):
        """Initialize distributed communication groups and models."""
        if self._distributed_initialized:
            return

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self._setup_model()
        self._setup_optimizer()
        self._distributed_initialized = True

    def _setup_model(self):
        """Initialize the model with FSDP wrapping."""
        # Load model configuration
        model_path = self.config.path

        # Create the model
        torch_dtype = "bfloat16" if self.config.bf16 else "float16"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, torch_dtype),
            trust_remote_code=True,
        )

        # Auto wrap policy for transformer layers
        # Get the transformer layer class from the model
        transformer_layer_cls = set()
        for name, module in model.named_modules():
            if any(
                layer_name in name for layer_name in ["layer", "block", "decoder_layer"]
            ):
                transformer_layer_cls.add(type(module))

        # Fallback to common transformer layer classes if none found
        if not transformer_layer_cls:
            transformer_layer_cls = {
                nn.TransformerEncoderLayer,
                nn.TransformerDecoderLayer,
            }

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls
        )

        # Wrap with FSDP
        self.model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=self._get_mixed_precision_config(),
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            use_orig_params=True,
        )

    def _get_mixed_precision_config(self):
        """Get mixed precision configuration for FSDP."""
        from torch.distributed.fsdp import MixedPrecision

        if getattr(self.config, "mixed_precision", False):
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        return None

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        optimizer_config = getattr(self.config, "optimizer_config", None)

        if optimizer_config:
            optimizer_name = getattr(optimizer_config, "name", "adamw").lower()

            if optimizer_name == "adamw":
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=getattr(optimizer_config, "lr", 1e-4),
                    betas=(
                        getattr(optimizer_config, "beta1", 0.9),
                        getattr(optimizer_config, "beta2", 0.999),
                    ),
                    eps=getattr(optimizer_config, "eps", 1e-8),
                    weight_decay=getattr(optimizer_config, "weight_decay", 0.01),
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

            # Setup learning rate scheduler if configured
            if (
                hasattr(optimizer_config, "lr_scheduler_type")
                and optimizer_config.lr_scheduler_type
            ):
                self._setup_scheduler(optimizer_config)
        else:
            # Default optimizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Setup gradient scaler for mixed precision
        if getattr(self.config, "mixed_precision", False):
            self.scaler = torch.cuda.amp.GradScaler()

    def _setup_scheduler(self, optimizer_config):
        """Setup learning rate scheduler."""
        if optimizer_config.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR

            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=optimizer_config.get("total_steps", 1000),
            )
        elif optimizer_config.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=optimizer_config.get("total_steps", 1000),
            )

    def _split_into_microbatches(
        self, input_: Dict, mb_spec: MicroBatchSpec
    ) -> List[Dict]:
        """Split input into microbatches based on the specification."""
        batch_size = len(input_["input_ids"])

        # Calculate actual number of microbatches
        if mb_spec.max_tokens_per_mb < int(1e12):
            # Use token-based splitting
            total_tokens = sum(len(ids) for ids in input_["input_ids"])
            n_mbs = max(
                mb_spec.n_mbs, math.ceil(total_tokens / mb_spec.max_tokens_per_mb)
            )
        else:
            n_mbs = mb_spec.n_mbs

        n_mbs = min(n_mbs, batch_size)  # Don't exceed batch size
        mb_size = batch_size // n_mbs

        microbatches = []
        for i in range(n_mbs):
            start_idx = i * mb_size
            end_idx = start_idx + mb_size if i < n_mbs - 1 else batch_size

            mb = {}
            for key, value in input_.items():
                if isinstance(value, (list, tuple)):
                    mb[key] = value[start_idx:end_idx]
                elif isinstance(value, torch.Tensor):
                    mb[key] = value[start_idx:end_idx]
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
        """Train the model on a batch of data."""
        self.model.train()

        # Split into microbatches
        microbatches = self._split_into_microbatches(input_, mb_spec)

        total_loss = 0.0
        total_tokens = 0.0

        # Zero gradients
        self.optimizer.zero_grad()

        # Process each microbatch
        for mb in microbatches:
            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**mb)
                    loss = loss_fn(outputs.logits, mb)
                    weight = loss_weight_fn(outputs.logits, mb)

                # Scale loss for accumulation
                scaled_loss = self.scaler.scale(loss / len(microbatches))
                scaled_loss.backward()
            else:
                outputs = self.model(**mb)
                loss = loss_fn(outputs.logits, mb)
                weight = loss_weight_fn(outputs.logits, mb)

                # Scale loss for accumulation
                scaled_loss = loss / len(microbatches)
                scaled_loss.backward()

            total_loss += loss.item() * weight
            total_tokens += weight

        # Normalize loss across microbatches and potentially across ranks
        if token_normalize_scope == "global" and dist.is_initialized():
            # Aggregate across all ranks
            loss_tensor = torch.tensor(
                [total_loss, total_tokens], device=torch.cuda.current_device()
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = loss_tensor.tolist()

        avg_loss = total_loss / max(total_tokens, 1e-8)

        # Optimizer step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Update learning rate
        if self.scheduler:
            self.scheduler.step()

        return {
            "loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "total_tokens": total_tokens,
        }

    @torch.no_grad()
    def forward(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, Dict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Run forward pass or inference on the model."""
        self.model.eval()

        # Split into microbatches
        microbatches = self._split_into_microbatches(input_, mb_spec)

        results = []

        for mb in microbatches:
            # Forward pass
            if (
                self.scaler
                and hasattr(self.config, "mixed_precision")
                and self.config.mixed_precision
            ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**mb)
            else:
                outputs = self.model(**mb)

            # Apply post hook if provided
            if post_hook:
                result = post_hook(outputs.logits, mb)
                results.append(result)
            else:
                results.append(outputs.logits)

        # Aggregate results
        if results:
            return aggregate_fn(results)
        return None

    def get_hf_model_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the model state dict in HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_distributed first.")

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            return self.model.state_dict()

    def save_model_to_hf(
        self,
        tokenizer: transformers.PreTrainedTokenizerFast,
        path: str,
        base_model_path: Optional[str] = None,
    ):
        """Save the model to HuggingFace format."""

        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_distributed first.")

        os.makedirs(path, exist_ok=True)

        # Save the state dict
        state_dict = self.get_hf_model_state_dict()
        torch.save(state_dict, os.path.join(path, "pytorch_model.bin"))

        # Save tokenizer
        tokenizer.save_pretrained(path)

        # Save config
        if base_model_path:
            config = transformers.AutoConfig.from_pretrained(base_model_path)
        else:
            config = transformers.AutoConfig.from_pretrained(self.config.path)
        config.save_pretrained(path)

    def load_model_from_hf(self, path: str):
        """Load the model from HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_distributed first.")

        state_dict = torch.load(
            os.path.join(path, "pytorch_model.bin"), map_location="cpu"
        )

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            self.model.load_state_dict(state_dict)

    def save_optimizer_state(self, path: str):
        """Save the optimizer state in a folder."""

        if self.optimizer is None:
            raise RuntimeError(
                "Optimizer not initialized. Call init_distributed first."
            )

        os.makedirs(path, exist_ok=True)

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

        # Save scheduler state if exists
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))

        # Save scaler state if exists
        if self.scaler is not None:
            torch.save(self.scaler.state_dict(), os.path.join(path, "scaler.pt"))

    def load_optimizer_state(self, path: str):
        """Load the optimizer state from a folder."""

        if self.optimizer is None:
            raise RuntimeError(
                "Optimizer not initialized. Call init_distributed first."
            )

        # Load optimizer state
        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu")
            )

        # Load scheduler state if exists
        scheduler_path = os.path.join(path, "scheduler.pt")
        if self.scheduler is not None and os.path.exists(scheduler_path):
            self.scheduler.load_state_dict(
                torch.load(scheduler_path, map_location="cpu")
            )

        # Load scaler state if exists
        scaler_path = os.path.join(path, "scaler.pt")
        if self.scaler is not None and os.path.exists(scaler_path):
            self.scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))
