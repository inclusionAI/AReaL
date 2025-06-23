import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import transformers

from arealite.api.cli_args import EngineConfig, MicroBatchSpec, TrainingArgs
from arealite.api.io_struct import FinetuneSpec
from arealite.api.llm_client_api import LLMClient
from realhf.api.cli_args import ParallelismConfig


class SPMDWrapper(abc.ABC):
    """A wrapper over the training/inference backends (e.g., FSDP, SGLang).
    We following the design of existing libraries, such as Megatron-LM and
    pytorch FSDP, which are mostly SPMD-based.
    """

    def __init__(self, args: TrainingArgs, engine_config: EngineConfig):
        self.args = args
        self.engine_config = engine_config

    def init_distributed(self, config: ParallelismConfig, ft_spec: FinetuneSpec):
        """Initialize distributed communication groups and models.

        Models may not be loaded during __init__, but when calling this method.
        """
        raise NotImplementedError()

    def train(self, mode: bool = True):
        """Set the module in training mode."""
        raise NotImplementedError()

    def eval(self):
        """Set the module in evaluation mode."""
        return self.train(False)

    def train_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
        version_steps: int,
        token_normalize_scope: Literal["global", "dp"] = "global",
    ) -> Dict:
        """Update the model with a batch of data and a loss function."""
        raise NotImplementedError()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function."""

        def _loss_fn(out, inp_):
            return float(loss_fn(out, inp_))

        return self.forward(
            input_=input_,
            mb_spec=mb_spec,
            post_hook=_loss_fn,
            aggregate_fn=sum,
        )

    def forward(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, Dict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Run the forward pass or inference on the model."""
        raise NotImplementedError()

    def get_version(self) -> int:
        raise NotImplementedError()

    def save_model_to_hf(
        self,
        path: str,
        tokenizer: Optional[transformers.PreTrainedTokenizerFast] = None,
        base_model_path: Optional[str] = None,
    ):
        raise NotImplementedError()

    def load_model_from_hf(self, path: str):
        raise NotImplementedError()

    def save_optimizer_state(self, path: str):
        """Save the optimizer state in a folder."""
        raise NotImplementedError()

    def load_optimizer_state(self, path: str):
        """Load the optimizer state in a folder."""
        raise NotImplementedError()

    def update_weights_to(self, llm_client: LLMClient):
        """Update the weights to the server by sending requests to the client."""
        raise NotImplementedError()


@dataclass
class EngineFactory:
    args: TrainingArgs

    def make_engine(self, engine_config: EngineConfig) -> SPMDWrapper:
        """Create an engine based on the configuration."""
        if engine_config.backend.type == "fsdp":
            from arealite.impl.fsdp_wrapper import FSDPEngine

            return FSDPEngine(self.args, engine_config)
        else:
            raise ValueError(f"Unsupported engine type: {engine_config.backend.type}")
