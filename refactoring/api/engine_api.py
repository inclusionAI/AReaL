import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal

import torch

from refactoring.api.cli_args import EngineConfig, MicroBatchSpec, TrainingArgs


class Engine(abc.ABC):
    """Defines the interface for modules after backend initialization.

    Different backends (e.g., Megatron, Transformers) will implement
    this interface to provide their own training and evaluation methods.
    """

    def __init__(self, args: TrainingArgs, config: EngineConfig):
        self.args = args
        self.config = config

    def init_process_group(self):
        raise NotImplementedError()

    def train_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[torch.Tensor, Dict], float],
        version_steps: int,
        token_normalize_scope: Literal["global", "dp"] = "global",
    ) -> Dict:
        """Update the model with a batch of data and a loss function.

        :param input_: The input data.
        :type input_: Dict
        :param mb_spec: The micro-batch specification, which defines how the input data
            should be split into micro-batches.
        :type mb_spec: MicroBatchSpec
        :param loss_fn: The loss function. It takes the output of the forward pass and the
            input data, returning the loss.
        :type loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor]
        :param loss_weight_fn: This function is used to calculate the number of valid tokens
            when normalizing loss across micro batches and DP ranks. Can be `lambda: 1`
            if just taking the average over batches.
        :type loss_weight_fn: Callable[[torch.Tensor, Dict], float]
        :param version_steps: The global step counter for this experiment,
            used by the backend to determine the learning rate schedule.
        :type version_steps: int
        :param global_normalize_scope: The scope of token-wise loss normalization. Choices:
            global: average across all micro batches across DP ranks.
            dp: average across micro batches in current DP rank.
            Default to "global".
        :type global_normalize_scope: Literal["global", "dp"]
        """

        raise NotImplementedError()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function.

        This method wraps :meth:`forward` with a customized ``post_hook`` and ``aggregate_fn``.
        :param input_: The input data.
        :type input_: Dict
        :param loss_fn: The loss function. It takes the output of the forward pass and the
            input data, returning the loss.
        :type loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor]
        :return: The aggregated scalar loss if on the last pipe stage.
        :rtype: torch.Tensor | None
        """

        def _loss_fn(out, inp_):

            # To prevent calling data reordering.

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
        """Run the forward pass or inference on the model. Note that it is
        gradient-free.

        To train the model, use :meth:`train_batch` instead.
        :param input_: The input data.
        :type input_: Dict
        :param post_hook: A function to apply to the output after the forward pass.
            It takes the output tensor and the input data, returning an arbitrary result.
            With a post_hook, we can process the output in mini-batches,
            reducing memory usage for operations such as gathering log-probabilities.
            If None, this function just returns the output tensor.
        :type post_hook: Callable[[torch.Tensor, Dict], Any] | None
        :param aggregate_fn: A function to aggregate the results of the post_hook.
        :type aggregate_fn: Callable[[List[Any]], Any]
        :return: The aggregated result of the post_hook from the last pipeline stage. Returns None otherwise.
        :rtype: Any | None
        """

        raise NotImplementedError()


@dataclass
class EngineFactory:
    args: TrainingArgs

    def make_engine(self, config: EngineConfig) -> Engine:
        """Create an engine based on the configuration."""
        if config.backend == "fsdp":
            from refactoring.impl.model.fsdp_engine import FSDPEngine

            return FSDPEngine(self.args, config)
        else:
            raise ValueError(f"Unsupported engine type: {config.backend}")
