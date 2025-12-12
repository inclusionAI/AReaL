import abc
from collections.abc import Callable
from typing import Any

import torch

from areal.api.engine_api import TrainEngine
from areal.utils.data import (
    MicroBatchList,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
)


class BaseTrainEngine(TrainEngine):
    """Base implementation of TrainEngine with common training/evaluation logic.

    This class provides default implementations for :meth:`train_batch`,
    :meth:`eval_batch`, and :meth:`forward_batch` that orchestrate the
    micro-batch splitting, forward/backward passes, and loss aggregation.

    Subclasses must implement the abstract methods to provide engine-specific
    behavior for micro-batch processing and loss computation.
    """

    @property
    def backend_requires_loss_tensor(self) -> bool:
        """Whether the backend framework requires loss to always be a tensor.

        Some backends like Megatron need loss to be a tensor even when
        forward_only=True, because their schedule still processes the loss.
        """
        return False

    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Execute a full training step on a batch.

        Performs forward pass, loss computation, backward pass, and optimizer step.

        Parameters
        ----------
        input_
            The input batch dictionary containing model inputs.
        loss_fn
            A function to compute the loss from model outputs.
        loss_weight_fn
            A function to compute per-sample loss weights for normalization.

        Returns
        -------
        dict[str, float]
            Training metrics including ``update_successful``, ``grad_norm``, and ``lr``.
        """
        self._ensure_ready()

        self.optimizer_zero_grad()

        mb_list, total_loss_weight = self.split_micro_batch(input_, loss_weight_fn)

        def process_output(
            logits: torch.Tensor, inputs: Any
        ) -> tuple[torch.Tensor, None]:
            loss = self.compute_loss(
                output=logits,
                inputs=inputs,
                loss_fn=loss_fn,
                total_loss_weight=total_loss_weight,
                loss_weight_fn=loss_weight_fn,
            )
            return loss, None

        self.forward_backward_batch(mb_list, process_output, forward_only=False)
        return self.optimizer_step()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate a batch without gradient computation.

        Parameters
        ----------
        input_
            The input batch dictionary containing model inputs.
        loss_fn
            A function to compute the loss from model outputs.
        loss_weight_fn
            A function to compute per-sample loss weights for normalization.

        Returns
        -------
        torch.Tensor | None
            The aggregated evaluation loss reduced across data parallel ranks.
        """
        self._ensure_ready()

        mb_list, total_loss_weight = self.split_micro_batch(input_, loss_weight_fn)

        losses: list[torch.Tensor] = []

        def process_output(
            logits: torch.Tensor, inputs: Any
        ) -> tuple[torch.Tensor | None, torch.Tensor]:
            loss = self.compute_loss(
                output=logits,
                inputs=inputs,
                loss_fn=loss_fn,
                total_loss_weight=total_loss_weight,
                loss_weight_fn=loss_weight_fn,
            )
            losses.append(loss.detach())
            # Some backends (e.g., Megatron) require loss to be a tensor even when forward_only.
            if self.backend_requires_loss_tensor:
                return loss, loss
            return None, loss

        self.forward_backward_batch(mb_list, process_output, forward_only=True)
        return self.aggregate_eval_losses(losses)

    @torch.no_grad()
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> torch.Tensor:
        """Perform forward-only inference on a batch.

        Parameters
        ----------
        input_
            The input batch dictionary containing model inputs.
        output_seqlens
            Optional sequence lengths for unpacking outputs. If None,
            inferred from ``cu_seqlens`` in the packed input.
        aggregate_fn
            Function to aggregate outputs from micro-batches (default: ``torch.cat``).

        Returns
        -------
        torch.Tensor
            The processed model outputs, padded and stacked along the batch dimension.
        """
        self._ensure_ready()

        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None

        mb_list, _ = self.split_micro_batch(input_)
        outputs: list[torch.Tensor] = []

        def process_output(
            logits: torch.Tensor, inputs: Any
        ) -> tuple[torch.Tensor | None, torch.Tensor]:
            result = self.process_forward_output(logits, inputs)
            outputs.append(result)
            # Some backends (e.g., Megatron) require loss to be a tensor even when forward_only.
            if self.backend_requires_loss_tensor:
                dummy_loss = torch.tensor(
                    1.0, device=result.device, requires_grad=False
                )
                return dummy_loss, result
            return None, result

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        def aggregate_and_reorder(result: list[torch.Tensor]) -> torch.Tensor:
            res = aggregate_fn(result)
            seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
            unpacked = unpack_sequence(res, lens=seqlens, dim=0)
            reordered = reorder_list(unpacked, mb_list.backward_indices)
            return pad_and_stack_tensors_along_first_dim(reordered)

        return self.aggregate_forward_outputs(outputs, aggregate_and_reorder)

    def _ensure_ready(self) -> None:
        """Hook for subclasses to verify engine state before batch processing.

        Subclasses should override this to check initialization status,
        verify model is on correct device, etc.
        """
        pass

    @abc.abstractmethod
    def split_micro_batch(
        self,
        input_: dict[str, Any],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor] | None = None,
    ) -> tuple[MicroBatchList, torch.Tensor]:
        """Split input batch into micro-batches for gradient accumulation.

        Parameters
        ----------
        input_ : dict[str, Any]
            The input batch dictionary.
        loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor], optional
            A function to compute the loss weight for each micro-batch.

        Returns
        -------
        tuple[MicroBatchList, torch.Tensor]
            The MicroBatchList (which is iterable) and total_loss_weight.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_compute_mb(
        self,
        mb_input: Any,
        process_output_fn: Callable[
            [torch.Tensor, Any], tuple[torch.Tensor | None, Any]
        ],
        **kwargs,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor | None, Any]]]:
        """Compute forward pass and prepare loss function closure for a single micro-batch.

        Parameters
        ----------
        mb_input : Any
            The micro-batch input data. The actual type depends on the engine implementation.
        process_output_fn : Callable[[torch.Tensor, Any], tuple[torch.Tensor | None, Any]]
            A function that processes the model output and returns (loss, result).
        **kwargs
            Additional keyword arguments for specific implementations.

        Returns
        -------
        tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor | None, Any]]]
            The model output (logits) and a callable that computes loss and returns
            (loss_or_none, result) for gradient accumulation.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_loss(
        self,
        output: torch.Tensor,
        inputs: Any,
        total_loss_weight: torch.Tensor,
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for a single micro-batch.

        Parameters
        ----------
        output
            Model forward output tensor.
        inputs
            Engine-specific micro-batch context returned by :meth:`split_micro_batch`.
            For FSDPEngine, this should be an instance of ``FSDPTrainContext``.
            Subclasses should define their own context dataclass and document
            the expected type.
        total_loss_weight
            Total loss weight for normalization across micro-batches.
        loss_fn
            Loss computation function.
        loss_weight_fn
            Function to compute loss weight from micro-batch inputs.

        Note
        ----
        This method is only used in :meth:`train_batch` and :meth:`eval_batch`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def aggregate_eval_losses(
        self,
        losses: list[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate evaluation losses from all micro-batches and reduce across data parallel group.

        Note
        ----
        This method is only used in :meth:`eval_batch`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def process_forward_output(
        self,
        output: torch.Tensor,
        inputs: Any,
    ) -> torch.Tensor:
        """Process forward output for a single micro-batch.

        Parameters
        ----------
        output
            Model forward output tensor.
        inputs
            Engine-specific micro-batch context returned by :meth:`split_micro_batch`.
            For FSDPEngine, this should be an instance of ``FSDPTrainContext``.
            Subclasses should define their own context dataclass and document
            the expected type.

        Note
        ----
        This method is only used in :meth:`forward_batch`.
        """
        raise NotImplementedError()

    def aggregate_forward_outputs(
        self,
        result: list[torch.Tensor],
        aggregate_fn: Callable[[list[torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate forward outputs from all micro-batches and broadcast across pipeline stages.

        Note
        ----
        This method is only used in :meth:`forward_batch`.
        """
        return aggregate_fn(result)
