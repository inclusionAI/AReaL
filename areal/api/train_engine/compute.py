from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

if TYPE_CHECKING:
    from areal.api.workflow_api import RolloutWorkflow
    from areal.utils.data import MicroBatchList


class TrainEngineComputeMixin(abc.ABC):
    @abc.abstractmethod
    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed training and load models.

        This method should be called after `create_process_group`.

        Parameters
        ----------
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self, mode: bool = True):
        """Set the engine to training mode.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the engine to training mode, by default True
        """
        raise NotImplementedError()

    def eval(self):
        """Set the engine to evaluation mode.

        This is a convenience method that calls `self.train(False)`.
        """
        return self.train(False)

    @abc.abstractmethod
    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        granularity: int = 1,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.
        Should note that this is a simple rollout engine method forwarding with
        distributed data management.

        Parameters
        ----------
        data : list[dict[str, Any]]
            A list of input data dictionaries.
        granularity : int, optional
            The granularity of the rollout, by default 1.
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str | None, optional
            The workflow to use for rollout generation, by default None.
        workflow_kwargs : dict[str, Any] | None, optional
            Keyword arguments to pass to the workflow constructor, by default None.

        Returns
        -------
        dict[str, Any]
            The rollout results.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        granularity: int = 1,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
    ) -> dict[str, Any]:
        """Prepare a batch of data for training from a dataloader.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The dataloader to fetch data from.
        granularity : int, optional
            The granularity of the rollout, by default 1.
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str | None, optional
            The workflow to use for rollout generation, by default None.
        workflow_kwargs : dict[str, Any] | None, optional
            Keyword arguments to pass to the workflow constructor, by default None.
        should_accept_fn : Callable[[dict[str, Any]], bool] | str | None, optional
            A function to filter trajectories, by default None.

        Returns
        -------
        dict[str, Any]
            The prepared batch data.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def optimizer_zero_grad(self):
        """Zero out all gradients in the optimizer."""
        raise NotImplementedError()

    @abc.abstractmethod
    def optimizer_step(self):
        """Perform a single optimization step.

        Returns
        -------
        dict[str, float]
            Training statistics containing ``update_successful``, ``grad_norm``, and ``lr``.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def lr_scheduler_step(self):
        """Advance the learning rate scheduler by one step."""
        raise NotImplementedError()

    def step_lr_scheduler(self):
        """This is an alias for `lr_scheduler_step()`."""
        return self.lr_scheduler_step()

    @abc.abstractmethod
    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> None:
        """Process micro-batches through forward and optionally backward pass.

        Parameters
        ----------
        mb_list : MicroBatchList
            The micro-batch list, which is iterable and yields MicroBatchItem tuples.
        process_output_fn : Callable[[torch.Tensor, dict[str, Any]], torch.Tensor | None]
            A function that processes the model output (logits) and returns the loss tensor.
            If the returned loss is not None, backward() will be called on it.
            Results can be collected via closure if needed.
            Signature: ``(logits: Tensor, inputs: dict) -> loss | None``
        forward_only : bool, optional
            If True, skip backward pass. Default is False.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Update the model with a batch of data and a loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : dict[str, Any]
            The input data for model forward pass and the loss function.
            Redundant entries are allowed.
        loss_fn : Callable[..., torch.Tensor]
            The loss function. For actor (is_critic=False), it receives
            (logprobs, entropy, input_data). For critic (is_critic=True),
            it receives (values, input_data). Returns a scalar normalized loss.
        loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor]
            A function used to calculate the weight of each micro-batch. Since
            loss_fn normalizes the loss for a micro-batch, we need a corresponding
            weight for each micro-batch to normalize the loss globally. The weight
            is usually the number of response tokens in the batch.

        Returns
        -------
        dict[str, float]
            Scalar statistics after training, e.g., the current learning rate,
            gradient norm, etc.
        """
        raise NotImplementedError()

    @torch.no_grad()
    @abc.abstractmethod
    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : dict[str, Any]
            The input data for model forward pass and the loss function.
            Redundant entries are allowed.
        loss_fn : Callable[..., torch.Tensor]
            The loss function. For actor (is_critic=False), it receives
            (logprobs, entropy, input_data). For critic (is_critic=True),
            it receives (values, input_data). Returns a scalar normalized loss.
        loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor]
            A function used to calculate the weight of each micro-batch. Since
            loss_fn normalizes the loss for a micro-batch, we need a corresponding
            weight for each micro-batch to normalize the loss globally. The weight
            is usually the number of response tokens in the batch.

        Returns
        -------
        torch.Tensor or None
            A scalar loss or None. The evaluation statistics should be aggregated
            with `stats_tracker`.
        """
        raise NotImplementedError()

    @torch.no_grad()
    @abc.abstractmethod
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> torch.Tensor:
        """Run the forward pass or inference on the model.

        Note
        ----
        This operation is gradient-free.

        Parameters
        ----------
        input_ : dict[str, Any]
            The input data for model forward pass. Redundant entries are allowed.
        output_seqlens : list[int], optional
            The desired output sequence lengths. If None, assumes that the output
            has the same lengths as inputs, by default None.
        aggregate_fn : Callable[[list[Any]], Any], optional
            A function to aggregate micro-batched outputs, by default torch.cat.

        Returns
        -------
        Any
            For actor (is_critic=False): logprobs tensor aggregated by `aggregate_fn`.
            For critic (is_critic=True): values tensor aggregated by `aggregate_fn`.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def forward(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> torch.Tensor:
        return self.forward_batch(input_, output_seqlens, aggregate_fn)

    @abc.abstractmethod
    def export_stats(self) -> dict[str, float]:
        """Export the statistics recorded in this engine process.

        Note
        ----
        Statistics will be all-reduced across the data parallel group
        and broadcasted from the last pipeline parallel stage.

        Returns
        -------
        dict[str, float]
            The exported scalar statistics.
        """
        raise NotImplementedError()
