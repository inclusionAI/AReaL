from __future__ import annotations

import abc
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import ParallelStrategy
from areal.api.io_struct import (
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.utils.data import (
    MicroBatchList,
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
)

if TYPE_CHECKING:
    from areal.api.workflow_api import RolloutWorkflow


@dataclass
class ForwardBackwardOutputs:
    mb_outputs: list[torch.Tensor] | None
    losses: list[torch.Tensor] | None


class TrainEngine(abc.ABC):
    def __init__(self):
        self.is_offload = None
        self.sp_group = None
        self.parallel_helper = None
        self.dp_group = None

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        """Initialize PyTorch distributed communication groups.

        Parameters
        ----------
        parallel_strategy : ParallelStrategy, optional
            The parallel strategy configuration for distributed training, by default None
        """
        raise NotImplementedError()

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

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        """Get the data parallel communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The data parallel communication group
        """
        raise NotImplementedError()

    @property
    def data_parallel_rank(self) -> int:
        """Get the rank of the current process in the data parallel group.

        Returns
        -------
        int
            The rank of the current process in the data parallel group
        """
        raise NotImplementedError()

    @property
    def data_parallel_world_size(self) -> int:
        """Get the world size of the data parallel group.

        Returns
        -------
        int
            The world size of the data parallel group
        """
        raise NotImplementedError()

    def current_data_parallel_head(self) -> int:
        """Get the current data parallel head rank.

        Returns
        -------
        int
            The rank of the current data parallel head
        """
        raise NotImplementedError()

    def is_data_parallel_head(self) -> bool:
        """Check if the current rank is the data parallel head of the current engine.

        Returns
        -------
        bool
            True if the current rank is the data parallel head, False otherwise
        """
        raise NotImplementedError()

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        """Get the context and model parallel communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The context and model parallel communication group
        """
        raise NotImplementedError()

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        """Get the CPU communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The CPU communication group
        """
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory of models."""

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

    def update_weights(self, meta: WeightUpdateMeta):
        """Update weights to the inference engine in a blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        """
        raise NotImplementedError()

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        """Connect to an inference engine for online training.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine to connect to
        """
        raise NotImplementedError()

    def set_version(self, version: int):
        """Set the current weight version in the training engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        raise NotImplementedError()

    def get_version(self) -> int:
        """Get the current weight version in the training engine.

        Returns
        -------
        int
            The current weight version number
        """
        raise NotImplementedError()

    def save(self, meta: SaveLoadMeta):
        """Save model weights and optimizer states for later use.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to save
        """
        raise NotImplementedError()

    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to load
        """
        raise NotImplementedError()

    def _split_micro_batch(
        self,
        input_: dict[str, Any],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor] | None = None,
    ) -> tuple[Iterable[dict[str, torch.Tensor]], MicroBatchList]:
        """Split input batch into micro-batches for gradient accumulation.

        This method prepares the input data for micro-batch processing by splitting
        the batch according to the configured micro-batch specification, computing
        total loss weights across all micro-batches, and creating an iterator that
        yields micro-batch dictionaries with necessary metadata.

        Parameters
        ----------
        input_ : dict[str, Any]
            The input batch dictionary.
        loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor], optional
            A function to compute the loss weight for each micro-batch.

        Returns
        -------
        tuple[Iterable[dict[str, torch.Tensor]], MicroBatchList]
            A tuple containing:
            - An iterable of micro-batch dictionaries.
            - A MicroBatchList object containing metadata about the micro-batches.
        """
        raise NotImplementedError()

    def _forward_compute_mb(
        self,
        mb_input: dict[str, Any],
        post_process_fn: Callable[[torch.Tensor, dict[str, Any]], Any],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, dict]]]:
        """Compute forward pass and prepare loss function for a single micro-batch.

        This method performs the forward pass on a single micro-batch and returns
        the output tensor along with a loss function closure. The loss function
        closure is used by the training framework to compute loss and perform
        backward pass during gradient accumulation.

        Parameters
        ----------
        mb_input : dict[str, Any]
            A dictionary containing the micro-batch input data. The exact structure
            depends on the engine implementation, but typically includes packed/padded
            tensors, padding metadata, and total loss weight.
        loss_fn : Callable[[torch.Tensor, dict[str, Any]], torch.Tensor], optional
            A function that computes the normalized loss given model output and
            input data. Optional for pure forward passes.
        loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor], optional
            A function that computes the weight for this micro-batch, typically
            the number of tokens. Used for proper loss scaling across micro-batches.
            Optional when `loss_fn` is not provided.
        **kwargs
            Additional keyword arguments that may be used by specific implementations,
            such as model reference for pipeline parallel, batch type, post-processing
            hooks, etc.

        Returns
        -------
        tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, dict]] | None]
            A tuple containing:
            - The model output tensor (logits) from the forward pass
            - A callable loss function that takes the output tensor and returns:
              - A loss tensor (scaled appropriately for gradient accumulation)
              - A dictionary with additional data
              If `loss_fn` is None (e.g., pure forward pass), the callable can be None.
        """
        raise NotImplementedError()

    def optimizer_zero_grad(self):
        """Zero out all gradients in the optimizer.

        This method clears the gradients of all model parameters before starting
        a new training step. For engines that use gradient accumulation across
        micro-batches, this should be called once at the beginning of each training
        step, not for each micro-batch.

        Note
        ----
        This should be called before starting a new training step, typically
        at the beginning of `train_batch`. For distributed engines, this may also
        clear gradient buffers used for gradient accumulation.
        """
        raise NotImplementedError()

    def optimizer_step(self):
        """Perform a single optimization step.

        This method executes one optimizer step, which typically includes gradient
        clipping (if configured), the optimizer update (e.g., AdamW step), and may
        integrate learning rate scheduling depending on the optimizer implementation.

        Returns
        -------
        dict[str, float]
            A dictionary containing training statistics:
            - `update_successful`: 1.0 if the update succeeded, 0.0 otherwise
              (e.g., if gradients were NaN/Inf and the update was skipped)
            - `grad_norm`: The gradient norm after clipping, or NaN if gradient
              clipping is disabled or not computed
            - `lr`: The current learning rate after the step
        """
        raise NotImplementedError()

    def lr_scheduler_step(self):
        """Advance the learning rate scheduler by one step.

        This method updates the learning rate according to the configured scheduler
        (e.g., cosine decay, linear warmup). The scheduler step is typically called
        after each optimizer step or periodically (e.g., once per PPO update).

        Note
        ----
        For some optimizers (e.g., Megatron's integrated optimizer), the learning
        rate scheduling may be integrated into the optimizer step, in which case
        this method may advance a separate scheduler or be a no-op.
        """
        raise NotImplementedError()

    def step_lr_scheduler(self):
        """Step the learning rate scheduler.

        Since PPO uses minibatch updates, this method should be called periodically
        (e.g., once per PPO step). It is separated from train_batch to allow
        for more flexible learning rate scheduling.
        """
        return self.lr_scheduler_step()

    def forward_backward_batch(
        self,
        data_iterator: Iterable[dict[str, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor] | None = None,
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor] | None = None,
        return_outputs: bool = False,
        forward_only: bool = False,
    ) -> ForwardBackwardOutputs:
        """Process micro-batches through forward and optionally backward pass.

        This method iterates over all micro-batches in the data iterator and processes
        them through the model. For each micro-batch, it performs forward pass and
        optionally backward pass (for training), collecting outputs or losses as
        specified by the parameters.

        Parameters
        ----------
        data_iterator : Iterable[dict[str, torch.Tensor]]
            `data_iterator` is typically produced by converting a `MicroBatchList` into
            an iterator (e.g., via `create_mb_iterator`), yielding per-micro-batch
            payloads and any metadata computed during splitting for downstream use.
        loss_fn : Callable[[torch.Tensor, dict[str, Any]], torch.Tensor], optional
            A function that computes the normalized loss given model output and
            input data. Required when `forward_only=False` or when `return_outputs=False`
            in forward-only mode. By default None.
        loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor], optional
            A function that computes the weight for each micro-batch, typically
            the number of tokens. Used for proper loss scaling. By default None.
        return_outputs : bool, optional
            If True, collect and return model outputs (logits) instead of losses.
            Only used when `forward_only=True`. By default False.
        forward_only : bool, optional
            If True, only perform forward pass (no backward pass or gradient computation).
            If False, perform both forward and backward passes for training. By default False.

        Returns
        -------
        ForwardBackwardOutputs
            A dataclass containing:
            - `mb_outputs`: List of output tensors (one per micro-batch) when
              `return_outputs=True` and `forward_only=True`. Each output may be
              processed by `output_post_hook` if provided. None otherwise.
            - `losses`: List of loss tensors (one per micro-batch) when collecting
              losses (training or eval mode). None when `return_outputs=True`.
        """
        raise NotImplementedError()

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
        input_ : Dict[str, Any]
            The input data for model forward pass and the loss function.
            Redundant entries are allowed.
        loss_fn : Callable[..., torch.Tensor]
            The loss function. For actor (is_critic=False), it receives
            (logprobs, entropy, input_data). For critic (is_critic=True),
            it receives (values, input_data). Returns a scalar normalized loss.
        loss_weight_fn : Callable[[Dict[str, Any]], torch.Tensor]
            A function used to calculate the weight of each micro-batch. Since
            loss_fn normalizes the loss for a micro-batch, we need a corresponding
            weight for each micro-batch to normalize the loss globally. The weight
            is usually the number of response tokens in the batch.

        Returns
        -------
        Dict[str, float]
            Scalar statistics after training, e.g., the current learning rate,
            gradient norm, etc.
        """
        self.optimizer_zero_grad()
        _data_iterator, _ = self._split_micro_batch(input_, loss_weight_fn)
        self.forward_backward_batch(_data_iterator, loss_fn, loss_weight_fn)
        return self.optimizer_step()

    @torch.no_grad()
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
        input_ : Dict[str, Any]
            The input data for model forward pass and the loss function.
            Redundant entries are allowed.
        loss_fn : Callable[..., torch.Tensor]
            The loss function. For actor (is_critic=False), it receives
            (logprobs, entropy, input_data). For critic (is_critic=True),
            it receives (values, input_data). Returns a scalar normalized loss.
        loss_weight_fn : Callable[[Dict[str, Any]], torch.Tensor]
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
        _data_iterator, _ = self._split_micro_batch(input_, loss_weight_fn)
        output = self.forward_backward_batch(
            _data_iterator, loss_fn, loss_weight_fn, forward_only=True
        )
        loss = torch.stack(output.losses).sum(dtype=torch.float32)
        dist.all_reduce(loss, group=self.dp_group)
        return loss

    @torch.no_grad()
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Run the forward pass or inference on the model.

        Note
        ----
        This operation is gradient-free.

        Parameters
        ----------
        input_ : Dict[str, Any]
            The input data for model forward pass. Redundant entries are allowed.
        output_seqlens : List[int], optional
            The desired output sequence lengths. If None, assumes that the output
            has the same lengths as inputs, by default None.
        aggregate_fn : Callable[[List[Any]], Any], optional
            A function to aggregate micro-batched outputs, by default torch.cat.

        Returns
        -------
        Any or None
            For actor (is_critic=False): logprobs tensor aggregated by `aggregate_fn`.
            For critic (is_critic=True): values tensor aggregated by `aggregate_fn`.
        """
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None

        _data_iterator, mb_list = self._split_micro_batch(input_)

        result = self.forward_backward_batch(
            _data_iterator,
            forward_only=True,
            return_outputs=True,
        )

        res = aggregate_fn(result.mb_outputs)
        output_seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
        unpacked = unpack_sequence(res, lens=output_seqlens, dim=0)
        reordered = reorder_list(unpacked, mb_list.backward_indices)
        return pad_and_stack_tensors_along_first_dim(reordered)

    @torch.no_grad()
    def forward(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> Any | None:
        """
        alias for forward_batch
        """
        return self.forward_batch(input_, output_seqlens, aggregate_fn)

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

    def onload(self):
        raise NotImplementedError()

    def offload(self):
        raise NotImplementedError()


class InferenceEngine(abc.ABC):
    def initialize(self, *args, **kwargs):
        """Initialize environments and launch the background thread for asynchronous distributed inference.

        For remote inference engines, this serves as a client and connects to the inference servers.
        For local inference engines, this creates an LLM engine on the local GPU.

        Parameters
        ----------
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments
        """
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory for the local inference engine."""
        raise NotImplementedError()

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        """Launch a local inference server via subprocess and return its connection info.

        By default, an `InferenceEngine` instance acts as a client that connects to an existing
        remote inference server without occupying GPU resources. This is the typical usage in
        SPMD mode, where each training process has an attached inference client.

        This method enables launching a local inference server process, which is useful for:

        1. **Single-controller mode**: Launch a local server to serve the `InferenceEngine`
           instance with direct GPU worker control.

        2. **Standalone inference**: Use AReaL's inference engine in independent scripts or notebooks
           for running agentic workflows without managing separate server processes.

        Parameters
        ----------
        server_args : Dict[str, Any]
            CLI arguments for the inference server (e.g., model path, GPU indices,
            port numbers, backend-specific settings)

        Returns
        -------
        LocalInfServerInfo
            Information about the launched server, including connection details and process metadata

        See Also
        --------
        teardown_server : Teardown the server launched by this method
        """
        raise NotImplementedError()

    def teardown_server(self):
        """Teardown the inference server launched by `launch_server`."""
        raise NotImplementedError()

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        raise NotImplementedError()

    def init_weights_update_group(self, meta: WeightUpdateMeta) -> Future[None]:
        """Initialize the weight update process group for distributed weight updates.

        This method should be called before performing any weight updates to ensure
        that the necessary communication groups are set up correctly.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update, such as the
            type of communication backend and allocation mode.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Returns
        -------
        Future[None]
            A future object representing the asynchronous initialization operation.
        """
        raise NotImplementedError()

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights in the inference engine in a non-blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        param_specs : List[ParamSpec]
            A list of parameter specifications for the weights to be updated

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        raise NotImplementedError()

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights in the inference engine from disk in a non-blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        raise NotImplementedError()

    def set_version(self, version: int) -> None:
        """Set the current weight version in the inference engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        raise NotImplementedError()

    def get_version(self) -> int:
        """Get the current weight version in the inference engine.

        Returns
        -------
        int
            The current weight version number
        """
        raise NotImplementedError()

    def submit(
        self,
        data: dict[str, Any],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        should_accept_fn: Callable | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Submit a request to the inference engine and return immediately.

        Should be used together with subsequent `wait`.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data for rollout. Used by the user's customized workflow implementation.
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "areal.workflow.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.
        should_accept_fn : Callable, optional
            A function used to decide whether to accept a specific trajectory, i.e., dynamic filtering.
            It takes a complete trajectory output by the workflow, and returns a bool, by default None.
        """
        raise NotImplementedError()

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for a specified number of requests to complete, with a timeout.

        Should be used together with preceding `submit`.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float, optional
            Timeout in seconds. Exceeding the timeout will raise a `TimeoutError`, by default None
        raise_timeout : bool, optional
            Whether to raise a `TimeoutError` when the timeout is exceeded,
            otherwise return an empty list, by default True

        Returns
        -------
        list[dict[str, Any] | None]
            A list of trajectory dictionaries. Each element may be None for rejected trajectories.

        Raises
        ------
        TimeoutError
            If the timeout is exceeded before enough trajectories are collected
        """
        raise NotImplementedError()

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch of requests to the inference engine and wait for the results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.

        See `workflow_api.py` for concrete implementation.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "areal.workflow.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.

        Returns
        -------
        Dict[str, Any]
            A concatenated batch of trajectory results
        """
        raise NotImplementedError()

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Asynchronously submit and wait until a full batch is ready with controlled staleness.

        See `workflow_api.py` for concrete implementation.

        .. warning::

            This method caches an internal data generator on the first call.
            The ``dataloader``, ``workflow``, ``workflow_kwargs``, and
            ``should_accept_fn`` parameters are captured at the first invocation
            and reused in all subsequent calls. Passing different arguments in
            later calls will **not** take effect.

            If you need to switch configurations mid-training, consider:

            - Using a separate inference engine instance
            - Using the :meth:`submit` / :meth:`wait` pattern for finer control

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from for batch preparation
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "areal.workflow.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.
        should_accept_fn : Callable, optional
            A function to decide whether to accept a trajectory, by default None

        Returns
        -------
        Dict[str, Any]
            A full batch of trajectory results with controlled staleness
        """
        raise NotImplementedError()

    def pause_generation(self):
        """Pause the generation of inference engine.

        Used during updating weights from distributed or disk.
        """
        raise NotImplementedError()

    def continue_generation(self):
        """Continue the generation of inference engine."""
        raise NotImplementedError()

    def pause(self):
        """Pause request submission for async rollout.

        Used during evaluation to prevent data over-generation.
        """
        raise NotImplementedError()

    def resume(self):
        """Resume request submission for async rollout."""
        raise NotImplementedError()

    def offload(self):
        """Offload model from GPU to CPU for inference engine."""
        raise NotImplementedError()

    def onload(self, tags: list[str] | None = None):
        """Onload model from CPU to GPU for inference engine.

        Parameters
        ----------
        tags : list[str], optional
            Tags to onload specific components. If None, onloads all components.
        """
        raise NotImplementedError()

    def export_stats(self) -> dict[str, float]:
        """Export the statistics recorded during workflow execution in the process.

        Workflow should only record scalar metrics like "rewards".
        These metrics will be reduced in the controller side.

        Note
        ----
        This method should be only called by the controller.

        Returns
        -------
        dict[str, float]
            The recorded scalar statistics.
        """
        raise NotImplementedError()
