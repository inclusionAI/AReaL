import abc
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import ParallelStrategy
from areal.api.io_struct import (
    ModelRequest,
    ModelResponse,
    ParamSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)

if TYPE_CHECKING:
    from areal.api.workflow_api import RolloutWorkflow


class DistributedBatchMemory:
    pass


class TrainController(abc.ABC):
    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        """Initialize PyTorch distributed communication groups.

        Parameters
        ----------
        parallel_strategy : ParallelStrategy, optional
            The parallel strategy configuration for distributed training, by default None
        """
        raise NotImplementedError()

    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed training and load models."""
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
    def parallelism_group(self) -> dist.ProcessGroup:
        """Get the global communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The global communication group
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy(self):
        """Destroy the engine and release GPU memory."""
        raise NotImplementedError()

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
    def upload_weights(self, meta: WeightUpdateMeta):
        """Upload weights to the inference engine."""
        raise NotImplementedError()

    def get_param_specs(
        self, weight_chunked_mem_mb: int = 1024
    ) -> List[List[ParamSpec]]:
        """Get the parameter specifications for the model.

        Parameters
        ----------
        weight_chunked_mem_mb : int, optional
            Memory size in MB for weight chunking, by default 1024

        Returns
        -------
        List[List[ParamSpec]]
            List of parameter specifications for the model
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
        """Save model weights (and optimizer states) for later use."""
        raise NotImplementedError()

    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file."""
        raise NotImplementedError()

    def step_lr_scheduler(self):
        """Step learning rate scheduler.

        Since PPO uses minibatch updates, this method just need to be called once after a few train_batch calls.
        It is separated from train_batch to allow for more flexible scheduling.
        """
        raise NotImplementedError()

    def train_batch(
        self,
        input_: DistributedBatchMemory,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> Dict[str, float]:
        """Update the model with a batch of data and a loss function."""
        raise NotImplementedError()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: DistributedBatchMemory,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function."""
        raise NotImplementedError()

    @torch.no_grad()
    def forward(
        self,
        input_: DistributedBatchMemory,
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, TensorDict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Run the forward pass or inference on the model. Note that it is gradient-free."""
        raise NotImplementedError()


class RolloutController(abc.ABC):
    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed inference and load models."""
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory for the local inference engine."""

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

    def update_weights(self, meta: WeightUpdateMeta) -> Future:
        """Update weights in the inference engine in a non-blocking manner.

        The reason for using a non-blocking API is that we want this API to be
        compatible with XCCL collective communications, e.g.::

            fut = rollout.update_weights(meta)
            actor.upload_weights(meta)
            fut.result()

        Note that the `upload_weights` API of `TrainEngine` is blocking.
        If this API is blocking as well, then we will not trigger `actor.upload_weights`,
        and weight updates will get stuck.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future
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
        data: Dict[str, Any],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> None:
        """Submit a request to the inference engine and return immediately.

        Should be used together with subsequent `wait`.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data for rollout. Used by the user's customized workflow implementation.
        workflow : RolloutWorkflow, optional
            The workflow instance to run. Note that a single workflow instance can run multiple data.
            Use `workflow` when you want to share some resources between different rollouts.
            Either `workflow` or `workflow_builder` should be specified, by default None.
        workflow_builder : Callable, optional
            A builder to create a workflow instance to run, guaranteed for source separation.
            Either `workflow` or `workflow_builder` should be specified, by default None.
        should_accept : Callable, optional
            A function used to decide whether to accept a specific trajectory, i.e., dynamic filtering.
            It takes a complete trajectory output by the workflow, and returns a bool, by default None.
        """
        raise NotImplementedError()

    def wait(self, count: int, timeout: float | None = None) -> TensorDict:
        """Wait for a specified number of requests to complete, with a timeout.

        Should be used together with preceding `submit`.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float, optional
            Timeout in seconds. Exceeding the timeout will raise a `TimeoutError`, by default None

        Returns
        -------
        TensorDict
            A concatenated batch of trajectories

        Raises
        ------
        TimeoutError
            If the timeout is exceeded before enough trajectories are collected
        """
        raise NotImplementedError()

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results.

        See `workflow_api.py` for concrete implementation.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow : RolloutWorkflow, optional
            The workflow instance to run, by default None
        workflow_builder : Callable, optional
            A builder to create a workflow instance, by default None
        should_accept : Callable, optional
            A function to decide whether to accept a trajectory, by default None

        Returns
        -------
        TensorDict
            A concatenated batch of trajectory results
        """
        raise NotImplementedError()

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        """Asynchronously submit and wait until a full batch is ready with controlled staleness.

        See `workflow_api.py` for concrete implementation.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from for batch preparation
        workflow : RolloutWorkflow, optional
            The workflow instance to run, by default None
        workflow_builder : Callable, optional
            A builder to create a workflow instance, by default None
        should_accept : Callable, optional
            A function to decide whether to accept a trajectory, by default None

        Returns
        -------
        TensorDict
            A full batch of trajectory results with controlled staleness
        """
        raise NotImplementedError()

    def pause(self):
        """Pause request submission for async rollout.

        Used during evaluation to prevent data over-generation.
        """
        raise NotImplementedError()

    def resume(self):
        """Resume request submission for async rollout."""
        raise NotImplementedError()

    def register_callback_to_all_worker(
        self, method: str, callback: Callable, **kwargs
    ):
        """
        Partial rollout api, register a callback function for the `method` method. After successful registration, the
        Controller will poll and call the `method` method in a background method. When the return
        value is obtained, it will be used as a parameter to call the `callback` callback.
        """
        raise NotImplementedError()

    def abort_all_requests(self) -> None:
        """Partial rollout api, abort all pending requests in the inference engine."""
        raise NotImplementedError()
