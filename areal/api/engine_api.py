import abc
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
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


@dataclass
class Scheduling:
    cpu: int
    gpu: int
    mem: int
    port_count: int
    cmd: str | None = None
    nodelist: str | None = None
    exclude: str | None = None
    partition: str | None = None
    container_image: str | None = None
    type: str | None = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    # time utils from "https://slurm.schedmd.com/sbatch.html"
    time_limit: Optional[str] = None  # see  "--time" option for format
    begin: Optional[str] = None  # see "--begin" option for format
    deadline: Optional[str] = None  # see "--deadline" option for format


class TrainEngine(abc.ABC):

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
    def parallelism_group(self) -> dist.ProcessGroup:
        """Get the global communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The global communication group
        """
        raise NotImplementedError()

    def get_scheduling_config(self) -> List[Scheduling]:
        """Get the scheduling configuration for the engine.

        This includes configuration such as container image, CPU/GPU/memory size.

        Returns
        -------
        Scheduling
            The scheduling configuration for the engine
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

    def upload_weights(self, meta: WeightUpdateMeta):
        """Upload weights to the inference engine in a blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        """
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

    def step_lr_scheduler(self):
        """Step the learning rate scheduler.

        Since PPO uses minibatch updates, this method should be called periodically
        (e.g., once per PPO step). It is separated from train_batch to allow
        for more flexible learning rate scheduling.
        """
        raise NotImplementedError()

    def train_batch(
        self,
        input_: Dict[str, Any],
        loss_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[Dict[str, Any]], torch.Tensor],
    ) -> Dict[str, float]:
        """Update the model with a batch of data and a loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : Dict[str, Any]
            The input data for model forward pass and the loss function.
            Redundant entries are allowed.
        loss_fn : Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]
            The loss function that takes the model's forward output and input_,
            and outputs a scalar normalized loss.
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
        raise NotImplementedError()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict[str, Any],
        loss_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[Dict[str, Any]], torch.Tensor],
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
        loss_fn : Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]
            The loss function that takes the model's forward output and input_,
            and outputs a scalar normalized loss.
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
        raise NotImplementedError()

    @torch.no_grad()
    def forward(
        self,
        input_: Dict[str, Any],
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, Dict[str, Any]], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
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
        post_hook : Callable[[torch.Tensor, Dict[str, Any]], Any], optional
            The post-processing function for micro-batched outputs. Post-processing
            the output on-the-fly during micro-batched forward can reduce peak
            memory usage, by default None.
        aggregate_fn : Callable[[List[Any]], Any], optional
            A function to aggregate micro-batched outputs, by default torch.cat.

        Returns
        -------
        Any or None
            The result produced by `post_hook` and `aggregate_fn`.
        """
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

    def wait(self, count: int, timeout: float | None = None) -> Dict[str, Any]:
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
        Dict[str, Any]
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
    ) -> Dict[str, Any]:
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
        Dict[str, Any]
            A concatenated batch of trajectory results
        """
        raise NotImplementedError()

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> Dict[str, Any]:
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
        Dict[str, Any]
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

    def get_scheduling_config(self) -> List[Scheduling]:
        """Get the scheduling configuration for the engine.

        This includes configuration such as container image, CPU/GPU/memory size.

        Returns
        -------
        Scheduling
            The scheduling configuration for the engine
        """
        raise NotImplementedError()
