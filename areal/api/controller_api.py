import abc
from collections.abc import Callable
from typing import Any

import torch

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import (
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Scheduler


class DistributedBatch(abc.ABC):
    """Abstract base class for data exchange between controller and engine.

    This class defines the interface for handling batched data operations
    between controller and engine components in a distributed environment.
    It supports two modes of data transfer:
    - Memory mode: Full data transfer through memory
    - File mode: Transfer only metadata between controller and engine
    """

    @classmethod
    def from_dict(cls, dataset: dict[str, torch.Tensor | Any]) -> "DistributedBatch":
        """Create a DistributedBatch from a dictionary format dataset.

        Parameters
        ----------
        dataset : Dict[str, Union[torch.Tensor, Any]]
            Dictionary format dataset to convert, supporting Tensor, scalar, and list types

        Returns
        -------
        DistributedBatch
            DistributedBatch instance created from the dictionary
        """
        raise NotImplementedError()

    @classmethod
    def from_list(
        cls, dataset: list[dict[str, torch.Tensor | Any]]
    ) -> "DistributedBatch":
        """Create a DistributedBatch from a list format dataset.

        Parameters
        ----------
        dataset : List[Dict[str, Union[torch.Tensor, Any]]]
            List format dataset to convert, supporting Tensor, scalar, and list types

        Returns
        -------
        DistributedBatch
            DistributedBatch instance created from the list
        """
        raise NotImplementedError()

    def chunk(self, dp_size: int) -> list["DistributedBatch"]:
        """Split the dataset across data parallel processes.

        This function preserves the original order of data, ensuring that
        the sequence of samples in the concatenated result matches the
        original dataset order.

        Parameters
        ----------
        dp_size : int
            Number of data parallel processes

        Returns
        -------
        list[DistributedBatch]
            List of DistributedBatch objects, one for each process
        """
        raise NotImplementedError()

    def chunk_by_ffd(self, group_size: int, dp_size: int) -> list["DistributedBatch"]:
        """Split data by sequence length using First Fit Decreasing algorithm.

        Parameters
        ----------
        group_size : int
            Size of each group
        dp_size : int
            Number of data parallel processes to split into

        Returns
        -------
        list[DistributedBatch]
            List of DistributedBatch objects
        """
        raise NotImplementedError()

    def union(self, other: "DistributedBatch") -> "DistributedBatch":
        """Merge another batch with this one.

        Parameters
        ----------
        other : DistributedBatch
            Another batch to merge with

        Returns
        -------
        DistributedBatch
            Merged batch
        """
        raise NotImplementedError()

    def get_data(self) -> dict[str, torch.Tensor | Any]:
        """Get all data from the DistributedBatch.

        Returns
        -------
        Dict[str, Union[torch.Tensor, Any]]
            Dictionary where keys are field names and values can be Tensor, scalar, or list types
            containing all values for that field across the entire batch.
        """
        raise NotImplementedError()

    @staticmethod
    def concat(data: list["DistributedBatch"]) -> "DistributedBatch":
        """Concatenate multiple batches into a single batch.

        Parameters
        ----------
        data : list[DistributedBatch]
            List of batches to concatenate

        Returns
        -------
        DistributedBatch
            Concatenated batch
        """
        raise NotImplementedError()

    def __getitem__(self, key: int | str):
        """Get an item from the batch.

        Parameters
        ----------
        key : int or str
            Index or key to retrieve

        Returns
        -------
        Dict[str, Union[torch.Tensor, Any]] or Union[torch.Tensor, Any]
            Retrieved item
        """
        raise NotImplementedError()

    def __setitem__(self, key: str, value: torch.Tensor | Any):
        """Set an item in the batch.

        Parameters
        ----------
        key : str
            Key to set
        value : Union[torch.Tensor, Any]
            Value to set (Tensor, scalar, or list)
        """
        raise NotImplementedError()

    def __delitem__(self, key: int | str):
        """Delete an item from the batch.

        Parameters
        ----------
        key : int or str
            Index or key to delete
        """
        raise NotImplementedError()

    def __getstate__(self):
        """Serialize the batch for pickle dump.

        Returns
        -------
        dict
            Dictionary containing the state to be serialized
        """
        raise NotImplementedError()

    def __setstate__(self, state):
        """Restore the batch from pickle load.

        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state
        """
        raise NotImplementedError()


class TrainController(abc.ABC):
    """A centralized controller that manages multiple distributed TrainEngine workers.

    TrainController serves as a high-level orchestrator for distributed training across
    multiple concurrent workers, each running TrainEngine instances. It provides a
    unified interface for coordinating training operations while abstracting away the
    complexities of inter-worker communication and data distribution.

    Key differences from TrainEngine:
        - Operates at a higher abstraction level, managing multiple engine instances
        - Does not directly perform collective communications (no rank and process group APIs)
        - Uses `DistributedBatch` for data that spans multiple workers
        - Provides centralized coordination for distributed training workflows

    The controller handles workload distribution, synchronization, and aggregation
    of results from the underlying TrainEngine workers, enabling scalable and
    efficient distributed training.
    """

    def __init__(
        self,
        train_engine: TrainEngine,
        config: TrainEngineConfig,
        scheduler: Scheduler,
    ):
        self.train_engine = train_engine
        self.config = config
        self.scheduler = scheduler

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

    def destroy(self):
        """Destroy the engine and release GPU memory of models."""
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

    def update_weights(self, meta: WeightUpdateMeta):
        """Update weights to the inference engine in a blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        """
        raise NotImplementedError()

    def connect_engine(self, engine: "InferenceEngine", meta: WeightUpdateMeta):
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

    def step_lr_scheduler(self):
        """Step the learning rate scheduler.

        Since PPO uses minibatch updates, this method should be called periodically
        (e.g., once per PPO step). It is separated from train_batch to allow
        for more flexible learning rate scheduling.
        """
        raise NotImplementedError()

    def train_batch(
        self,
        input_: DistributedBatch,
        loss_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Update the model with a batch of data and a loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : DistributedBatch
            The distributed input data for model forward pass and the loss function.
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
        input_: DistributedBatch,
        loss_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : DistributedBatch
            The distributed input data for model forward pass and the loss function.
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
        input_: DistributedBatch,
        output_seqlens: list[int] | None = None,
        post_hook: Callable[[torch.Tensor, dict[str, Any]], Any] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Run the forward pass or inference on the model.

        Note
        ----
        This operation is gradient-free.

        Parameters
        ----------
        input_ : DistributedBatch
            The distributed input data for model forward pass. Redundant entries are allowed.
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
