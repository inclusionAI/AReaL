from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict

from arealite.api.cli_args import (
    InferenceEngineConfig,
    MicroBatchSpec,
    RolloutControllerConfig,
    TrainControllerConfig,
)
from arealite.api.engine_api import InferenceEngine, TrainEngine
from arealite.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    LLMResponse,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from arealite.api.workflow_api import RolloutWorkflow
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.scheduler.base import Scheduler


class TrainController(ABC):
    # TrainController可以通过同名接口调用所有TrainEngine/actor/critic的方法
    # 除此之外没有别的方法了
    # 虽然方法相同，但是传数据集的参数类型不同:
    #   Engine data: List[Dict[str, Any]]
    #   Controller data: DistributedBatch
    def __init__(
        self,
        train_engine: TrainEngine,
        config: TrainControllerConfig,
        scheduler: Scheduler,
    ):
        self.train_engine = train_engine
        self.config = config
        self.scheduler = scheduler

    def initialize(self):
        """Initialize environments for distributed training and load models."""
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        pass

    def upload_weights(self, meta: WeightUpdateMeta):
        """Upload weights to the inference engine."""
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
        input_: Dict,
    ) -> Dict[str, float]:
        """Update the model with a batch of data and a loss function."""
        raise NotImplementedError()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function."""
        raise NotImplementedError()

    @torch.no_grad()
    def forward(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, Dict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Run the forward pass or inference on the model. Note that it is gradient-free."""
        raise NotImplementedError()

    def train_distributed_batch(
        self, input_: DistributedBatchMemory
    ) -> Dict[str, float]:
        """Update the model with a batch of data."""
        raise NotImplementedError()


class RolloutController(ABC):
    # RolloutController可以通过同名接口调用所有InferenceEngine的方法
    # 除此之外没有别的方法了
    # 虽然方法相同，但是传数据集的参数类型不同:
    #   Engine data: List[Dict[str, Any]]
    #   Controller data: DistributedBatch
    def __init__(
        self,
        inf_engine: InferenceEngine,
        config: RolloutControllerConfig,
        scheduler: Scheduler,
    ):
        self.inf_engine = inf_engine
        self.config = config
        self.scheduler = scheduler

    def initialize(self):
        """Initialize environments for distributed inference and load models."""
        raise NotImplementedError()

    def update_weights(self, meta: WeightUpdateMeta) -> Future:
        """Update weights in the inference engine."""
        raise NotImplementedError()

    def submit(self, data: Dict[str, Any], workflow: RolloutWorkflow) -> None:
        """Asynchronously submit a request to the inference engine. Exits immediately."""
        raise NotImplementedError()

    def wait(self, count: int, timeout: int) -> DistributedBatchMemory:
        """Wait for a specified number of requests to complete, with a timeout."""
        raise NotImplementedError()

    def rollout(
        self, data: List[Dict[str, Any]], workflow: RolloutWorkflow
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        raise NotImplementedError()

    def rollout_distributed_batch(
        self, input_: DistributedBatchMemory, workflow: RolloutWorkflow
    ) -> DistributedBatchMemory:
        """Submit a batch of requests to the inference engine and wait for the results."""
        raise NotImplementedError()
