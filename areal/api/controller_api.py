from abc import ABC
from concurrent.futures import Future
from typing import Any, Callable, Dict, List

import torch
from tensordict import TensorDict

from areal.api.cli_args import MicroBatchSpec
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import (
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Scheduler
from areal.api.workflow_api import RolloutWorkflow


class TrainController(ABC):
    def __init__(
        self,
        train_engine: TrainEngine,
        scheduler: Scheduler,
    ):
        self.train_engine = train_engine
        self.scheduler = scheduler

    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed training and load models."""
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory."""

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


class RolloutController(ABC):
    def __init__(
        self,
        inf_engine: InferenceEngine,
        scheduler: Scheduler,
    ):
        self.inf_engine = inf_engine
        self.scheduler = scheduler

    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed inference and load models."""
        raise NotImplementedError()

    def update_weights(self, meta: WeightUpdateMeta) -> Future:
        """Update weights in the inference engine."""
        raise NotImplementedError()

    def submit(self, data: Dict[str, Any], workflow: RolloutWorkflow) -> None:
        """Asynchronously submit a request to the inference engine. Exits immediately."""
        raise NotImplementedError()

    def wait(self, count: int, timeout: int) -> TensorDict:
        """Wait for a specified number of requests to complete, with a timeout."""
        raise NotImplementedError()

    def rollout(
        self, data: List[Dict[str, Any]], workflow: RolloutWorkflow
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        raise NotImplementedError()
