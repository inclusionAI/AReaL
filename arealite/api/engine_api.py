import abc
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict

from arealite.api.cli_args import MicroBatchSpec
from arealite.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    LLMResponse,
    SaveLoadMeta,
    WeightUpdateMeta,
)

if TYPE_CHECKING:
    from arealite.api.workflow_api import RolloutWorkflow


@dataclass
class Scheduling:
    cpu: int
    gpu: int
    mem: int
    nodelist: str = None
    exclude: str = None
    partition: str = None
    container_image: str = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    # time utils from "https://slurm.schedmd.com/sbatch.html"
    time_limit: Optional[str] = None  # see  "--time" option for format
    begin: Optional[str] = None  # see "--begin" option for format
    deadline: Optional[str] = None  # see "--deadline" option for format


class TrainEngine(abc.ABC):

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
        """Initialize environments for distributed training and load models."""
        raise NotImplementedError()

    def get_scheduling_config(self) -> Scheduling:
        """Get the scheduling configuration for the engine, e.g., image, cpu/gpu/memory size."""
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
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> Dict[str, float]:
        """Update the model with a batch of data and a loss function."""
        raise NotImplementedError()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
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


class InferenceEngine(abc.ABC):

    def initialize(self, addr: str | None, ft_spec):
        """Initialize environments for distributed inference and load models."""
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        pass

    def update_weights(self, meta: WeightUpdateMeta) -> Future:
        """Update weights in the inference engine."""
        raise NotImplementedError()

    async def agenerate(self, req: LLMRequest) -> LLMResponse:
        """Asynchronously generate a response for the given request."""
        raise NotImplementedError()

    def submit(self, data: Dict[str, Any], workflow: "RolloutWorkflow") -> None:
        """Asynchronously submit a request to the inference engine. Exits immediately."""
        raise NotImplementedError()

    def wait(self, count: int, timeout: float) -> TensorDict:
        """Wait for a specified number of requests to complete, with a timeout."""
        raise NotImplementedError()

    def rollout(
        self, data: List[Dict[str, Any]], workflow: "RolloutWorkflow"
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        raise NotImplementedError()
