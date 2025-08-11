import abc
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.io_struct import (
    FinetuneSpec,
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
    nodelist: Optional[str] = None
    exclude: Optional[str] = None
    partition: Optional[str] = None
    container_image: Optional[str] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    # time utils from "https://slurm.schedmd.com/sbatch.html"
    time_limit: Optional[str] = None  # see  "--time" option for format
    begin: Optional[str] = None  # see "--begin" option for format
    deadline: Optional[str] = None  # see "--deadline" option for format


class TrainEngine(abc.ABC):

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
        """Initialize environments for distributed training and load models."""
        raise NotImplementedError()

    @property
    def parallelism_group(self) -> dist.ProcessGroup:
        """The global communication group of this engine."""
        raise NotImplementedError()

    def get_scheduling_config(self) -> Scheduling:
        """Get the scheduling configuration for the engine, e.g., image, cpu/gpu/memory size."""
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory."""

    def train(self, mode: bool = True):
        """Set the engine to the train mode."""
        raise NotImplementedError()

    def eval(self):
        """Set the engine to the eval mode."""
        return self.train(False)

    def upload_weights(self, meta: WeightUpdateMeta):
        """Upload weights to the inference engine (in a blocking manner)."""
        raise NotImplementedError()

    def get_param_specs(
        self, weight_chunked_mem_mb: int = 1024
    ) -> List[List[ParamSpec]]:
        """Get the parameter specifications for the model."""
        raise NotImplementedError()

    def set_version(self, version: int):
        """Set the current weight version in the train engine."""
        raise NotImplementedError()

    def get_version(self) -> int:
        """Get the current weight version in the train engine."""
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
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> Dict[str, float]:
        """Update the model with a batch of data and a loss function."""
        raise NotImplementedError()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: TensorDict,
        loss_fn: Callable[[torch.Tensor, TensorDict], torch.Tensor],
        loss_weight_fn: Callable[[TensorDict], float],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function."""
        raise NotImplementedError()

    @torch.no_grad()
    def forward(
        self,
        input_: TensorDict,
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, TensorDict], Any] | None = None,
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

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request."""
        raise NotImplementedError()

    def update_weights(self, meta: WeightUpdateMeta) -> Future:
        """Update weights in the inference engine in a non-blocking manner."""
        raise NotImplementedError()

    def set_version(self, version: int) -> None:
        """Set the current weight version in the inference engine."""
        raise NotImplementedError()

    def get_version(self) -> int:
        """Get the current weight version in the inference engine."""
        raise NotImplementedError()

    def submit(
        self,
        data: Dict[str, Any],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
    ) -> None:
        """Asynchronously submit a request to the inference engine. Exits immediately."""
        raise NotImplementedError()

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        """Wait for a specified number of requests to complete, with a timeout."""
        raise NotImplementedError()

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
    ) -> TensorDict:
        """Submit a batch of requests to the inference engine and wait for the results."""
        raise NotImplementedError()

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: Optional["RolloutWorkflow"] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> TensorDict:
        """Asynchronously submit and wait until a full batch is ready."""
        raise NotImplementedError()

    def pause(self):
        """Pause request submission for async rollout. Used during evaluation to prevent data over generation."""
        raise NotImplementedError()

    def resume(self):
        """Resume request submission for async rollout."""
        raise NotImplementedError()
