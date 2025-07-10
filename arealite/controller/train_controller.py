from abc import ABC, abstractmethod
from typing import Any, Dict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict
import asyncio

from arealite.api.cli_args import MicroBatchSpec, TrainEngineConfig, TrainControllerConfig
from arealite.api.engine_api import TrainEngine
from arealite.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    LLMResponse,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from arealite.api.scheduler_api import SchedulerClient
from realhf.base.names import worker

if TYPE_CHECKING:
    from arealite.api.workflow_api import RolloutWorkflow


class TrainController(ABC):
    # TrainController可以通过同名接口调用所有TrainEngine/actor/critic的方法
    # 除此之外没有别的方法了
    # 虽然方法相同，但是传数据集的参数类型不同:
    #   Engine data: List[Dict[str, Any]]
    #   Controller data: DistributedBatch
    def __init__(self, train_engine: TrainEngine, config: TrainControllerConfig, scheduler: SchedulerClient):
        self.train_engine = train_engine
        self.config = config
        self.scheduler = scheduler

    def initialize(self):
        """Initialize environments for distributed training and load models."""
        # todo
        self.scheduler.submit()
        engines = self.scheduler.wait()
        self.engines = engines

        tasks = [
            self.scheduler.initialize_engine(engine_id, self.train_engine)
            for engine_id in self.engines
        ]

        loop = asyncio.get_running_loop()
        return loop.run_until_complete(asyncio.gather(*tasks))

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        pass

    def _rpc_call(self, method, *args, **kwargs):
        tasks = [
            self.scheduler.call_engine(engine_id, method, args, kwargs)
            for engine_id in self.engines
        ]

        loop = asyncio.get_running_loop()
        return loop.run_until_complete(asyncio.gather(*tasks))

    def upload_weights(self, meta: WeightUpdateMeta):
        """Upload weights to the inference engine."""
        return self._rpc_call("upload_weights", meta)

    def save(self, meta: SaveLoadMeta):
        """Save model weights (and optimizer states) for later use."""
        return self._rpc_call("save", meta)

    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file."""
        return self._rpc_call("load", meta)

    def step_lr_scheduler(self):
        """Step learning rate scheduler.

        Since PPO uses minibatch updates, this method just need to be called once after a few train_batch calls.
        It is separated from train_batch to allow for more flexible scheduling.
        """
        return self._rpc_call("step_lr_scheduler")

    def train_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> Dict[str, float]:
        """Update the model with a batch of data and a loss function."""
        self._rpc_call("train_batch". input_, )
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