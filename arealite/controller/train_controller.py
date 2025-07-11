from abc import ABC, abstractmethod
from typing import Any, Dict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict
import asyncio

from arealite.api.cli_args import MicroBatchSpec, TrainEngineConfig, TrainControllerConfig
from arealite.api.controller_api import TrainController
from arealite.api.engine_api import TrainEngine
from arealite.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    LLMResponse,
    SaveLoadMeta,
    WeightUpdateMeta, AllocationMode,
)
from arealite.api.scheduler_api import SchedulerClient, EngineSchedulingConfig, ContainerSpec
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronInitConfig
from realhf.base.names import worker
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
import logging

class DistributedTrainController(TrainController):
    # TrainController可以通过同名接口调用所有TrainEngine/actor/critic的方法
    # 除此之外没有别的方法了
    # 虽然方法相同，但是传数据集的参数类型不同:
    #   Engine data: List[Dict[str, Any]]
    #   Controller data: DistributedBatch
    def __init__(self, train_engine: TrainEngine, config: TrainControllerConfig, scheduler: SchedulerClient):
        super().__init__(train_engine, config, scheduler)
        self.allocate_mode = AllocationMode.from_str(config.allocation_mode)
        self.dp_world_size = self.allocate_mode.train_world_size // self.allocate_mode.train_dp_size

        # todo
        # @dataclass
        # class Scheduling:
        #     cpu: int
        #     gpu: int
        #     mem: int
        #     nodelist: str = None
        #     exclude: str = None
        #     partition: str = None
        #     container_image: str = None
        #     env_vars: Dict[str, str] = field(default_factory=dict)
        #     # time utils from "https://slurm.schedmd.com/sbatch.html"
        #     time_limit: Optional[str] = None  # see  "--time" option for format
        #     begin: Optional[str] = None  # see "--begin" option for format
        #     deadline: Optional[str] = None  # see "--deadline" option for format


    def initialize(self):
        """Initialize environments for distributed training and load models."""
        scheduling = self.train_engine.get_scheduling_config()
        # todo：支持多容器
        engine_scheduling_config = EngineSchedulingConfig(replicas=self.allocate_mode.train_world_size)
        engine_scheduling_config.specs.append(ContainerSpec(
            cpu=scheduling.cpu,
            mem=scheduling.mem,
            gpu=scheduling.gpu,
            container_image=self.config.container_image,
            cmd=self.config.cmd,
            env_vars=scheduling.env_vars,
        ))
        self.scheduler.submit(engine_scheduling_config)

        # todo: 等待调度完成，job状态为running
        self.scheduler.wait(5*60, )

        engines = self.scheduler.get_engines()
        # engine info
        self.engines = engines

        server_addrs = [f"{engine.ip}:{engine.port[0]}" for engine in engines if engine.port]

        # todo: 不能写死remote megatron, 让engine抽象出接口
        tasks = [
            self.scheduler.initialize_engine(engine.engine_id, self.train_engine, RemoteMegatronInitConfig(addrs=server_addrs, global_rank=index, world_size=self.allocate_mode.train_world_size))
            for index, engine in enumerate(self.engines)
        ]

        loop = asyncio.get_running_loop()
        return loop.run_until_complete(asyncio.gather(*tasks))

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        pass

    async def _rpc_call(self, method, *args, **kwargs):
        logging.info(f"[train controller] start to rpc call, method: {method}, args: {args}, kwargs: {kwargs}")

        tasks = [
            self.scheduler.call_engine(engine.engine_id, method, args, kwargs)
            for engine in self.engines
        ]
        results = await asyncio.gather(*tasks)

        logging.info(f"[train controller] end to rpc call, method: {method}, args: {args}, kwargs: {kwargs}")
        return results

    async def _rpc_call_tasks(self, tasks):
        results = await asyncio.gather(*tasks)
        return results

    def upload_weights(self, meta: WeightUpdateMeta):
        """Upload weights to the inference engine."""
        return asyncio.run(self._rpc_call("upload_weights", meta))

    def save(self, meta: SaveLoadMeta):
        """Save model weights (and optimizer states) for later use."""
        return asyncio.run(self._rpc_call("save", meta))

    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file."""
        return asyncio.run(self._rpc_call("load", meta))

    def step_lr_scheduler(self):
        """Step learning rate scheduler.

        Since PPO uses minibatch updates, this method just need to be called once after a few train_batch calls.
        It is separated from train_batch to allow for more flexible scheduling.
        """
        return asyncio.run(self._rpc_call("step_lr_scheduler"))

    def train_distributed_batch(
        self,
        input_: DistributedBatchMemory
    ) -> Dict[str, float]:
        """Update the model with a batch of data and a loss function."""
        # self._rpc_call("train_batch". input_, )
        batches = input_.split(self.dp_world_size)
        assert len(self.engines) % self.dp_world_size == 0
        tasks = []
        for index, engine in enumerate(self.engines):
            batch_index = index // self.dp_world_size
            batch_data = batches[batch_index]
            tasks.append(
                self.scheduler.call_engine(engine.engine_id, "train_distributed_batch", batch_data)
            )

        results = asyncio.run(self._rpc_call_tasks(*tasks))


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
        output_seqlens: List[List[int]] | None = None,
    ) -> Any | None:
        """Run the forward pass or inference on the model. Note that it is gradient-free."""
        raise NotImplementedError()