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
AllocationMode
)
from arealite.api.scheduler_api import SchedulerClient
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronInitConfig
from realhf.base.names import worker

if TYPE_CHECKING:
    from arealite.api.workflow_api import RolloutWorkflow
from arealite.api.controller_api import RolloutController
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.api.scheduler_api import SchedulerClient, EngineSchedulingConfig, ContainerSpec

class DistributedRolloutController(RolloutController):
    # RolloutController可以通过同名接口调用所有InferenceEngine的方法
    # 除此之外没有别的方法了
    # 虽然方法相同，但是传数据集的参数类型不同:
    #   Engine data: List[Dict[str, Any]]
    #   Controller data: DistributedBatch
    def __init__(self, inf_engine, config, scheduler):
        super().__init__(inf_engine, config, scheduler)
        self.allocate_mode = AllocationMode.from_str(config.allocation_mode)
        self.dp_world_size = self.allocate_mode.gen_world_size // self.allocate_mode.gen_dp_size

    async def _rpc_call(self, method, *args, **kwargs):
        tasks = [
            self.scheduler.call_engine(engine.engine_id, method, args, kwargs)
            for engine in self.engines
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def _rpc_call_tasks(self, tasks):
        results = await asyncio.gather(*tasks)
        return results

    def initialize(self):
        """Initialize environments for distributed inference and load models."""
        """Initialize environments for distributed training and load models."""
        scheduling = self.inf_engine.get_scheduling_config()
        # todo：支持多容器
        engine_scheduling_config = EngineSchedulingConfig(replicas=self.allocate_mode.gen_world_size)
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
        self.scheduler.wait(5 * 60, )

        engines = self.scheduler.get_engines()
        # engine info
        self.engines = engines

        server_addrs = [f"{engine.ip}:{engine.port[0]}" for engine in engines if engine.port]

        # todo: 不能写死remote megatron, 让engine抽象出接口
        tasks = [
            self.scheduler.initialize_engine(engine.engine_id, self.inf_engine,)
            for index, engine in enumerate(self.engines)
        ]

        loop = asyncio.get_running_loop()
        return loop.run_until_complete(asyncio.gather(*tasks))

    def update_weights(self, meta: WeightUpdateMeta) -> None:
        """Update weights in the inference engine."""
        self._rpc_call("update_weights", meta)
        return

    def submit(self, data: DistributedBatchMemory, workflow: RolloutWorkflow) -> None:
        """Asynchronously submit a request to the inference engine. Exits immediately."""
        raise NotImplementedError()

    def wait(self, count: int, timeout: int) -> DistributedBatchMemory:
        """Wait for a specified number of requests to complete, with a timeout."""
        raise NotImplementedError()

    def rollout(
        self,
        data: DistributedBatchMemory,
        workflow: RolloutWorkflow
    ) -> DistributedBatchMemory:
        """Submit a batch of requests to the inference engine and wait for the results."""
        batches = data.split(self.dp_world_size)
        assert len(self.engines) % self.dp_world_size == 0
        tasks = []
        for index, engine in enumerate(self.engines):
            batch_index = index//self.dp_world_size
            batch_data = batches[batch_index]
            tasks.append(
                self.scheduler.call_engine(engine.engine_id, "rollout", batch_data, workflow)
            )

        dbds = asyncio.run(self._rpc_call_tasks(*tasks))

        result = DistributedBatchMemory(None)
        for dbd in dbds:
            result = result.merge(dbd)

        return result
