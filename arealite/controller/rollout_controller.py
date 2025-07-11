from abc import ABC, abstractmethod
from typing import Any, Dict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict
import asyncio

from arealite.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    LLMResponse,
    SaveLoadMeta,
    WeightUpdateMeta,
AllocationMode
)
from arealite.extension.asystem.remote_megatron_engine import RemoteInferenceInitConfig

if TYPE_CHECKING:
    from arealite.api.workflow_api import RolloutWorkflow
from arealite.api.controller_api import RolloutController
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.scheduler.base import Scheduler, SchedulingConfig, ContainerSpec
import logging

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
        logging.info(f"[rollout controller] start to  rpc call, method: {method}, args: {args}, kwargs: {kwargs}")

        tasks = [
            self.scheduler.call_engine(worker.id, method, args, kwargs)
            for worker in self.workers
        ]
        results = await asyncio.gather(*tasks)

        logging.info(f"[rollout controller] end to rpc call, method: {method}, args: {args}, kwargs: {kwargs}")
        return results

    async def _rpc_call_tasks(self, tasks):
        results = await asyncio.gather(*tasks)
        return results

    def initialize(self):
        """Initialize environments for distributed inference and load models."""
        scheduling = self.inf_engine.get_scheduling_config()
        # todo：支持多容器
        scheduling_config = SchedulingConfig(replicas=self.allocate_mode.gen_world_size)
        scheduling_config.specs.append(ContainerSpec(
            cpu=scheduling.cpu,
            mem=scheduling.mem,
            gpu=scheduling.gpu,
            container_image=self.config.container_image,
            cmd=self.config.cmd,
            env_vars=scheduling.env_vars,
        ))
        self.scheduler.create_workers(worker_scheduling_config)

        self.workers = self.scheduler.get_workers(timeout=5*60)

        server_addrs = [f"{worker.ip}:{worker.port[0]}" for worker in self.workers if worker.ports]

        tasks = [
            self.scheduler.initialize_engine(worker.id, self.inf_engine, RemoteInferenceInitConfig(addrs=server_addrs, global_rank=index, world_size=self.allocate_mode.gen_world_size))
            for index, worker in enumerate(self.workers)
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

    def rollout_distributed_batch(
        self,
        data: DistributedBatchMemory,
        workflow: RolloutWorkflow
    ) -> DistributedBatchMemory:
        """Submit a batch of requests to the inference engine and wait for the results."""
        batches = data.split(self.dp_world_size)
        assert len(self.workers) % self.dp_world_size == 0
        tasks = []
        for index, worker in enumerate(self.workers):
            batch_index = index//self.dp_world_size
            batch_data = batches[batch_index]
            tasks.append(
                self.scheduler.call_engine(worker.id, "rollout", batch_data, workflow)
            )

        datasets = asyncio.run(self._rpc_call_tasks(*tasks))

        result = DistributedBatchMemory(None)
        for dataset in datasets:
            result = result.merge(dataset)

        return result
