import asyncio
from typing import List, Dict, Any
from tensordict import TensorDict, stack

from arealite.api.engine_api import InferenceEngine
from arealite.api.io_struct import (
    WeightUpdateMeta,
    AllocationMode
)

from arealite.api.cli_args import RolloutControllerConfig
from arealite.api.workflow_api import RolloutWorkflow
from arealite.api.controller_api import RolloutController
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.scheduler.base import Scheduler, SchedulingConfig, ContainerSpec
from arealite.extension.asystem.remote_sglang_engine import RemoteSGLangInitConfig
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class DistributedRolloutController(RolloutController):
    # RolloutController可以通过同名接口调用所有InferenceEngine的方法
    # 除此之外没有别的方法了
    # 虽然方法相同，但是传数据集的参数类型不同:
    #   Engine data: List[Dict[str, Any]]
    #   Controller data: DistributedBatch
    def __init__(self, inf_engine: InferenceEngine, config: RolloutControllerConfig, scheduler: Scheduler):
        super().__init__(inf_engine, config, scheduler)
        self.dp_world_size = 16 // 2

    async def _rpc_call(self, method, *args, **kwargs):
        logging.info(f"[rollout controller] start to  rpc call, method: {method}, args: {args}, kwargs: {kwargs}")

        tasks = [
            self.scheduler.call_engine(worker.id, method, args, kwargs)
            for worker in self.workers
        ]
        results = await asyncio.gather(*tasks)

        logging.info(f"[rollout controller] end to rpc call, method: {method}, args: {args}, kwargs: {kwargs}")
        return results

    async def _rpc_call_tasks(self, *tasks):
        results = await asyncio.gather(*tasks)
        return results

    def initialize(self):
        """Initialize environments for distributed inference and load models."""
        scheduling = self.inf_engine.get_scheduling_config()
        # todo：支持多容器
        scheduling_config = SchedulingConfig(replicas=16)
        scheduling_config = {"num_workers": 16}
        self.scheduler.create_workers(scheduling_config)

        self.workers = self.scheduler.get_workers(timeout=5*60)
        self.workers = self.workers[0:16]
        server_addrs = [f"{worker.ip}:{worker.ports[0]}" for worker in self.workers if worker.ports]

        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [
                executor.submit(
                    self.scheduler.create_engine,
                    worker.id,
                    self.inf_engine,
                    RemoteSGLangInitConfig(server_addrs=server_addrs)
                )
                for worker in self.workers
            ]
            print(f"submit workers: {len(futures)}")
            try:
                for future in as_completed(futures):
                    future.result()  # 可加异常处理
            except KeyboardInterrupt:
                print("收到Ctrl+C，正在终止所有初始化任务...")
                # 取消所有未完成的future
                for f in futures:
                    f.cancel()
                raise  # 重新抛出异常，主程序能感知


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
        batches = data.split(2)
        assert len(self.workers) % self.dp_world_size == 0
        futures = []

        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            for index, worker in enumerate(self.workers):
                batch_index = index//self.dp_world_size
                batch_data = batches[batch_index]
                futures.append(executor.submit(
                    self.scheduler.call_engine,
                    worker.id,
                    "rollout",
                    batch_data,
                    workflow
                ))

            results = []
            try:
                for future in as_completed(futures):
                    result = future.result()  # 可加异常处理
                    results.append(result)
            except KeyboardInterrupt:
                print("收到Ctrl+C，正在终止所有初始化任务...")
                # 取消所有未完成的future
                for f in futures:
                    f.cancel()
                raise  # 重新抛出异常，主程序能感知


        batchdata = DistributedBatchMemory(None)
        for dataset in results:
            batchdata = batchdata.merge(dataset)

        return batchdata

    def rollout(
        self, data: List[Dict[str, Any]], workflow: RolloutWorkflow
    ) -> TensorDict:
        futures = []

        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            for index, worker in enumerate(self.workers):
                futures.append(executor.submit(
                    self.scheduler.call_engine,
                    worker.id,
                    "rollout",
                    data,
                    workflow
                ))

            results = []
            try:
                for future in as_completed(futures):
                    result = future.result()  # 可加异常处理
                    results.append(result)
            except KeyboardInterrupt:
                print("收到Ctrl+C，正在终止所有初始化任务...")
                # 取消所有未完成的future
                for f in futures:
                    f.cancel()
                raise  # 重新抛出异常，主程序能感知

        return stack(results, dim=0)

