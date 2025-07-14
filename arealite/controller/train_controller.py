from abc import ABC, abstractmethod
from typing import Any, Dict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from arealite.api.cli_args import TrainControllerConfig
from arealite.api.controller_api import TrainController
from arealite.api.engine_api import TrainEngine
from arealite.api.io_struct import (
    SaveLoadMeta,
    WeightUpdateMeta, AllocationMode,
)
from arealite.scheduler.base import Scheduler, SchedulingConfig, ContainerSpec
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronInitConfig
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
import logging

class DistributedTrainController(TrainController):
    # TrainController可以通过同名接口调用所有TrainEngine/actor/critic的方法
    # 除此之外没有别的方法了
    # 虽然方法相同，但是传数据集的参数类型不同:
    #   Engine data: List[Dict[str, Any]]
    #   Controller data: DistributedBatch
    def __init__(self, train_engine: TrainEngine, config: TrainControllerConfig, scheduler: Scheduler):
        super().__init__(train_engine, config, scheduler)
        # self.allocate_mode = AllocationMode.from_str(config.allocation_mode)
        self.dp_world_size = 16 // 2

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
        scheduling_config = {"num_workers": 16}
        self.scheduler.create_workers(scheduling_config)

        self.workers = self.scheduler.get_workers(timeout=60*6)

        self.workers = self.workers[16:]
        server_addrs = [f"{worker.ip}:{worker.ports[0]}" for worker in self.workers if worker.ports]
        print(f"self.workers: {len(self.workers)}")
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [
                executor.submit(
                    self.scheduler.create_engine,
                    worker.id,
                    self.train_engine,
                    RemoteMegatronInitConfig(server_addrs=server_addrs, global_rank=index, world_size=16)
                )
                for index, worker in enumerate(self.workers)
            ]
            try:
                for future in as_completed(futures):
                    future.result()  # 可加异常处理
            except KeyboardInterrupt:
                print("收到Ctrl+C，正在终止所有初始化任务...")
                # 取消所有未完成的future
                for f in futures:
                    f.cancel()
                raise  # 重新抛出异常，主程序能感知
        # todo: 不能写死remote megatron, 让engine抽象出接口

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        pass

    async def _rpc_call(self, method, *args, **kwargs):
        logging.info(f"[train controller] start to rpc call, method: {method}, args: {args}, kwargs: {kwargs}")

        tasks = [
            self.scheduler.call_engine(worker.work_id, method, args, kwargs)
            for worker in self.workers
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
        batches = input_.split(2)

        assert len(self.workers) % self.dp_world_size == 0
        tasks = []
        for index, worker in enumerate(self.workers):
            batch_index = index // self.dp_world_size
            batch_data = batches[batch_index]
            tasks.append(
                self.scheduler.call_engine(worker.id, "train_distributed_batch", batch_data)
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
