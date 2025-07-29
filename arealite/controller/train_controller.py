import resource
import time
from abc import ABC, abstractmethod
from typing import Any, Dict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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
    WeightUpdateMeta,
    AllocationMode,
)
from arealite.scheduler.base import Scheduler, SchedulingConfig, ContainerSpec, ScheduleStrategy
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronInitConfig
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
import logging
from realhf.base import stats_tracker
logger = logging.getLogger("DistributedTrainController")


class DistributedTrainController(TrainController):
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
        super().__init__(train_engine, config, scheduler)
        self.allocate_mode = AllocationMode.from_str(config.allocation_mode)
        self.dp_world_size = self.allocate_mode.train_world_size // self.allocate_mode.train_dp_size

    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed training and load models."""
        scheduling = self.train_engine.get_scheduling_config()
        scheduling_config = SchedulingConfig(replicas=self.allocate_mode.train_world_size)

        target = kwargs.get("colocation_with")
        scheduling_config.schedule_strategy = ScheduleStrategy(type="colocation", uid=target.uid) if target else None

        workerSpec = ContainerSpec(
            cpu=0,
            mem=0,
            gpu=scheduling.gpu,
            cmd="bash /storage/openpsi/codes/dh183333/arealite-test-bugfix/AReaL/arealite/scheduler/scripts/launch-worker.sh",
            env_vars=scheduling.env_vars.copy() if scheduling.env_vars is not None else {},
            portCount=1
        )
        workerSpec.env_vars["REAL_PACKAGE_PATH"] = "/storage/openpsi/codes/dh183333/arealite-test-bugfix/AReaL"
        workerSpec.env_vars["WORKER_IMAGE"] = "/storage/openpsi/images/areal-25.01-sglang-bf16-editable-metrics-xccl-20250716.sif"
        workerSpec.env_vars["WORKER_LOG_DIR"] = "/storage/openpsi/experiments/logs/root/{experiment_name}/{trial_name}".format(
            experiment_name=self.config.experiment_name, trial_name=self.config.trial_name)
        workerSpec.env_vars["WORKER_TYPE"] = "model-worker"

        engineSpec = ContainerSpec(
            cpu=0,
            mem=0,
            gpu=0,
            cmd="bash /storage/openpsi/codes/dh183333/arealite-test-bugfix/AReaL/arealite/scheduler/scripts/launch-hybrid-server.sh",
            env_vars=scheduling.env_vars.copy() if scheduling.env_vars is not None else {},
            portCount=1
        )
        engineSpec.env_vars["REAL_PACKAGE_PATH"] = "/storage/openpsi/codes/Asystem-HybridEngine"
        engineSpec.env_vars["WORKER_IMAGE"] = "/storage/openpsi/images/hybrid-engine-13060133-20250724003115.sif"
        engineSpec.env_vars["WORKER_LOG_DIR"] = "/storage/openpsi/experiments/logs/root/{experiment_name}/{trial_name}".format(
            experiment_name=self.config.experiment_name, trial_name=self.config.trial_name)
        engineSpec.env_vars["WORKER_TYPE"] = "model-worker"
        engineSpec.env_vars["WORK_MODE"] = "TRAINING"

        scheduling_config.specs.append(workerSpec)
        scheduling_config.specs.append(engineSpec)

        self.scheduler.create_workers("train",scheduling_config, schedule_strategy = target)

        self.workers = self.scheduler.get_workers("train", timeout=60*5)

        server_addrs = [f"{worker.ip}:{worker.ports[0]}" for worker in self.workers if worker.ports]

        print(f"self.workers: {len(self.workers)}")
        # todo: 等待megatron server启动完成
        time.sleep(100)
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [
                executor.submit(
                    self.scheduler.create_engine,
                    worker.id,
                    self.train_engine,
                    RemoteMegatronInitConfig(
                        server_addrs=server_addrs, global_rank=index, world_size=self.allocate_mode.train_world_size
                    ),
                )
                for index, worker in enumerate(self.workers)
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                raise
        # todo: 不能写死remote megatron, 让engine抽象出接口

    def destroy(self):
        """Destroy the engine and release GPU memory."""
        pass

    def _rpc_call(self, method, *args, **kwargs):
        logging.info(f"[train controller] start to  rpc call, method: {method}, args: {args}, kwargs: {kwargs}")
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [
                executor.submit(
                    self.scheduler.call_engine,
                    worker.id,
                    method,
                    *args,
                    **kwargs
                )
                for worker in self.workers
            ]

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

    def train_distributed_batch(
        self, input_: DistributedBatchMemory
    ) -> Dict[str, float]:
        """Update the model with a batch of data and a loss function."""
        logger.info(f"start to train_distributed_batch")
        print(f"start to train_distributed_batch")
        batches = input_.split(self.allocate_mode.train_dp_size)
        time.sleep(10)
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"内存使用111: {mem_usage / 1024:.2f} MB")
        assert len(self.workers) % self.dp_world_size == 0
        logger.info(f"controller debug111")
        print("start to train_distributed_batch111")
        print(f"batches: {len(batches)}, workers: {len(self.workers)}")
        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            for index, worker in enumerate(self.workers):
                logger.info(f"controller debug111: {index}")
                t1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"start to train_distributed_batch111: {index}, {t1}")
                batch_index = index // self.dp_world_size
                t2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"controller debug222: {index}, {t2}", flush=True)
                batch_data = batches[batch_index]
                t3 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"controller debug333: {index}, {t3}", flush=True)
                futures.append(executor.submit(
                    self.scheduler.call_engine,
                    worker.id,
                    "train_distributed_batch",
                    batch_data
                ))
                t4 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"controller debug444: {index}, {t4}", flush=True)

            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                raise

        with (
            stats_tracker.record_timing("distributed_train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            for train_stat in results:
                stats_tracker.scalar(**train_stat)

        return results

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
