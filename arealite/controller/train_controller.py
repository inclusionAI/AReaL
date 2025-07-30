import resource
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
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
        logger.info(f"scheduling config: {scheduling_config}")

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
        workerSpec.env_vars["WORKER_TYPE"] = "training-worker"

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
        engineSpec.env_vars["WORKER_TYPE"] = "training-engine"
        engineSpec.env_vars["WORK_MODE"] = "TRAINING"

        scheduling_config.specs.append(workerSpec)
        scheduling_config.specs.append(engineSpec)

        self.scheduler.create_workers("train",scheduling_config)

        self.workers = self.scheduler.get_workers("train", timeout=60*5)

        server_addrs = [f"{worker.ip}:{worker.ports[0]}" for worker in self.workers if worker.ports]

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
        logging.info(f"start to  rpc call, method: {method}, args: {args}, kwargs: {kwargs}")
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
                logger.info("receive ctrl+c, terminating all initialization tasks...")
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

    def notify_event(self, event: str, global_step: int) -> None:
        """Notify workers about training start/end events.
        
        Args:
            event: "train_start" or "train_end"
            global_step: Current global step
        """
        self._rpc_call("notify_event", event, global_step)
        return None

    def train_distributed_batch(
        self, input_: DistributedBatchMemory
    ) -> Dict[str, float]:
        """Update the model with a batch of data and a loss function."""
        logger.info(f"start to train_distributed_batch")
        batches = input_.split(self.allocate_mode.train_dp_size)
        time.sleep(10)
        assert len(self.workers) % self.dp_world_size == 0
        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            for index, worker in enumerate(self.workers):
                batch_index = index // self.dp_world_size
                batch_data = batches[batch_index]
                futures.append(executor.submit(
                    self.scheduler.call_engine,
                    worker.id,
                    "train_distributed_batch",
                    batch_data
                ))
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
