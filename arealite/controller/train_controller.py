import os
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch import Tensor

from arealite.api.cli_args import TrainControllerConfig
from arealite.api.controller_api import TrainController
from arealite.api.engine_api import TrainEngine
from arealite.api.io_struct import (
    SaveLoadMeta,
    WeightUpdateMeta,
    AllocationMode,
)
from arealite.controller.utils import create_engine_with_retry
from arealite.scheduler.base import Scheduler, SchedulingConfig, ContainerSpec, ScheduleStrategy
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronInitConfig
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from realhf.base import stats_tracker, logging
logger = logging.getLogger("DistributedTrainController")


class DistributedTrainController(TrainController):
    def __init__(
        self,
        train_engine: TrainEngine,
        config: TrainControllerConfig,
        scheduler: Scheduler,
        *args,
        **kwargs
    ):
        super().__init__(train_engine, config, scheduler)
        self.allocate_mode = AllocationMode.from_str(config.allocation_mode)
        self.role = kwargs.get("role", "train")
        self.world_size = self.allocate_mode.train_world_size
        self.dp_size = self.allocate_mode.train_dp_size
        self.tp_size = self.allocate_mode.train_tp_size
        self.pp_size = self.allocate_mode.train_pp_size



    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed training and load models."""
        scheduling = self.train_engine.get_scheduling_config()
        scheduling_config = SchedulingConfig(replicas=self.world_size)

        target = kwargs.get("colocation_with")
        scheduling_config.schedule_strategy = ScheduleStrategy(type="colocation", uid=target.uid) if target else None
        logger.info(f"scheduling config: {scheduling_config}")

        arealite_path = os.environ["REAL_PACKAGE_PATH"]
        engine_path = os.environ["ENGINE_PATH"]
        workerSpec = ContainerSpec(
            cpu=0,
            mem=0,
            gpu=scheduling.gpu,
            cmd=f"bash {arealite_path}/arealite/scheduler/scripts/launch-worker.sh".format(arealite_path=arealite_path),
            env_vars=scheduling.env_vars.copy() if scheduling.env_vars is not None else {},
            portCount=1
        )
        workerSpec.env_vars["REAL_PACKAGE_PATH"] = arealite_path
        workerSpec.env_vars["WORKER_IMAGE"] = "/storage/openpsi/images/areal-25.01-sglang-bf16-editable-metrics-xccl-20250716.sif"
        workerSpec.env_vars["WORKER_LOG_DIR"] = "/storage/openpsi/experiments/logs/root/{experiment_name}/{trial_name}".format(
            experiment_name=self.config.experiment_name, trial_name=self.config.trial_name)
        workerSpec.env_vars["WORKER_TYPE"] = "training-worker"

        engineSpec = ContainerSpec(
            cpu=0,
            mem=0,
            gpu=0,
            cmd=f"bash {arealite_path}/arealite/scheduler/scripts/launch-hybrid-server.sh".format(arealite_path=arealite_path),
            env_vars=scheduling.env_vars.copy() if scheduling.env_vars is not None else {},
            portCount=1
        )
        engineSpec.env_vars["ENGINE_PACKAGE_PATH"] = engine_path
        engineSpec.env_vars["WORKER_IMAGE"] = "/storage/openpsi/images/hybrid-engine-13250134-20250731173300.sif"
        engineSpec.env_vars["WORKER_LOG_DIR"] = "/storage/openpsi/experiments/logs/root/{experiment_name}/{trial_name}".format(
            experiment_name=self.config.experiment_name, trial_name=self.config.trial_name)
        engineSpec.env_vars["WORKER_TYPE"] = "training-engine"
        engineSpec.env_vars["WORK_MODE"] = "TRAINING"
        engineSpec.env_vars["GLOO_SOCKET_IFNAME"] = "eth0"
        engineSpec.env_vars["NCCL_SOCKET_IFNAME"] = "eth0"
        engineSpec.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        engineSpec.env_vars["USE_MAX_V2"] = "1"

        scheduling_config.specs.append(workerSpec)
        scheduling_config.specs.append(engineSpec)

        self.scheduler.create_workers(self.role,scheduling_config)

        self.workers = self.scheduler.get_workers(self.role, timeout=1800)

        server_addrs = [f"{worker.ip}:{worker.ports[0]}" for worker in self.workers if worker.ports]
        # FIXME: @chucai
        time.sleep(60)
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [
                executor.submit(
                    partial(
                        create_engine_with_retry,
                        self.scheduler.create_engine,
                        worker.id,
                        self.train_engine,
                        RemoteMegatronInitConfig(
                            server_addrs=server_addrs, global_rank=index, world_size=self.world_size
                        ),
                    )
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
        logger.info(f"start to  rpc call, method: {method}, args: {args}, kwargs: {kwargs}")
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
        batches = input_.split(self.dp_size)
        # dp_world_size = self.tp_size * self.pp_size
        # assert len(self.workers) % dp_world_size == 0
        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            for index, worker in enumerate(self.workers):
                batch_index = index % self.dp_size
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

    # @torch.no_grad
    # def compute_logprobs_with_distributed(self, input_: DistributedBatchMemory) -> Tensor:
    #     """Update the model with a batch of data and a loss function."""
    #     logger.info(f"start to compute_logprobs_with_distributed")
    #     batches = input_.split(self.dp_size)
    #     dp_world_size = self.tp_size * self.pp_size
    #     assert len(self.workers) % dp_world_size == 0
    #     futures = []
    #     results = []
    #     with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
    #         for index, worker in enumerate(self.workers):
    #             batch_index = index // dp_world_size
    #             batch_data = batches[batch_index]
    #             futures.append(executor.submit(
    #                 self.scheduler.call_engine,
    #                 worker.id,
    #                 "compute_logprobs_with_distributed",
    #                 batch_data
    #             ))
    #         try:
    #             for future in as_completed(futures):
    #                 result = future.result()
    #                 results.append(result)
    #         except KeyboardInterrupt:
    #             for f in futures:
    #                 f.cancel()
    #             raise
    #
    #     # cat tensor from dp head
    #     tensors_from_dp_heads = results[::dp_world_size]
    #     concatenated_result = torch.cat(tensors_from_dp_heads, dim=0)
    #     return concatenated_result

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

