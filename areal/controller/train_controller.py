from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, List

import torch

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import TrainEngineConfig
from areal.api.controller_api import DistributedBatch, TrainController
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    ParamSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Job, Scheduler, ScheduleStrategy, Worker
from areal.controller.batch import DistributedBatchMemory
from areal.controller.utils import create_engine_with_retry, rpc_call
from areal.utils import logging
from areal.utils.http import wait_future_ordered

logger = logging.getLogger("DistributedTrainController")


class DistributedTrainController(TrainController):
    def __init__(
        self, train_engine: TrainEngine, config: TrainEngineConfig, scheduler: Scheduler
    ):
        super().__init__(train_engine, config, scheduler)

        self.role: str = "train"
        self.group_size: int
        self.alloc_mode: AllocationMode
        self.workers: List[Worker]
        self.engine_dp_ranks: List[int]

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        assert self.workers is not None, "Workers are not created"
        self.custom_function_call("create_process_group", parallel_strategy)

    def initialize(
        self,
        alloc_mode_str: str,
        ft_spec: FinetuneSpec,
        schedule_strategy: ScheduleStrategy,
        **kwargs,
    ):
        """Initialize environments for distributed training and load models."""
        self.alloc_mode = AllocationMode.from_str(alloc_mode_str)
        self.ft_spec = ft_spec

        # todo: group size is a sampling parameter and an attribute of the data, should be moved to DistributedBatch
        self.group_size = kwargs.get("group_size", 1)

        job = Job(
            replicas=self.alloc_mode.train.world_size,
            tasks=self.train_engine.get_scheduling_config(),
            schedule_strategy=schedule_strategy,
            role=self.role,
        )
        logger.info(f"Start to create job: {job}")
        self.scheduler.create_workers(job)
        # after get workers, all rpc server is ready
        self.workers = self.scheduler.get_workers(self.role, timeout=1800)

        logger.info(f"Start to initialize engine")
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            create_engine: Callable[..., Any] = partial(
                create_engine_with_retry,
                self.scheduler.create_engine,
                60,  # max_retries
                10,  # retry_delay
            )
            futures: List[Future] = [
                executor.submit(
                    create_engine,
                    worker.id,
                    self.train_engine,
                    None,
                    self.ft_spec,
                    self.alloc_mode.train,
                )
                for worker in self.workers
            ]
            try:
                wait_future_ordered(futures, exit_on_exception=True)
            except Exception as e:
                logger.error(f"Failed to initialize engine: {e}")
                raise

        logger.info(f"Start to get rank info from engine")
        self.engine_dp_ranks = rpc_call(
            self.scheduler, self.workers, "data_parallel_rank"
        )
        logger.info(f"Initialize train engines succeeded!")

    def destroy(self):
        self.scheduler.delete_workers()

    def train(self, mode: bool = True):
        self.custom_function_call("train", mode)

    def upload_weights(self, meta: WeightUpdateMeta):
        self.custom_function_call("upload_weights", meta)

    def get_param_specs(
        self, weight_chunked_mem_mb: int = 1024
    ) -> List[List[ParamSpec]]:
        ret: List[List[List[ParamSpec]]] = self.custom_function_call(
            "get_param_specs", weight_chunked_mem_mb
        )
        flattened = [inner for outer in ret for inner in outer]
        return flattened

    def set_version(self, version: int):
        return self.custom_function_call("set_version", version)

    def get_version(self) -> List[int]:
        return self.custom_function_call("get_version")

    def save(self, meta: SaveLoadMeta):
        self.custom_function_call("save", meta)

    def load(self, meta: SaveLoadMeta):
        self.custom_function_call("load", meta)

    def step_lr_scheduler(self):
        self.custom_function_call("step_lr_scheduler")

    def custom_function_call(self, method: str, *args, **kwargs):
        return rpc_call(self.scheduler, self.workers, method, None, *args, **kwargs)

    def _align_batches_with_dp(
        self, input_: DistributedBatch, rebalance=True
    ) -> List[DistributedBatch]:
        if rebalance:
            inputs = input_.chunk_by_ffd(self.group_size, self.alloc_mode.train.dp_size)
        else:
            inputs = input_.chunk(self.alloc_mode.train.dp_size)

        batches = []
        for dp_rank in self.engine_dp_ranks:
            batches.append(inputs[dp_rank])

        return batches

    def train_batch(
        self,
        input_: DistributedBatch,
        loss_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[Dict[str, Any]], torch.Tensor],
    ) -> List[Dict[str, float]]:

        batches = self._align_batches_with_dp(input_, True)
        train_stats = rpc_call(
            self.scheduler,
            self.workers,
            "train_batch",
            batches,
            loss_fn,
            loss_weight_fn,
        )

        return train_stats

    def ppo_update(
        self,
        input_: DistributedBatch,
    ) -> List[Dict[str, float]]:

        batches = self._align_batches_with_dp(input_, True)
        train_stats = rpc_call(
            self.scheduler,
            self.workers,
            "ppo_update",
            batches,
        )

        return train_stats

    def eval_batch(
        self,
        input_: DistributedBatch,
        loss_fn: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[Dict[str, Any]], torch.Tensor],
    ) -> List[torch.Tensor]:

        batches = self._align_batches_with_dp(input_, True)
        eval_stats = rpc_call(
            self.scheduler, self.workers, "eval_batch", batches, loss_fn, loss_weight_fn
        )

        return eval_stats

    def compute_logp(
        self,
        input_: DistributedBatch,
        *args,
        **kwargs,
    ):
        batches = self._align_batches_with_dp(input_, False)
        logps = rpc_call(
            self.scheduler,
            self.workers,
            "compute_logp",
            batches,
            *args,
            **kwargs,
        )
        logger.info(f"debug: type logps: {type(logps)}")
        return logps

    def compute_advantages(
        self,
        input_: DistributedBatch,
        *args,
        **kwargs,
    ):
        batches = self._align_batches_with_dp(input_, False)
        advantages = rpc_call(
            self.scheduler,
            self.workers,
            "compute_advantages",
            batches,
            *args,
            **kwargs,
        )
        return DistributedBatchMemory.concat(advantages)

    def forward(
        self,
        input_: DistributedBatch,
        output_seqlens: List[int] | None = None,
        post_hook: Callable[[torch.Tensor, Dict[str, Any]], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> List[Any]:
        batches = self._align_batches_with_dp(input_, False)
        forward_stats = rpc_call(
            self.scheduler,
            self.workers,
            "forward",
            batches,
            output_seqlens,
            post_hook,
            aggregate_fn,
        )

        return forward_stats
