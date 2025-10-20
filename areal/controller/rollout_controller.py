from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, List

from tensordict import TensorDict, stack

from areal.api.cli_args import InferenceEngineConfig
from areal.api.controller_api import RolloutController, DistributedBatch
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import AllocationMode, WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow

from areal.api.scheduler_api import Job, Scheduler, ScheduleStrategy, Worker
from areal.controller.utils import create_engine_with_retry, rpc_call
from areal.utils.data import concat_padded_tensors
from areal.utils import logging
from areal.utils.http import wait_future_ordered
from areal.controller.batch import DistributedBatchMemory


logger = logging.getLogger("DistributedRolloutController")


class DistributedRolloutController(RolloutController):
    def __init__(
        self,
        inf_engine: InferenceEngine,
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    ):
        super().__init__(inf_engine, config, scheduler)
        self.role: str = "rollout"
        self.alloc_mode: AllocationMode
        self.enable_colocate_mode: bool
        self.dp_world_size: int
        self.dp_head_workers: List[Worker]

    def initialize(
        self,
        config,
        target: str,
    ):
        self.alloc_mode = AllocationMode.from_str(config.allocation_mode)
        self.dp_world_size = self.alloc_mode.gen.world_size // self.alloc_mode.gen.dp_size

        tasks = self.inf_engine.get_scheduling_config()
        tasks[0].cmd = "python3 -m areal.scheduler.rpc.rpc_server"

        job = Job(
            replicas=self.alloc_mode.gen.world_size,
            tasks=tasks,
            schedule_strategy=ScheduleStrategy(type="colocation", target=target) if target else None,
            role=self.role,
        )
        logger.info(f"Start to create job: {job}")
        self.scheduler.create_workers(job, config=config)
        logger.info(f"[dzq_debug] create_worker finished.")

        workers = self.scheduler.get_workers(self.role, timeout=1800)
        self.dp_head_workers = [worker for idx, worker in enumerate(workers) if idx % self.dp_world_size == 0]
        assert len(self.dp_head_workers) == self.alloc_mode.gen.dp_size
        logger.info(f"[dzq_debug] create_worker finished. {len(self.dp_head_workers)} "
                    f"w0:{self.dp_head_workers[0]},"
                    f"w1: {self.dp_head_workers[1]}")
        with ThreadPoolExecutor(max_workers=len(self.dp_head_workers)) as executor:
            create_engine: Callable[..., Any] = partial(
                create_engine_with_retry,
                self.scheduler.create_engine,
                60,  # max_retries
                10,  # retry_delay
            )

            futures = [
                executor.submit(
                        create_engine,
                        worker.id,
                        self.inf_engine,
                        None,
                        None,
                        self.dp_world_size,
                )
                for worker in self.dp_head_workers
            ]

            try:
                wait_future_ordered(futures, exit_on_exception=True)
            except Exception as e:
                logger.error(f"Failed to initialize engine: {e}")
                raise

    def destroy(self):
        self.scheduler.delete_workers()

    def __del__(self):
        self.destroy()

    def update_weights(self, meta: WeightUpdateMeta) -> None:
        """Update weights in the inference engine."""
        self.custom_function_call("update_weights", None, meta)
        return None

    def prepare_batch(self, data: DistributedBatch, workflow: RolloutWorkflow) -> None:
        """Asynchronously submit a request to the inference engine. Exits immediately."""
        batches = data.chunk(self.alloc_mode.gen.dp_size)
        self.custom_function_call("prepare_batch", batches, workflow)
        return None

    def rollout_batch(
        self,
        data: DistributedBatch,
        workflow: RolloutWorkflow
    ) -> DistributedBatch:
        """Submit a batch of requests to the inference engine and wait for the results."""
        batches = data.chunk(self.alloc_mode.gen.dp_size)
        batch_results = self.custom_function_call("rollout_batch", batches, workflow) # [DistributedBatchMemory]
        assert len(batch_results) > 0
        size = int(batch_results[0]["input_ids"].shape[0])
        bs = size * len(batch_results)
        # results 转成List(Dict[key, tensor])
        list_results = []
        for result in batch_results:
            list_results.append(result.get_data())
        padded = concat_padded_tensors(list_results)
        if isinstance(padded, dict):
            padded = TensorDict(padded, batch_size=[bs])
        return DistributedBatchMemory.from_dict(padded.to_dict())

    def set_version(self, version: int) -> None:
        self.custom_function_call("set_version", None, version)
        return None

    def get_version(self) -> int:
        results = self.custom_function_call("get_version", None)
        return results[0]

    def pause(self):
        self.custom_function_call("pause", None)

    def resume(self):
        self.custom_function_call("resume", None)

    def submit(self, data: DistributedBatch):
        batches = data.chunk(self.alloc_mode.gen.dp_size)
        self.custom_function_call("submit", batches)

    def wait(self, counts: List[int], timeout: float | None = None)->DistributedBatch:
        assert len(counts) == len(self.dp_head_workers)
        results = self.custom_function_call("wait", counts, timeout)
        return DistributedBatch.concat(results)

    def custom_function_call(self, method: str, batches, *args, **kwargs):
        return rpc_call(self.scheduler, self.dp_head_workers, method, batches, args, kwargs)
