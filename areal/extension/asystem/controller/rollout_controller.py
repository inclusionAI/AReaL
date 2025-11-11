"""ASystem RolloutController implementation.

This module provides the ASystem-specific RolloutController that inherits from
the base RolloutController and overrides the initialize method.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from areal.api.alloc_mode import AllocationMode
from areal.api.io_struct import WeightUpdateMeta
from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.scheduler_api import Job, Scheduler
from areal.controller.rollout_controller import TASK_RUNNER_MAX_QSIZE
from areal.controller.rollout_controller import (
    RolloutController as BaseRolloutController,
)
from areal.core import StalenessManager
from areal.core.async_task_runner import AsyncTaskRunner
from areal.extension.asystem.remote_hybrid_inference_worker import (
    RemoteHypidInferenceInitConfig,
)
from areal.extension.asystem.controller.util import execute_parallel_tasks
from areal.utils import logging


class RolloutController(BaseRolloutController):
    """ASystem-specific RolloutController.

    This controller inherits from the base RolloutController and overrides
    the initialize method to provide ASystem-specific initialization behavior.
    """

    def __init__(
        self,
        inf_engine: type[InferenceEngine],
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    ):
        """Initialize the ASystem RolloutController.

        Parameters
        ----------
        inf_engine : type[InferenceEngine]
            The inference engine class (not instance) to create on workers
        config : InferenceEngineConfig
            Configuration for the inference engines
        scheduler : Scheduler
            Scheduler for managing workers
        """
        super().__init__(inf_engine, config, scheduler)

    async def _async_initialize(self, job: Job, *args, **kwargs):
        # Create workers via scheduler
        self.logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        self.logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        self.logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        self.logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.inf_engine
        engine_path = f"{engine_class.__module__}.{engine_class.__name__}"

        # Create and initialize engines on workers
        self.logger.info("Creating engines...")
        tasks = [
            self.scheduler.create_engine(
                worker_id=worker.id,
                engine=engine_path,
                config=self.config,
            )
            for worker in self.workers
        ]
        await asyncio.gather(*tasks)
        self.logger.info("Engine created on all workers!")

        self.logger.info("Calling engine initialization...")
        init_configs = self._build_engine_initialize_config(
            enable_colocate_mode=job.scheduling_strategy.type == "colocation"
        )

        # after building init configs, we reset self.workers so that each worker is a sglang instance
        self.workers = self.workers[:: self.alloc_mode.gen_instance_size]
        assert len(init_configs) == len(self.workers)

        tasks = [
            self.scheduler.async_call_engine(worker.id, "initialize", init_config)
            for worker, init_config in zip(self.workers, init_configs)
        ]
        await asyncio.gather(*tasks)
        self.logger.info("All engines are initialized...")

    def initialize(
        self,
        role: str,
        alloc_mode: AllocationMode,
        *args,
        **kwargs,
    ):
        """Initialize environments and launch the background thread for asynchronous distributed inference.

        This method is overridden to provide ASystem-specific initialization behavior.
        Currently, it passes without performing any initialization.

        Parameters
        ----------
        role : str
            Worker role name
        alloc_mode : AllocationMode
            The allocation mode configuration for distributed setup
        *args
            Variable length argument list passed to engine initialization
        **kwargs
            Arbitrary keyword arguments passed to engine initialization
        """
        self.logger = logging.getLogger("[RolloutController]")
        self.logger.info("Initializing ASystem RolloutController")

        # Get scheduling config from kwargs or use defaults
        self._worker_role = role
        self.alloc_mode = alloc_mode

        job = Job(
            replicas=alloc_mode.gen.world_size,
            tasks=list(self.config.scheduling_specs),
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )
        self.logger.info(f"job: {job}")

        # Use asyncio.run to call async scheduler methods synchronously
        asyncio.run(
            self._async_initialize(
                job,
                *args,
                **kwargs,
            )
        )

        # Initialize AsyncTaskRunner for task execution
        self.runner = AsyncTaskRunner(
            max_queue_size=TASK_RUNNER_MAX_QSIZE,
            enable_tracing=self.config.enable_rollout_tracing,
        )
        self.runner.initialize(logger=self.logger)

        # Initialize thread pool for weight updates
        self.executor = ThreadPoolExecutor(max_workers=alloc_mode.gen.dp_size)

        # Initialize staleness manager for global capacity control
        max_concurrent_rollouts = (
            self.config.max_concurrent_rollouts or self.config.consumer_batch_size
        )
        consumer_batch_size = self.config.consumer_batch_size

        self.staleness_manager = StalenessManager(
            max_concurrent_rollouts=max_concurrent_rollouts,
            consumer_batch_size=consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

    def _build_engine_initialize_config(
        self, enable_colocate_mode: bool
    ) -> list[RemoteHypidInferenceInitConfig]:
        init_configs = []

        # Get master address
        master_addr = f"{self.workers[0].ip}:{self.workers[0].engine_ports[1]}"

        for index, worker in enumerate(self.workers):
            if index % self.alloc_mode.gen_instance_size != 0:
                continue

            main_server_addrs = [
                f"{worker.ip}:{worker.engine_ports[0]}"
                for worker in self.workers[
                    index : index + self.alloc_mode.gen_instance_size
                ]
                if worker.engine_ports
            ]
            free_addrs = [
                [
                    f"{worker.ip}:{port}"
                    for worker in self.workers[
                        index : index + self.alloc_mode.gen_instance_size
                    ]
                    for port in worker.engine_ports[1:]
                ]
            ]

            init_config = RemoteHypidInferenceInitConfig(
                main_server_addrs=main_server_addrs,
                free_addrs=free_addrs,
                world_size=self.alloc_mode.gen.world_size,
                global_ranks=list(
                    range(index, index + self.alloc_mode.gen_instance_size)
                ),
                master_addr=master_addr,
                enable_colocate_mode=enable_colocate_mode,
            )
            init_configs.append(init_config)

        return init_configs

    async def update_weights(self, meta: WeightUpdateMeta) -> None:
        self.logger.info("begin update_weights")
        execute_parallel_tasks(self.workers, self.scheduler, "update_weights", meta)
        self.logger.info("finish update_weights")
