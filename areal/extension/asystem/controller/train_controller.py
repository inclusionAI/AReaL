"""ASystem TrainController implementation.

This module provides the ASystem-specific TrainController that inherits from
the base TrainController and overrides the initialize method.
"""

import asyncio

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import AllocationMode, FinetuneSpec
from areal.api.scheduler_api import Job, Scheduler
from areal.controller.train_controller import TrainController as BaseTrainController
from areal.extension.asystem.remote_hybrid_train_worker import RemoteMegatronInitConfig
from areal.utils import logging

logger = logging.getLogger("TrainController")


class TrainController(BaseTrainController):
    """ASystem-specific TrainController.

    This controller inherits from the base TrainController and overrides
    the initialize method to provide ASystem-specific initialization behavior.
    """

    def __init__(
        self,
        train_engine: type[TrainEngine],
        config: TrainEngineConfig,
        scheduler: Scheduler,
    ):
        """Initialize the ASystem TrainController.

        Parameters
        ----------
        train_engine : type[TrainEngine]
            The engine class (not instance) to instantiate on each worker
        config : TrainEngineConfig
            Configuration for training engines
        scheduler : Scheduler
            Scheduler for worker management
        """
        super().__init__(train_engine, config, scheduler)

    def initialize(
        self,
        role: str,
        alloc_mode: AllocationMode,
        ft_spec: FinetuneSpec,
        **kwargs,
    ):
        """Initialize environments for distributed training and load models.

        This method is overridden to provide ASystem-specific initialization behavior.
        Currently, it passes without performing any initialization.

        Parameters
        ----------
        role : str
            Role identifier for the workers
        alloc_mode : AllocationMode
            Allocation mode configuration for distributed setup
        ft_spec : FinetuneSpec
            Finetune specification for model initialization
        **kwargs
            Additional keyword arguments passed to engine initialization
        """
        self.logger = logging.getLogger("[TrainController]")

        # Store configuration
        self._worker_role = role
        self.alloc_mode = alloc_mode
        self.parallel_strategy = alloc_mode.train

        # Create job for scheduler
        job = Job(
            replicas=alloc_mode.train.world_size,
            tasks=list(self.config.scheduling_specs),
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

        # Create workers via scheduler
        self.logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        self.logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        self.logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        self.logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.train_engine
        engine_path = f"{engine_class.__module__}.{engine_class.__name__}"

        # Create and initialize engines on workers
        asyncio.run(self._async_create_engines(engine_path))
        asyncio.run(self._async_initialize(job, ft_spec, **kwargs))

        # Identify DP head workers
        # todo: @chucai, implement this, record rank info in hybrid train worker and implement is_data_parallel_head...
        # self._identify_dp_heads()

        self.logger.info("TrainController initialization complete")

    async def _async_initialize(self, job: Job, ft_spec: FinetuneSpec, **kwargs):
        # Initialize engines
        self.logger.info("Calling engine initialization...")
        init_configs = self._build_engine_initialize_config(
            enable_colocate_mode=job.scheduling_strategy.type == "colocation"
        )

        assert len(init_configs) == len(self.workers)

        tasks = [
            self.scheduler.async_call_engine(
                worker.id, "initialize", init_config, _should_bcast=False
            )
            for worker, init_config in zip(self.workers, init_configs)
        ]

        await asyncio.gather(*tasks)
        self.logger.info("All engines are initialized!")

    def _build_engine_initialize_config(
        self, enable_colocate_mode: bool
    ) -> list[RemoteMegatronInitConfig]:
        server_addrs = [
            f"{worker.ip}:{worker.engine_ports[0]}" for worker in self.workers
        ]
        return [
            RemoteMegatronInitConfig(
                server_addrs=server_addrs,
                global_rank=index,
                world_size=self.alloc_mode.train.world_size,
                enable_colocate_mode=enable_colocate_mode,
            )
            for index, worker in enumerate(self.workers)
        ]
