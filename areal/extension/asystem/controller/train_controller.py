"""ASystem TrainController implementation.

This module provides the ASystem-specific TrainController that inherits from
the base TrainController and overrides the initialize method.
"""

import asyncio
from copy import deepcopy

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import AllocationMode, FinetuneSpec
from areal.api.scheduler_api import Job, Scheduler
from areal.controller.train_controller import TrainController as BaseTrainController
from areal.utils import logging
from areal.utils.network import find_free_ports

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

        if alloc_mode.gen_backend == "sglang":
            self.config.scheduling_spec.env_vars["NCCL_CUMEM_ENABLE"] = "0"
            self.config.scheduling_spec.env_vars["NCCL_NVLS_ENABLE"] = "0"

        self.parallel_strategy = alloc_mode.train

        # Create job for scheduler
        job = Job(
            replicas=alloc_mode.train.world_size,
            tasks=[
                deepcopy(self.config.scheduling_spec)
                for _ in range(alloc_mode.train.world_size)
            ],
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )
        # Create environment variables to mimic torchrun
        # FIXME: here master_addr and master_port only work in the local setting
        port = find_free_ports(1)[0]
        for i, task in enumerate(job.tasks):
            task.env_vars["RANK"] = str(i)
            task.env_vars["WORLD_SIZE"] = str(alloc_mode.train.world_size)
            task.env_vars["LOCAL_RANK"] = str(
                0
            )  # because we have only set 1 CUDA_VISIBLE_DEVICES for each process
            # TODO: find a real master addr with scheduler
            task.env_vars["MASTER_ADDR"] = "localhost"
            task.env_vars["MASTER_PORT"] = str(port)

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
        asyncio.run(self._async_initialize_engines(ft_spec, **kwargs))

        # Identify DP head workers
        self._identify_dp_heads()

        self.logger.info("TrainController initialization complete")
