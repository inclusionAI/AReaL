"""ASystem RolloutController implementation.

This module provides the ASystem-specific RolloutController that inherits from
the base RolloutController and overrides the initialize method.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.scheduler_api import Job, Scheduler
from areal.controller.rollout_controller import TASK_RUNNER_MAX_QSIZE
from areal.controller.rollout_controller import (
    RolloutController as BaseRolloutController,
)
from areal.core import StalenessManager
from areal.core.async_task_runner import AsyncTaskRunner
from areal.utils import logging

logger = logging.getLogger("RolloutController")


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

        # Get scheduling config from kwargs or use defaults
        self._worker_role = role
        self.config.scheduling_spec.cpu *= alloc_mode.gen_instance_size
        self.config.scheduling_spec.mem *= alloc_mode.gen_instance_size
        self.config.scheduling_spec.gpu = alloc_mode.gen_instance_size
        job = Job(
            replicas=alloc_mode.gen.dp_size,
            tasks=[self.config.scheduling_spec for _ in range(alloc_mode.gen.dp_size)],
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

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
