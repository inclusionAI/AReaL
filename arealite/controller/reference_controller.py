import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch

from arealite.api.cli_args import TrainControllerConfig
from arealite.api.controller_api import TrainController
from arealite.api.engine_api import TrainEngine
from arealite.api.io_struct import AllocationMode, SaveLoadMeta, WeightUpdateMeta
from arealite.controller.train_controller import DistributedTrainController
from arealite.controller.utils import create_engine_with_retry
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronInitConfig
from arealite.scheduler.base import (
    ContainerSpec,
    Scheduler,
    ScheduleStrategy,
    SchedulingConfig,
)
from realhf.base import logging

logger = logging.getLogger("DistributedReferenceController")


class DistributedReferenceController(DistributedTrainController):
    def __init__(
        self,
        train_engine: TrainEngine,
        config: TrainControllerConfig,
        scheduler: Scheduler,
        *args,
        **kwargs,
    ):
        super().__init__(train_engine, config, scheduler)
        self.allocate_mode = AllocationMode.from_str(config.allocation_mode)
        self.role = kwargs.get("role", "ref")
        self.world_size = self.allocate_mode.reference_world_size
        self.dp_size = self.allocate_mode.reference_dp_size
        self.tp_size = self.allocate_mode.reference_tp_size
        self.pp_size = self.allocate_mode.reference_pp_size
        self.storage_prefix = config.storage_prefix
