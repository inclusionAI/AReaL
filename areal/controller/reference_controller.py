from areal.api.cli_args import TrainControllerConfig
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import AllocationMode
from areal.controller.train_controller import DistributedTrainController
from areal.scheduler.base import (
    Scheduler,
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
