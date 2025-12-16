from __future__ import annotations

import dataclasses
import gc
from typing import TYPE_CHECKING

import torch.distributed as dist

from areal.api.alloc_mode import FSDPParallelStrategy, ParallelStrategy
from areal.api.train_engine import TrainEngineDistMixin
from areal.platforms import current_platform
from areal.utils import logging
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.distributed import patch_dist_group_timeout
from areal.utils.fsdp.parallel import ParallelHelper

if TYPE_CHECKING:
    from areal.engine.fsdp.protocol import FSDPEngineProtocol


class FSDPDistMixin(TrainEngineDistMixin):
    def create_process_group(
        self: FSDPEngineProtocol, parallel_strategy: ParallelStrategy | None = None
    ):
        patch_dist_group_timeout(DIST_GROUP_DEFAULT_TIMEOUT)

        backend = current_platform.communication_backend
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            self.own_global_group = True

        self._cpu_group = dist.new_group(
            timeout=DIST_GROUP_DEFAULT_TIMEOUT, backend="gloo"
        )

        # FSDP-specific process group setup
        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()

        self.logger = logging.getLogger(f"[FSDP Engine Rank {dist.get_rank()}]")

        parallel_strategy = self._make_parallel_strategy(parallel_strategy)

        self.parallel_helper = ParallelHelper.from_parallel_strategy(parallel_strategy)

        self.logger.info(
            f"Initializing device mesh with parallel dims {str(self.parallel_helper)}."
        )

        self.world_mesh = self.parallel_helper.world_mesh

        self.dp_group = self.world_mesh["dp"].get_group()
        self.sp_group = self.world_mesh["sp"].get_group()

        # Sequence and model parallel group (sp+tp)
        self.mp_group = self.world_mesh["sp_tp"].get_group()

        self.rank = dist.get_rank()

        self.dp_head = dist.get_process_group_ranks(self.mp_group)[0]
        self.dp_rank = dist.get_rank(self.dp_group)

        self.logger.info(f"Data parallel head {self.dp_head} and rank {self.dp_rank}")

    @property
    def data_parallel_group(self: FSDPEngineProtocol) -> dist.ProcessGroup:
        return self.dp_group

    @property
    def data_parallel_rank(self: FSDPEngineProtocol) -> int:
        return self.dp_rank

    @property
    def data_parallel_world_size(self: FSDPEngineProtocol) -> int:
        return self.parallel_helper.dp_size

    def current_data_parallel_head(self: FSDPEngineProtocol) -> int:
        return self.dp_head

    def is_data_parallel_head(self: FSDPEngineProtocol) -> bool:
        return self.rank == self.dp_head

    @property
    def context_and_model_parallel_group(self: FSDPEngineProtocol) -> dist.ProcessGroup:
        return self.mp_group

    @property
    def cpu_group(self: FSDPEngineProtocol) -> dist.ProcessGroup:
        assert self.initialized
        return self._cpu_group

    def destroy(self: FSDPEngineProtocol):
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        current_platform.empty_cache()
        gc.collect()
        # NOTE: if `own_global_group` is true, we assume that
        # no communications are needed after `destroy`, so we
        # directly destroy all groups. Otherwise, process group
        # handles still exist and we expect another engine to
        # clean up these groups.
        if dist.is_initialized() and self.own_global_group:
            dist.destroy_process_group()
        self.initialized = False

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> FSDPParallelStrategy:
        return FSDPParallelStrategy(
            **dataclasses.asdict(parallel_strategy),
        )
