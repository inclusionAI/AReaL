from __future__ import annotations

import dataclasses
import gc
from typing import TYPE_CHECKING

import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel

from areal.api.alloc_mode import MegatronParallelStrategy, ParallelStrategy
from areal.api.train_engine import TrainEngineDistMixin
from areal.platforms import current_platform
from areal.utils import logging
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT

if TYPE_CHECKING:
    from areal.engine.megatron.protocol import MegatronEngineProtocol


class MegatronDistMixin(TrainEngineDistMixin):
    def create_process_group(
        self: MegatronEngineProtocol,
        parallel_strategy: ParallelStrategy | None = None,
    ):
        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()
        self.parallel_strategy = self._make_parallel_strategy(parallel_strategy)
        backend = current_platform.communication_backend
        if not dist.is_initialized():
            # TODO: Handle the condition when WORLD_SIZE and RANK is not set in launcher
            # NOTE: device_id **SHOULD NOT** be passed into init_process_group,
            # otherwise initializing the NCCL weight update group will be wrong!
            dist.init_process_group(
                backend=backend,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            # Initialize Megatron parallel states
            # NOTE: we assume all MegatronEngine has the same parallel strategy.
            vpp_size = self.parallel_strategy.virtual_pipeline_parallel_size
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.parallel_strategy.tensor_parallel_size,
                pipeline_model_parallel_size=self.parallel_strategy.pipeline_parallel_size,
                virtual_pipeline_model_parallel_size=vpp_size if vpp_size > 1 else None,
                use_sharp=False,
                order="tp-cp-ep-dp-pp",
                context_parallel_size=self.parallel_strategy.context_parallel_size,
                expert_model_parallel_size=self.parallel_strategy.expert_parallel_size,
                expert_tensor_parallel_size=self.parallel_strategy.expert_tensor_parallel_size,
                distributed_timeout_minutes=int(
                    DIST_GROUP_DEFAULT_TIMEOUT.seconds / 60
                ),
            )
            # Set megatron model parallel seed
            tensor_parallel.model_parallel_cuda_manual_seed(self.seed)
            self.own_global_group = True
        self.logger = logging.getLogger(f"[Megatron Engine Rank {dist.get_rank()}]")
        self._context_and_model_parallel_group = None
        self._init_context_and_model_parallel_group()
        # This is needed for barrier synchronization when models are moved to CPU
        self._cpu_group = dist.new_group(
            timeout=DIST_GROUP_DEFAULT_TIMEOUT, backend="gloo"
        )
        self.process_group_initialized = True

    @property
    def data_parallel_rank(self: MegatronEngineProtocol) -> int:
        assert self.process_group_initialized
        return mpu.get_data_parallel_rank()

    @property
    def data_parallel_world_size(self: MegatronEngineProtocol) -> int:
        assert self.process_group_initialized
        return mpu.get_data_parallel_world_size()

    @property
    def data_parallel_group(self: MegatronEngineProtocol) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return mpu.get_data_parallel_group()

    def current_data_parallel_head(self: MegatronEngineProtocol) -> int:
        """Get the rank of the head of the current data parallel group."""
        assert self.process_group_initialized
        ranks = dist.get_process_group_ranks(self.context_and_model_parallel_group)
        return ranks[0]

    def is_data_parallel_head(self: MegatronEngineProtocol) -> bool:
        assert self.process_group_initialized
        ranks = dist.get_process_group_ranks(self.context_and_model_parallel_group)
        return ranks[0] == self.rank

    @property
    def pipeline_parallel_rank(self: MegatronEngineProtocol) -> int:
        assert self.process_group_initialized
        return mpu.get_pipeline_model_parallel_rank()

    def is_pipeline_parallel_head(self: MegatronEngineProtocol) -> bool:
        assert self.process_group_initialized
        return self.is_pp_head

    @property
    def context_and_model_parallel_group(
        self: MegatronEngineProtocol,
    ) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return self._context_and_model_parallel_group

    @property
    def cpu_group(self: MegatronEngineProtocol) -> dist.ProcessGroup:
        assert self.process_group_initialized
        return self._cpu_group

    def destroy(self: MegatronEngineProtocol):
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            self.model = None
        gc.collect()
        current_platform.empty_cache()
        gc.collect()
        self.process_group_initialized = False
        # NOTE: if `own_global_group` is true, we assume that
        # no communications are needed after `destroy`, so we
        # directly destroy all groups. Otherwise, process group
        # handles still exist and we expect another engine to
        # clean up these groups.
        if dist.is_initialized() and self.own_global_group:
            mpu.destroy_model_parallel()
            dist.destroy_process_group()
            self.own_global_group = False

    def _make_parallel_strategy(
        self: MegatronEngineProtocol,
        parallel_strategy: ParallelStrategy,
    ) -> MegatronParallelStrategy:
        base_strategy = dataclasses.asdict(parallel_strategy)
        vpp_size = self.mcore_config.virtual_pipeline_parallel_size
        return MegatronParallelStrategy(
            use_sequence_parallel=parallel_strategy.tensor_parallel_size > 1,
            virtual_pipeline_parallel_size=vpp_size,
            **base_strategy,
        )

    def _init_context_and_model_parallel_group(
        self: MegatronEngineProtocol,
    ) -> None:
        # Initialize context and model parallel groups, which are only used in AReaL
        # for data distribution
        rank_generator = mpu.RankGenerator(
            tp=self.parallel_strategy.tensor_parallel_size,
            ep=1,
            dp=self.parallel_strategy.data_parallel_size,
            pp=self.parallel_strategy.pipeline_parallel_size,
            cp=self.parallel_strategy.context_parallel_size,
            order="tp-cp-ep-dp-pp",
            rank_offset=0,
        )
        context_and_model_parallel_ranks = rank_generator.get_ranks("tp-cp-pp")
        # create context and model_parallel_groups
        for dp_rank, ranks in enumerate(context_and_model_parallel_ranks):
            group = mpu.create_group(
                ranks,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                pg_options=mpu.get_nccl_options("tp-cp-pp", {}),
                group_desc="CONTEXT_AND_MODEL_PARALLEL_GROUP",
            )
            if dp_rank == mpu.get_data_parallel_rank():
                self._context_and_model_parallel_group = group
