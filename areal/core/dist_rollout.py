from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.platforms import current_platform
from areal.utils.data import (
    all_gather_tensor_container,
    broadcast_tensor_container,
    concat_padded_tensors,
    get_batch_size,
    tensor_container_to,
)
from areal.utils.datapack import ffd_allocate


@dataclass
class RedistributedData:
    all_data: List[Dict[str, Any]]
    data: Dict[str, Any]
    rank: int
    group_indices: List[List[int]]


def _slice_tensor_dict(data: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Slices tensors in a dictionary along the first dimension."""
    sliced_data = {}
    batch_size = -1
    if "attention_mask" in data:
        batch_size = data["attention_mask"].shape[0]
    for key, value in data.items():
        if torch.is_tensor(value) and value.shape[0] == batch_size:
            sliced_data[key] = value[start:end]
        else:
            sliced_data[key] = value
    return sliced_data


def redistribute(
    data: Dict[str, Any], granularity: int = 1, group=None
) -> RedistributedData:
    """Redistribute a batch across a process group.

    This function only accepts padded data which must have an "attention_mask" field,
    Each tensor should have shape [bs, seqlen, *] or [bs].

    This function will divide the global batch into segments each with consecutive
    `granularity` sequences, and then redistribute the segments (e.g., for GRPO).
    """
    all_gathered = all_gather_tensor_container(data, group=group)

    all_data = []
    for d in all_gathered:
        bs = get_batch_size(d)
        assert bs % granularity == 0, f"Batch size {bs} not divisible by granularity {granularity}"
        all_data += [
            _slice_tensor_dict(d, i, i + granularity) for i in range(0, bs, granularity)
        ]

    seqlens = [d["attention_mask"].sum().item() for d in all_data]

    # Remove pad positions
    for d in all_data:
        l = d["attention_mask"].sum(-1).max().item()
        attn_mask_shape = d["attention_mask"].shape
        for k, v in d.items():
            if (
                torch.is_tensor(v)
                and len(v.shape) >= 2
                and v.shape[:2] == attn_mask_shape[:2]
            ):
                d[k] = v[:, :l]

    # No capacity limit leads to balanced partition across this group
    group_indices = ffd_allocate(
        seqlens, capacity=int(1e12), min_groups=dist.get_world_size(group)
    )
    local_indices = group_indices[dist.get_rank(group=group)]

    data = concat_padded_tensors([all_data[i] for i in local_indices])
    return RedistributedData(
        all_data=all_data,
        data=data,
        rank=dist.get_rank(group=group),
        group_indices=group_indices,
    )


class DistRolloutCoordinator:
    def __init__(self, rollout_engine: InferenceEngine, train_engine: TrainEngine):

        self.rollout_engine = rollout_engine
        self.train_engine = train_engine

    def _broadcast_and_redistribute_batch(
        self,
        batch: Dict[str, Any] | None,
        granularity: int = 1,
    ) -> Dict[str, Any]:
        """Broadcast and redistribute batch across distributed workers.

        This helper encapsulates:
        1. Redistribution within data parallel group (for load balancing)
        2. Broadcasting to context and model parallel group
        3. Synchronization barriers

        Parameters
        ----------
        batch : Dict[str, Any] | None
            Batch data from data parallel head, None for other ranks
        granularity : int, default=1
            Granularity for redistribution within data parallel group.
            - For single-turn rollouts: Use actor.config.group_size (GRPO grouping)
            - For multi-turn rollouts: Use 1 (default, per-completion redistribution)
            - For custom scenarios: Use custom value (e.g., n_trajs for agent trajectories)

        Returns
        -------
        Dict[str, Any]
            Redistributed and broadcast batch available on all ranks
        """
        # if batch is not None:
        #     redist = redistribute(
        #         batch,
        #         granularity=granularity,
        #         group=self.train_engine.data_parallel_group,
        #     )
        #     batch = redist.data

        dist.barrier(device_ids=[current_platform.current_device()])
        current_platform.synchronize()

        batch = broadcast_tensor_container(
            batch,
            src_rank=self.train_engine.current_data_parallel_head(),
            group=self.train_engine.context_and_model_parallel_group,
        )

        dist.barrier(device_ids=[current_platform.current_device()])
        current_platform.synchronize()

        return batch

    def rollout_batch(
        self,
        data: List[Dict[str, Any]],
        granularity: int = 1,
        workflow: Optional[RolloutWorkflow] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> Dict[str, Any]:
        """Generate rollout batch with distributed coordination (synchronous).

        This method orchestrates distributed rollout generation:
        - Only data parallel heads generate rollouts (avoid redundancy)
        - Results are transferred to device and redistributed
        - Batch is broadcast to all workers
        - Synchronization barriers ensure consistency

        Must call connect_engine() before using this method.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Input data batch for rollout generation
        granularity : int, default=1
            Granularity for redistribution within data parallel group.
            - For single-turn rollouts: Set to actor.config.group_size (GRPO grouping)
            - For multi-turn rollouts: Use default value of 1 (per-completion redistribution)
            - For custom scenarios: Use custom value (e.g., n_trajs for agent trajectories)
        workflow : RolloutWorkflow, optional
            Workflow defining rollout logic
        workflow_builder : Callable, optional
            Builder function for workflow
        should_accept : Callable, optional
            Filter function for accepting samples

        Returns
        -------
        Dict[str, Any]
            Generated rollout batch on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        batch = None
        if self.train_engine.is_data_parallel_head():
            batch = self.rollout_engine.rollout_batch(
                data,
                workflow=workflow,
                workflow_builder=workflow_builder,
                should_accept=should_accept,
            )
            batch = tensor_container_to(batch, current_platform.current_device())

        return self._broadcast_and_redistribute_batch(batch, granularity=granularity)

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        granularity: int = 1,
        workflow: Optional[RolloutWorkflow] = None,
        workflow_builder: Optional[Callable] = None,
        should_accept: Callable | None = None,
    ) -> Dict[str, Any]:
        """Prepare async rollout batch with distributed coordination.

        Similar to rollout_batch but uses prepare_batch for async training,
        where rollout generation happens concurrently with training.

        Must call connect_engine() before using this method.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            Dataloader to pull samples from
        granularity : int, default=1
            Granularity for redistribution within data parallel group.
            - For single-turn rollouts: Set to actor.config.group_size (GRPO grouping)
            - For multi-turn rollouts: Use default value of 1 (per-completion redistribution)
            - For custom scenarios: Use custom value (e.g., n_trajs for agent trajectories)
        workflow : RolloutWorkflow, optional
            Workflow defining rollout logic
        workflow_builder : Callable, optional
            Builder function for workflow
        should_accept : Callable, optional
            Filter function for accepting samples based on staleness

        Returns
        -------
        Dict[str, Any]
            Prepared rollout batch on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        batch = None
        if self.train_engine.is_data_parallel_head():
            batch = self.rollout_engine.prepare_batch(
                dataloader,
                workflow=workflow,
                workflow_builder=workflow_builder,
                should_accept=should_accept,
            )
            batch = tensor_container_to(batch, current_platform.current_device())

        return self._broadcast_and_redistribute_batch(batch, granularity=granularity)
