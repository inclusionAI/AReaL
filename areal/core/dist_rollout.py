from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    tensor_container_to,
)
from areal.utils.datapack import ffd_allocate


@dataclass
class RedistributedData:
    all_data: list[dict[str, Any]]
    data: dict[str, Any]
    rank: int
    group_indices: list[list[int]]


def _remove_padding_from_trajectory(d: dict[str, Any]) -> dict[str, Any]:
    """Remove padding from a single trajectory dict based on attention_mask.

    Modifies the dict in-place and returns it.
    """
    if "attention_mask" not in d:
        return d
    max_sequence_length = int(d["attention_mask"].sum(-1).max().item())
    attn_mask_shape = d["attention_mask"].shape
    for k, v in d.items():
        if (
            torch.is_tensor(v)
            and len(v.shape) >= 2
            and v.shape[:2] == attn_mask_shape[:2]
        ):
            d[k] = v[:, :max_sequence_length]
    return d


def redistribute_trajectories(
    trajectories: list[dict[str, Any]],
    group=None,
) -> RedistributedData:
    """Redistribute a list of trajectory dicts across a process group.

    Each trajectory dict should contain tensors with shape [batch_size, seqlen, *],
    where batch_size can vary per trajectory. This function gathers trajectories
    from all ranks and redistributes them for load balancing based on sequence lengths.

    Parameters
    ----------
    trajectories : list[dict[str, Any]]
        List of trajectory dictionaries from the local rank. Each trajectory
        contains tensors with shape [batch_size, seqlen, ...].
    group : dist.ProcessGroup, optional
        The process group for communication. If None, uses the default group.

    Returns
    -------
    RedistributedData
        Contains:
        - all_data: All trajectories gathered from all ranks (with padding removed)
        - data: Concatenated trajectories assigned to the local rank
        - rank: Local rank in the group
        - group_indices: Assignment of trajectory indices to each rank
    """
    # All-gather trajectories from all ranks
    all_gathered = all_gather_tensor_container(trajectories, group=group)

    # Flatten the list of lists into a single list of trajectories
    all_data = []
    for traj_list in all_gathered:
        all_data.extend(traj_list)

    # Compute sequence lengths for load balancing
    seqlens = [d["attention_mask"].sum().item() for d in all_data]

    # Remove pad positions from each trajectory
    for d in all_data:
        _remove_padding_from_trajectory(d)

    # Allocate trajectories to ranks using first-fit-decreasing
    # No capacity limit leads to balanced partition across this group
    group_indices = ffd_allocate(
        seqlens, capacity=int(1e12), min_groups=dist.get_world_size(group)
    )
    local_indices = group_indices[dist.get_rank(group=group)]

    # Concatenate assigned trajectories for this rank
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

    def _broadcast_and_redistribute_trajectories(
        self,
        trajectories: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Broadcast and redistribute trajectories across distributed workers.

        This helper encapsulates:
        1. Redistribution within data parallel group (for load balancing)
        2. Broadcasting to context and model parallel group
        3. Synchronization barriers

        Parameters
        ----------
        trajectories : list[dict[str, Any]] | None
            List of trajectory dicts from data parallel head, None for other ranks.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory.

        Returns
        -------
        dict[str, Any]
            Redistributed and broadcast batch available on all ranks (concatenated)
        """
        if trajectories is not None:
            redist = redistribute_trajectories(
                trajectories,
                group=self.train_engine.data_parallel_group,
            )
            batch = redist.data
        else:
            batch = None

        current_platform.synchronize()
        dist.barrier(group=self.train_engine.cpu_group)

        batch = broadcast_tensor_container(
            batch,
            src_rank=self.train_engine.current_data_parallel_head(),
            group=self.train_engine.context_and_model_parallel_group,
        )

        current_platform.synchronize()
        dist.barrier(group=self.train_engine.cpu_group)

        return batch

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> dict[str, Any]:
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
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            Workflow defining rollout logic
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).

        Returns
        -------
        Dict[str, Any]
            Generated rollout batch on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        trajectories = None
        if self.train_engine.is_data_parallel_head():
            trajectories = self.rollout_engine.rollout_batch(
                data,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
            )
            trajectories = tensor_container_to(
                trajectories, current_platform.current_device()
            )

        return self._broadcast_and_redistribute_trajectories(trajectories)

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> dict[str, Any]:
        """Prepare async rollout batch with distributed coordination.

        Similar to rollout_batch but uses prepare_batch for async training,
        where rollout generation happens concurrently with training.

        Must call connect_engine() before using this method.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            Dataloader to pull samples from
        workflow : RolloutWorkflow | type[RolloutWorkflow] | str
            Workflow defining rollout logic
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            Filter function for accepting samples based on staleness
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        dynamic_bs : bool, optional
            If True, enables dynamic batch sizing. Default is False.

        Returns
        -------
        Dict[str, Any]
            Prepared rollout batch on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        trajectories = None
        if self.train_engine.is_data_parallel_head():
            trajectories = self.rollout_engine.prepare_batch(
                dataloader,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
                dynamic_bs=dynamic_bs,
            )
            trajectories = tensor_container_to(
                trajectories, current_platform.current_device()
            )

        return self._broadcast_and_redistribute_trajectories(trajectories)
