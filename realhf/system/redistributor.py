# Copyright 2025 Ant Group Inc.

import asyncio
import dataclasses
import itertools
from collections import defaultdict
from typing import *

from realhf.api.cli_args import ClusterSpecConfig


class GlobalStorageTracker:
    def __init__(self, world_size: int):
        self.lock = asyncio.Lock()
        self.storages: List[Dict[Hashable, List[str]]]
        self.storages = [{} for _ in range(world_size)]
        self.data_owner: Dict[Tuple[Hashable, str], int]
        self.data_owner = {}

    async def add_data(self, rank: int, ids: List[Hashable], key: str, is_owner: bool):
        async with self.lock:
            for data_id in ids:
                if data_id not in self.storages[rank]:
                    self.storages[rank][data_id] = [key]
                else:
                    if key not in self.storages[rank][data_id]:
                        self.storages[rank][data_id].append(key)
                if is_owner:
                    self.data_owner[(data_id, key)] = rank

    def add_data_synced(self, rank: int, ids: List[Hashable], key: str, is_owner: bool):
        for data_id in ids:
            if data_id not in self.storages[rank]:
                self.storages[rank][data_id] = [key]
            else:
                if key not in self.storages[rank][data_id]:
                    self.storages[rank][data_id].append(key)
            if is_owner:
                self.data_owner[(data_id, key)] = rank

    async def clear_data(self, ids: List[Hashable]):
        async with self.lock:
            for storage in self.storages:
                for i in ids:
                    if i in storage:
                        storage.pop(i)
            keys = list(self.data_owner.keys())
            for i, k in keys:
                if i in ids:
                    self.data_owner.pop((i, k))


@dataclasses.dataclass
class RedistribStep:
    comm_type: str
    root: int | None
    srcs: List[int] | None
    dsts: List[int] | None
    ids: List[List[Hashable]]
    keys: List[str]

    def __repr__(self) -> str:
        if self.comm_type == "gather":
            return f"Gather {self.keys} to {self.root} from {self.srcs}."
        if self.comm_type == "scatter":
            return f"Scatter {self.keys} from {self.root} to {self.dsts}."
        if self.comm_type == "bcast":
            return f"Bcast {self.keys} from {self.root} to {self.dsts}."
        raise NotImplementedError()


class RedistribPlanner:
    def __init__(
        self, cluster_config: ClusterSpecConfig, storage_tracker: GlobalStorageTracker
    ):
        self.cluster_config = cluster_config
        self.storage_tracker = storage_tracker

    def derive_plan(
        self,
        dests: Dict[int, List[Hashable]],
        keys: List[str],
        pattern: str = "gather-scatter",
    ) -> List[RedistribStep]:
        if pattern == "gather-scatter":
            return self.derive_plan_gather_scatter(dests, keys)
        elif pattern == "bcast":
            return self.derive_plan_bcast(dests, keys)
        raise NotImplementedError(f"Unknown data redistribution pattern: {pattern}")

    def derive_plan_gather_scatter(
        self, dests: Dict[int, List[Hashable]], keys: List[str]
    ) -> List[RedistribStep]:
        self.dests = dests

        all_data_ids = set()
        for all_samples in dests.values():
            for data_id in all_samples:
                all_data_ids.add(data_id)

        transfer_plan = []
        for key in keys:
            owners = sorted(
                list(
                    set(
                        [
                            self.storage_tracker.data_owner[(i, key)]
                            for i in all_data_ids
                        ]
                    )
                )
            )
            gather_ids = []
            for owner in owners:
                this_owner_ids = []
                for i in all_data_ids:
                    if (
                        i in self.storage_tracker.storages[owner]
                        and key in self.storage_tracker.storages[owner][i]
                    ):
                        this_owner_ids.append(i)
                gather_ids.append(sorted(this_owner_ids))
            gather_step = RedistribStep(
                comm_type="gather",
                root=owners[0],
                srcs=owners,
                dsts=None,
                ids=gather_ids,
                keys=[key],
            )

            scatter_dsts = sorted([i for i in dests if dests[i]])
            scatter_ids = [sorted(dests[i]) for i in scatter_dsts]
            scatter_step = RedistribStep(
                comm_type="scatter",
                root=owners[0],
                dsts=scatter_dsts,
                srcs=None,
                ids=scatter_ids,
                keys=[key],
            )
            transfer_plan += [gather_step, scatter_step]

        # Prune the plan.
        pop_indices = []
        for idx, step in enumerate(transfer_plan):
            # 1. Omit the gather step if data has already been gathered before.
            if step.comm_type == "gather":
                all_gather_ids = list(itertools.chain.from_iterable(step.ids))
                key = step.keys[0]
                if any(
                    i not in self.storage_tracker.storages[step.root]
                    for i in all_gather_ids
                ):
                    continue
                if any(
                    key not in self.storage_tracker.storages[step.root][i]
                    for i in all_gather_ids
                ):
                    continue
                pop_indices.append(idx)
            # 2. Omit the gather + scatter step if data has already exists in all dst GPUs.
            if step.comm_type == "scatter":
                key = step.keys[0]
                all_exists = True
                for dst, ids in zip(step.dsts, step.ids):
                    if any(i not in self.storage_tracker.storages[dst] for i in ids):
                        all_exists = False
                        break
                    if any(
                        key not in self.storage_tracker.storages[dst][i] for i in ids
                    ):
                        all_exists = False
                        break
                if all_exists:
                    pop_indices.append(idx)
                    pop_indices.append(idx - 1)
        for pop_idx in reversed(sorted(set(pop_indices))):
            transfer_plan.pop(pop_idx)

        # Merging the gather/scatter of different keys
        gather_plan = {}
        scatter_plan = {}
        for step in transfer_plan:
            if step.comm_type == "gather":
                plan_key = (
                    step.root,
                    tuple(sorted(step.srcs)),
                    tuple([tuple(sorted(ids)) for ids in step.ids]),
                )
                if plan_key not in gather_plan:
                    gather_plan[plan_key] = step
                else:
                    assert all(
                        k not in gather_plan[plan_key].keys for k in step.keys
                    ), (
                        gather_plan[plan_key],
                        step,
                        plan_key,
                    )
                    gather_plan[plan_key].keys += step.keys
            if step.comm_type == "scatter":
                plan_key = (
                    step.root,
                    tuple(sorted(step.dsts)),
                    tuple([tuple(sorted(ids)) for ids in step.ids]),
                )
                if plan_key not in scatter_plan:
                    scatter_plan[plan_key] = step
                else:
                    assert all(
                        k not in scatter_plan[plan_key].keys for k in step.keys
                    ), (
                        scatter_plan[plan_key],
                        step,
                        plan_key,
                    )
                    scatter_plan[plan_key].keys += step.keys

        # Prioritize gather over scatter
        return list(gather_plan.values()) + list(scatter_plan.values())

    def derive_plan_bcast(
        self, dests: Dict[int, List[Hashable]], keys: List[str] | Tuple[str]
    ) -> List[RedistribStep]:
        assert isinstance(keys, (list, tuple)), type(keys)
        keys = list(keys)
        self.dests = dests

        # Get all requried data IDs.
        all_data_ids = set()
        for all_samples in self.dests.values():
            for data_id in all_samples:
                all_data_ids.add(data_id)

        # The producers for each required data.
        id2gpu_src = {}
        for data_id in all_data_ids:
            for key in keys:
                id2gpu_src[(data_id, key)] = []
                for gpu_id, ids2keys in enumerate(self.storage_tracker.storages):
                    if data_id in ids2keys and key in ids2keys[data_id]:
                        id2gpu_src[(data_id, key)].append(gpu_id)

        # The consumers for each requried data.
        id2gpu_dst = {}
        for data_id in all_data_ids:
            for key in keys:
                id2gpu_dst[(data_id, key)] = []
                for gpu_id, ids in self.dests.items():
                    if data_id in ids:
                        id2gpu_dst[(data_id, key)].append(gpu_id)

        self.transfer_plan = {}

        for data_id, key in itertools.product(all_data_ids, keys):
            source_gpus = id2gpu_src[(data_id, key)]
            target_gpus = id2gpu_dst[(data_id, key)]

            assert len(source_gpus) > 0, (data_id, key, id2gpu_src, id2gpu_dst)

            # Omit data transfer if it exists in the target GPU
            target_gpus = [gpu for gpu in target_gpus if gpu not in source_gpus]
            if not target_gpus:
                continue

            # Find the "nearest" GPU for data transfer.
            best_src = self._select_best_bcast_source(source_gpus, target_gpus)

            self.transfer_plan[(data_id, key)] = {"src": best_src, "dsts": target_gpus}

        return self._group_bcast_transfers()

    def _on_same_node(self, i, j) -> bool:
        return (i // self.cluster_config.n_gpus_per_node) == (
            j // self.cluster_config.n_gpus_per_node
        )

    def _select_best_bcast_source(self, source_gpus, target_gpus):
        same_node_counts = {}
        for src in source_gpus:
            same_node_count = sum(
                1 for dst in target_gpus if self._on_same_node(src, dst)
            )
            same_node_counts[src] = same_node_count

        # Find the source that maximizes locality.
        max_same_node = max(same_node_counts.values())
        best_sources = [
            src for src, count in same_node_counts.items() if count == max_same_node
        ]

        # Find the source with the smallest workload.
        src_load = defaultdict(int)
        for plan in self.transfer_plan.values():
            src_gpu = plan["src"]
            src_load[src_gpu] += len(plan["dsts"])
        return min(best_sources, key=lambda src: src_load[src])

    def _group_bcast_transfers(self) -> List[RedistribStep]:
        # Group data ids that should be transferred from "src" to "dsts"
        src_to_transfers = defaultdict(lambda: defaultdict(list))
        for (data_id, key), plan in self.transfer_plan.items():
            src_to_transfers[(plan["src"], key)][tuple(sorted(plan["dsts"]))].append(
                data_id
            )

        stages = []
        while any(src_to_transfers.values()):
            stage = []
            used_dsts = set()
            used_srcs = set()

            for (src, key), transfers in list(src_to_transfers.items()):
                if src in used_srcs:
                    continue
                if not transfers:
                    continue

                # Find a transfer that can be concurrent executed.
                pop_key = None
                for i, dsts in enumerate(transfers):
                    if not any(dst in used_dsts for dst in dsts):
                        pop_key = dsts
                        break

                if pop_key is not None:
                    data_ids = transfers.pop(pop_key)
                    stage.append(
                        RedistribStep(
                            comm_type="bcast",
                            root=src,
                            srcs=[src],
                            keys=[key],
                            dsts=pop_key,
                            ids=data_ids,
                        )
                    )
                    used_dsts.update(dsts)
                    used_srcs.add(src)

            if stage:
                stages += stage
            else:
                for (src, key), transfers in list(src_to_transfers.items()):
                    if transfers:
                        dsts, data_ids = transfers.pop(0)
                        stages.append(
                            RedistribStep(
                                comm_type="bcast",
                                srcs=[src],
                                root=src,
                                dsts=dsts,
                                ids=data_ids,
                                keys=[key],
                            )
                        )
                        break

        return stages
