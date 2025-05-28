# Copyright 2025 Ant Group Inc.

import bisect
import dataclasses
import itertools
from collections import defaultdict
from typing import *

import numpy as np
import torch
import torch.distributed as dist

from realhf.api.core.config import ModelName, ModelShardID
from realhf.api.core.data_api import SequenceSample
from realhf.base import constants, logging
from realhf.base.topology import ProcessTopology, new_or_get_group
from realhf.impl.model.comm.global_comm import filter_match_mwids
from realhf.system.redistributor import RedistribStep

BCAST_GROUPS = {}
GATHER_GROUPS = {}
SCATTER_GROUPS = {}

logger = logging.getLogger("data_manager", "system")


def find_minimal_superset(A: List[Set[int]], B: Set[int]) -> Set[int] | None:
    min_size = float("inf")
    result = None
    for S in A:
        if B.issubset(S):
            if len(S) < min_size:
                min_size = len(S)
                result = S
    return result


class DataManager:

    def __init__(
        self,
        model_topos: Dict[ModelName, ProcessTopology],
        msid2mwid: Optional[Dict[ModelShardID, int]] = None,
        data_transfer_pairs: Optional[List[Tuple[ModelName, ModelName]]] = None,
    ):
        self.model_topos = model_topos
        self.msid2mwid = msid2mwid
        self.data_transfer_pairs = data_transfer_pairs

        self.storage: Dict[Hashable, SequenceSample] = {}

    def setup_process_groups(self):
        if self.msid2mwid is None or self.data_transfer_pairs is None:
            return

        model_topos = self.model_topos
        msid2mwid = self.msid2mwid
        data_transfer_pairs = self.data_transfer_pairs

        # Stores the ranks given a (model_name, dp_rank) pair.
        # These workers correspond to a complete set of model parameters sharded by TP+PP.
        mw_dp_ranks: Dict[Tuple[ModelName, int], List[int]] = {}

        mw_ranks: Dict[ModelName, List[int]] = {}

        # Stores the dp_head (i.e., tp_rank=0, pp_rank=-1) ranks given a model_name.
        mw_dp_head_ranks: Dict[ModelName, List[int]] = defaultdict(list)

        assert msid2mwid is not None
        for model_name, topo in model_topos.items():
            mw_ranks[model_name] = filter_match_mwids(
                model_name,
                topo,
                msid2mwid,
            )
            mw_dp_head_ranks[model_name] = filter_match_mwids(
                model_name,
                topo,
                msid2mwid,
                pipe=topo.get_dim("pipe") - 1,
                tensor=0,
            )
            dp_size = topo.get_dim("data")
            for dp_i in range(dp_size):
                mw_dp_ranks[model_name, dp_i] = filter_match_mwids(
                    model_name,
                    topo,
                    msid2mwid,
                    data=dp_i,
                )

        for src, dst in data_transfer_pairs:
            src_topo = model_topos[src]
            dst_topo = model_topos[dst]

            ranks = tuple(sorted(mw_dp_head_ranks[src]))
            GATHER_GROUPS[ranks] = new_or_get_group(
                list(ranks), backend="nccl" if constants.use_cuda() else "gloo"
            )

            for rank in ranks:
                scatter_ranks = tuple(sorted(set([rank] + mw_ranks[dst])))
                SCATTER_GROUPS[scatter_ranks] = new_or_get_group(
                    list(scatter_ranks),
                    backend="nccl" if constants.use_cuda() else "gloo",
                )

            # Construct all src-dst pairs, from any src dp rank to any dst dp rank.
            # Note that a dp rank corresponds to multiple parameter shards (TP+PP),
            # so each pair is a group-to-group communication.
            # Since the models in the source group have duplicate data (TP+PP),
            # we just use its "head" as the broadcast source,
            # and broadcast to all the ranks in the destination group.
            for src_dp, dst_dp in itertools.product(
                range(src_topo.get_dim("data")), range(dst_topo.get_dim("data"))
            ):
                src_mw_rank = mw_dp_head_ranks[src][src_dp]
                dst_mw_ranks = mw_dp_ranks[dst, dst_dp]
                # The src and dst groups can be disjoint or overlapped.
                # If they are disjoint, we need to include the src_mw_rank in the group.
                # Otherwise, we only need to include the dst_mw_ranks.
                if src_mw_rank not in dst_mw_ranks:
                    _ranks = [src_mw_rank] + dst_mw_ranks
                else:
                    _ranks = dst_mw_ranks
                key = tuple(sorted(_ranks))
                BCAST_GROUPS[key] = new_or_get_group(
                    _ranks, backend="nccl" if constants.use_cuda() else "gloo"
                )

    def storage_size(self):
        return len(self.storage)

    def store(self, x: SequenceSample):
        assert len(x.ids) == 1
        assert x.ids[0] not in self.storage
        self.storage[x.ids[0]] = x

    def update(self, x: SequenceSample):
        self.storage[x.ids[0]].update_(x)

    def get(self, data_id: Hashable):
        return self.storage[data_id]

    def has_data(self, data_id: Hashable):
        return data_id in self.storage

    def remove(self, ids: List[Hashable]):
        for data_id in ids:
            if data_id in self.storage:
                del self.storage[data_id]

    def clear_data(self):
        self.storage.clear()

    def _bcast_recv(
        self,
        step: RedistribStep,
        data_infos: Dict[Hashable, SequenceSample],
    ):
        assert len(step.keys) == 1
        ids = step.ids
        key = step.keys[0]
        dtype = data_infos[ids[0]].dtypes[key]
        total_len = sum(sum(data_infos[_id].seqlens[key][0]) for _id in ids)
        trailing_shape = data_infos[ids[0]].trailing_shapes[key]

        buf = torch.zeros(
            (total_len, *trailing_shape),
            dtype=dtype,
            device=constants.current_device(),
        )

        if len(step.dsts) == 1:
            dist.recv(buf, src=step.root)
        else:
            global BCAST_GROUPS
            group = BCAST_GROUPS[tuple(sorted([step.root] + list(step.dsts)))]
            dist.broadcast(buf, src=step.root, group=group)

        # Split the received data and put it into the storage.
        offset = 0
        for _id in ids:
            seqlens = data_infos[_id].seqlens[key]
            assert len(seqlens) == 1
            seqlen = sum(seqlens[0])
            if buf is not None:
                vs = buf[offset : offset + seqlen]
            else:
                vs = None
            offset = offset + seqlen
            with SequenceSample.disable_validation():
                s = SequenceSample(
                    keys=[key],
                    dtypes={key: vs.dtype if vs is not None else None},
                    trailing_shapes={key: vs.shape[1:] if vs is not None else None},
                    ids=[_id],
                    seqlens={key: seqlens},
                    data={key: vs},
                    metadata={},
                )
            if _id in self.storage:
                self.storage[_id].update_(s)
            else:
                self.storage[_id] = s

    def _bcast_send(self, step: RedistribStep):
        ids = step.ids
        for _id in ids:
            self.storage[_id].to_device(constants.current_device())
        assert len(step.keys) == 1
        key = step.keys[0]
        vs = torch.cat(
            [self.storage[_id].data[key] for _id in ids],
            dim=0,
        )
        if len(step.dsts) == 1:
            dist.send(vs, dst=step.dsts[0])
        else:
            global BCAST_GROUPS
            group = BCAST_GROUPS[tuple(sorted([step.root] + list(step.dsts)))]
            dist.broadcast(vs, src=step.root, group=group)

    def _run_bcast(
        self, step: RedistribStep, data_infos: Dict[Hashable, SequenceSample]
    ):
        if dist.get_rank() in step.dsts:
            self._bcast_recv(step, data_infos=data_infos)

        if dist.get_rank() == step.root:
            self._bcast_send(step)

    def _pad_data(self, x: torch.Tensor, maxlen: int):
        assert x.dtype == torch.float32
        assert len(x.shape) == 1
        if maxlen > x.numel():
            return torch.nn.functional.pad(x, (0, maxlen - x.numel()), value=0.0)
        return x

    def _run_gather(
        self, step: RedistribStep, data_infos: Dict[Hashable, SequenceSample]
    ):
        # It's possible that some DP rank is not involved.
        # Create dummpy data to make the gather happy.
        gather_ranks = find_minimal_superset(
            [set(k) for k in GATHER_GROUPS.keys()], set(step.srcs)
        )
        assert gather_ranks is not None, (
            set(step.srcs),
            [set(k) for k in GATHER_GROUPS.keys()],
        )
        gather_ranks = sorted(list(gather_ranks))

        pgroup = GATHER_GROUPS[tuple(gather_ranks)]

        if dist.get_rank() not in gather_ranks:
            return

        maxlen = 0
        for ids in step.ids:
            infos = [data_infos[i] for i in ids]
            maxlen = max(
                maxlen,
                sum(
                    [
                        sum([sum(info.seqlens[key][0]) for info in infos])
                        for key in step.keys
                    ]
                ),
            )

        if dist.get_rank() == step.root:
            gather_list = [
                torch.empty(
                    maxlen, device=constants.current_device(), dtype=torch.float32
                )
                for _ in gather_ranks
            ]
            is_valid_gather = [i in step.srcs for i in gather_ranks]
        else:
            gather_list = None

        if dist.get_rank() in step.srcs:
            local_gather_idx = step.srcs.index(dist.get_rank())
            ids = step.ids[local_gather_idx]
            for i in ids:
                self.storage[i].to_device(constants.current_device())
            samples = [self.storage[i] for i in ids]
            data = torch.cat(
                [
                    sample.data[key].float().flatten()
                    for sample in samples
                    for key in step.keys
                ]
            )
            data = self._pad_data(data, maxlen)
        else:
            data = torch.empty(
                maxlen, device=constants.current_device(), dtype=torch.float32
            )

        dist.gather(
            data,
            gather_list,
            dst=step.root,
            group=pgroup,
        )

        if dist.get_rank() != step.root:
            del data
            return

        cnt = 0
        for is_valid, buf in zip(is_valid_gather, gather_list):
            if not is_valid:
                continue
            ids = step.ids[cnt]
            offset = 0
            for i in ids:
                for key in step.keys:
                    seqlen = data_infos[i].seqlens[key][0]
                    dtype = data_infos[i].dtypes[key]
                    trailing_shape = data_infos[i].trailing_shapes[key]
                    size = int(np.prod(trailing_shape) * sum(seqlen))
                    data = buf[offset : offset + size].to(dtype)
                    offset += size
                    # with SequenceSample.disable_validation():
                    s = SequenceSample(
                        keys=[key],
                        dtypes={key: dtype},
                        trailing_shapes={key: trailing_shape},
                        ids=[i],
                        seqlens={key: [seqlen]},
                        data={key: data},
                        metadata={},
                    )
                    if i in self.storage:
                        self.storage[i].update_(s)
                    else:
                        self.storage[i] = s
            cnt += 1
        assert cnt == len(step.srcs) == len(step.ids)
        del data

    def _run_scatter(
        self, step: RedistribStep, data_infos: Dict[Hashable, SequenceSample]
    ):
        if dist.get_rank() != step.root and dist.get_rank() not in step.dsts:
            return

        maxlen = 0
        for ids in step.ids:
            infos = [data_infos[i] for i in ids]
            maxlen = max(
                maxlen,
                sum(
                    [
                        sum([sum(info.seqlens[key][0]) for info in infos])
                        for key in step.keys
                    ]
                ),
            )

        buf = torch.empty(
            maxlen, device=constants.current_device(), dtype=torch.float32
        )

        if dist.get_rank() == step.root:
            # Scatter destinations include all DP, TP, and PP ranks
            # and data is duplicated among TP/PP groups
            # We allocate new memory for DP ranks, but use the same pointer
            # for all TP and PP ranks to save memory.
            scatter_clusters = []
            for idx, ids in enumerate(step.ids):
                for _ids, idx_list in scatter_clusters:
                    if set(ids) == set(_ids):
                        idx_list.append(idx)
                        break
                else:
                    scatter_clusters.append((ids, [idx]))
            scatter_list = [None for _ in range(len(step.ids))]
            before_pad = []
            for ids, idx_list in scatter_clusters:
                for i in ids:
                    self.storage[i].to_device(constants.current_device())
                samples = [self.storage[i] for i in ids]
                data = torch.cat(
                    [
                        sample.data[key].float().flatten()
                        for sample in samples
                        for key in step.keys
                    ]
                )
                before_pad.append(data)

            maxlen = max([x.shape[0] for x in before_pad])
            after_pad = [self._pad_data(x, maxlen) for x in before_pad]
            for (ids, idx_list), data in zip(scatter_clusters, after_pad):
                for idx in idx_list:
                    scatter_list[idx] = data

            assert all([torch.is_tensor(t) for t in scatter_list])

            if step.root not in step.dsts:
                idx = bisect.bisect(step.dsts, step.root)
                scatter_list.insert(idx, buf)
        else:
            scatter_list = None

        key = tuple(sorted(set([step.root] + step.dsts)))
        dist.scatter(buf, scatter_list, src=step.root, group=SCATTER_GROUPS[key])

        if dist.get_rank() not in step.dsts:
            return

        local_dst_idx = step.dsts.index(dist.get_rank())
        ids = step.ids[local_dst_idx]
        offset = 0
        for i in ids:
            for key in step.keys:
                seqlen = data_infos[i].seqlens[key][0]
                dtype = data_infos[i].dtypes[key]
                trailing_shape = data_infos[i].trailing_shapes[key]
                size = int(np.prod(trailing_shape) * sum(seqlen))
                data = buf[offset : offset + size].to(dtype)
                offset += size
                # with SequenceSample.disable_validation():
                s = SequenceSample(
                    keys=[key],
                    dtypes={key: dtype},
                    trailing_shapes={key: trailing_shape},
                    ids=[i],
                    seqlens={key: [seqlen]},
                    data={key: data},
                    metadata={},
                )
                if i in self.storage:
                    self.storage[i].update_(s)
                else:
                    self.storage[i] = s

    def redistribute(
        self,
        data_info: SequenceSample,
        plan: List[RedistribStep],
    ):
        data_infos = {x.ids[0]: x for x in data_info.unpack()}

        for step in plan:
            if step.comm_type == "bcast":
                self._run_bcast(step, data_infos)
            elif step.comm_type == "gather":
                self._run_gather(step, data_infos)
            elif step.comm_type == "scatter":
                self._run_scatter(step, data_infos)
