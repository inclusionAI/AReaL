# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import itertools
from collections import defaultdict
from typing import *

import numpy as np
import scipy.optimize
import torch.distributed
import torch.nn as nn

from realhf.api.core import model_api
from realhf.api.core.config import ModelName, ModelShardID
from realhf.base import constants, topology
from realhf.impl.model.comm.global_comm import filter_match_mwids
from realhf.impl.model.nn.flatten_param import (
    ContiguousParamSpec,
    build_param_spec,
    param_intervals_from_keys,
    param_size_from_keys,
)
from realhf.impl.model.nn.real_llm_base import keys_from_layer_indices
from realhf.impl.model.nn.real_llm_parallel import (
    partition_pipeline_layers,
    pipeline_repartition_strategy,
)

_TRAINABLE: Dict[ModelName, bool] = {}


def set_trainable(model_name: ModelName, trainable: bool):
    _TRAINABLE[model_name] = trainable


def is_trainable(model_name: ModelName) -> bool:
    return _TRAINABLE[model_name]


@dataclasses.dataclass(unsafe_hash=True)
class ParamReallocPair:
    src: ModelName
    src_dp_rank: int
    src_tp_rank: int
    src_pp_rank: int
    dst: ModelName
    dst_tp_rank: int
    dst_pp_rank: int


@dataclasses.dataclass(unsafe_hash=True)
class ParamReallocModelPair:
    src: ModelName
    dst: ModelName


@dataclasses.dataclass
class ParamReallocInfo:
    # Groups for parameter synchronization.
    param_realloc_model_group: Dict[
        ParamReallocModelPair, torch.distributed.ProcessGroup
    ]
    param_realloc_model_cpu_group: Dict[
        ParamReallocModelPair, torch.distributed.ProcessGroup
    ]
    param_realloc_groups: Dict[ParamReallocPair, torch.distributed.ProcessGroup]
    param_realloc_src_ranks: Dict[ParamReallocPair, int]
    param_realloc_dst_ranks: Dict[ParamReallocPair, List[int]]


def _max_match(_src_ranks: List[int], _grouped_dst_ranks: List[List[int]]):
    cost_matrix = []
    for source in _src_ranks:
        costs = []
        for destinations in _grouped_dst_ranks:
            cost = 0 if source in destinations else 1
            costs.append(cost)
        cost_matrix.append(costs)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def _group_mwids_by_node(ranks: List[int]) -> Dict[int, List[int]]:
    node2ranks = defaultdict(list)
    for r in ranks:
        node2ranks[r // 8].append(r)
    return {k: node2ranks[k] for k in sorted(node2ranks.keys())}


def _squeeze_mwids_by_node(ranks: List[int]) -> List[int]:
    node2ranks = _group_mwids_by_node(ranks)
    return [ranks[0] for ranks in node2ranks.values()]


def _assign_src_to_dsts(
    node2srcs: Dict[int, List[int]], node2dsts: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    """Assign nodes with a greedy algorithm.

    All ranks in the values of node2srcs have the data required by all dst ranks.
    Source ranks can be assigned to zero or multiple destination ranks.
    All destination ranks must be assigned to exactly one src rank.

    Args:
        node2srcs (Dict[int, List[int]]): Node index -> source ranks.
        node2dsts (Dict[int, List[int]]): Node index -> destination ranks.

    Returns:
        Dict[int, List[int]]: src rank -> dst ranks.
    """
    # First, assign all destination ranks to source nodes.
    # If a destination node is also a source node, assign it to itself.
    # Otherwise, find the node with the minimum workload for load balancing.
    dst_src_nodes = {}
    for dst_node in node2dsts.keys():
        if dst_node in node2srcs:
            # dst node is also src node
            dst_src_nodes[dst_node] = dst_node
    src_node_workloads = {k: int(k in dst_src_nodes) for k in node2srcs}
    assert sum(src_node_workloads.values()) == len(dst_src_nodes)
    for dst_node in node2dsts.keys():
        if dst_node not in node2srcs:
            # find a source node with the minimum workload
            src_node = min(src_node_workloads, key=src_node_workloads.get)
            dst_src_nodes[dst_node] = src_node
    assert all(dst_node in dst_src_nodes for dst_node in node2dsts)

    # Revert the key-value of the dict.
    src_dst_nodes = defaultdict(list)
    for dst_node, src_node in dst_src_nodes.items():
        src_dst_nodes[src_node].append(dst_node)

    # Next, find an appropriate source rank on each source node.
    # If the source rank is also the destination rank, assign it to the destination rank.
    # Otherwise, assign the first one to the destination ranks.
    assignment = {}
    for src_node, dst_nodes in src_dst_nodes.items():
        assigned = False
        src_ranks = node2srcs[src_node]
        dst_ranks = sum([node2dsts[dst_node] for dst_node in dst_nodes], start=[])
        for s in src_ranks:
            if s in dst_ranks:
                assignment[s] = dst_ranks
                assigned = True
                break
        if not assigned:
            assignment[src_ranks[0]] = dst_ranks
    assert len(set(assignment.keys())) == len(assignment.keys())
    assert len(set(sum(assignment.values(), []))) == len(sum(assignment.values(), []))

    return assignment


def _create_param_realloc_groups(
    from_topo: topology.ProcessTopology,
    to_topo: topology.ProcessTopology,
    src: ModelName,
    dst: ModelName,
    msid2mwid: Dict[ModelShardID, int],
    param_realloc_groups: Dict[ParamReallocPair, torch.distributed.ProcessGroup],
    param_realloc_src_ranks: Dict[ParamReallocPair, int],
    param_realloc_dst_ranks: Dict[ParamReallocPair, List[int]],
):
    mwid2msid: Dict[int, Dict[ModelName, ModelShardID]] = defaultdict(dict)
    for k, v in msid2mwid.items():
        mwid2msid[v][k.model_name] = k
    for pp_i, pp_j in itertools.product(
        range(from_topo.get_dim("pipe")), range(to_topo.get_dim("pipe"))
    ):
        # create tensor reshard groups
        src_tp_size = from_topo.get_dim("tensor")
        dst_tp_size = to_topo.get_dim("tensor")

        for tp_j in range(dst_tp_size):
            _all_dst_ranks = filter_match_mwids(
                dst, to_topo, msid2mwid, pipe=pp_j, tensor=tp_j
            )
            if src_tp_size > dst_tp_size:
                factor = src_tp_size // dst_tp_size
                tp_is = list(range(factor * tp_j, factor * (tp_j + 1)))
                _all_src_ranks = [
                    filter_match_mwids(
                        src, from_topo, msid2mwid, tensor=tp_i, pipe=pp_i
                    )
                    for tp_i in tp_is
                ]
            else:
                factor = dst_tp_size // src_tp_size
                _all_src_ranks = [
                    filter_match_mwids(
                        src,
                        from_topo,
                        msid2mwid,
                        tensor=tp_j // factor,
                        pipe=pp_i,
                    )
                ]
            # All GPUs in _src_ranks have the data required by (pp_j, tp_j)
            for _src_ranks in _all_src_ranks:
                # NOTE: inter-node communication cost is significantly larger than intra-node communication cost.
                # We only select one sender per host/node to prevent multiple senders occupying the same network bandwidth.
                # This is not the optimal solution for intra-node communication
                # because there may exist a source rank that is also dst rank,
                # but we forcely select the first source rank on each node here.
                assignment = _assign_src_to_dsts(
                    _group_mwids_by_node(_src_ranks),
                    _group_mwids_by_node(_all_dst_ranks),
                )
                _idle_src_ranks = [r for r in _src_ranks if r not in assignment]
                for _src_rank in _idle_src_ranks:
                    dp_i, tp_i = (
                        from_topo.get_coord(
                            mwid2msid[_src_rank][src].parallelism_rank
                        ).data,
                        from_topo.get_coord(
                            mwid2msid[_src_rank][src].parallelism_rank
                        ).tensor,
                    )
                    key = ParamReallocPair(
                        src=src,
                        src_dp_rank=dp_i,
                        src_tp_rank=tp_i,
                        src_pp_rank=pp_i,
                        dst=dst,
                        dst_tp_rank=tp_j,
                        dst_pp_rank=pp_j,
                    )
                    param_realloc_dst_ranks[key] = []
                    param_realloc_groups[key] = None
                    param_realloc_src_ranks[key] = _src_rank
                for _src_rank, _dst_ranks in assignment.items():
                    dp_i, tp_i = (
                        from_topo.get_coord(
                            mwid2msid[_src_rank][src].parallelism_rank
                        ).data,
                        from_topo.get_coord(
                            mwid2msid[_src_rank][src].parallelism_rank
                        ).tensor,
                    )
                    key = ParamReallocPair(
                        src=src,
                        src_dp_rank=dp_i,
                        src_tp_rank=tp_i,
                        src_pp_rank=pp_i,
                        dst=dst,
                        dst_tp_rank=tp_j,
                        dst_pp_rank=pp_j,
                    )
                    param_realloc_dst_ranks[key] = _dst_ranks
                    if _src_rank not in _dst_ranks:
                        _dst_ranks = [_src_rank] + _dst_ranks
                    assert len(set(_dst_ranks)) == len(_dst_ranks)
                    if len(_dst_ranks) > 1:
                        if torch.distributed.is_initialized():
                            param_realloc_groups[key] = topology.new_or_get_group(
                                _dst_ranks
                            )
                        else:
                            # for estimating parameter realloc cost
                            param_realloc_groups[key] = 1
                    else:
                        param_realloc_groups[key] = None
                    param_realloc_src_ranks[key] = _src_rank


def setup_param_realloc(
    model_topos: Optional[Dict[str, topology.ProcessTopology]] = None,
    msid2mwid: Optional[Dict[ModelShardID, int]] = None,
    param_realloc_pairs: Optional[List[Tuple[ModelName, ModelName]]] = None,
) -> ParamReallocInfo:
    param_realloc_groups = {}
    param_realloc_src_ranks = {}
    param_realloc_dst_ranks = {}
    param_realloc_model_group = {}
    param_realloc_model_cpu_group = {}
    if param_realloc_pairs is not None:
        for src, dst in param_realloc_pairs:
            _create_param_realloc_groups(
                model_topos[src],
                model_topos[dst],
                src,
                dst,
                msid2mwid,
                param_realloc_groups,
                param_realloc_src_ranks,
                param_realloc_dst_ranks,
            )
            pair_mw_ranks = set()
            topo1 = model_topos[src]
            topo2 = model_topos[dst]
            for i in range(topo1.world_size()):
                pair_mw_ranks.add(
                    msid2mwid[ModelShardID.from_parallelism_rank(src, topo1, i)]
                )
            for j in range(topo2.world_size()):
                pair_mw_ranks.add(
                    msid2mwid[ModelShardID.from_parallelism_rank(dst, topo2, j)]
                )
            param_realloc_model_group[ParamReallocModelPair(src, dst)] = (
                topology.new_or_get_group(list(sorted(pair_mw_ranks)))
            )
            param_realloc_model_cpu_group[ParamReallocModelPair(src, dst)] = (
                topology.new_or_get_group(list(sorted(pair_mw_ranks)), backend="gloo")
            )
    return ParamReallocInfo(
        param_realloc_groups=param_realloc_groups,
        param_realloc_src_ranks=param_realloc_src_ranks,
        param_realloc_dst_ranks=param_realloc_dst_ranks,
        param_realloc_model_group=param_realloc_model_group,
        param_realloc_model_cpu_group=param_realloc_model_cpu_group,
    )


@dataclasses.dataclass
class ReparallelizeSenderStep:
    rank: int
    sender_tp_portion_id: int
    receiver_tp_portion_id: int
    param_keys: List[str]
    param_intervals_cpu: List[Tuple[int, int]]
    param_intervals_cuda: torch.Tensor
    max_interval_size: int
    param_size: int
    group: torch.distributed.ProcessGroup
    dst_ranks: List[int]
    remove: bool = False


@dataclasses.dataclass
class ReparallelizeReceiverStep:
    rank: int
    sender_tp_portion_id: int
    receiver_tp_portion_id: int
    sender_param_intervals_cpu: List[Tuple[int, int]]
    sender_param_intervals_cuda: torch.Tensor
    sender_max_interval_size: int
    receiver_param_intervals_cpu: List[Tuple[int, int]]
    receiver_param_intervals_cuda: torch.Tensor
    receiver_max_interval_size: int
    param_size: int
    param_keys: List[str]
    param_dtype: torch.dtype
    src: int
    dst_ranks: List[int]
    group: torch.distributed.ProcessGroup


def _derive_reparallelize_comm_plan(
    from_model_name: ModelName,
    to_model_name: ModelName,
    from_topo: topology.ProcessTopology,
    to_topo: topology.ProcessTopology,
    from_model_config: model_api.ReaLModelConfig,
    to_model_config: model_api.ReaLModelConfig,
    pg_info: ParamReallocInfo,
    dtype: Optional[torch.dtype] = torch.float16,
) -> List[ReparallelizeReceiverStep | ReparallelizeSenderStep]:
    src_tp_size = from_topo.get_dim("tensor")
    dst_tp_size = to_topo.get_dim("tensor")
    assert src_tp_size % dst_tp_size == 0 or dst_tp_size % src_tp_size == 0
    for k, v in dataclasses.asdict(to_model_config).items():
        if k not in ["is_critic"] and v != getattr(from_model_config, k):
            raise ValueError(
                f"Can't load a checkpoint with different config (key `{k}`, "
                f"value in checkpoint is `{v}`, current value is `{getattr(from_model_config, k)}`)."
            )
    if from_model_config.n_kv_heads > 1 and (
        from_model_config.n_kv_heads % src_tp_size == 0
    ) != (from_model_config.n_kv_heads % dst_tp_size == 0):
        raise ValueError("Whether to partition kv heads should remain the same.")

    from_layer_mapping = partition_pipeline_layers(
        from_model_config,
        from_topo.get_dim("pipe"),
    )
    from_layer_mapping = {
        k: list(range(v[0], v[1])) for k, v in from_layer_mapping.items()
    }
    to_layer_mapping = partition_pipeline_layers(
        to_model_config,
        to_topo.get_dim("pipe"),
    )
    to_layer_mapping = {k: list(range(v[0], v[1])) for k, v in to_layer_mapping.items()}
    repart_strat = pipeline_repartition_strategy(from_layer_mapping, to_layer_mapping)

    from_model_head_param_point_to_embedding = (
        from_model_config.tied_embedding
        and not from_model_config.is_critic
        and from_topo.get_dim("pipe") == 1
    )
    to_model_head_param_point_to_embedding = (
        to_model_config.tied_embedding
        and not to_model_config.is_critic
        and to_topo.get_dim("pipe") == 1
    )
    if constants.has_model_name(from_model_name):
        with constants.model_scope(from_model_name):
            from_layer_indices = from_layer_mapping[constants.pipe_parallel_rank()]
            from_model_param_specs, _ = build_param_spec(
                from_layer_indices,
                from_model_config,
                tp_size=from_topo.get_dim("tensor"),
                dp_size=from_topo.get_dim("data"),
                pp_size=from_topo.get_dim("pipe"),
                head_param_point_to_embedding=from_model_head_param_point_to_embedding,
            )
    if constants.has_model_name(to_model_name):
        with constants.model_scope(to_model_name):
            to_layer_indices = to_layer_mapping[constants.pipe_parallel_rank()]
            to_model_param_specs, _ = build_param_spec(
                to_layer_indices,
                to_model_config,
                tp_size=to_topo.get_dim("tensor"),
                pp_size=to_topo.get_dim("pipe"),
                dp_size=to_topo.get_dim("data"),
                head_param_point_to_embedding=to_model_head_param_point_to_embedding,
            )

    comm_plan = []

    src_dp_size = from_topo.get_dim("data")
    src_pp_size = from_topo.get_dim("pipe")
    dst_pp_size = to_topo.get_dim("pipe")

    # derive a global NCCL communication plan
    for (pp_i, pp_j), layer_indices in repart_strat.items():
        if len(layer_indices) == 0:
            continue

        for tp_i in range(src_tp_size):
            if dst_tp_size > src_tp_size:
                factor = dst_tp_size // src_tp_size
                tp_js = [i + factor * tp_i for i in range(factor)]
                receiver_tp_portion_id = 0
            else:
                factor = src_tp_size // dst_tp_size
                tp_js = [tp_i // factor]
                receiver_tp_portion_id = tp_i % factor
            for sender_tp_portion_id, tp_j in enumerate(tp_js):

                for dp_i in range(src_dp_size):
                    key = ParamReallocPair(
                        src=from_model_name,
                        src_dp_rank=dp_i,
                        src_tp_rank=tp_i,
                        src_pp_rank=pp_i,
                        dst=to_model_name,
                        dst_tp_rank=tp_j,
                        dst_pp_rank=pp_j,
                    )
                    src = pg_info.param_realloc_src_ranks[key]
                    group = pg_info.param_realloc_groups[key]
                    dst_ranks = pg_info.param_realloc_dst_ranks[key]

                    param_keys = None
                    param_intervals_cpu = receiver_param_intervals_cpu = None
                    param_intervals_cuda = receiver_param_intervals_cuda = None
                    max_interval_size = max_receiver_interval_size = None
                    param_keys = keys_from_layer_indices(
                        from_model_config, layer_indices
                    )
                    param_size = param_size_from_keys(
                        config=from_model_config,
                        src_tp_size=src_tp_size,
                        sd_keys=param_keys,
                        src2dst_tp_size=max(dst_tp_size // src_tp_size, 1),
                        src2dst_tp_rank=sender_tp_portion_id,
                        head_param_point_to_embedding=from_model_head_param_point_to_embedding,
                    )
                    if torch.distributed.is_initialized():
                        # torch.distributed is not initialized when estimating param realloc cost
                        if torch.distributed.get_rank() == src:
                            param_intervals_cpu = param_intervals_from_keys(
                                model_name=from_model_name,
                                config=from_model_config,
                                tp_size=src_tp_size,
                                param_spec=from_model_param_specs,
                                sd_keys=param_keys,
                                portion_size=max(dst_tp_size // src_tp_size, 1),
                                portion_rank=sender_tp_portion_id,
                                head_param_point_to_embedding=from_model_head_param_point_to_embedding,
                            )
                            param_intervals_cuda = torch.tensor(
                                param_intervals_cpu,
                                dtype=torch.long,
                                device=constants.current_device(),
                            )
                            max_interval_size = max(
                                y - x for x, y in param_intervals_cpu
                            )
                        if torch.distributed.get_rank() in dst_ranks:
                            receiver_param_intervals_cpu = param_intervals_from_keys(
                                model_name=to_model_name,
                                config=to_model_config,
                                tp_size=dst_tp_size,
                                param_spec=to_model_param_specs,
                                sd_keys=param_keys,
                                portion_size=max(src_tp_size // dst_tp_size, 1),
                                portion_rank=receiver_tp_portion_id,
                                head_param_point_to_embedding=to_model_head_param_point_to_embedding,
                            )
                            receiver_param_intervals_cuda = torch.tensor(
                                receiver_param_intervals_cpu,
                                dtype=torch.long,
                                device=constants.current_device(),
                            )
                            max_receiver_interval_size = max(
                                y - x for x, y in receiver_param_intervals_cpu
                            )

                    for dst_rank in dst_ranks:
                        comm_plan.append(
                            ReparallelizeReceiverStep(
                                rank=dst_rank,
                                sender_tp_portion_id=sender_tp_portion_id,
                                receiver_tp_portion_id=receiver_tp_portion_id,
                                param_keys=param_keys,
                                sender_param_intervals_cpu=param_intervals_cpu,
                                sender_param_intervals_cuda=param_intervals_cuda,
                                sender_max_interval_size=max_interval_size,
                                receiver_param_intervals_cpu=receiver_param_intervals_cpu,
                                receiver_param_intervals_cuda=receiver_param_intervals_cuda,
                                receiver_max_interval_size=max_receiver_interval_size,
                                param_size=param_size,
                                param_dtype=dtype,
                                src=src,
                                dst_ranks=dst_ranks,
                                group=group,
                            )
                        )
                    comm_plan.append(
                        ReparallelizeSenderStep(
                            rank=src,
                            sender_tp_portion_id=sender_tp_portion_id,
                            receiver_tp_portion_id=receiver_tp_portion_id,
                            param_keys=param_keys,
                            param_intervals_cpu=param_intervals_cpu,
                            param_intervals_cuda=param_intervals_cuda,
                            max_interval_size=max_interval_size,
                            param_size=param_size,
                            group=group,
                            dst_ranks=dst_ranks,
                        )
                    )
    for i, step in enumerate(comm_plan):
        if isinstance(step, ReparallelizeReceiverStep):
            continue
        step: ReparallelizeSenderStep
        required_by_nex_steps = False
        for nex_step in comm_plan[i + 1 :]:
            if (
                isinstance(nex_step, ReparallelizeSenderStep)
                and nex_step.rank == step.rank
                and nex_step.param_keys == step.param_keys
            ):
                required_by_nex_steps = True
                break
        step.remove = not required_by_nex_steps

    return comm_plan


@dataclasses.dataclass
class ReparallelizeTraget:
    comm_plan: List[Union[ReparallelizeSenderStep, ReparallelizeReceiverStep]]
    to_param_spec: Dict[str, ContiguousParamSpec]
    to_param_size: int
    to_layers_handle: nn.ModuleList
    to_layer_start_idx: int
    to_layer_end_idx: int
