# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import itertools
import os
import socket
from collections import defaultdict
from typing import *

import torch.distributed

from realhf.api.core.config import ModelName, ModelShardID
from realhf.base import constants, gpu_utils, name_resolve, names, network, topology


@dataclasses.dataclass
class NCCLProcessGroupInfo:
    world_size: int
    global_rank: int
    local_gpu_id: int
    # 3D parallelism groups of each model.
    model_groups: Dict[str, torch.distributed.ProcessGroup]


def filter_match_mwids(
    model_name: ModelName,
    topo: topology.ProcessTopology,
    msid2mwid: Dict[ModelShardID, int],
    **conditions,
) -> List[int]:
    if len(conditions) == 0:
        mwids_this_model = [
            msid2mwid[ModelShardID.from_parallelism_rank(model_name, topo, j)]
            for j in range(topo.world_size())
        ]
    else:
        mwids_this_model = [
            msid2mwid[ModelShardID.from_parallelism_rank(model_name, topo, j)]
            for j in topo.filter_match(**conditions)
        ]
    mwids_this_model = sorted(mwids_this_model)
    assert len(set(mwids_this_model)) == len(mwids_this_model)
    return list(mwids_this_model)


def setup_global_comm(
    args,
    expr_name: str,
    trial_name: str,
    worker_index: int,
    model_topos: Optional[Dict[str, topology.ProcessTopology]] = None,
    msid2mwid: Optional[Dict[ModelShardID, int]] = None,
    backend: str = "nccl",
) -> NCCLProcessGroupInfo:
    peers: List[int] = list(
        sorted(
            map(
                int,
                name_resolve.get_subtree(
                    names.distributed_peer(
                        expr_name,
                        trial_name,
                        gpu_utils.GLOBAL_PROCESS_GROUP_NAME,
                    )
                ),
            )
        )
    )
    assert len(peers) == len(set(peers)), f"Duplicated trainer worker index. {peers}"
    world_size = len(peers)
    global_rank = peers.index(worker_index)

    mw_ranks = {}
    if model_topos is not None:
        assert msid2mwid is not None
        for model_name, topo in model_topos.items():
            mw_ranks[model_name] = filter_match_mwids(model_name, topo, msid2mwid)

    if (
        "GPU_DEVICES_ISOLATED" not in os.environ
        and "RAY" not in os.environ["REAL_MODE"]
        and constants.use_cuda()
    ):
        raise RuntimeError(
            "GPU devices not isolated in slurm or local mode. This should not happen."
        )

    if constants.use_cuda():
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
            local_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            local_gpu_id = int(
                os.environ["CUDA_VISIBLE_DEVICES"].split(",")[worker_index]
            )
    else:
        local_gpu_id = global_rank

    pg_master_name = names.distributed_master(
        expr_name, trial_name, gpu_utils.GLOBAL_PROCESS_GROUP_NAME
    )

    if worker_index == 0:
        host_ip = socket.gethostbyname(socket.gethostname())
        port = network.find_free_port(
            experiment_name=expr_name,
            trial_name=trial_name,
            lockfile_root=os.path.join(constants.get_cache_path(args), "ports"),
        )
        pg_init_addr = f"tcp://{host_ip}:{port}"
        name_resolve.add(pg_master_name, pg_init_addr, keepalive_ttl=300)
    else:
        try:
            pg_init_addr = name_resolve.wait(pg_master_name, timeout=300)
        except TimeoutError:
            raise TimeoutError(
                f"global_rank={global_rank} worker_index={worker_index} wait for process group init timeout."
            )

    if not constants.use_cuda():
        backend = "gloo"
    torch_dist_kwargs = dict(
        world_size=world_size,
        rank=global_rank,
        init_method=pg_init_addr,
        backend=backend,
        timeout=constants.NCCL_DEFAULT_TIMEOUT,
    )
    if constants.use_cuda():
        torch.cuda.set_device(
            0
        )  # initialize CUDA here with only a single visible device
    # This environment variable is used by DeepSpeed.
    os.environ["LOCAL_RANK"] = "0"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            **torch_dist_kwargs, group_name=gpu_utils.GLOBAL_PROCESS_GROUP_NAME
        )

    model_groups = {}
    for model_name, ranks in mw_ranks.items():
        model_groups[model_name] = topology.new_or_get_group(ranks, backend=backend)
        constants.set_parallelism_group(model_name, model_groups[model_name], ranks)
        cpu_group = topology.new_or_get_group(ranks, backend="gloo")
        constants.set_cpu_parallelism_group(model_name, cpu_group)

    self_group = None
    for i in range(world_size):
        group = topology.new_or_get_group([i], backend=backend)
        if i == global_rank:
            self_group = group
            constants.set_self_group(self_group)

    # logger.info(f"Setup process group finishes for worker_index={worker_index}")

    return NCCLProcessGroupInfo(
        world_size=world_size,
        global_rank=global_rank,
        local_gpu_id=local_gpu_id,
        model_groups=model_groups,
    )
