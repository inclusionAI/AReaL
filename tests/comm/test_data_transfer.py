# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import os
import pathlib
import pickle
import uuid
from typing import *

import numpy as np
import pytest
import torch
import torch.distributed as dist

from realhf.api.cli_args import ClusterSpecConfig
from realhf.api.core.config import ModelName, ModelShardID
from realhf.api.core.data_api import SequenceSample
from realhf.base import constants, testing, topology
from realhf.base.testing import LocalMultiProcessTest, init_global_constants
from realhf.system.data_manager import DataManager
from realhf.system.redistributor import GlobalStorageTracker, RedistribPlanner


def get_data_manager(
    from_model_name,
    to_model_name,
    from_pp_dp_tp,
    to_pp_dp_tp,
):

    from_num_pp, from_num_dp, from_num_tp = from_pp_dp_tp
    to_num_pp, to_num_dp, to_num_tp = to_pp_dp_tp

    from_world_size = from_num_dp * from_num_tp * from_num_pp
    to_world_size = to_num_dp * to_num_tp * to_num_pp

    from_topo = topology.PipeDataTensorParallelTopology(
        num_dp=from_num_dp,
        num_tp=from_num_tp,
        num_pp=from_num_pp,
        sequence_parallel=False,
        gradient_checkpointing=False,
        max_prompt_len=None,
        gradient_accumulation_fusion=False,
    )
    to_topo = topology.PipeDataTensorParallelTopology(
        num_dp=to_num_dp,
        num_tp=to_num_tp,
        num_pp=to_num_pp,
        sequence_parallel=False,
        gradient_checkpointing=False,
        max_prompt_len=None,
        gradient_accumulation_fusion=False,
    )

    model_topos = {from_model_name: from_topo, to_model_name: to_topo}

    msid2mwid = {}
    for i in range(dist.get_world_size()):
        # We assume the `from_model` occupies the first serveral GPUs,
        # while the `to_model` occupies GPUs from the last one.
        # For example, when the world size of `from_model` is 6 and
        # the world size of `to_model` is 4, the GPU layout is:
        # GPU 0-3: from_model (shard 0-3)
        # GPU 4-5: from_model (shard 4-5) + to_model (shard 0-1)
        # GPU 6-7: to_model (shard 2-3)
        _model_names = []
        if i < from_world_size:
            _model_names.append(from_model_name)
        if i >= dist.get_world_size() - to_world_size:
            _model_names.append(to_model_name)
        for _model_name in _model_names:
            if _model_name == from_model_name:
                coord = model_topos[_model_name].get_coord(i)
            else:
                coord = model_topos[_model_name].get_coord(
                    i + to_world_size - dist.get_world_size()
                )
            k = ModelShardID(
                _model_name,
                dp_rank=coord.data,
                tp_rank=coord.tensor,
                pp_rank=coord.pipe,
                topo=model_topos[_model_name],
            )
            msid2mwid[k] = i

    init_global_constants(
        num_dp=from_num_dp,
        num_tp=from_num_tp,
        num_pp=from_num_pp,
        topo=from_topo,
        model_name=from_model_name,
        sequence_parallel=False,
        msid2mwid=msid2mwid,
    )

    init_global_constants(
        num_dp=to_num_dp,
        num_tp=to_num_tp,
        num_pp=to_num_pp,
        model_name=to_model_name,
        sequence_parallel=False,
        msid2mwid=msid2mwid,
    )

    return DataManager(
        model_topos=model_topos,
        msid2mwid=msid2mwid,
        data_transfer_pairs=[(from_model_name, to_model_name)],
    )


def recursive_assert_equal(x1, x2):
    if type(x1) != type(x2):
        raise AssertionError(f"{type(x1)} != {type(x2)}")
    if isinstance(x1, dict):
        assert set(x1.keys()) == set(x2.keys())
        for k in x1.keys():
            recursive_assert_equal(x1[k], x2[k])
    elif dataclasses.is_dataclass(x1):
        for f in dataclasses.fields(x1):
            recursive_assert_equal(getattr(x1, f.name), getattr(x2, f.name))
    elif isinstance(x1, torch.Tensor):
        assert torch.allclose(x1, x2), (x1, x2)
    elif isinstance(x1, list):
        assert len(x1) == len(x2)
        for a, b in zip(x1, x2):
            recursive_assert_equal(a, b)
    else:
        assert x1 == x2


def _test_data_transfer(
    tmp_path,
    from_pp_dp_tp: Tuple,
    to_pp_dp_tp: Tuple,
):

    from_model_name = ModelName("data_transfer_test", 0)
    from_topo = topology.PipeDataTensorParallelTopology(
        num_pp=from_pp_dp_tp[0],
        num_tp=from_pp_dp_tp[-1],
        num_dp=from_pp_dp_tp[1],
        sequence_parallel=True,
        gradient_checkpointing=True,
        gradient_accumulation_fusion=True,
    )
    to_model_name = ModelName("data_transfer_test", 1)
    to_topo = topology.PipeDataTensorParallelTopology(
        num_pp=to_pp_dp_tp[0],
        num_tp=to_pp_dp_tp[-1],
        num_dp=to_pp_dp_tp[1],
        sequence_parallel=True,
        gradient_checkpointing=True,
        gradient_accumulation_fusion=True,
    )

    data_manager = get_data_manager(
        from_model_name,
        to_model_name,
        from_pp_dp_tp,
        to_pp_dp_tp,
    )
    data_manager.setup_process_groups()

    storage_tracker = GlobalStorageTracker(dist.get_world_size())
    planner = RedistribPlanner(ClusterSpecConfig(), storage_tracker)

    key = "input_ids"

    world_size = dist.get_world_size()
    samples = []
    for dp_rank in range(from_pp_dp_tp[1]):
        gpu_id = data_manager.msid2mwid[
            ModelShardID(
                from_model_name,
                dp_rank=dp_rank,
                tp_rank=0,
                pp_rank=from_pp_dp_tp[0] - 1,
                topo=from_topo,
            )
        ]
        storage_tracker.add_data_synced(
            gpu_id,
            ids=[str(i + dp_rank * world_size) for i in range(world_size)],
            key=key,
            is_owner=True,
        )

        seqlens = torch.randint(10, 1000, size=(world_size,))
        dist.all_reduce(seqlens)
        input_ids = torch.randint(
            0,
            10000,
            size=(int(sum(seqlens)),),
        )
        dist.all_reduce(input_ids)

        s = SequenceSample.from_default(
            ids=[str(i + dp_rank * world_size) for i in range(world_size)],
            seqlens=seqlens.numpy().tolist(),
            data=dict(input_ids=input_ids),
        )

        if dist.get_rank() == 0:
            for ss in s.unpack():
                with open(os.path.join(tmp_path, f"{ss.ids[0]}.pkl"), "wb") as f:
                    pickle.dump(ss, f)

        samples.append(s)
        if dist.get_rank() == gpu_id:
            for ss in s.unpack():
                data_manager.store(ss)

    dist.barrier()

    all_ids = list(map(str, range(world_size * from_topo.get_dim("data"))))
    np.random.shuffle(all_ids)
    _all_ids = [all_ids]
    dist.broadcast_object_list(_all_ids, src=0)
    all_ids = _all_ids[0]

    dests = {}
    for rank in range(to_topo.world_size()):
        coord = to_topo.get_coord(rank)
        dp_size = to_topo.get_dim("data")
        gpu_id = data_manager.msid2mwid[
            ModelShardID(
                to_model_name,
                dp_rank=coord.data,
                tp_rank=coord.tensor,
                pp_rank=coord.pipe,
                topo=to_topo,
            )
        ]
        size_per_dp = len(all_ids) // dp_size
        dests[gpu_id] = [str(coord.data * size_per_dp + i) for i in range(size_per_dp)]

    for gpu_id in range(world_size):
        if gpu_id not in dests:
            dests[gpu_id] = []

    plan = planner.derive_plan(dests, keys=[key])
    data_manager.redistribute(SequenceSample.gather(samples), plan)
    dist.barrier()

    for i, s in data_manager.storage.items():
        with open(os.path.join(tmp_path, f"{i}.pkl"), "rb") as f:
            ss = pickle.load(f)
        recursive_assert_equal(ss, s)
    print("success")


parallelism = [(4, 1, 1), (2, 2, 2), (1, 8, 1), (3, 2, 1), (2, 1, 2), (1, 2, 2)]


@pytest.mark.parametrize("from_pp_dp_tp", parallelism)
@pytest.mark.parametrize("to_pp_dp_tp", parallelism)
@pytest.mark.distributed
def test_data_transfer(
    tmp_path,
    from_pp_dp_tp: Tuple,
    to_pp_dp_tp: Tuple,
):
    constants.set_force_cpu(True)
    test_impl = LocalMultiProcessTest(
        world_size=16,
        func=_test_data_transfer,
        timeout_secs=300,
        tmp_path=tmp_path,
        from_pp_dp_tp=from_pp_dp_tp,
        to_pp_dp_tp=to_pp_dp_tp,
    )
    test_impl.launch()
