# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import itertools
import os
import platform
import socket
import time
from collections import defaultdict
from typing import *

import realhf.base.logging as logging
import realhf.base.name_resolve as name_resolve
import realhf.base.names as names
import realhf.base.network as network

logger = logging.getLogger("System-GPU", "system")

GPU_DEVICES_ISOLATED = False
GLOBAL_PROCESS_GROUP_NAME = "master"


def gpu_count():
    """Returns the number of gpus on a node.

    Ad-hoc to frl cluster.
    """
    if platform.system() == "Darwin":
        return 0
    elif platform.system() == "Windows":
        try:
            import torch

            return torch.cuda.device_count()
        except ImportError:
            return 0
    else:
        dev_directories = list(os.listdir("/dev/"))
        for cnt in itertools.count():
            if "nvidia" + str(cnt) in dev_directories:
                continue
            else:
                break
        return cnt


def set_cuda_device(device):
    """Set the default cuda-device.

    Useful on multi-gpu nodes. Should be called in every gpu-thread.
    """
    # logger.info(f"Setting device to {device}.")
    if device != "cpu":
        import torch

        torch.cuda.set_device(device)


def reveal_pg_identity(expr_name, trial_name, worker_index):
    master_group_name = names.distributed_peer(
        expr_name, trial_name, GLOBAL_PROCESS_GROUP_NAME
    )
    name_resolve.add_subentry(master_group_name, str(worker_index))


def isolate_cuda_device(
    worker_type: str,
    rank: int,
    world_size: int,
    experiment_name: str,
    trial_name: str,
):
    """Isolate CUDA_VISIBLE_DEVICES for each Slurm jobstep.

    To distinguish the concept of job/jobstep/worker/task, check scheduler/slurm/utils.py.
    A slurm job with multiple jobsteps will not set CUDA_VISIBLE_DEVICES properly.
    For example, if a job has 2 jobsteps, each with 1 GPU, and is allocated onto GPU 0 and 1,
    then CUDA_VISIBLE_DEVICES of these jobsteps will be 0,1, instead of 0 and 1.
    We use this function in `apps.remote` to isolate CUDA_VISIBLE_DEVICES for each jobstep.

    Args:
        worker_type (str): .
        rank (int): Rank of the **jobstep**.
        world_size (int): Size of the **jobsteps**, aka SLURM_NPROCS. However, we may call this function
            in other schedulers (e.g. local scheduler), so we don't use SLURM_NPROCS directly.
        experiment_name (str): .
        trial_name (str): .
    """
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        return

    name_resolve_identifier = f"__type_{worker_type}"
    name_resolve.add_subentry(
        names.distributed_local_peer(
            experiment_name,
            trial_name,
            socket.gethostname(),
            name_resolve_identifier,
        ),
        rank,
    )
    name_resolve.add_subentry(
        names.distributed_peer(experiment_name, trial_name, name_resolve_identifier),
        rank,
    )
    logger.debug(
        f"Worker type {worker_type} rank {rank} waiting for peers, world size {world_size}..."
    )
    while (
        len(
            name_resolve.get_subtree(
                names.distributed_peer(
                    experiment_name, trial_name, name_resolve_identifier
                )
            )
        )
        < world_size
    ):
        time.sleep(0.1)
    # logger.info(f"Rank {rank} discovers all peers, resolving local rank...")
    local_peer_name = names.distributed_local_peer(
        experiment_name,
        trial_name,
        socket.gethostname(),
        name_resolve_identifier,
    )
    local_peers = list(
        [
            str(x)
            for x in sorted([int(x) for x in name_resolve.get_subtree(local_peer_name)])
        ]
    )
    # logger.info(f"Rank {rank} discovers local peers with global ranks {local_peers}")

    local_peer_index = local_peers.index(str(rank))
    n_local_peers = len(local_peers)
    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    n_visible_devices = len(visible_devices)
    if n_visible_devices == 0:
        raise RuntimeError(
            f"No visible cuda devices: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
    if n_visible_devices == n_local_peers:
        local_gpu_id = visible_devices[local_peer_index]
    elif n_visible_devices == 1:
        local_gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
    elif n_visible_devices % n_local_peers == 0:
        # A process occupies multiple GPUs, e.g., TP generation server
        factor = n_visible_devices // n_local_peers
        local_gpu_id = visible_devices[factor * local_peer_index]
    else:
        if not os.environ.get("REAL_MODE") == "LOCAL":
            raise RuntimeError(
                f"Unresolvable CUDA_VISIBLE_DEVICES {os.environ['CUDA_VISIBLE_DEVICES']} on host {network.gethostname()}, "
                f"local peers (global ranks) {local_peers}, local peer index {local_peer_index}."
            )
        # In the local mode, all processes use GPUs in a round-robin manner
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        local_gpu_id = int(devices[local_peer_index % len(devices)])

    # logger.info(
    #     f"Worker type {worker_type} rank {rank} running on host {socket.gethostname()}, "
    #     f"local peer index: {local_peer_index}, local gpu id {local_gpu_id}."
    # )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)
    os.environ["GPU_DEVICES_ISOLATED"] = "1"
