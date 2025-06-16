# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import os
import re
import subprocess
from typing import List

import numpy as np


def parse_node_id(node_name: str, prefix: str) -> int:
    return int(node_name.split(prefix)[-1])


def parse_nodelist(cluster_config, nodelist: str, prefix: str) -> List[str]:
    if not nodelist.startswith(prefix):
        raise ValueError(
            f"Node list `{nodelist}` does not start with hostname prefix `{prefix}`."
        )
    n = len(str(cluster_config.n_nodes))
    nodelist = nodelist.replace(prefix, "")
    if "[" not in nodelist:
        return [prefix + nodelist]
    else:
        nodelist = nodelist.strip("[]")
        node_ids = []
        nodelist = nodelist.split(",")
        for node_repr in nodelist:
            if "-" not in node_repr:
                node_ids.append(int(node_repr))
            else:
                start, end = map(int, node_repr.split("-"))
                node_ids += list(range(start, end + 1))
        return [f"{prefix}{node_id:0{n}d}" for node_id in node_ids]


def are_ones_contiguous(binary_array: np.ndarray):
    one_indices = np.where(binary_array == 1)[0]
    if len(one_indices) == 0:
        return False
    return np.all(np.diff(one_indices) == 1)


def slurm_hostname_key(hostname):
    """Custom sorting key function to sort Slurm hostnames."""
    # Extract node number from hostname
    match = re.match(r"(\D+)(\d+)", hostname)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    else:
        return (hostname,)


def check_slurm_availability():

    slurm_available = (
        int(
            subprocess.run(
                "squeue",
                shell=True,
                stdout=open(os.devnull, "wb"),
                stderr=open(os.devnull, "wb"),
            ).returncode
        )
        == 0
    )
    return slurm_available
