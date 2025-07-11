# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import math
from typing import List, Optional, Tuple, Union

import numpy as np

from realhf.api.cli_args import ClusterSpecConfig, ParallelismConfig
from realhf.api.core.dfg import MFCDef
from realhf.base.slurm_utils import are_ones_contiguous, parse_nodelist


@dataclasses.dataclass
class DeviceMesh:
    # number of total nodes, n_gpus_per_node=8
    n_nodes: int
    n_gpus_per_node: int
    # a 2D binary array of current device mesh name
    # shape: (n_nodes, n_gpus_per_node)
    mapping: np.ndarray

    def to_dict(self):
        return dict(
            n_nodes=self.n_nodes,
            n_gpus_per_node=self.n_gpus_per_node,
            mapping=self.mapping.tolist(),
        )

    def split(self, split_n_gpus: int) -> Tuple["DeviceMesh", "DeviceMesh"]:
        """Split the current device into two parts, with the first part having
        `split_n_gpus` GPUs.

        The second part will have the remaining GPUs.
        """
        assert self._is_valid_mapping()
        assert (
            self.mapping.sum() > split_n_gpus
        ), f"split size {split_n_gpus} should be smaller than the number of current GPUs {self.mapping.sum()}"

        sub_mapping2 = self.mapping.copy()
        t = split_n_gpus
        cnt = 0
        while t > 0:
            i, j = cnt // self.n_gpus_per_node, cnt % self.n_gpus_per_node
            if sub_mapping2[i, j] == 0:
                cnt += 1
            else:
                sub_mapping2[i, j] = 0
                t -= 1
        sub_mapping1 = self.mapping - sub_mapping2

        d1, d2 = (
            DeviceMesh(
                n_nodes=self.n_nodes,
                n_gpus_per_node=self.n_gpus_per_node,
                mapping=sub_mapping1,
            ),
            DeviceMesh(
                n_nodes=self.n_nodes,
                n_gpus_per_node=self.n_gpus_per_node,
                mapping=sub_mapping2,
            ),
        )
        assert d1._is_valid_mapping()
        assert d2._is_valid_mapping()
        return (d1, d2)

    @staticmethod
    def from_dict(d):
        device_mesh = DeviceMesh(**d)
        device_mesh.mapping = np.array(d["mapping"])
        return device_mesh

    def __post_init__(self):
        assert self._is_valid_mapping()

    def __eq__(self, other: "DeviceMesh"):
        return np.all(self.mapping == other.mapping)

    def __op_assertion(self, other: "DeviceMesh"):
        assert self.n_nodes == other.n_nodes
        assert self.n_gpus_per_node == other.n_gpus_per_node

    def overlap(self, other: "DeviceMesh") -> bool:
        self.__op_assertion(other)
        return np.any(self.mapping & other.mapping)

    def contain(self, other: "DeviceMesh") -> bool:
        self.__op_assertion(other)
        return np.all(self.mapping & other.mapping == self.mapping)

    def contained_by(self, other: "DeviceMesh") -> bool:
        self.__op_assertion(other)
        return np.all(self.mapping & other.mapping == other.mapping)

    def sub_device_meshes(self, min_n_gpus: int = 4) -> List["DeviceMesh"]:
        """Find sub device meshes of this device mesh with at least min_n_gpus
        gpus.

        Sub device meshes have following constraints:
            1. Sub device meshes have the same cluster mesh.
            2. Sub device meshes of multiple nodes must contain consecutive nodes
               in the cluster mesh.
            3. Sub device meshes can only be of shape 1x1, 1x2, 1x4, 1x8 or Nx8
            4. If sub device meshes are of shape 1x2 or 1x4, the start GPU id
               must be 0, 2, 4, 6 for 1x2 and 0, 4 for 1x4.
        """
        sub_mappings = []
        rows, cols = np.where(self.mapping == 1)

        unique_rows = np.unique(rows)
        for row in unique_rows:
            this_cols = cols[rows == row]
            assert (
                self.n_gpus_per_node % min_n_gpus == 0
                or min_n_gpus % self.n_gpus_per_node == 0
            )
            n_gpus = min_n_gpus
            while n_gpus < min(self.n_gpus_per_node, np.sum(self.mapping)):
                for start in range(np.min(this_cols), self.n_gpus_per_node, n_gpus):
                    sub_mapping = np.zeros(
                        (self.n_nodes, self.n_gpus_per_node), dtype=np.int32
                    )
                    sub_mapping[row, start : start + n_gpus] = 1
                    sub_mappings.append(sub_mapping)
                n_gpus *= 2

        for n_rows in range(1, len(unique_rows) + 1):
            for start in range(0, len(unique_rows) - n_rows + 1):
                sub_mapping = np.zeros(
                    (self.n_nodes, self.n_gpus_per_node), dtype=np.int32
                )
                sub_mapping[start : start + n_rows, :] = 1
                sub_mappings.append(sub_mapping)

        return [
            DeviceMesh(
                n_nodes=self.n_nodes,
                n_gpus_per_node=self.n_gpus_per_node,
                mapping=sub_mapping,
            )
            for sub_mapping in sub_mappings
        ]

    def _is_valid_mapping(self) -> bool:
        if self.mapping.shape != (self.n_nodes, self.n_gpus_per_node):
            raise RuntimeError(
                f"Invalid mapping shape {self.mapping.shape} n_nodes={self.n_nodes} "
                f"n_gpus_per_node={self.n_gpus_per_node}"
            )
        if not np.all(np.logical_or(self.mapping == 0, self.mapping == 1)):
            raise RuntimeError(f"Invalid mapping value {self.mapping}")

        assert math.log(self.n_gpus_per_node, 2).is_integer()

        if self.mapping.sum() >= self.n_gpus_per_node:
            if not (
                self.mapping.sum() % self.n_gpus_per_node == 0
                and np.all(
                    np.logical_or(
                        self.mapping.sum(1) == self.n_gpus_per_node,
                        self.mapping.sum(1) == 0,
                    )
                )
            ):
                raise RuntimeError(
                    f"Invalid mapping sum {self.mapping}. "
                    "If using GPUs more than an entire node, "
                    "only several complete nodes are allowed."
                )
        if not are_ones_contiguous(self.mapping.flatten()):
            raise RuntimeError(f"mapping devices are not contiguous {self.mapping}")
        return True


def make_device_mesh_from_name(
    cluster: ClusterSpecConfig,
    global_mesh_name: str,
    name: str,
    n_gpus_per_node: int = 8,
):
    """
    DeviceMesh name format: <prefix><node_indices>[:<gpu_ids>]
        slurm_nodelist is the name of slurm nodes the mesh is on, should follow slurm convention,
        for example "slurmd-[40-43]" or "slurmd-[01,11,13-14]" with prefix slurmd-,
        if n_nodes=1, gpu_ids are the gpu id list delimited by comma if n_gpus < n_gpus_per_node,
        for example "0,1,2,3" or "0,1". An example of full device mesh name
        in this situation is "slurmd-40:0,1,2,3"

    Note: cluster device mesh name must occupy entire nodes.
    """
    prefix = cluster.node_name_prefix
    node_list = parse_nodelist(cluster, global_mesh_name, prefix)
    n_nodes = len(node_list)

    gpu_ids = None
    if ":" in name:
        node_names, gpu_ids = name.split(":")
        gpu_ids = list(map(int, gpu_ids.split(",")))
        assert all(gpu_id < n_gpus_per_node for gpu_id in gpu_ids)
    else:
        node_names = name
    node_names = parse_nodelist(cluster, node_names, prefix)
    mapping = np.zeros((n_nodes, n_gpus_per_node), dtype=np.int32)
    if gpu_ids is None:
        node_indices = [node_list.index(node_name) for node_name in node_names]
        mapping[node_indices, :] = 1
    else:
        assert len(node_names) == 1
        node_index = node_list.index(node_names[0])
        mapping[node_index, gpu_ids] = 1

    return DeviceMesh(
        n_nodes=n_nodes,
        n_gpus_per_node=n_gpus_per_node,
        mapping=mapping,
    )


def find_parallel_strategies(
    device_mesh: DeviceMesh,
) -> List[ParallelismConfig]:
    n_gpus = np.sum(device_mesh.mapping)
    res = []
    for num_tp in [1, 2, 4, 8]:
        if n_gpus >= num_tp:
            assert n_gpus % num_tp == 0
            num_dp_pp = n_gpus // num_tp
            num_pp = 1
            while num_pp <= num_dp_pp:
                num_dp_tp = n_gpus // num_pp
                valid = (
                    num_dp_tp in [1, 2, 4, 8] or num_dp_tp % 8 == 0
                ) and num_dp_pp % num_pp == 0
                if valid:
                    res.append(ParallelismConfig(num_pp, num_tp, num_dp_pp // num_pp))
                num_pp += 1
    return res


@dataclasses.dataclass
class RPCAllocation:
    rpc: MFCDef
    device_mesh: DeviceMesh
    parallel: ParallelismConfig

    def __post_init__(self):
        world_size = (
            self.parallel.tensor_parallel_size
            * self.parallel.pipeline_parallel_size
            * self.parallel.data_parallel_size
        )
        assert world_size == self.device_mesh.mapping.sum(), (
            "World size of ParallelismConfig does not match number of GPUs in device mesh"
            f"world_size {world_size} != n GPUs {self.device_mesh.mapping.sum()}"
        )

    def to_dict(self):
        return dict(
            rpc=self.rpc.name,
            device_mesh=self.device_mesh.to_dict(),
            parallel=dataclasses.asdict(self.parallel),
        )

    @staticmethod
    def from_dict(d):
        return RPCAllocation(
            rpc=d["rpc"],
            device_mesh=DeviceMesh.from_dict(d["device_mesh"]),
            parallel=ParallelismConfig(**d["parallel"]),
        )
