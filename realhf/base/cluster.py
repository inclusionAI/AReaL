# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import getpass
import json
import os
import re
import subprocess
from typing import Dict, List, Optional, Union

CLUSTER_SPEC_PATH = os.environ.get("CLUSTER_SPEC_PATH", "")


def get_user_tmp():
    user = getpass.getuser()
    user_tmp = os.path.join("/home", user, ".cache", "realhf")
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp


class ClusterSpec:
    def __init__(self):
        self.__loaded = False

    def load_spec_from_file(self, file_path: str):
        try:
            with open(file_path, "r") as f:
                spec: Dict = json.load(f)
        except FileNotFoundError:
            if file_path == "":
                spec = dict(
                    cluster_type="local",
                    cluster_name="local",
                    fileroot=get_user_tmp(),
                )
            else:
                raise FileNotFoundError(f"Cluster spec file not found: {file_path}")

        self.__cluster_type = spec["cluster_type"]
        self.__cluster_name = spec["cluster_name"]
        self.__fileroot = spec["fileroot"]
        self.__node_type_from_node_name_re = spec.get("node_type_from_node_name", None)
        self.__gpu_type_from_node_name_re = spec.get("gpu_type_from_node_name", None)
        self.__gpu_type = spec.get("gpu_type", None)
        self.__default_mount = spec.get("default_mount", None)
        self.__gpu_image = spec.get("gpu_image", None)
        self.__gpu_infer_image = spec.get("gpu_infer_image", self.__gpu_image)
        self.__cpu_image = spec.get("cpu_image", None)
        self.__node_name_prefix = spec.get("node_name_prefix", "NODE")
        # self.__n_nodes decides number of digits in slurm hostnames
        # e.g. if __n_nodes = 32, then the hostnames will be NODE{:02d}
        #      if __n_nodes = 128, then the hostnames will be NODE{:03d}
        self.__n_nodes = int(spec.get("n_nodes", 32))
        self.__n_gpus_per_node = int(spec.get("n_gpus_per_node", 8))
        assert isinstance(self.__n_nodes, int)

        self.__loaded = True

    @property
    def name(self):
        assert self.__loaded
        return self.__cluster_name

    @property
    def gpu_type(self):
        assert self.__loaded
        return self.__gpu_type

    def node_type_from_node_name(self, node_name: str) -> str:
        """Mapping nodename to slurm node type, including "g1", "g2", "g8",
        "a100"."""
        if self.__cluster_type != "slurm":
            raise NotImplementedError(
                "Only slurm cluster uses node_type_from_node_name."
            )
        assert self.__loaded
        for regex, node_type in self.__node_type_from_node_name_re.items():
            if re.match(regex, node_name):
                return node_type
        raise NotImplementedError(node_name)

    def gpu_type_from_node_name(self, node_name: str) -> str:
        """Mapping nodename to slurm GPU type, including "geforce" and
        "tesla"."""
        if self.__cluster_type != "slurm":
            raise NotImplementedError(
                "Only slurm cluster uses gpu_type_from_node_name."
            )
        assert self.__loaded
        for regex, gpu_type in self.__gpu_type_from_node_name_re.items():
            if re.match(regex, node_name):
                return gpu_type
        raise NotImplementedError(node_name)

    @property
    def fileroot(self) -> str:
        """Return the root directory of the file system in the cluster.

        When running experiments, files such as logs, checkpoints,
        caches will be saved under this directory.
        """
        assert self.__loaded
        return self.__fileroot

    @fileroot.setter
    def fileroot(self, root: str):
        # Used for testing
        self.__fileroot = root

    @property
    def default_mount(self) -> str:
        """Directories that should be mounted to container that runs
        workers."""
        assert self.__loaded
        return self.__default_mount

    @property
    def gpu_image(self) -> str:
        """Return the default image for containers of GPU trainer workers."""
        assert self.__loaded
        return self.__gpu_image

    @property
    def gpu_infer_image(self) -> str:
        """Return the default image for containers of GPU inference workers."""
        assert self.__loaded
        return self.__gpu_infer_image

    @property
    def cpu_image(self) -> str:
        """Return the default image for containers of CPU workers."""
        assert self.__loaded
        return self.__cpu_image

    @property
    def node_name_prefix(self) -> str:
        """Return the prefix of node names in slurm format."""
        assert self.__loaded
        return self.__node_name_prefix

    @property
    def n_nodes(self) -> int:
        return self.__n_nodes

    @property
    def suffix_n_digits(self) -> int:
        return len(str(self.__n_nodes))

    @property
    def n_gpus_per_node(self) -> int:
        return self.__n_gpus_per_node

    @property
    def cluster_type(self) -> str:
        return self.__cluster_type


def node_name_is_node_type(
    node_name: str, node_type: Optional[Union[List[str], str]] = None
) -> bool:
    assert spec is not None
    if node_type is None:
        return True
    if not isinstance(node_type, list):
        node_type = [node_type]
    nt_condition = []
    for nt in node_type:
        if nt not in ["g1", "g2", "g8", "a100"]:
            raise ValueError(f"Unknown node type {nt}.")
        else:
            cond = spec.node_type_from_node_name(node_name) == nt
        nt_condition.append(cond)
    return any(nt_condition)


spec = ClusterSpec()
spec.load_spec_from_file(CLUSTER_SPEC_PATH)
