# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import json
import os
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from realhf.api.cli_args import BaseExperimentConfig


class ClusterSpec:
    def __init__(self):
        # Set default values to comfort ray
        from realhf.api.cli_args import BaseExperimentConfig

        self.load_spec_from_args(BaseExperimentConfig())

        self.__loaded = False

    def load_spec_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cluster spec file not found: {file_path}")

        with open(file_path, "r") as f:
            spec: Dict = json.load(f)

        self.__cluster_type = spec["cluster_type"]
        self.__cluster_name = spec["cluster_name"]
        self.__fileroot = spec["fileroot"]
        self.__gpu_type = spec.get("gpu_type", None)
        self.__mount = spec.get("default_mount", None)
        self.__gpu_image = spec.get("gpu_image", None)
        self.__gpu_infer_image = spec.get("gpu_infer_image", self.__gpu_image)
        self.__cpu_image = spec.get("cpu_image", None)
        self.__node_name_prefix = spec.get("node_name_prefix", "slurmd-")
        # self.__n_nodes decides number of digits in slurm hostnames
        # e.g. if __n_nodes = 32, then the hostnames will be slurmd-{:02d}
        #      if __n_nodes = 128, then the hostnames will be slurmd-{:03d}
        self.__n_nodes = int(spec.get("n_nodes", 32))
        self.__n_gpus_per_node = int(spec.get("n_gpus_per_node", 8))
        assert isinstance(self.__n_nodes, int)

        self.__loaded = True

    def load_spec_from_args(self, args: "BaseExperimentConfig"):
        self.__cluster_type = args.mode
        self.__cluster_name = args.cluster.cluster_name
        self.__fileroot = args.cluster.fileroot
        self.__gpu_type = args.cluster.gpu_type
        self.__mount = args.cluster.mount
        self.__gpu_image = args.cluster.gpu_image
        self.__gpu_infer_image = args.cluster.gpu_infer_image
        self.__cpu_image = args.cluster.cpu_image
        self.__node_name_prefix = args.cluster.node_name_prefix
        self.__n_nodes = args.cluster.n_nodes
        self.__n_gpus_per_node = args.cluster.n_gpus_per_node
        self.__loaded = True

    @property
    def name(self):
        assert self.__loaded
        return self.__cluster_name

    @property
    def gpu_type(self):
        assert self.__loaded
        return self.__gpu_type

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
    def mount(self) -> str:
        """Directories that should be mounted to container that runs
        workers."""
        assert self.__loaded
        return self.__mount

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


spec = ClusterSpec()


def init_cluster_spec(args: "BaseExperimentConfig"):
    global spec
    CLUSTER_SPEC_PATH = os.environ.get("CLUSTER_SPEC_PATH", "")
    if args.cluster.config_path:
        spec.load_spec_from_file(args.cluster.config_path)
    elif CLUSTER_SPEC_PATH:
        spec.load_spec_from_file(CLUSTER_SPEC_PATH)
    else:
        spec.load_spec_from_args(args)
