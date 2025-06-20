# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import json
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from realhf.api.cli_args import ClusterSpecConfig


def load_spec_from_file(config: "ClusterSpecConfig"):
    with open(config.config_path, "r") as f:
        spec: Dict = json.load(f)

    config.cluster_name = spec["cluster_name"]
    config.fileroot = spec["fileroot"]
    config.gpu_type = spec.get("gpu_type", None)
    config.mount = spec.get("default_mount", None)
    config.gpu_image = spec.get("gpu_image", None)
    config.gpu_infer_image = spec.get("gpu_infer_image", config.gpu_image)
    config.cpu_image = spec.get("cpu_image", None)
    config.node_name_prefix = spec.get("node_name_prefix", "slurmd-")
    config.n_nodes = int(spec.get("n_nodes", 32))
    config.n_gpus_per_node = int(spec.get("n_gpus_per_node", 8))
