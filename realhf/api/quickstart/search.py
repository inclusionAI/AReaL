# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
from typing import List, Optional

from realhf.api.cli_args import ParallelismConfig
from realhf.api.core.dfg import MFCDef
from realhf.api.quickstart.device_mesh import DeviceMesh


@dataclasses.dataclass
class RPCExecution:
    rpc: MFCDef
    device_mesh: DeviceMesh
    parallel_strategy: ParallelismConfig
    time_cost: Optional[int] = None
    mem: Optional[int] = None
    static_mem: Optional[int] = None

    def __repr__(self):
        return f"RPCExecution({self.rpc}, {self.device_mesh}, {self.parallel_strategy})"

    def __hash__(self):
        return hash(
            (
                self.rpc.name,
                self.device_mesh.cluster_mesh,
                self.device_mesh.device_mesh_name,
                str(self.parallel_strategy),
            )
        )


@dataclasses.dataclass
class RPCInstance:
    rpc: MFCDef
    iteration_id: int
    parents: List[MFCDef]
    children: List[MFCDef]

    @property
    def name(self):
        return f"{self.rpc.name}:{self.iteration_id}"

    def __repr__(self):
        if len(self.parents) == 0 and len(self.children) == 0:
            return f"RPCInstance({self.rpc.name}, {self.iteration_id})"
        else:
            return (
                f"RPCInstance({self.rpc.name}, {self.iteration_id}, "
                f"{self.parents}, {self.children})"
            )

    def __hash__(self):
        return hash((self.rpc.name, self.iteration_id))
