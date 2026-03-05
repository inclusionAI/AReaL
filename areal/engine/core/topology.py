"""Unified topology management for parallel training engines.

Provides a common interface (TopologyManager protocol) for managing parallel
process groups across FSDP, Megatron, and Archon engines.

Canonical dimension names:
    tp  - tensor parallelism
    dp  - data parallelism
    cp  - context parallelism (FSDP calls this "sp" historically)
    pp  - pipeline parallelism
    ep  - expert parallelism
    etp - expert tensor parallelism
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from areal.api.alloc_mode import ParallelStrategy
from areal.infra.platforms import current_platform

__all__ = [
    "TopologyManager",
    "DeviceMeshTopology",
]


@runtime_checkable
class TopologyManager(Protocol):
    """Protocol for unified parallel topology management.

    Engine-specific implementations wrap their native topology objects
    while providing a common interface for size queries, enabled flags,
    and process group access.
    """

    # Size properties
    @property
    def tp_size(self) -> int: ...

    @property
    def dp_size(self) -> int: ...

    @property
    def pp_size(self) -> int: ...

    @property
    def cp_size(self) -> int: ...

    @property
    def ep_size(self) -> int: ...

    @property
    def etp_size(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    # Enabled flags
    @property
    def tp_enabled(self) -> bool: ...

    @property
    def dp_enabled(self) -> bool: ...

    @property
    def pp_enabled(self) -> bool: ...

    @property
    def cp_enabled(self) -> bool: ...

    @property
    def ep_enabled(self) -> bool: ...

    # Access methods
    def get_group(self, name: str) -> dist.ProcessGroup | None: ...

    def get_mesh(self, name: str) -> DeviceMesh | None: ...

    # Computed properties
    @property
    def gradient_divide_factor(self) -> int: ...

    @property
    def seq_len_divisor(self) -> int: ...

    @property
    def context_and_model_parallel_size(self) -> int: ...


class DeviceMeshTopology:
    """TopologyManager implementation backed by PyTorch DeviceMesh.

    Consolidates mesh-building logic from ParallelHelper (FSDP) and
    ArchonParallelDims (Archon). Supports all 5 parallelism dimensions
    including pipeline parallelism.

    Args:
        strategy: A ParallelStrategy defining the parallel dimensions.
        device_type: Device type for mesh creation (e.g., "cuda", "npu").
            If None, uses the current platform's device type.
    """

    def __init__(
        self,
        strategy: ParallelStrategy,
        device_type: str | None = None,
    ) -> None:
        self._strategy = strategy
        self._device_type = device_type or current_platform.device_type
        self._world_mesh: DeviceMesh | None = None
        self._meshes: dict[str, DeviceMesh] = {}
        self._validate()

    def _validate(self) -> None:
        s = self._strategy
        tp, dp, cp, pp, ep, etp = (
            s.tp_size,
            s.dp_size,
            s.cp_size,
            s.pp_size,
            s.ep_size,
            s.etp_size,
        )

        for name, val in [("tp", tp), ("dp", dp), ("cp", cp), ("pp", pp)]:
            if val < 1:
                raise ValueError(f"{name} size must be >= 1, got {val}")

        expected = dp * cp * tp * pp
        if expected != s.world_size:
            raise ValueError(
                f"dp({dp}) * cp({cp}) * tp({tp}) * pp({pp}) = {expected} "
                f"!= world_size({s.world_size})"
            )

        if ep > 1:
            if etp not in (1, tp):
                raise ValueError(
                    f"etp must be 1 or equal to tp, got etp={etp}, tp={tp}"
                )
            if etp == tp:
                if ep % cp != 0 or (dp * cp) % ep != 0:
                    raise ValueError(
                        f"When etp=tp, ep must divide cp and dp*cp must divide ep. "
                        f"Got dp={dp}, cp={cp}, ep={ep}"
                    )
            else:
                if ep % (cp * tp) != 0 or (dp * cp * tp) % ep != 0:
                    raise ValueError(
                        f"When etp=1, ep must divide cp*tp and dp*cp*tp must divide ep. "
                        f"Got dp={dp}, cp={cp}, tp={tp}, ep={ep}"
                    )

    # =========================================================================
    # Mesh Creation
    # =========================================================================

    def build_mesh(self) -> DeviceMesh:
        if self._strategy.ep_size > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    def _build_mesh_without_ep(self) -> DeviceMesh:
        s = self._strategy
        dims = [s.pp_size, s.dp_size, s.cp_size, s.tp_size]
        names = ["pp", "dp_shard", "cp", "tp"]

        mesh = init_device_mesh(
            self._device_type, tuple(dims), mesh_dim_names=tuple(names)
        )

        # Store base dimension meshes
        self._meshes["pp"] = mesh["pp"]
        self._meshes["dp_shard"] = mesh["dp_shard"]
        self._meshes["cp"] = mesh["cp"]
        self._meshes["tp"] = mesh["tp"]

        # dp mesh: for data loading (same as dp_shard when no HSDP)
        self._meshes["dp"] = mesh["dp_shard"]._flatten(mesh_dim_name="dp")

        # dp_cp mesh: dp_shard * cp (for FSDP sharding and loss all-reduce)
        self._meshes["dp_cp"] = mesh["dp_shard", "cp"]._flatten(mesh_dim_name="dp_cp")

        # cp_tp mesh: cp * tp (context + model parallel)
        self._meshes["cp_tp"] = mesh["cp", "tp"]._flatten(mesh_dim_name="cp_tp")

        # pp_cp_tp mesh: pp * cp * tp (context and model parallel group)
        self._meshes["pp_cp_tp"] = mesh["pp", "cp", "tp"]._flatten(
            mesh_dim_name="pp_cp_tp"
        )

        # FSDP backward-compatible aliases (sp = cp)
        self._meshes["sp"] = self._meshes["cp"]
        self._meshes["dp_sp"] = self._meshes["dp_cp"]
        self._meshes["sp_tp"] = self._meshes["cp_tp"]
        self._meshes["dp_shard_cp"] = self._meshes["dp_cp"]

        self._world_mesh = mesh
        return mesh

    def _build_mesh_with_ep(self) -> DeviceMesh:
        s = self._strategy
        dp, cp, tp, ep, etp = (
            s.dp_size,
            s.cp_size,
            s.tp_size,
            s.ep_size,
            s.etp_size,
        )

        if etp == tp:
            dp_mod_ep = dp * cp // ep
            dp_in_ep = ep // cp
        else:
            dp_mod_ep = dp * cp * tp // ep
            dp_in_ep = ep // (cp * tp)

        dims = [s.pp_size, dp_mod_ep, dp_in_ep, cp, tp]
        names = ["pp", "dp_mod_ep", "dp_in_ep", "cp", "tp"]

        mesh = init_device_mesh(
            self._device_type, tuple(dims), mesh_dim_names=tuple(names)
        )

        # Store base dimension meshes
        self._meshes["pp"] = mesh["pp"]
        self._meshes["dp_mod_ep"] = mesh["dp_mod_ep"]
        self._meshes["dp_in_ep"] = mesh["dp_in_ep"]
        self._meshes["cp"] = mesh["cp"]
        self._meshes["tp"] = mesh["tp"]

        # dp mesh: for data loading
        self._meshes["dp"] = mesh["dp_mod_ep", "dp_in_ep"]._flatten(mesh_dim_name="dp")

        # dp_cp mesh: dp * cp (for FSDP sharding and loss all-reduce)
        self._meshes["dp_cp"] = mesh["dp_mod_ep", "dp_in_ep", "cp"]._flatten(
            mesh_dim_name="dp_cp"
        )

        # cp_tp mesh: cp * tp
        self._meshes["cp_tp"] = mesh["cp", "tp"]._flatten(mesh_dim_name="cp_tp")

        # pp_cp_tp mesh: pp * cp * tp
        self._meshes["pp_cp_tp"] = mesh["pp", "cp", "tp"]._flatten(
            mesh_dim_name="pp_cp_tp"
        )

        # EP mesh: flatten based on ETP mode
        if etp == tp:
            ep_mesh_dims = ["dp_in_ep"]
            if cp > 1:
                ep_mesh_dims.append("cp")

            if len(ep_mesh_dims) > 1:
                mesh[tuple(ep_mesh_dims)]._flatten(mesh_dim_name="ep")
            else:
                mesh["dp_in_ep"]._flatten(mesh_dim_name="ep")
            self._meshes["ep"] = mesh["ep"]

            # ep_tp mesh: 2D mesh [ep, tp] for ExpertTensorParallel
            self._meshes["ep_tp"] = mesh["ep", "tp"]
        else:
            self._meshes["ep"] = mesh["dp_in_ep", "cp", "tp"]._flatten(
                mesh_dim_name="ep"
            )

        # FSDP backward-compatible aliases (sp = cp)
        self._meshes["sp"] = self._meshes["cp"]
        self._meshes["dp_sp"] = self._meshes["dp_cp"]
        self._meshes["sp_tp"] = self._meshes["cp_tp"]
        self._meshes["dp_shard_cp"] = self._meshes["dp_cp"]
        self._meshes["dp_shard_mod_ep"] = self._meshes["dp_mod_ep"]
        self._meshes["dp_shard_in_ep"] = self._meshes["dp_in_ep"]

        self._world_mesh = mesh
        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    # =========================================================================
    # Size Properties
    # =========================================================================

    @property
    def tp_size(self) -> int:
        return self._strategy.tp_size

    @property
    def dp_size(self) -> int:
        return self._strategy.dp_size

    @property
    def pp_size(self) -> int:
        return self._strategy.pp_size

    @property
    def cp_size(self) -> int:
        return self._strategy.cp_size

    @property
    def ep_size(self) -> int:
        return self._strategy.ep_size

    @property
    def etp_size(self) -> int:
        return self._strategy.etp_size

    @property
    def world_size(self) -> int:
        return self._strategy.world_size

    # =========================================================================
    # Enabled Flags
    # =========================================================================

    @property
    def tp_enabled(self) -> bool:
        return self._strategy.tp_size > 1

    @property
    def dp_enabled(self) -> bool:
        return self._strategy.dp_size > 1

    @property
    def pp_enabled(self) -> bool:
        return self._strategy.pp_size > 1

    @property
    def cp_enabled(self) -> bool:
        return self._strategy.cp_size > 1

    @property
    def ep_enabled(self) -> bool:
        return self._strategy.ep_size > 1

    @property
    def etp_enabled(self) -> bool:
        return self._strategy.etp_size > 1

    # =========================================================================
    # Access Methods
    # =========================================================================

    def get_mesh(self, name: str) -> DeviceMesh | None:
        # Ensure mesh is built
        _ = self.world_mesh
        return self._meshes.get(name)

    def get_group(self, name: str) -> dist.ProcessGroup | None:
        submesh = self.get_mesh(name)
        if submesh is None:
            return None
        return submesh.get_group()

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def gradient_divide_factor(self) -> int:
        return self._strategy.dp_size * self._strategy.cp_size

    @property
    def seq_len_divisor(self) -> int:
        return self._strategy.tp_size * self._strategy.cp_size

    @property
    def context_and_model_parallel_size(self) -> int:
        return self._strategy.cp_size * self._strategy.tp_size * self._strategy.pp_size
