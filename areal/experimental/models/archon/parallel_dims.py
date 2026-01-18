# Adapted from torchtitan: torchtitan/distributed/parallel_dims.py

from dataclasses import dataclass, field

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from areal.utils import logging

logger = logging.getLogger("ArchonParallelDims")

__all__ = [
    "ArchonParallelDims",
]


@dataclass
class ArchonParallelDims:
    """Parallel dimensions for Archon engine.

    Archon Engine uses PyTorch-native distributed APIs (FSDP2, TP, CP, EP) inspired by torchtitan.

    Note: etp (Expert Tensor Parallel) is always 1 - not supported yet.
    Note: dp_replicate is always 1 - no HSDP support yet.

    Attributes:
        dp_shard: FSDP shard dimension (data parallel). -1 means auto-compute.
        tp: Tensor Parallel size.
        cp: Context Parallel size (implemented as Ulysses SP, not Ring Attention).
        ep: Expert Parallel size.
        world_size: Total number of processes.
        device_type: Device type for mesh creation (e.g., "cuda", "npu").

    Mesh semantics:
        - total_gpu = dp_shard × cp × tp
        - fsdp_size = dp_shard × cp  (CP ranks participate in weight sharding)

    Available mesh dimensions (when EP disabled):
        - dp: dp_shard (for data loading)
        - dp_shard_cp: dp_shard * cp (for FSDP sharding)
        - dp_cp: dp_shard * cp (for loss all-reduce)
        - cp: Context Parallel
        - tp: Tensor Parallel

    Available mesh dimensions (when EP enabled):
        - dp: dp_shard (for data loading)
        - dp_shard_cp: dp_shard * cp (for FSDP sharding of dense params)
        - dp_cp: dp_shard * cp (for loss all-reduce)
        - cp: Context Parallel
        - tp: Tensor Parallel
        - ep: Expert Parallel (flattened from dp_shard_in_ep + cp [+ tp if etp=1])
        - dp_shard_mod_ep: dp_shard_cp * tp / ep (for FSDP sharding of MoE experts)

    Example (dp_shard=2, cp=2, tp=2, ep=1, 8 GPUs):
        - fsdp_size = dp_shard × cp = 2 × 2 = 4
        - Mesh dims: (dp_shard=2, cp=2, tp=2)

    Example (dp_shard=2, cp=1, tp=2, ep=2, 4 GPUs):
        - ep borrows from dp_shard_in_ep * cp * tp (since etp=1)
        - dp_shard_mod_ep = dp_shard * cp * tp / ep = 2 * 1 * 2 / 2 = 2
        - dp_shard_in_ep = ep / (cp * tp) = 2 / (1 * 2) = 1
        - Mesh dims: (dp_shard_mod_ep=2, dp_shard_in_ep=1, cp=1, tp=2)
    """

    dp_shard: int = -1  # FSDP shard dimension, -1 means auto
    tp: int = 1  # Tensor Parallel size
    cp: int = 1  # Context Parallel size (Ulysses SP)
    ep: int = 1  # Expert Parallel size (ETP is always 1)
    world_size: int = 1
    device_type: str = "cuda"

    # Internal state
    _world_mesh: DeviceMesh | None = field(default=None, repr=False)
    _meshes: dict[str, DeviceMesh] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.dp_shard < 0:
            self.dp_shard = self.world_size // (self.tp * self.cp)

        expected_world_size = self.dp_shard * self.tp * self.cp
        if expected_world_size != self.world_size:
            raise ValueError(
                f"dp_shard * tp * cp must equal world_size, "
                f"got {self.dp_shard} * {self.tp} * {self.cp} = {expected_world_size}, "
                f"world_size={self.world_size}"
            )

        # Validate EP constraints (ETP=1 only, so EP borrows from dp_shard * cp * tp)
        if self.ep > 1:
            # EP borrows all cp and tp and some dp_shard degree
            if self.ep % (self.cp * self.tp) != 0:
                raise ValueError(
                    f"ep must be divisible by cp * tp, "
                    f"got ep={self.ep}, cp={self.cp}, tp={self.tp}"
                )
            if (self.dp_shard * self.cp * self.tp) % self.ep != 0:
                raise ValueError(
                    f"dp_shard * cp * tp must be divisible by ep, "
                    f"got {self.dp_shard} * {self.cp} * {self.tp} = {self.dp_shard * self.cp * self.tp}, ep={self.ep}"
                )

    # =========================================================================
    # Mesh Creation
    # =========================================================================

    def build_mesh(self) -> DeviceMesh:
        """Build device mesh for all parallelism dimensions."""
        if self.ep > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    def _build_mesh_without_ep(self) -> DeviceMesh:
        """Build mesh when EP is disabled."""
        # Always include all dimensions, even if size=1
        # This ensures submeshes like mp (cp × tp) can always be created
        dims = [self.dp_shard, self.cp, self.tp]
        names = ["dp_shard", "cp", "tp"]

        logger.info(f"Building 3-D device mesh with {names}, {dims}")
        mesh = init_device_mesh(
            self.device_type, tuple(dims), mesh_dim_names=tuple(names)
        )

        self._meshes["dp_shard"] = mesh["dp_shard"]
        self._meshes["cp"] = mesh["cp"]
        self._meshes["tp"] = mesh["tp"]

        # dp mesh: for data loading
        self._meshes["dp"] = mesh["dp_shard"]._flatten(mesh_dim_name="dp")

        # dp_shard_cp mesh: for FSDP param sharding
        self._meshes["dp_shard_cp"] = mesh["dp_shard", "cp"]._flatten(
            mesh_dim_name="dp_shard_cp"
        )

        # dp_cp mesh: for loss all-reduce
        self._meshes["dp_cp"] = mesh["dp_shard", "cp"]._flatten(mesh_dim_name="dp_cp")

        # cp_tp mesh: for context parallel × tensor parallel (data broadcast group)
        self._meshes["cp_tp"] = mesh["cp", "tp"]._flatten(mesh_dim_name="cp_tp")

        self._world_mesh = mesh
        return mesh

    def _build_mesh_with_ep(self) -> DeviceMesh:
        """Build mesh when EP is enabled.

        With EP (etp=1), dp_shard and ep are derived submeshes:
        - ep = dp_shard_in_ep * cp * tp
        - dp_shard = dp_shard_mod_ep * dp_shard_in_ep
        """
        # Since etp=1, ep borrows from dp_shard_in_ep * cp * tp
        dp_shard_mod_ep = self.dp_shard * self.cp * self.tp // self.ep
        dp_shard_in_ep = self.ep // (self.cp * self.tp)

        # Always include all dimensions, even if size=1
        # This ensures submeshes like mp (cp × tp) can always be created
        dims = [dp_shard_mod_ep, dp_shard_in_ep, self.cp, self.tp]
        names = ["dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp"]

        logger.info(f"Building 4-D device mesh with {names}, {dims}")
        mesh = init_device_mesh(
            self.device_type, tuple(dims), mesh_dim_names=tuple(names)
        )

        self._meshes["dp_shard_mod_ep"] = mesh["dp_shard_mod_ep"]
        self._meshes["dp_shard_in_ep"] = mesh["dp_shard_in_ep"]
        self._meshes["cp"] = mesh["cp"]
        self._meshes["tp"] = mesh["tp"]

        # dp mesh: for data loading
        self._meshes["dp"] = mesh["dp_shard_mod_ep", "dp_shard_in_ep"]._flatten(
            mesh_dim_name="dp"
        )

        # dp_shard_cp mesh: for FSDP param sharding
        self._meshes["dp_shard_cp"] = mesh[
            "dp_shard_mod_ep", "dp_shard_in_ep", "cp"
        ]._flatten(mesh_dim_name="dp_shard_cp")

        # dp_cp mesh: for loss all-reduce
        self._meshes["dp_cp"] = mesh[
            "dp_shard_mod_ep", "dp_shard_in_ep", "cp"
        ]._flatten(mesh_dim_name="dp_cp")

        # ep mesh: for expert parallel
        self._meshes["ep"] = mesh["dp_shard_in_ep", "cp", "tp"]._flatten(
            mesh_dim_name="ep"
        )

        # cp_tp mesh: for context parallel × tensor parallel (data broadcast group)
        self._meshes["cp_tp"] = mesh["cp", "tp"]._flatten(mesh_dim_name="cp_tp")

        self._world_mesh = mesh
        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        """Lazily build and return the device mesh."""
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    # =========================================================================
    # Enabled Flags
    # =========================================================================

    @property
    def fsdp_enabled(self) -> bool:
        """Whether FSDP is enabled (dp_shard > 1 or cp > 1)."""
        return self.dp_shard > 1 or self.cp > 1

    @property
    def dp_enabled(self) -> bool:
        """Whether data parallelism is enabled (dp_shard > 1)."""
        return self.dp_shard > 1

    @property
    def dp_cp_enabled(self) -> bool:
        """Whether data or context parallelism is enabled."""
        return self.dp_enabled or self.cp_enabled

    @property
    def tp_enabled(self) -> bool:
        """Whether tensor parallelism is enabled."""
        return self.tp > 1

    @property
    def cp_enabled(self) -> bool:
        """Whether context parallelism is enabled."""
        return self.cp > 1

    @property
    def ep_enabled(self) -> bool:
        """Whether expert parallelism is enabled."""
        return self.ep > 1

    # =========================================================================
    # Utilities
    # =========================================================================

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        """Gradient divide factor for FSDP (consistent with data parallelism degree).

        This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        Although the FSDP sharding of experts is done on a mesh of a different size than
        other parameters, the gradient division factor should be consistent with data.
        """
        return self.dp_shard * self.cp

    @property
    def seq_len_divisor(self) -> int:
        """Minimum divisor for sequence length (for TP and CP compatibility)."""
        return self.tp * self.cp * 2

    def get_mesh(self, name: str) -> DeviceMesh | None:
        """Get submesh by name, return None if not available.

        This method safely retrieves a submesh without throwing an exception
        if the dimension doesn't exist (e.g., cp=1 means no "cp" mesh).

        Args:
            name: Mesh dimension name. Available names depend on EP status:
                - Without EP: 'dp', 'dp_shard_cp', 'dp_cp', 'dp_shard', 'cp', 'tp', 'cp_tp'
                - With EP: 'dp', 'dp_shard_cp', 'dp_cp', 'ep', 'dp_shard_mod_ep',
                          'dp_shard_in_ep', 'cp', 'tp', 'cp_tp'

        Returns:
            DeviceMesh for the requested dimension, or None if not available.
        """
        # Ensure mesh is built
        _ = self.world_mesh
        return self._meshes.get(name)

    def get_group(self, name: str) -> dist.ProcessGroup | None:
        """Get process group by name, return None if not available.

        Args:
            name: Mesh dimension name. Available names depend on EP status:
                - Without EP: 'dp', 'dp_shard_cp', 'dp_cp', 'dp_shard', 'cp', 'tp', 'cp_tp'
                - With EP: 'dp', 'dp_shard_cp', 'dp_cp', 'ep', 'dp_shard_mod_ep',
                          'dp_shard_in_ep', 'cp', 'tp', 'cp_tp'

        Returns:
            ProcessGroup for the requested dimension, or None if not available.
        """
        submesh = self.get_mesh(name)
        if submesh is None:
            return None
        return submesh.get_group()
