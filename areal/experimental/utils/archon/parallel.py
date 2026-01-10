# Adapted from torchtitan: torchtitan/distributed/parallel_dims.py

from dataclasses import dataclass, field

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

__all__ = [
    "ArchonParallelDims",
]


@dataclass
class ArchonParallelDims:
    """Parallel dimensions for Archon engine.

    Archon Engine uses PyTorch-native distributed APIs (FSDP2, TP, CP) inspired by torchtitan.

    Attributes:
        dp_shard: FSDP shard dimension (data parallel). -1 means auto-compute.
        tp: Tensor Parallel size.
        cp: Context Parallel size (implemented as Ulysses SP, not Ring Attention).
        world_size: Total number of processes.
        device_type: Device type for mesh creation (e.g., "cuda", "npu").

    Mesh semantics:
        - total_gpu = dp_shard × cp × tp
        - GPU index = dp_rank * cp * tp + cp_rank * tp + tp_rank
        - Creates 3D mesh (dp, cp, tp)
        - FSDP shards only on dp dimension; CP group members share identical weights
        - Sub-meshes derived via _flatten: mp (model parallel = cp + tp)

    Example (dp=2, cp=2, tp=2, 8 GPUs):
        3D Mesh shape: (dp=2, cp=2, tp=2)

        Layout:
                dp=0                    dp=1
            cp=0    cp=1            cp=0    cp=1
          tp=0 tp=1 tp=0 tp=1     tp=0 tp=1 tp=0 tp=1
          [0]  [1]  [2]  [3]      [4]  [5]  [6]  [7]

        Process Groups:
            dp groups: [0,4], [1,5], [2,6], [3,7]  (same cp_rank and tp_rank) - for FSDP
            cp groups: [0,2], [1,3], [4,6], [5,7]  (same dp_rank and tp_rank)
            tp groups: [0,1], [2,3], [4,5], [6,7]  (same dp_rank and cp_rank)
            mp groups: [0,1,2,3], [4,5,6,7]        (same dp_rank) - model parallel
    """

    dp_shard: int = -1  # FSDP shard dimension, -1 means auto
    tp: int = 1  # Tensor Parallel size
    cp: int = 1  # Context Parallel size (Ulysses SP)
    world_size: int = 1
    device_type: str = "cuda"  # Device type for mesh creation

    # Internal state
    _world_mesh: DeviceMesh | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.dp_shard < 0:
            self.dp_shard = self.world_size // (self.tp * self.cp)

        expected_world_size = self.dp_shard * self.tp * self.cp
        assert expected_world_size == self.world_size, (
            f"dp_shard * tp * cp must equal world_size, "
            f"got {self.dp_shard} * {self.tp} * {self.cp} = {expected_world_size}, "
            f"world_size={self.world_size}"
        )

    # =========================================================================
    # Mesh Creation
    # =========================================================================

    @property
    def world_mesh(self) -> DeviceMesh:
        """Lazily build and cache the device mesh."""
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh(self.device_type)
        return self._world_mesh

    def build_mesh(self, device_type: str) -> DeviceMesh:
        """Build device mesh for FSDP, CP, and TP.

        Creates a 3D mesh (dp, cp, tp) where product equals world_size.
        Sub-meshes are derived via _flatten for model parallel group.

        Root mesh dims: dp, cp, tp
        Derived sub-meshes:
            - mp: flattened (cp, tp) for model parallel group

        FSDP shards only on dp dimension. CP group members share identical weights.
        """
        mesh = init_device_mesh(
            device_type,
            (self.dp_shard, self.cp, self.tp),
            mesh_dim_names=("dp", "cp", "tp"),
        )

        # Create sub-mesh via _flatten
        # mp: model parallel group (cp + tp) for context_and_model_parallel_group
        mesh["cp", "tp"]._flatten(mesh_dim_name="mp")

        return mesh

    # =========================================================================
    # Process Groups
    # =========================================================================

    @property
    def dp_group(self) -> ProcessGroup:
        """Return the DP (FSDP) process group."""
        return self.world_mesh["dp"].get_group()

    @property
    def tp_group(self) -> ProcessGroup:
        """Return the TP process group."""
        return self.world_mesh["tp"].get_group()

    @property
    def cp_group(self) -> ProcessGroup:
        """Return the CP process group."""
        return self.world_mesh["cp"].get_group()

    @property
    def mp_group(self) -> ProcessGroup:
        """Return the model parallel group (cp + tp)."""
        return self.world_mesh["mp"].get_group()

    # =========================================================================
    # Enabled Flags
    # =========================================================================

    @property
    def dp_enabled(self) -> bool:
        """Whether data parallelism (FSDP) is enabled."""
        return self.dp_shard > 1

    @property
    def tp_enabled(self) -> bool:
        """Whether tensor parallelism is enabled."""
        return self.tp > 1

    @property
    def cp_enabled(self) -> bool:
        """Whether context parallelism is enabled."""
        return self.cp > 1

    # =========================================================================
    # Rank Accessors
    # =========================================================================

    @property
    def dp_rank(self) -> int:
        """Get data parallel rank (not including CP)."""
        return dist.get_rank() // (self.cp * self.tp)

    @property
    def cp_rank(self) -> int:
        """Get context parallel rank within CP group."""
        return (dist.get_rank() // self.tp) % self.cp

    @property
    def tp_rank(self) -> int:
        """Get tensor parallel rank within TP group."""
        return dist.get_rank() % self.tp

    # =========================================================================
    # Utilities
    # =========================================================================

    @property
    def seq_len_divisor(self) -> int:
        """Minimum divisor for sequence length (for TP and CP compatibility)."""
        return self.tp * self.cp
