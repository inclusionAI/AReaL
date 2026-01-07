# Adapted from torchtitan: torchtitan/distributed/parallel_dims.py

from dataclasses import dataclass

from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

__all__ = [
    "ArchonParallelDims",
]


@dataclass
class ArchonParallelDims:
    """Parallel dimensions for Archon engine.

    Archon Engine uses PyTorch-native distributed APIs (FSDP2, TP, CP) inspired by torchtitan.
    This dataclass configures the parallel dimensions and builds the device mesh.

    Attributes:
        dp_shard: FSDP shard dimension (data parallel). -1 means auto-compute.
        tp: Tensor Parallel size.
        cp: Context Parallel size.
        world_size: Total number of processes.
        device_type: Device type for mesh creation (e.g., "cuda", "npu").

    Supported configurations:
        - FSDP only (tp=1, cp=1)
        - FSDP + TP (tp>1, cp=1)
        - FSDP + TP + CP (tp>1, cp>1) - requires custom Archon model with SDPA

    Note on CP:
        - CP requires custom Archon model (not HuggingFace model)
        - CP only works with SDPA attention (for hook compatibility)
        - When CP is enabled, fsdp mesh includes both dp_shard and cp dimensions
        - seq_len must be divisible by tp * (cp * 2)
    """

    dp_shard: int = -1  # FSDP shard dimension, -1 means auto
    tp: int = 1  # Tensor Parallel size
    cp: int = 1  # Context Parallel size
    world_size: int = 1
    device_type: str = "cuda"  # Device type for mesh creation

    # Internal state
    _world_mesh: DeviceMesh | None = None

    def __post_init__(self):
        if self.dp_shard < 0:
            # Auto-compute dp_shard from world_size, tp, and cp
            self.dp_shard = self.world_size // (self.tp * self.cp)

        expected_world_size = self.dp_shard * self.tp * self.cp
        assert expected_world_size == self.world_size, (
            f"dp_shard * tp * cp must equal world_size, "
            f"got {self.dp_shard} * {self.tp} * {self.cp} = {expected_world_size}, "
            f"world_size={self.world_size}"
        )

    @property
    def world_mesh(self) -> DeviceMesh:
        """Lazily build and cache the device mesh."""
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh(self.device_type)
        return self._world_mesh

    def build_mesh(self, device_type: str) -> DeviceMesh:
        """Build device mesh for FSDP, TP, and optionally CP.

        Args:
            device_type: Device type (e.g., "cuda", "npu").

        Returns:
            DeviceMesh with dimensions (fsdp, tp). Always 2D mesh even when tp=1.

        Mesh layout:
            - fsdp: includes dp_shard * cp (CP shares FSDP's all-gather/reduce-scatter)
            - tp: Tensor Parallel dimension (always present, even when tp=1)
            - cp: Context Parallel dimension (for attention splitting)

        Example (world_size=16, dp=2, tp=2, cp=2):
            Dimensions: (fsdp=4, tp=2) where fsdp = dp_shard * cp
            Note: CP is handled via context manager, not explicit mesh dim
        """
        # FSDP mesh includes both dp_shard and cp
        # CP uses FSDP's all-gather for weights and reduce-scatter for gradients
        fsdp_size = self.dp_shard * self.cp

        # Always create 2D mesh: (fsdp, tp), even when tp=1
        # This ensures tp_group is always available
        return init_device_mesh(
            device_type,
            (fsdp_size, self.tp),
            mesh_dim_names=("fsdp", "tp"),
        )

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def fsdp_enabled(self) -> bool:
        # FSDP is enabled if dp_shard > 1 or cp > 1
        # (CP requires FSDP's all-gather/reduce-scatter)
        return self.dp_shard > 1 or self.cp > 1

    @property
    def dp_group(self) -> ProcessGroup:
        return self.world_mesh["fsdp"].get_group()

    @property
    def tp_group(self) -> ProcessGroup:
        """Return the TP process group. Always available since mesh always has tp dim."""
        return self.world_mesh["tp"].get_group()

    @property
    def seq_len_divisor(self) -> int:
        """Minimum divisor for sequence length.

        Sequence Parallel requires seq_len divisible by TP degree.
        Context Parallel requires seq_len divisible by 2 * CP degree (for load balancing).
        """
        return self.tp * (self.cp * 2)
