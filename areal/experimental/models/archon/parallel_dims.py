# Adapted from torchtitan: torchtitan/distributed/parallel_dims.py

import functools
from dataclasses import dataclass, field

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from areal.api.alloc_mode import ParallelStrategy
from areal.engine.core.topology import DeviceMeshTopology
from areal.utils import logging


@functools.cache
def _get_logger() -> logging.Logger:
    """Get rank-aware logger for this module."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon ParallelDims Rank {rank}]")


__all__ = [
    "ArchonParallelDims",
]


@dataclass
class ArchonParallelDims:
    """Parallel dimensions for Archon engine.

    Archon Engine uses PyTorch-native distributed APIs inspired by torchtitan.

    Note: dp_replicate is always 1 - no HSDP support yet.

    Attributes:
        dp_shard: FSDP shard dimension (data parallel). -1 means auto-compute.
        tp: Tensor Parallel size.
        cp: Context Parallel size (implemented as Ulysses SP, not Ring Attention).
        pp: Pipeline Parallel size.
        ep: Expert Parallel size.
        etp: Expert Tensor Parallel size (must be 1 or equal to tp).
        world_size: Total number of processes.
        device_type: Device type for mesh creation (e.g., "cuda", "npu").

    Mesh semantics:
        - total_gpu = pp × dp_shard × cp × tp
        - fsdp_size = dp_shard × cp  (CP ranks participate in weight sharding)

    Expert Parallelism Strategy Selection:
        The strategy for MoE expert layers depends on EP and ETP settings:

        | EP  | TP  | etp | Strategy              | Expert Weight Sharding       |
        |-----|-----|-----|-----------------------|------------------------------|
        | 1   | 1   | -   | None                  | Replicate                    |
        | 1   | >1  | -   | TensorParallel        | [Shard(1/2)]                 |
        | >1  | 1   | -   | ExpertParallel        | [Shard(0)]                   |
        | >1  | >1  | 1   | ExpertParallel        | [Shard(0)] (TP borrowed by EP) |
        | >1  | >1  | tp  | ExpertTensorParallel  | [Shard(0), Shard(1/2)]       |

        When EP>1 and TP>1:
        - etp=1: TP dimension is borrowed by EP for token dispatch. Experts use
          only EP sharding [Shard(0)]. EP borrows from dp_shard × cp × tp.
        - etp=tp: TP dimension remains independent. Experts use 2D sharding
          [Shard(0), Shard(1/2)] combining EP and TP. EP borrows from
          dp_shard × cp only.

    Available mesh dimensions (when EP disabled):
        - pp: Pipeline Parallel
        - dp_shard: FSDP shard dimension
        - cp: Context Parallel
        - tp: Tensor Parallel
        - dp: dp_shard (for data loading)
        - dp_shard_cp: dp_shard * cp (for FSDP sharding)
        - dp_cp: dp_shard * cp (for loss all-reduce)
        - pp_cp_tp: pp * cp * tp (for context and model parallel group / data broadcast)

    Available mesh dimensions (when EP enabled):
        - pp: Pipeline Parallel
        - dp_shard_mod_ep: dp_shard * cp / ep when etp=tp, else dp_shard * cp * tp / ep
        - dp_shard_in_ep: ep / cp when etp=tp, else ep / (cp * tp)
        - cp: Context Parallel
        - tp: Tensor Parallel
        - dp: dp_shard (for data loading)
        - dp_shard_cp: dp_shard * cp (for FSDP sharding of dense params)
        - dp_cp: dp_shard * cp (for loss all-reduce)
        - pp_cp_tp: pp * cp * tp (for context and model parallel group / data broadcast)
        - ep: Expert Parallel (flattened from dp_shard_in_ep * cp * etp)
        - ep_tp: 2D mesh [ep, tp] for ExpertTensorParallel (only when etp=tp)

    Example (pp=1, dp_shard=2, cp=2, tp=2, ep=1, etp=1, 8 GPUs):
        - fsdp_size = dp_shard × cp = 2 × 2 = 4
        - Mesh dims: (pp=1, dp_shard=2, cp=2, tp=2)

    Example (pp=2, dp_shard=2, cp=1, tp=2, ep=1, etp=1, 8 GPUs):
        - fsdp_size = dp_shard × cp = 2 × 1 = 2
        - Mesh dims: (pp=2, dp_shard=2, cp=1, tp=2)

    Example (pp=1, dp_shard=2, cp=1, tp=2, ep=2, etp=1, 4 GPUs):
        - ep borrows from dp_shard_in_ep * cp * tp (since etp=1)
        - dp_shard_mod_ep = dp_shard * cp * tp / ep = 2 * 1 * 2 / 2 = 2
        - dp_shard_in_ep = ep / (cp * tp) = 2 / (1 * 2) = 1
        - Mesh dims: (pp=1, dp_shard_mod_ep=2, dp_shard_in_ep=1, cp=1, tp=2)

    Example (pp=1, dp_shard=2, cp=1, tp=2, ep=2, etp=2, 4 GPUs):
        - ep borrows from dp_shard_in_ep * cp only (since etp=tp)
        - dp_shard_mod_ep = dp_shard * cp / ep = 2 * 1 / 2 = 1
        - dp_shard_in_ep = ep / cp = 2 / 1 = 2
        - Mesh dims: (pp=1, dp_shard_mod_ep=1, dp_shard_in_ep=2, cp=1, tp=2)
        - ep_tp mesh: 2D mesh [ep=2, tp=2] for 2D expert weight sharding
    """

    dp_replicate: int = 1  # HSDP replicate dimension (not supported yet)
    dp_shard: int = -1  # FSDP shard dimension, -1 means auto
    cp: int = 1  # Context Parallel size (Ulysses SP)
    tp: int = 1  # Tensor Parallel size
    pp: int = 1  # Pipeline Parallel size
    ep: int = 1  # Expert Parallel size
    etp: int = 1  # Expert Tensor Parallel size (1 or tp)
    world_size: int = 1
    device_type: str = "cuda"

    # Internal state
    _topology: DeviceMeshTopology | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.dp_shard < 0:
            self.dp_shard = self.world_size // (self.tp * self.cp * self.pp)

        # Build a ParallelStrategy to feed into DeviceMeshTopology
        strategy = ParallelStrategy(
            tensor_parallel_size=self.tp,
            pipeline_parallel_size=self.pp,
            data_parallel_size=self.dp_shard,
            context_parallel_size=self.cp,
            expert_parallel_size=self.ep,
            expert_tensor_parallel_size=self.etp,
        )
        # DeviceMeshTopology validates all constraints
        self._topology = DeviceMeshTopology(strategy, device_type=self.device_type)

    # =========================================================================
    # Mesh Creation (delegated to DeviceMeshTopology)
    # =========================================================================

    @property
    def topology(self) -> DeviceMeshTopology:
        assert self._topology is not None
        return self._topology

    def build_mesh(self) -> DeviceMesh:
        """Build device mesh for all parallelism dimensions."""
        return self.topology.build_mesh()

    @property
    def world_mesh(self) -> DeviceMesh:
        """Lazily build and return the device mesh."""
        return self.topology.world_mesh

    # =========================================================================
    # Enabled Flags
    # =========================================================================

    @property
    def dp_replicate_enabled(self) -> bool:
        """Whether HSDP replication is enabled (dp_replicate > 1)."""
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        """Whether FSDP sharding is enabled (dp_shard > 1)."""
        return self.dp_shard > 1

    @property
    def dp_enabled(self) -> bool:
        """Whether any data parallelism is enabled (dp_replicate > 1 or dp_shard > 1)."""
        return self.dp_replicate_enabled or self.dp_shard_enabled

    @property
    def dp_cp_enabled(self) -> bool:
        """Whether data or context parallelism is enabled (for loss all-reduce)."""
        return self.dp_enabled or self.cp_enabled

    @property
    def fsdp_enabled(self) -> bool:
        """Whether FSDP is enabled (dp_shard > 1 or cp > 1)."""
        return self.dp_shard_enabled or self.cp_enabled

    @property
    def cp_enabled(self) -> bool:
        """Whether context parallelism is enabled."""
        return self.cp > 1

    @property
    def tp_enabled(self) -> bool:
        """Whether tensor parallelism is enabled."""
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        """Whether pipeline parallelism is enabled."""
        return self.pp > 1

    @property
    def ep_enabled(self) -> bool:
        """Whether expert parallelism is enabled."""
        return self.ep > 1

    @property
    def etp_enabled(self) -> bool:
        """Whether expert tensor parallelism is enabled (etp > 1)."""
        return self.etp > 1

    # =========================================================================
    # Utilities (delegated to DeviceMeshTopology)
    # =========================================================================

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        """Gradient divide factor for FSDP (consistent with data parallelism degree)."""
        return self.topology.gradient_divide_factor

    @property
    def seq_len_divisor(self) -> int:
        """Minimum divisor for sequence length (for TP and CP compatibility).

        Note: Archon multiplies by 2 for ring attention compatibility.
        """
        return self.tp * self.cp * 2

    @property
    def context_and_model_parallel_size(self) -> int:
        """Context and model parallel size (cp * tp * pp)."""
        return self.topology.context_and_model_parallel_size

    def get_mesh(self, name: str) -> DeviceMesh | None:
        """Get submesh by name, return None if not available."""
        return self.topology.get_mesh(name)

    def get_group(self, name: str) -> dist.ProcessGroup | None:
        """Get process group by name, return None if not available."""
        return self.topology.get_group(name)
