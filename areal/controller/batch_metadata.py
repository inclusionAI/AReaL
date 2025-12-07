from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TensorMetadata:
    """Metadata for a tensor field."""

    shape: tuple[int, ...]
    dtype: str
    device: str = "cpu"

    def __repr__(self) -> str:
        return f"TensorMetadata(shape={self.shape}, dtype={self.dtype}, device={self.device})"


@dataclass
class ShardMetadata:
    """Metadata for a single (sub-)shard stored on one node.

    A logical batch can be composed of multiple shards, and a single physical
    shard can be split into multiple logical sub-shards via offset and batch_size.
    """

    node_id: str
    node_addr: str
    shard_id: str
    batch_size: int
    offset: int = 0
    fields: dict[str, TensorMetadata] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ShardMetadata(node_id={self.node_id}, node_addr={self.node_addr}, "
            f"shard_id={self.shard_id}, offset={self.offset}, "
            f"batch_size={self.batch_size}, fields={list(self.fields.keys())})"
        )


@dataclass
class BatchMetadata:
    """Metadata for a distributed batch sharded across multiple nodes."""

    batch_id: str
    global_step: int
    total_batch_size: int
    shards: list[ShardMetadata] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"BatchMetadata(batch_id={self.batch_id}, global_step={self.global_step}, "
            f"total_batch_size={self.total_batch_size}, num_shards={len(self.shards)}, "
            f"shards={self.shards})"
        )

    def get_all_node_addrs(self) -> set[str]:
        """Get all unique node addresses in this batch."""
        return {shard.node_addr for shard in self.shards}
