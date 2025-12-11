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
class ShardId:
    """Identifier for a shard, composed of task_id and key."""

    task_id: str
    key: str

    def __str__(self) -> str:
        return f"{self.task_id}:{self.key}"

    def __repr__(self) -> str:
        return f"ShardId(task_id={self.task_id}, key={self.key})"

    def __hash__(self) -> int:
        return hash((self.task_id, self.key))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShardId):
            return False
        return self.task_id == other.task_id and self.key == other.key

    @classmethod
    def from_string(cls, s: str, default_key: str = "default") -> ShardId:
        if ":" in s:
            parts = s.split(":", 1)
            return cls(task_id=parts[0], key=parts[1])
        return cls(task_id=s, key=default_key)


@dataclass
class ShardMetadata:
    """Metadata for a single (sub-)shard stored on one node.

    A logical batch can be composed of multiple shards, and a single physical
    shard can be split into multiple logical sub-shards via offset and batch_size.
    """

    node_id: str
    node_addr: str
    shard_id: ShardId
    tensor_metadata: TensorMetadata | None = None

    def __repr__(self) -> str:
        return (
            f"ShardMetadata(node_id={self.node_id}, node_addr={self.node_addr}, "
            f"shard_id={self.shard_id}, tensor_metadata={self.tensor_metadata})"
        )


@dataclass
class BatchMetadata:
    """Metadata for a distributed batch sharded across multiple nodes."""

    batch_id: str
    shards: list[ShardMetadata] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"BatchMetadata(batch_id={self.batch_id}, num_shards={len(self.shards)}, "
            f"shards={self.shards})"
        )

    def get_all_node_addrs(self) -> set[str]:
        """Get all unique node addresses in this batch."""
        return {shard.node_addr for shard in self.shards}
