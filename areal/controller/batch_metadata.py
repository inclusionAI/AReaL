from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TensorMetadata:
    """Metadata for a single tensor field.

    Attributes
    ----------
    shape : tuple[int, ...]
        Shape of the tensor
    dtype : str
        Data type of the tensor (e.g., 'torch.float32', 'torch.int64')
    device : str
        Device where the tensor is stored (e.g., 'cpu', 'cuda:0')
    """

    shape: tuple[int, ...]
    dtype: str
    device: str = "cpu"

    def __repr__(self) -> str:
        return f"TensorMetadata(shape={self.shape}, dtype={self.dtype}, device={self.device})"


@dataclass
class ScalarMetadata:
    """Metadata for a scalar or list field.

    Attributes
    ----------
    value_type : str
        Type of the value (e.g., 'int', 'float', 'str', 'list')
    length : int
        For list types, the length of the list; for scalar types, always 1
    """

    value_type: str
    length: int = 1

    def __repr__(self) -> str:
        return f"ScalarMetadata(value_type={self.value_type}, length={self.length})"


@dataclass
class ShardMetadata:
    """Metadata for a single (sub-)shard stored on one node.

    A logical batch can be composed of multiple shards, and a single physical
    shard can be split into multiple logical sub-shards via the ``offset`` and
    ``batch_size`` fields.

    Attributes
    ----------
    node_id : str
        Identifier of the node storing this shard
    node_addr : str
        Network address (host:port) of the node's HTTP server
    shard_id : str
        Unique identifier for the *physical* shard on the node
    offset : int
        Starting index (inclusive) within the physical shard along batch dimension
    batch_size : int
        Number of samples in this logical sub-shard
    fields : dict[str, TensorMetadata | ScalarMetadata]
        Metadata for each field in the shard
    """

    node_id: str
    node_addr: str
    shard_id: str
    batch_size: int
    offset: int = 0
    fields: dict[str, TensorMetadata | ScalarMetadata] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ShardMetadata(node_id={self.node_id}, node_addr={self.node_addr}, "
            f"shard_id={self.shard_id}, offset={self.offset}, "
            f"batch_size={self.batch_size}, fields={list(self.fields.keys())})"
        )


@dataclass
class BatchMetadata:
    """Metadata for a distributed batch.

    This structure describes a batch that may be sharded across multiple nodes.
    The control plane uses this metadata for coordination without transferring
    the actual tensor data.

    Attributes
    ----------
    batch_id : str
        Unique identifier for this batch
    global_step : int
        Global training step associated with this batch
    total_batch_size : int
        Total number of samples across all shards
    shards : list[ShardMetadata]
        Metadata for each shard in the batch
    """

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
        """Get all unique node addresses in this batch.

        Returns
        -------
        set[str]
            Set of node addresses (host:port)
        """
        return {shard.node_addr for shard in self.shards}
