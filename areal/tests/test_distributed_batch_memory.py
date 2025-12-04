"""Tests for the refactored DistributedBatchMemory with metadata support."""

import pytest
import torch

from areal.controller.batch import DistributedBatchMemory
from areal.controller.batch_metadata import (
    BatchMetadata,
    ScalarMetadata,
    ShardMetadata,
    TensorMetadata,
)


class TestDistributedBatchMemoryBasic:
    """Test basic functionality of DistributedBatchMemory (backward compatibility)."""

    def test_from_dict_basic(self):
        """Test creating a batch from dictionary."""
        data = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
        batch = DistributedBatchMemory.from_dict(data)

        assert batch.dataset is not None
        assert batch._is_local is True
        assert batch.metadata is None
        assert len(batch) == 2

    def test_get_data_local(self):
        """Test getting data from local batch."""
        data = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
        batch = DistributedBatchMemory.from_dict(data)
        result = batch.get_data()

        assert "input_ids" in result
        assert "attention_mask" in result
        assert torch.equal(result["input_ids"], data["input_ids"])

    def test_chunk(self):
        """Test chunking a batch."""
        data = {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        }
        batch = DistributedBatchMemory.from_dict(data)
        chunks = batch.chunk(2)

        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2

    def test_concat(self):
        """Test concatenating batches."""
        batch1 = DistributedBatchMemory.from_dict(
            {
                "input_ids": torch.tensor([[1, 2]]),
            }
        )
        batch2 = DistributedBatchMemory.from_dict(
            {
                "input_ids": torch.tensor([[3, 4]]),
            }
        )
        result = DistributedBatchMemory.concat([batch1, batch2])

        assert len(result) == 2
        data = result.get_data()
        assert data["input_ids"].shape == (2, 2)


class TestDistributedBatchMemoryMetadata:
    """Test metadata-based functionality."""

    def test_create_metadata_for_local_data(self):
        """Test creating metadata for local data."""
        data = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": [0, 1],
        }
        metadata = DistributedBatchMemory.create_metadata_for_local_data(
            data,
            node_id="test-node",
            node_addr="localhost:8765",
            global_step=10,
        )

        assert metadata.total_batch_size == 2
        assert metadata.global_step == 10
        assert len(metadata.shards) == 1
        assert metadata.shards[0].node_id == "test-node"
        assert "input_ids" in metadata.shards[0].fields
        assert "labels" in metadata.shards[0].fields

    def test_from_metadata(self):
        """Test creating batch from metadata."""
        metadata = BatchMetadata(
            batch_id="test-batch",
            global_step=5,
            total_batch_size=10,
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    batch_size=10,
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(10, 5),
                            dtype="torch.int64",
                        ),
                    },
                ),
            ],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        assert batch.dataset is None
        assert batch._is_local is False
        assert batch.metadata == metadata

    def test_concat_with_metadata(self):
        """Test concatenating batches with metadata."""
        metadata1 = BatchMetadata(
            batch_id="batch-1",
            global_step=1,
            total_batch_size=5,
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    batch_size=5,
                    fields={},
                ),
            ],
        )
        metadata2 = BatchMetadata(
            batch_id="batch-2",
            global_step=2,
            total_batch_size=3,
            shards=[
                ShardMetadata(
                    node_id="node-1",
                    node_addr="localhost:8766",
                    shard_id="shard-1",
                    batch_size=3,
                    fields={},
                ),
            ],
        )

        batch1 = DistributedBatchMemory.from_metadata(metadata1)
        batch2 = DistributedBatchMemory.from_metadata(metadata2)
        result = DistributedBatchMemory.concat([batch1, batch2])

        assert result.metadata is not None
        assert result.metadata.total_batch_size == 8
        assert len(result.metadata.shards) == 2
        assert result.metadata.global_step == 2  # max of input steps

    def test_serialization(self):
        """Test serialization and deserialization."""
        import pickle

        metadata = BatchMetadata(
            batch_id="test",
            global_step=1,
            total_batch_size=10,
            shards=[],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        # Serialize
        serialized = pickle.dumps(batch)
        # Deserialize
        deserialized = pickle.loads(serialized)

        assert deserialized.metadata.batch_id == "test"
        assert deserialized._is_local is False

    def test_chunk_metadata(self):
        """Test chunking with metadata mode."""
        from areal.controller.batch_metadata import TensorMetadata

        metadata = BatchMetadata(
            batch_id="test-batch",
            global_step=5,
            total_batch_size=100,
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    batch_size=50,
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(50, 128), dtype="torch.int64"
                        ),
                    },
                ),
                ShardMetadata(
                    node_id="node-1",
                    node_addr="localhost:8766",
                    shard_id="shard-1",
                    batch_size=50,
                    fields={
                        "input_ids": TensorMetadata(
                            shape=(50, 128), dtype="torch.int64"
                        ),
                    },
                ),
            ],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        # Chunk into 2 groups
        chunks = batch.chunk(2)

        assert len(chunks) == 2
        assert all(chunk.metadata is not None for chunk in chunks)
        assert all(chunk._is_local is False for chunk in chunks)
        # Total size should be preserved
        total_size = sum(chunk.metadata.total_batch_size for chunk in chunks)
        assert total_size == 100

    def test_union_metadata(self):
        """Test union with metadata mode."""
        metadata1 = BatchMetadata(
            batch_id="batch-1",
            global_step=1,
            total_batch_size=30,
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="localhost:8765",
                    shard_id="shard-0",
                    batch_size=30,
                    fields={},
                ),
            ],
        )
        metadata2 = BatchMetadata(
            batch_id="batch-2",
            global_step=2,
            total_batch_size=20,
            shards=[
                ShardMetadata(
                    node_id="node-1",
                    node_addr="localhost:8766",
                    shard_id="shard-1",
                    batch_size=20,
                    fields={},
                ),
            ],
        )

        batch1 = DistributedBatchMemory.from_metadata(metadata1)
        batch2 = DistributedBatchMemory.from_metadata(metadata2)
        result = batch1.union(batch2)

        assert result.metadata is not None
        assert result.metadata.total_batch_size == 50
        assert len(result.metadata.shards) == 2
        assert result.metadata.global_step == 2  # max of input steps
        assert result._is_local is False

    def test_get_total_size_metadata(self):
        """Test _get_total_size with metadata mode."""
        metadata = BatchMetadata(
            batch_id="test",
            global_step=1,
            total_batch_size=123,
            shards=[],
        )
        batch = DistributedBatchMemory.from_metadata(metadata)

        assert len(batch) == 123
        assert batch._get_total_size() == 123


class TestBatchMetadata:
    """Test metadata structures."""

    def test_tensor_metadata(self):
        """Test TensorMetadata creation."""
        meta = TensorMetadata(
            shape=(32, 128),
            dtype="torch.float32",
            device="cuda:0",
        )
        assert meta.shape == (32, 128)
        assert meta.dtype == "torch.float32"
        assert meta.device == "cuda:0"

    def test_scalar_metadata(self):
        """Test ScalarMetadata creation."""
        meta = ScalarMetadata(
            value_type="int",
            length=1,
        )
        assert meta.value_type == "int"
        assert meta.length == 1

    def test_shard_metadata(self):
        """Test ShardMetadata creation."""
        meta = ShardMetadata(
            node_id="node-0",
            node_addr="localhost:8765",
            shard_id="shard-0",
            batch_size=32,
            fields={
                "input_ids": TensorMetadata(
                    shape=(32, 128),
                    dtype="torch.int64",
                ),
            },
        )
        assert meta.node_id == "node-0"
        assert meta.batch_size == 32
        assert "input_ids" in meta.fields

    def test_batch_metadata_node_addrs(self):
        """Test getting all node addresses from batch metadata."""
        metadata = BatchMetadata(
            batch_id="test",
            global_step=1,
            total_batch_size=64,
            shards=[
                ShardMetadata(
                    node_id="node-0",
                    node_addr="192.168.1.10:8765",
                    shard_id="shard-0",
                    batch_size=32,
                    fields={},
                ),
                ShardMetadata(
                    node_id="node-1",
                    node_addr="192.168.1.11:8765",
                    shard_id="shard-1",
                    batch_size=32,
                    fields={},
                ),
            ],
        )
        addrs = metadata.get_all_node_addrs()
        assert len(addrs) == 2
        assert "192.168.1.10:8765" in addrs
        assert "192.168.1.11:8765" in addrs


class TestRPCDistributedBatchReturn:
    """Test RPC automatic distributed batch return functionality."""

    def test_handle_tensor_return(self):
        """Test handling tensor return from engine method."""
        from areal.scheduler.rpc.rpc_server import _handle_distributed_batch_return

        # Mock engine with get_version method
        class MockEngine:
            def get_version(self):
                return 42

        engine = MockEngine()

        # Test tensor return
        tensor_result = torch.randn(10, 5)
        batch = _handle_distributed_batch_return(
            tensor_result,
            distributed_batch_target_key="logits",
            engine=engine,
        )

        # Should return DistributedBatchMemory with metadata
        assert isinstance(batch, DistributedBatchMemory)
        assert batch.metadata is not None
        assert batch.metadata.global_step == 42
        assert batch.metadata.total_batch_size == 10
        assert len(batch.metadata.shards) == 1
        assert "logits" in batch.metadata.shards[0].fields

    def test_handle_dict_return(self):
        """Test handling dict return from engine method."""
        from areal.scheduler.rpc.rpc_server import _handle_distributed_batch_return

        # Mock engine
        class MockEngine:
            def get_version(self):
                return 100

        engine = MockEngine()

        # Test dict return
        dict_result = {
            "logits": torch.randn(8, 10, 50),
            "values": torch.randn(8, 10),
            "metadata": "some_string",  # non-tensor field
        }
        batch = _handle_distributed_batch_return(
            dict_result,
            distributed_batch_target_key=None,
            engine=engine,
        )

        # Should return DistributedBatchMemory with metadata
        assert isinstance(batch, DistributedBatchMemory)
        assert batch.metadata is not None
        assert batch.metadata.global_step == 100
        assert batch.metadata.total_batch_size == 8
        assert "logits" in batch.metadata.shards[0].fields
        assert "values" in batch.metadata.shards[0].fields

    def test_handle_non_tensor_return(self):
        """Test that non-tensor returns are passed through."""
        from areal.scheduler.rpc.rpc_server import _handle_distributed_batch_return

        class MockEngine:
            def get_version(self):
                return 0

        engine = MockEngine()

        # Test non-tensor returns
        int_result = 42
        result = _handle_distributed_batch_return(int_result, None, engine)
        assert result == 42

        str_result = "hello"
        result = _handle_distributed_batch_return(str_result, None, engine)
        assert result == "hello"

        dict_result = {"loss": 0.5, "accuracy": 0.9}  # no tensors
        result = _handle_distributed_batch_return(dict_result, None, engine)
        assert result == dict_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
