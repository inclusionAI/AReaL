"""Integration tests for RTensor with RPC server."""

import subprocess
import sys
import time
import uuid

import orjson
import pytest
import requests
import torch

from areal.scheduler.rpc.rtensor import RTensor, TensorShardInfo
from areal.scheduler.rpc.serialization import serialize_value
from areal.utils.proc import kill_process_tree

RPC_SERVER_PORT = 8077


@pytest.fixture(scope="module")
def rpc_server():
    """Start RPC server for integration tests."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "areal.scheduler.rpc.rpc_server",
            "--host",
            "localhost",
            "--port",
            str(RPC_SERVER_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    max_attempts = 20
    for _ in range(max_attempts):
        try:
            resp = requests.get(f"http://localhost:{RPC_SERVER_PORT}/health", timeout=1)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("RPC server failed to start")

    yield f"localhost:{RPC_SERVER_PORT}"

    kill_process_tree(proc.pid)


class TestRTensorIntegration:
    """Integration tests using real RPC server."""

    def test_single_shard_storage_and_retrieval(self, rpc_server):
        """Test storing and retrieving a single tensor shard (InferenceEngine workflow)."""
        # Create tensor and shard ID
        tensor = torch.randn(5, 10).cpu()
        shard_id = str(uuid.uuid4())

        # Create RTensor manually
        rtensor = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=shard_id,
                    node_addr=rpc_server,
                    size=tensor.shape[0],
                    seqlens=[int(tensor.shape[0])],
                )
            ],
            data=tensor.to("meta"),
        )

        # Verify RTensor structure
        assert len(rtensor.shards) == 1
        assert rtensor.shards[0].shard_id == shard_id
        assert rtensor.shards[0].node_addr == rpc_server
        assert rtensor.shards[0].size == tensor.shape[0]

        # Store on server
        serialized_tensor = serialize_value(tensor)
        resp = requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized_tensor),
        )
        assert resp.status_code == 200

        # Retrieve via RTensor.to_local()
        localized = rtensor.to_local()

        assert isinstance(localized, torch.Tensor)
        assert localized.shape == tensor.shape
        assert torch.allclose(localized, tensor)

    def test_batched_shard_storage_and_retrieval(self, rpc_server):
        """Test batched tensor splitting and retrieval (TrainEngine workflow)."""
        # Create a batched tensor (total batch size = 3 + 5 + 2 = 10)
        batch_tensor = torch.randn(10, 8).cpu()

        # Create layout describing how batch should be split
        layout = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr=rpc_server,
                    size=3,
                    seqlens=[3],
                ),
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr=rpc_server,
                    size=5,
                    seqlens=[5],
                ),
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr=rpc_server,
                    size=2,
                    seqlens=[2],
                ),
            ],
            data=torch.empty(0, device="meta"),
        )

        # Create batched RTensor (stores shards locally via from_batched)
        rtensor = RTensor.from_batched(
            batch_tensor, layout=layout, node_addr=rpc_server
        )

        # Verify shards match layout
        assert len(rtensor.shards) == 3
        assert rtensor.shards[0].size == 3
        assert rtensor.shards[1].size == 5
        assert rtensor.shards[2].size == 2

        # Upload each shard to server (simulating distributed storage)
        for i, shard_info in enumerate(rtensor.shards):
            start = sum(s.size for s in rtensor.shards[:i])
            end = start + shard_info.size
            shard_tensor = batch_tensor[start:end]

            serialized = serialize_value(shard_tensor)
            resp = requests.put(
                f"http://{rpc_server}/data/{shard_info.shard_id}",
                data=orjson.dumps(serialized),
            )
            assert resp.status_code == 200

        # Retrieve and verify reconstruction
        reconstructed = rtensor.to_local()

        assert reconstructed.shape == batch_tensor.shape
        assert torch.allclose(reconstructed, batch_tensor)

    def test_localize_nested_structure(self, rpc_server):
        """Test localizing nested structures containing RTensors."""
        # Create tensors
        tensor1 = torch.randn(3, 4).cpu()
        tensor2 = torch.randn(2, 6).cpu()

        # Store on server
        shard_id1 = str(uuid.uuid4())
        shard_id2 = str(uuid.uuid4())

        for shard_id, tensor in [(shard_id1, tensor1), (shard_id2, tensor2)]:
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )

        # Create nested structure with RTensors
        nested = {
            "logits": RTensor(
                shards=[
                    TensorShardInfo(
                        shard_id=shard_id1,
                        node_addr=rpc_server,
                        size=tensor1.shape[0],
                        seqlens=[int(tensor1.shape[0])],
                    )
                ],
                data=torch.empty(tensor1.shape, device="meta"),
            ),
            "metadata": {"count": 3},
            "values": RTensor(
                shards=[
                    TensorShardInfo(
                        shard_id=shard_id2,
                        node_addr=rpc_server,
                        size=tensor2.shape[0],
                        seqlens=[int(tensor2.shape[0])],
                    )
                ],
                data=torch.empty(tensor2.shape, device="meta"),
            ),
        }

        # Localize entire structure
        localized = RTensor.localize(nested)

        # Verify structure
        assert isinstance(localized["logits"], torch.Tensor)
        assert isinstance(localized["values"], torch.Tensor)
        assert localized["metadata"]["count"] == 3
        assert torch.allclose(localized["logits"], tensor1)
        assert torch.allclose(localized["values"], tensor2)

    def test_remotize_and_localize_roundtrip(self, rpc_server):
        """Test remotize with pre-existing layout and localize roundtrip."""
        # Create a layout RTensor with actual shard IDs
        shard_id = str(uuid.uuid4())
        layout = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=shard_id,
                    node_addr=rpc_server,
                    size=4,
                    seqlens=[10, 10, 10, 10],  # Match sequence length
                )
            ],
            data=None,
        )

        # Simulate output with tensors
        output = {
            "logits": torch.randn(4, 10).cpu(),
            "score": 0.95,
        }

        # remotize using existing layout (creates new shard IDs)
        remotized = RTensor.remotize(
            output,
            layout=layout,
            node_addr=rpc_server,
        )

        # Verify RTensor was created
        assert isinstance(remotized["logits"], RTensor)
        assert remotized["score"] == 0.95

        # Store tensor on server using the NEW shard_id created by remotize
        from areal.scheduler.rpc.rtensor import fetch

        actual_shard_id = remotized["logits"].shards[0].shard_id
        tensor_from_local = fetch(actual_shard_id)
        serialized = serialize_value(tensor_from_local)
        resp = requests.put(
            f"http://{rpc_server}/data/{actual_shard_id}",
            data=orjson.dumps(serialized),
        )
        assert resp.status_code == 200

        # Localize (fetches remote tensor)
        localized = RTensor.localize(remotized)

        assert isinstance(localized["logits"], torch.Tensor)
        assert torch.allclose(localized["logits"], output["logits"])
        assert localized["score"] == 0.95

    def test_clear_batch_data(self, rpc_server):
        """Test clearing stored tensor shards."""
        # Store some tensors
        shard_ids = []
        for i in range(3):
            tensor = torch.randn(2, 3).cpu()
            shard_id = str(uuid.uuid4())
            shard_ids.append(shard_id)

            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )

        # Clear the shards
        resp = requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": shard_ids},
        )
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "ok"
        assert result["cleared_count"] == 3

        # Verify shards are gone
        for shard_id in shard_ids:
            resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
            assert resp.status_code == 404

    def test_sequence_padding_on_concatenation(self, rpc_server):
        """Test that RTensor handles variable sequence lengths with padding."""
        # Create tensors with different sequence lengths
        tensor1 = torch.randn(2, 5, 8).cpu()  # batch=2, seq=5
        tensor2 = torch.randn(3, 10, 8).cpu()  # batch=3, seq=10
        tensor3 = torch.randn(1, 7, 8).cpu()  # batch=1, seq=7

        # Store on server
        shards = []
        for i, tensor in enumerate([tensor1, tensor2, tensor3]):
            shard_id = str(uuid.uuid4())
            shards.append(
                TensorShardInfo(
                    shard_id=shard_id,
                    node_addr=rpc_server,
                    size=tensor.shape[0],
                    seqlens=[int(tensor.shape[0])] * tensor.shape[1],  # total tokens
                )
            )

            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )

        # Create RTensor and localize (should pad to max_len=10)
        rtensor = RTensor(
            shards=shards,
            data=torch.empty(6, 10, 8, device="meta"),  # total batch=6, max_seq=10
        )
        result = rtensor.to_local()

        # Verify shape and padding
        assert result.shape == (6, 10, 8)
        assert torch.allclose(result[0:2, 0:5], tensor1)  # First 2 items, first 5 seq
        assert torch.allclose(result[2:5, 0:10], tensor2)  # Next 3 items, all 10 seq
        assert torch.allclose(result[5:6, 0:7], tensor3)  # Last item, first 7 seq
        # Verify padding is zeros
        assert torch.allclose(result[0:2, 5:10], torch.zeros(2, 5, 8))
        assert torch.allclose(result[5:6, 7:10], torch.zeros(1, 3, 8))

    def test_dispatch_merge_with_mixed_types(self, rpc_server):
        """RTensor + scalars + lists dispatch/merge."""
        tensor = torch.arange(60).reshape(6, 10).float().cpu()
        layout = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id="", node_addr=rpc_server, size=2, seqlens=[10, 15]
                ),
                TensorShardInfo(
                    shard_id="", node_addr=rpc_server, size=4, seqlens=[12, 18, 20, 14]
                ),
            ],
            data=torch.empty(0, device="meta"),
        )
        rtensor = RTensor.from_batched(tensor, layout=layout, node_addr=rpc_server)

        # Upload shards
        for i, shard_info in enumerate(rtensor.shards):
            start = sum(s.size for s in rtensor.shards[:i])
            end = start + shard_info.size
            shard_tensor = tensor[start:end]
            serialized = serialize_value(shard_tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_info.shard_id}",
                data=orjson.dumps(serialized),
            )

        batch = {"rtensor": rtensor, "scalar": 0.5, "list": [1, 2, 3]}
        group_indices = [[0], [1]]
        splits, _ = RTensor.data_parallel_dispatch(
            batch, dp_size=2, group_indices=group_indices
        )

        # Verify splits
        assert len(splits) == 2
        assert splits[0]["scalar"] == 0.5
        assert splits[1]["scalar"] == 0.5
        assert splits[0]["list"] == [1, 2, 3]

        # Modify RTensor data in splits
        splits[0]["rtensor"].data = splits[0]["rtensor"].data + 1.0
        splits[1]["rtensor"].data = splits[1]["rtensor"].data + 2.0

        # Merge back
        merged = RTensor.data_parallel_merge(splits, group_indices)

        assert merged["scalar"] == 0.5
        assert merged["list"] == [1, 2, 3]
        assert isinstance(merged["rtensor"], RTensor)

    def test_end_to_end_remotize_dispatch_merge(self, rpc_server):
        """Full workflow roundtrip."""
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
        batch = {
            "attention_mask": attention_mask,
            "logits": torch.randn(3, 4, 10).cpu(),
            "values": torch.randn(3, 4).cpu(),
        }

        # Extract layout
        layout = RTensor.extract_layout(batch, layouts={}, node_addr=rpc_server)

        # remotize
        remotized = RTensor.remotize(batch, layout=layout, node_addr=rpc_server)

        # Store shards on server
        for key, value in remotized.items():
            if isinstance(value, RTensor):
                for shard in value.shards:
                    # Fetch from local storage and upload to server
                    from areal.scheduler.rpc.rtensor import fetch

                    tensor = fetch(shard.shard_id)
                    serialized = serialize_value(tensor)
                    requests.put(
                        f"http://{rpc_server}/data/{shard.shard_id}",
                        data=orjson.dumps(serialized),
                    )

        # Dispatch
        group_indices = [[0]]
        splits, _ = RTensor.data_parallel_dispatch(
            remotized, dp_size=1, group_indices=group_indices
        )

        # Merge (single worker, so just return)
        merged = RTensor.data_parallel_merge([splits[0]], group_indices)

        # Verify roundtrip
        merged_localized = RTensor.localize(merged)
        assert torch.allclose(merged_localized["logits"], batch["logits"])
        assert torch.allclose(merged_localized["values"], batch["values"])

    def test_from_batched_integration(self, rpc_server):
        """from_batched with actual storage."""
        batch_tensor = torch.arange(100).reshape(10, 10).float().cpu()
        layout = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id="", node_addr=rpc_server, size=4, seqlens=[40]
                ),
                TensorShardInfo(
                    shard_id="", node_addr=rpc_server, size=6, seqlens=[60]
                ),
            ],
            data=torch.empty(0, device="meta"),
        )

        rtensor = RTensor.from_batched(
            batch_tensor, layout=layout, node_addr=rpc_server
        )

        # Upload shards to server
        from areal.scheduler.rpc.rtensor import fetch

        for i, shard_info in enumerate(rtensor.shards):
            tensor = fetch(shard_info.shard_id)
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_info.shard_id}",
                data=orjson.dumps(serialized),
            )

        # Retrieve via to_local
        reconstructed = rtensor.to_local()

        assert torch.allclose(reconstructed, batch_tensor)


class TestDataParallelOps:
    """Tests for data parallel dispatch and merge operations."""

    def test_data_parallel_dispatch_rtensor(self):
        """Test dispatching RTensor across DP groups."""
        # Create RTensor with 3 shards
        batch_tensor = torch.randn(10, 8).cpu()
        rtensor = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node1",
                    size=3,
                    seqlens=[15],
                ),
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node2",
                    size=5,
                    seqlens=[25],
                ),
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node3",
                    size=2,
                    seqlens=[10],
                ),
            ],
            data=batch_tensor,
        )

        # Dispatch to 2 DP groups with custom group indices
        group_indices = [[0, 2], [1]]  # Group 0: shards 0,2; Group 1: shard 1
        split_rtensors, _ = RTensor.data_parallel_dispatch(
            rtensor, dp_size=2, group_indices=group_indices
        )

        # Verify split
        assert len(split_rtensors) == 2
        assert len(split_rtensors[0].shards) == 2  # Group 0 has 2 shards
        assert len(split_rtensors[1].shards) == 1  # Group 1 has 1 shard
        assert split_rtensors[0].data.shape == (5, 8)  # 3+2 rows
        assert split_rtensors[1].data.shape == (5, 8)  # 5 rows

    def test_data_parallel_dispatch_nested_structures(self):
        """Test dispatching nested structures with RTensors."""
        # Create nested structure
        rtensor = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node1",
                    size=4,
                    seqlens=[20],
                ),
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node2",
                    size=6,
                    seqlens=[30],
                ),
            ],
            data=torch.randn(10, 5).cpu(),
        )

        nested = {
            "rtensor": rtensor,
            "scalar": 42,
            "list": [rtensor, 99],
        }

        # Dispatch to 2 DP groups
        group_indices = [[0], [1]]
        results, _ = RTensor.data_parallel_dispatch(
            nested, dp_size=2, group_indices=group_indices
        )

        # Verify structure
        assert len(results) == 2
        assert isinstance(results[0]["rtensor"], RTensor)
        assert results[0]["scalar"] == 42  # Scalars replicated
        assert results[1]["scalar"] == 42
        assert len(results[0]["list"]) == 2
        assert isinstance(results[0]["list"][0], RTensor)
        assert results[0]["list"][1] == 99  # Scalar in list replicated

    def test_data_parallel_merge_rtensor(self):
        """Test merging RTensors from DP groups."""
        # Create two RTensors from different DP groups
        rtensor1 = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node1",
                    size=3,
                    seqlens=[15],
                )
            ],
            data=torch.randn(3, 8).cpu(),
        )
        rtensor2 = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node2",
                    size=5,
                    seqlens=[25],
                )
            ],
            data=torch.randn(5, 8).cpu(),
        )

        # Merge
        group_indices = [[0], [1]]
        merged = RTensor.data_parallel_merge([rtensor1, rtensor2], group_indices)

        # Verify merge
        assert isinstance(merged, RTensor)
        assert len(merged.shards) == 2
        assert merged.data.shape == (8, 8)  # 3+5 rows

    def test_data_parallel_merge_nested_structures(self):
        """Test merging nested structures from DP groups."""
        # Create results from 2 DP groups
        results = [
            {
                "rtensor": RTensor(
                    shards=[
                        TensorShardInfo(
                            shard_id=str(uuid.uuid4()),
                            node_addr="node1",
                            size=3,
                            seqlens=[15],
                        )
                    ],
                    data=torch.randn(3, 4).cpu(),
                ),
                "scalar": 0.5,
            },
            {
                "rtensor": RTensor(
                    shards=[
                        TensorShardInfo(
                            shard_id=str(uuid.uuid4()),
                            node_addr="node2",
                            size=2,
                            seqlens=[10],
                        )
                    ],
                    data=torch.randn(2, 4).cpu(),
                ),
                "scalar": 0.5,
            },
        ]

        # Merge
        group_indices = [[0], [1]]
        merged = RTensor.data_parallel_merge(results, group_indices)

        # Verify structure
        assert isinstance(merged, dict)
        assert isinstance(merged["rtensor"], RTensor)
        assert len(merged["rtensor"].shards) == 2
        assert merged["rtensor"].data.shape == (5, 4)  # 3+2 rows
        assert merged["scalar"] == 0.5  # First scalar returned

    def test_data_parallel_merge_rejects_raw_tensors(self):
        """Test that merge rejects raw tensors."""
        results = [torch.randn(3, 4), torch.randn(2, 4)]
        group_indices = [[0], [1]]

        # Should raise TypeError
        with pytest.raises(
            TypeError, match="Regular tensors not allowed in merge - only RTensors"
        ):
            RTensor.data_parallel_merge(results, group_indices)

    def test_data_parallel_merge_with_mismatched_group_indices(self):
        """Test merge order depends on group_indices."""
        rtensor1 = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node1",
                    size=3,
                    seqlens=[15],
                )
            ],
            data=torch.randn(3, 8).cpu(),
        )
        rtensor2 = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node2",
                    size=5,
                    seqlens=[25],
                )
            ],
            data=torch.randn(5, 8).cpu(),
        )

        # Merge with different group_indices affects order
        merged1 = RTensor.data_parallel_merge(
            [rtensor1, rtensor2], group_indices=[[0], [1]]
        )
        merged2 = RTensor.data_parallel_merge(
            [rtensor1, rtensor2], group_indices=[[1], [0]]
        )

        # Different group_indices produce different shard orders
        assert merged1.shards[0].size == 3
        assert merged2.shards[0].size == 5

    def test_data_parallel_dispatch_kwargs_only(self):
        """Test dispatch with only kwargs (no args)."""
        rtensor = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node1",
                    size=4,
                    seqlens=[20],
                ),
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node2",
                    size=6,
                    seqlens=[30],
                ),
            ],
            data=torch.randn(10, 5).cpu(),
        )

        nested = {"batch": rtensor, "lr": 0.001}
        group_indices = [[0], [1]]
        results, _ = RTensor.data_parallel_dispatch(
            nested, dp_size=2, group_indices=group_indices
        )

        assert len(results) == 2
        assert results[0]["lr"] == 0.001
        assert results[1]["lr"] == 0.001
        assert isinstance(results[0]["batch"], RTensor)
        assert isinstance(results[1]["batch"], RTensor)


class TestRTensorExtractLayout:
    """Test layout extraction from various input structures."""

    def test_extract_layout_from_attention_mask(self):
        """Verify seqlens extracted correctly from attention_mask."""
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
        batch = {"attention_mask": attention_mask, "input_ids": torch.randn(3, 4)}
        node_addr = "localhost:8080"

        layout = RTensor.extract_layout(batch, layouts={}, node_addr=node_addr)

        assert isinstance(layout, RTensor)
        assert len(layout.shards) == 1
        assert layout.shards[0].size == 3
        assert layout.shards[0].seqlens == [3, 2, 4]
        assert layout.shards[0].node_addr == node_addr

    def test_extract_layout_missing_attention_mask(self):
        """RuntimeError when attention_mask missing."""
        batch = {"input_ids": torch.randn(3, 4)}

        with pytest.raises(RuntimeError, match="attention_mask.*not found"):
            RTensor.extract_layout(batch, layouts={}, node_addr="localhost:8080")

    def test_extract_layout_non_dict_batch(self):
        """RuntimeError for non-dict input without RTensor in layouts."""
        batch = [torch.randn(3, 4), torch.randn(2, 5)]

        with pytest.raises(RuntimeError, match="dict batch"):
            RTensor.extract_layout(batch, layouts={}, node_addr="localhost:8080")

    def test_extract_layout_with_existing_rtensor(self):
        """Returns existing RTensor from layouts."""
        existing_rtensor = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node1",
                    size=5,
                    seqlens=[10, 15, 20, 12, 8],
                )
            ],
            data=torch.empty(5, 20, device="meta"),
        )
        layouts = {"input": existing_rtensor}
        batch = {"output": torch.randn(5, 10)}

        layout = RTensor.extract_layout(
            batch, layouts=layouts, node_addr="localhost:8080"
        )

        assert layout is existing_rtensor


class TestRTensorFromBatched:
    """Test from_batched with various tensor shapes and truncation."""

    def test_from_batched_with_sequence_truncation(self):
        """Verify truncation to max seqlens."""
        from areal.scheduler.rpc.rtensor import fetch

        batch_tensor = torch.randn(4, 20, 128).cpu()
        layout = RTensor(
            shards=[
                TensorShardInfo(shard_id="", node_addr="node1", size=1, seqlens=[10]),
                TensorShardInfo(shard_id="", node_addr="node1", size=1, seqlens=[15]),
                TensorShardInfo(shard_id="", node_addr="node1", size=1, seqlens=[8]),
                TensorShardInfo(shard_id="", node_addr="node1", size=1, seqlens=[12]),
            ],
            data=torch.empty(0, device="meta"),
        )

        rtensor = RTensor.from_batched(batch_tensor, layout=layout, node_addr="node1")

        assert len(rtensor.shards) == 4
        shard0 = fetch(rtensor.shards[0].shard_id)
        shard1 = fetch(rtensor.shards[1].shard_id)
        shard2 = fetch(rtensor.shards[2].shard_id)
        shard3 = fetch(rtensor.shards[3].shard_id)

        assert shard0.shape == (1, 10, 128)
        assert shard1.shape == (1, 15, 128)
        assert shard2.shape == (1, 8, 128)
        assert shard3.shape == (1, 12, 128)

    def test_from_batched_4d_tensors(self):
        """Test 4D tensor handling with truncation."""
        from areal.scheduler.rpc.rtensor import _pad_cat_dim0, fetch

        batch_tensor = torch.randn(8, 10, 32, 32).cpu()
        # Seqlens matching the sequence dimension
        layout = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id="", node_addr="node1", size=3, seqlens=[10, 10, 10]
                ),
                TensorShardInfo(
                    shard_id="", node_addr="node1", size=5, seqlens=[10] * 5
                ),
            ],
            data=torch.empty(0, device="meta"),
        )

        rtensor = RTensor.from_batched(batch_tensor, layout=layout, node_addr="node1")

        # Fetch shards from local storage and reconstruct manually
        shards = [fetch(shard.shard_id) for shard in rtensor.shards]
        reconstructed = _pad_cat_dim0(shards)

        assert reconstructed.shape == batch_tensor.shape
        assert torch.allclose(reconstructed, batch_tensor)

    def test_from_batched_1d_tensors(self):
        """Test 1D tensors (no padding)."""
        from areal.scheduler.rpc.rtensor import fetch

        batch_tensor = torch.randn(100).cpu()
        layout = RTensor(
            shards=[
                TensorShardInfo(shard_id="", node_addr="node1", size=40, seqlens=[40]),
                TensorShardInfo(shard_id="", node_addr="node1", size=30, seqlens=[30]),
                TensorShardInfo(shard_id="", node_addr="node1", size=30, seqlens=[30]),
            ],
            data=torch.empty(0, device="meta"),
        )

        rtensor = RTensor.from_batched(batch_tensor, layout=layout, node_addr="node1")

        # Fetch shards from local storage and reconstruct
        shards = [fetch(shard.shard_id) for shard in rtensor.shards]
        reconstructed = torch.cat(shards)

        assert torch.allclose(reconstructed, batch_tensor)

    def test_from_batched_mismatched_layout_size(self):
        """ValueError on size mismatch."""
        batch_tensor = torch.randn(10, 5).cpu()
        layout = RTensor(
            shards=[
                TensorShardInfo(shard_id="", node_addr="node1", size=3, seqlens=[15]),
                TensorShardInfo(shard_id="", node_addr="node1", size=5, seqlens=[25]),
            ],
            data=torch.empty(0, device="meta"),
        )

        with pytest.raises(ValueError, match="does not match layout total size"):
            RTensor.from_batched(batch_tensor, layout=layout, node_addr="node1")


class TestRTensorErrorHandling:
    """Test error handling for network and storage failures."""

    def test_to_local_with_missing_shard(self, rpc_server):
        """RuntimeError on HTTP 404."""
        rtensor = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id="nonexistent-shard-id",
                    node_addr=rpc_server,
                    size=3,
                    seqlens=[10, 15, 12],
                )
            ],
            data=torch.empty(3, 20, device="meta"),
        )

        with pytest.raises(RuntimeError, match="Failed to fetch shard"):
            rtensor.to_local()

    def test_to_local_with_server_error(self, rpc_server):
        """RuntimeError on deleted shard."""
        from areal.scheduler.rpc.rtensor import remove, store

        tensor = torch.randn(2, 5).cpu()
        shard_id = str(uuid.uuid4())
        store(shard_id, tensor)

        rtensor = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=shard_id, node_addr=rpc_server, size=2, seqlens=[8, 12]
                )
            ],
            data=torch.empty(2, 5, device="meta"),
        )

        remove(shard_id)

        with pytest.raises(RuntimeError):
            rtensor.to_local()

    def test_split_tensor_size_mismatch(self):
        """ValueError on layout mismatch."""
        tensor = torch.randn(10, 5).cpu()
        layout = RTensor(
            shards=[
                TensorShardInfo(shard_id="", node_addr="node1", size=6, seqlens=[30]),
                TensorShardInfo(shard_id="", node_addr="node1", size=6, seqlens=[30]),
            ],
            data=torch.empty(0, device="meta"),
        )

        with pytest.raises(ValueError, match="does not match layout total size"):
            RTensor.split_tensor(tensor, layout)

    def test_from_batched_with_non_cpu_tensor(self):
        """ValueError for non-CPU, non-meta tensors."""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_tensor = torch.randn(5, 10, device="cuda")
        layout = RTensor(
            shards=[
                TensorShardInfo(shard_id="", node_addr="node1", size=5, seqlens=[25])
            ],
            data=torch.empty(0, device="meta"),
        )

        with pytest.raises(ValueError, match="CPU or meta device"):
            RTensor.from_batched(batch_tensor, layout=layout, node_addr="node1")


class TestRTensorFFDAllocation:
    """Test FFD allocation with realistic sequence length distributions."""

    def test_data_parallel_dispatch_with_ffd_realistic_seqlens(self):
        """Balanced allocation with realistic seqlens."""
        seqlens = [
            512,
            1024,
            256,
            2048,
            128,
            1536,
            768,
            384,
            896,
            640,
            1280,
            192,
            448,
            1792,
            320,
            960,
        ]
        shards = [
            TensorShardInfo(
                shard_id=str(uuid.uuid4()), node_addr="node1", size=1, seqlens=[sl]
            )
            for sl in seqlens
        ]
        batch_tensor = torch.randn(len(seqlens), 10).cpu()
        rtensor = RTensor(shards=shards, data=batch_tensor)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=4
        )

        # Verify all shards assigned
        assert len(group_indices) == 4
        all_indices = [idx for group in group_indices for idx in group]
        assert sorted(all_indices) == list(range(16))

        # Verify balanced allocation (within reasonable variance)
        group_totals = [sum(seqlens[i] for i in group) for group in group_indices]
        max_total = max(group_totals)
        min_total = min(group_totals)
        assert (max_total - min_total) / max_total < 0.3  # Within 30% variance

    def test_data_parallel_dispatch_with_imbalanced_seqlens(self):
        """Extreme imbalance handling."""
        seqlens = [4096, 128, 128, 128, 128, 128]
        shards = [
            TensorShardInfo(
                shard_id=str(uuid.uuid4()), node_addr="node1", size=1, seqlens=[sl]
            )
            for sl in seqlens
        ]
        batch_tensor = torch.randn(len(seqlens), 10).cpu()
        rtensor = RTensor(shards=shards, data=batch_tensor)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=3
        )

        # Large item likely isolated in one group
        group_totals = [sum(seqlens[i] for i in group) for group in group_indices]
        assert max(group_totals) >= 4096

    def test_data_parallel_dispatch_merge_roundtrip_with_ffd(self):
        """Order preserved after dispatch/merge roundtrip."""
        seqlens = [512, 256, 1024, 128, 768, 384]
        shards = [
            TensorShardInfo(
                shard_id=str(uuid.uuid4()), node_addr="node1", size=1, seqlens=[sl]
            )
            for sl in seqlens
        ]
        batch_tensor = (
            torch.arange(len(seqlens) * 10).reshape(len(seqlens), 10).float().cpu()
        )
        rtensor = RTensor(shards=shards, data=batch_tensor)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=3
        )
        merged = RTensor.data_parallel_merge(split_rtensors, group_indices)

        # Verify shard order preserved
        assert len(merged.shards) == len(seqlens)
        for i, shard in enumerate(merged.shards):
            assert shard.seqlens == [seqlens[i]]

        # Verify data preserved
        assert torch.allclose(merged.data, batch_tensor)


class TestRTensorConcurrency:
    """Test concurrent operations on storage."""

    def test_concurrent_storage_writes(self, rpc_server):
        """20 threads store different shards."""
        import threading

        shard_ids = [str(uuid.uuid4()) for _ in range(20)]
        tensors = [torch.randn(2, 3).cpu() for _ in range(20)]

        def store_shard(shard_id, tensor):
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )

        threads = [
            threading.Thread(target=store_shard, args=(sid, t))
            for sid, t in zip(shard_ids, tensors)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify all shards retrievable
        for shard_id in shard_ids:
            resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
            assert resp.status_code == 200

    def test_concurrent_storage_reads(self, rpc_server):
        """10 threads fetch same shard."""
        import threading

        tensor = torch.randn(5, 8).cpu()
        shard_id = str(uuid.uuid4())
        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}", data=orjson.dumps(serialized)
        )

        results = [None] * 10

        def fetch_shard(idx):
            rtensor = RTensor(
                shards=[
                    TensorShardInfo(
                        shard_id=shard_id,
                        node_addr=rpc_server,
                        size=5,
                        seqlens=[40],
                    )
                ],
                data=torch.empty(5, 8, device="meta"),
            )
            results[idx] = rtensor.to_local()

        threads = [threading.Thread(target=fetch_shard, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify all fetched tensors identical
        for result in results:
            assert torch.allclose(result, tensor)

    def test_concurrent_clear_operations(self, rpc_server):
        """3 threads clear overlapping shards."""
        import threading

        shard_ids = [str(uuid.uuid4()) for _ in range(10)]
        for shard_id in shard_ids:
            tensor = torch.randn(2, 2).cpu()
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}", data=orjson.dumps(serialized)
            )

        # Overlapping shard sets
        shard_sets = [
            shard_ids[:5],
            shard_ids[3:8],
            shard_ids[6:],
        ]

        def clear_shards(shard_list):
            requests.delete(
                f"http://{rpc_server}/data/clear",
                json={"shard_ids": shard_list},
            )

        threads = [threading.Thread(target=clear_shards, args=(s,)) for s in shard_sets]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify all shards deleted (no errors)
        for shard_id in shard_ids:
            resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
            assert resp.status_code == 404


class TestRTensorComplexPadding:
    """Test padding with complex tensor shapes."""

    def test_pad_cat_4d_tensors_variable_all_dims(self):
        """4D padding correctness."""
        from areal.scheduler.rpc.rtensor import _pad_cat_dim0

        tensors = [
            torch.randn(2, 5, 8, 8).cpu(),
            torch.randn(3, 10, 6, 10).cpu(),
            torch.randn(1, 7, 12, 6).cpu(),
        ]

        result = _pad_cat_dim0(tensors)

        assert result.shape == (6, 10, 12, 10)
        # Verify original data preserved
        assert torch.allclose(result[0:2, :5, :8, :8], tensors[0])
        assert torch.allclose(result[2:5, :10, :6, :10], tensors[1])
        assert torch.allclose(result[5:6, :7, :12, :6], tensors[2])
        # Verify padding is zeros
        assert torch.allclose(result[0:2, 5:, :, :], torch.zeros(2, 5, 12, 10))

    def test_pad_cat_dimension_mismatch(self):
        """ValueError on ndim mismatch."""
        from areal.scheduler.rpc.rtensor import _pad_cat_dim0

        tensors = [torch.randn(2, 5).cpu(), torch.randn(3, 5, 8).cpu()]

        with pytest.raises(ValueError, match="dimension mismatch"):
            _pad_cat_dim0(tensors)

    def test_localize_with_3d_nested_padding(self, rpc_server):
        """Nested structures with 3D tensors."""
        tensor1 = torch.randn(2, 5, 16).cpu()
        tensor2 = torch.randn(3, 8, 16).cpu()

        shard_id1 = str(uuid.uuid4())
        shard_id2 = str(uuid.uuid4())

        for shard_id, tensor in [(shard_id1, tensor1), (shard_id2, tensor2)]:
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )

        nested = {
            "encoder": RTensor(
                shards=[
                    TensorShardInfo(
                        shard_id=shard_id1,
                        node_addr=rpc_server,
                        size=2,
                        seqlens=[10, 8],
                    )
                ],
                data=torch.empty(2, 5, 16, device="meta"),
            ),
            "decoder": RTensor(
                shards=[
                    TensorShardInfo(
                        shard_id=shard_id2,
                        node_addr=rpc_server,
                        size=3,
                        seqlens=[15, 12, 10],
                    )
                ],
                data=torch.empty(3, 8, 16, device="meta"),
            ),
        }

        localized = RTensor.localize(nested)

        assert isinstance(localized["encoder"], torch.Tensor)
        assert isinstance(localized["decoder"], torch.Tensor)
        assert torch.allclose(localized["encoder"], tensor1)
        assert torch.allclose(localized["decoder"], tensor2)


class TestRTensorEdgeCases:
    """Test edge cases like empty batches and single-item batches."""

    def test_empty_batch_dispatch(self):
        """Empty shards list handling."""
        rtensor = RTensor(shards=[], data=torch.tensor([]).cpu())

        # Empty RTensor should raise RuntimeError in FFD allocation
        # This is expected behavior - can't allocate 0 items to 4 groups
        with pytest.raises(
            RuntimeError, match="Number of values.*smaller than min_groups"
        ):
            RTensor.data_parallel_dispatch(rtensor, dp_size=4)

    def test_single_item_batch(self):
        """Single shard dispatch with fewer groups."""
        rtensor = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node1",
                    size=5,
                    seqlens=[25],
                )
            ],
            data=torch.randn(5, 8).cpu(),
        )

        # Dispatch to single group works
        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=1
        )

        assert len(split_rtensors) == 1
        assert len(split_rtensors[0].shards) == 1
        assert split_rtensors[0].shards[0].size == 5

    def test_cat_with_empty_rtensor_list(self):
        """RTensor.cat([]) returns empty RTensor."""
        result = RTensor.cat([])

        assert isinstance(result, RTensor)
        assert len(result.shards) == 0

    def test_remotize_with_none_values(self):
        """None preserved in structures."""
        layout = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=str(uuid.uuid4()),
                    node_addr="node1",
                    size=4,
                    seqlens=[20],
                )
            ],
            data=torch.empty(4, 10, device="meta"),
        )
        obj = {"logits": torch.randn(4, 10).cpu(), "mask": None, "score": 0.95}

        remotized = RTensor.remotize(obj, layout=layout, node_addr="node1")

        assert remotized["mask"] is None
        assert remotized["score"] == 0.95
        assert isinstance(remotized["logits"], RTensor)


class TestRTensorMemoryCleanup:
    """Test memory cleanup and storage stats."""

    def test_storage_stats_accuracy(self, rpc_server):
        """Verify cleared_count and bytes."""
        shard_ids = []
        for i in range(5):
            tensor = torch.randn(10, 20).cpu()
            shard_id = str(uuid.uuid4())
            shard_ids.append(shard_id)
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}", data=orjson.dumps(serialized)
            )

        resp = requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": shard_ids},
        )
        result = resp.json()

        assert result["status"] == "ok"
        assert result["cleared_count"] == 5

    def test_clear_batches_nested_structure(self, rpc_server):
        """collect_shards on nested dict."""
        tensors = [torch.randn(3, 5).cpu(), torch.randn(2, 4).cpu()]
        shard_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        for shard_id, tensor in zip(shard_ids, tensors):
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}", data=orjson.dumps(serialized)
            )

        nested = {
            "batch1": RTensor(
                shards=[
                    TensorShardInfo(
                        shard_id=shard_ids[0],
                        node_addr=rpc_server,
                        size=3,
                        seqlens=[15],
                    )
                ],
                data=torch.empty(3, 5, device="meta"),
            ),
            "batch2": {
                "inner": RTensor(
                    shards=[
                        TensorShardInfo(
                            shard_id=shard_ids[1],
                            node_addr=rpc_server,
                            size=2,
                            seqlens=[8],
                        )
                    ],
                    data=torch.empty(2, 4, device="meta"),
                )
            },
        }

        shards_by_node = RTensor.collect_shards(nested)
        assert rpc_server in shards_by_node
        assert set(shards_by_node[rpc_server]) == set(shard_ids)

        # Clear all shards
        resp = requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": shard_ids},
        )
        assert resp.status_code == 200

        # Verify deletion
        for shard_id in shard_ids:
            resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
            assert resp.status_code == 404

    def test_storage_cleanup_after_localize(self, rpc_server):
        """Shards persist after fetch."""
        tensor = torch.randn(4, 6).cpu()
        shard_id = str(uuid.uuid4())
        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}", data=orjson.dumps(serialized)
        )

        rtensor = RTensor(
            shards=[
                TensorShardInfo(
                    shard_id=shard_id,
                    node_addr=rpc_server,
                    size=4,
                    seqlens=[20],
                )
            ],
            data=torch.empty(4, 6, device="meta"),
        )

        localized = rtensor.to_local()
        assert torch.allclose(localized, tensor)

        # Verify shard still on server (not auto-deleted)
        resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
        assert resp.status_code == 200
