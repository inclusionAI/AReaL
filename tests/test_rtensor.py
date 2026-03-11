"""Integration tests for RTensor with RPC server."""

import subprocess
import sys
import time
import uuid

import orjson
import pytest
import requests
import torch

from areal.infra.rpc.rtensor import (
    RTensor,
    TensorShardInfo,
    extract_layouts_from_container,
)
from areal.infra.rpc.serialization import serialize_value
from areal.infra.utils.proc import kill_process_tree
from areal.utils.network import find_free_ports


@pytest.fixture(scope="module")
def rpc_server():
    """Start RPC server for integration tests."""
    RPC_SERVER_PORT = find_free_ports(1)[0]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "areal.infra.rpc.rpc_server",
            "--host",
            "localhost",
            "--port",
            str(RPC_SERVER_PORT),
            "--experiment-name",
            "test-rtensor",
            "--trial-name",
            "trial0",
            "--role",
            "master",
            "--worker-index",
            "0",
        ],
        stdout=sys.stdout,
        stderr=sys.stdout,
    )

    # Wait for server to be ready
    max_attempts = 60
    for _ in range(max_attempts):
        try:
            resp = requests.get(f"http://localhost:{RPC_SERVER_PORT}/health", timeout=1)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
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
            shard=TensorShardInfo(
                shard_id=shard_id,
                node_addr=rpc_server,
                size=tensor.shape[0],
                seqlens=[int(tensor.shape[0])],
            ),
            data=tensor.to("meta"),
        )

        # Verify RTensor structure
        assert rtensor.shard.shard_id == shard_id
        assert rtensor.shard.node_addr == rpc_server
        assert rtensor.shard.size == tensor.shape[0]

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
                shard=TensorShardInfo(
                    shard_id=shard_id1,
                    node_addr=rpc_server,
                    size=tensor1.shape[0],
                    seqlens=[int(tensor1.shape[0])],
                ),
                data=torch.empty(tensor1.shape, device="meta"),
            ),
            "metadata": {"count": 3},
            "values": RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_id2,
                    node_addr=rpc_server,
                    size=tensor2.shape[0],
                    seqlens=[int(tensor2.shape[0])],
                ),
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
            shard=TensorShardInfo(
                shard_id=shard_id,
                node_addr=rpc_server,
                size=4,
                seqlens=[10, 10, 10, 10],  # Match sequence length
            ),
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
        from areal.infra.rpc.rtensor import fetch

        actual_shard_id = remotized["logits"].shard.shard_id
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


class TestRTensorExtractLayout:
    """Test layout extraction from various input structures."""

    def test_extract_layout_from_attention_mask(self):
        """Verify seqlens extracted correctly from attention_mask."""
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
        batch = {"attention_mask": attention_mask, "input_ids": torch.randn(3, 4)}
        node_addr = "localhost:8080"

        layout = RTensor.extract_layout(batch, layouts={}, node_addr=node_addr)

        assert isinstance(layout, RTensor)
        assert layout.shard.size == 3
        assert layout.shard.seqlens == [3, 2, 4]
        assert layout.shard.node_addr == node_addr

    def test_extract_layout_missing_attention_mask(self):
        """Returns None when attention_mask missing."""
        batch = {"input_ids": torch.randn(3, 4)}

        layout = RTensor.extract_layout(batch, node_addr="localhost:8080")

        assert layout is None

    def test_extract_layout_non_dict_batch(self):
        """Returns None for non-dict, non-tensor input."""
        batch = [torch.randn(3, 4), torch.randn(2, 5)]

        layout = RTensor.extract_layout(batch, node_addr="localhost:8080")

        assert layout is None

    def test_extract_layout_with_existing_rtensor(self):
        """Returns existing RTensor from layouts."""
        existing_rtensor = RTensor(
            shard=TensorShardInfo(
                shard_id=str(uuid.uuid4()),
                node_addr="node1",
                size=5,
                seqlens=[10, 15, 20, 12, 8],
            ),
            data=torch.empty(5, 20, device="meta"),
        )
        layouts = {"input": existing_rtensor}
        batch = {"output": torch.randn(5, 10)}

        layout = RTensor.extract_layout(
            batch, layouts=layouts, node_addr="localhost:8080"
        )

        assert layout is existing_rtensor

    def test_extract_layout_standalone_tensor(self):
        """Verify layout created from standalone tensor shape."""
        tensor = torch.randn(3, 10)
        layout = RTensor.extract_layout(tensor, node_addr="localhost:8080")
        assert isinstance(layout, RTensor)
        assert layout.shard.size == 3
        assert layout.shard.seqlens == [10, 10, 10]

    def test_extract_layout_standalone_1d_tensor(self):
        """Verify 1D tensor gets seqlens=[1]."""
        tensor = torch.randn(5)
        layout = RTensor.extract_layout(tensor, node_addr="localhost:8080")
        assert isinstance(layout, RTensor)
        assert layout.shard.size == 5
        assert layout.shard.seqlens == [1, 1, 1, 1, 1]

    def test_extract_layout_float_dict_returns_none(self):
        """Dict without attention_mask (e.g., stats dict) returns None."""
        obj = {"lr": 0.001, "grad_norm": 1.5}
        layout = RTensor.extract_layout(obj, node_addr="localhost:8080")
        assert layout is None


class TestRTensorErrorHandling:
    """Test error handling for network and storage failures."""

    def test_to_local_with_missing_shard(self, rpc_server):
        """RuntimeError on HTTP 404."""
        rtensor = RTensor(
            shard=TensorShardInfo(
                shard_id="nonexistent-shard-id",
                node_addr=rpc_server,
                size=3,
                seqlens=[10, 15, 12],
            ),
            data=torch.empty(3, 20, device="meta"),
        )

        with pytest.raises(RuntimeError, match="Failed to fetch shard"):
            rtensor.to_local()

    def test_to_local_with_server_error(self, rpc_server):
        """RuntimeError on deleted shard."""
        from areal.infra.rpc.rtensor import remove, store

        tensor = torch.randn(2, 5).cpu()
        shard_id = str(uuid.uuid4())
        store(shard_id, tensor)

        rtensor = RTensor(
            shard=TensorShardInfo(
                shard_id=shard_id, node_addr=rpc_server, size=2, seqlens=[8, 12]
            ),
            data=torch.empty(2, 5, device="meta"),
        )

        remove(shard_id)

        with pytest.raises(RuntimeError):
            rtensor.to_local()


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
                shard=TensorShardInfo(
                    shard_id=shard_id,
                    node_addr=rpc_server,
                    size=5,
                    seqlens=[40],
                ),
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
                shard=TensorShardInfo(
                    shard_id=shard_id1,
                    node_addr=rpc_server,
                    size=2,
                    seqlens=[10, 8],
                ),
                data=torch.empty(2, 5, 16, device="meta"),
            ),
            "decoder": RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_id2,
                    node_addr=rpc_server,
                    size=3,
                    seqlens=[15, 12, 10],
                ),
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

    def test_remotize_with_none_values(self):
        """None preserved in structures."""
        layout = RTensor(
            shard=TensorShardInfo(
                shard_id=str(uuid.uuid4()),
                node_addr="node1",
                size=4,
                seqlens=[20],
            ),
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
                shard=TensorShardInfo(
                    shard_id=shard_ids[0],
                    node_addr=rpc_server,
                    size=3,
                    seqlens=[15],
                ),
                data=torch.empty(3, 5, device="meta"),
            ),
            "batch2": {
                "inner": RTensor(
                    shard=TensorShardInfo(
                        shard_id=shard_ids[1],
                        node_addr=rpc_server,
                        size=2,
                        seqlens=[8],
                    ),
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
            shard=TensorShardInfo(
                shard_id=shard_id,
                node_addr=rpc_server,
                size=4,
                seqlens=[20],
            ),
            data=torch.empty(4, 6, device="meta"),
        )

        localized = rtensor.to_local()
        assert torch.allclose(localized, tensor)

        # Verify shard still on server (not auto-deleted)
        resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
        assert resp.status_code == 200


class TestAutoRemotize:
    """Test auto_remotize method with various input types."""

    def test_auto_remotize_list_of_dicts(self, rpc_server):
        """Test remotizing list of dicts with different attention masks."""
        # Create two trajectory dicts with different seqlens
        traj1 = {
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "input_ids": torch.randn(2, 4),
            "logits": torch.randn(2, 4).cpu(),
        }
        traj2 = {
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
            ),
            "input_ids": torch.randn(3, 5),
            "logits": torch.randn(3, 5).cpu(),
        }

        result = RTensor.auto_remotize([traj1, traj2], node_addr=rpc_server)

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)
        assert isinstance(result[0]["logits"], RTensor)
        assert isinstance(result[1]["logits"], RTensor)
        # Verify different shard_ids (per-trajectory isolation)
        assert result[0]["logits"].shard.shard_id != result[1]["logits"].shard.shard_id
        # Verify seqlens match attention masks
        assert result[0]["logits"].shard.seqlens == [3, 2]
        assert result[1]["logits"].shard.seqlens == [4, 3, 2]

    def test_auto_remotize_list_of_tensors(self, rpc_server):
        """Test remotizing list of standalone tensors."""
        tensors = [torch.randn(2, 5).cpu(), torch.randn(3, 7).cpu()]

        result = RTensor.auto_remotize(tensors, node_addr=rpc_server)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, RTensor) for r in result)
        assert result[0].shard.size == 2
        assert result[0].shard.seqlens == [5, 5]
        assert result[1].shard.size == 3
        assert result[1].shard.seqlens == [7, 7, 7]

    def test_auto_remotize_list_with_none(self, rpc_server):
        """Test remotizing list with None values interspersed."""
        traj_dict = {
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "logits": torch.randn(1, 4).cpu(),
        }

        result = RTensor.auto_remotize(
            [traj_dict, None, traj_dict], node_addr=rpc_server
        )

        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], dict)
        assert result[1] is None
        assert isinstance(result[2], dict)
        assert isinstance(result[0]["logits"], RTensor)
        assert isinstance(result[2]["logits"], RTensor)

    def test_auto_remotize_single_dict(self, rpc_server):
        """Test remotizing single dict (not wrapped in list)."""
        traj_dict = {
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "logits": torch.randn(1, 4).cpu(),
        }

        result = RTensor.auto_remotize(traj_dict, node_addr=rpc_server)

        assert isinstance(result, dict)
        assert isinstance(result["logits"], RTensor)
        assert result["logits"].shard.seqlens == [3]

    def test_auto_remotize_standalone_tensor(self, rpc_server):
        """Test remotizing standalone tensor (not in dict or list)."""
        tensor = torch.randn(2, 5).cpu()

        result = RTensor.auto_remotize(tensor, node_addr=rpc_server)

        assert isinstance(result, RTensor)
        assert result.shard.size == 2
        assert result.shard.seqlens == [5, 5]

    def test_auto_remotize_none(self):
        """Test that None input returns None."""
        result = RTensor.auto_remotize(None, node_addr="localhost:8080")
        assert result is None

    def test_auto_remotize_scalar(self):
        """Test that scalar values pass through unchanged."""
        result_int = RTensor.auto_remotize(42, node_addr="localhost:8080")
        assert result_int == 42

        result_bool = RTensor.auto_remotize(True, node_addr="localhost:8080")
        assert result_bool is True

        result_float = RTensor.auto_remotize(3.14, node_addr="localhost:8080")
        assert result_float == 3.14

    def test_auto_remotize_float_dict(self):
        """Test that dict without attention_mask returns unchanged."""
        obj = {"lr": 0.001, "grad_norm": 1.5}
        result = RTensor.auto_remotize(obj, node_addr="localhost:8080")
        assert result == obj

    def test_auto_remotize_empty_list(self):
        """Test that empty list returns empty list."""
        result = RTensor.auto_remotize([], node_addr="localhost:8080")
        assert result == []

    def test_auto_remotize_roundtrip(self, rpc_server):
        """Test remotize->localize roundtrip for trajectory dict."""
        original_traj = {
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "logits": torch.randn(2, 4).cpu(),
        }
        original_logits = original_traj["logits"].clone()

        # Remotize
        remotized = RTensor.auto_remotize(original_traj, node_addr=rpc_server)

        # Store tensors on server using the NEW shard_ids created by auto_remotize
        from areal.infra.rpc.rtensor import fetch

        for key in ["attention_mask", "logits"]:
            if isinstance(remotized[key], RTensor):
                actual_shard_id = remotized[key].shard.shard_id
                tensor_from_local = fetch(actual_shard_id)
                serialized = serialize_value(tensor_from_local)
                resp = requests.put(
                    f"http://{rpc_server}/data/{actual_shard_id}",
                    data=orjson.dumps(serialized),
                )
                assert resp.status_code == 200

        # Localize (fetches remote tensors)
        localized = RTensor.localize(remotized)

        assert isinstance(localized["logits"], torch.Tensor)
        assert torch.allclose(localized["logits"], original_logits, atol=1e-5)

    def test_extract_layouts_from_container_traj_list(self, rpc_server):
        """Test that extract_layouts_from_container returns list[RTensor] for traj-list."""
        # Arrange: Create traj-list input like compute_logp receives
        rt1 = RTensor(
            shard=TensorShardInfo(
                size=2, seqlens=[5, 12], shard_id="s1", node_addr=rpc_server
            ),
            data=torch.empty(2, 20, device="meta"),
        )
        rt2 = RTensor(
            shard=TensorShardInfo(
                size=1, seqlens=[8], shard_id="s2", node_addr=rpc_server
            ),
            data=torch.empty(1, 20, device="meta"),
        )
        traj_list = [
            {"input_ids": rt1, "attention_mask": rt1},
            {"input_ids": rt2, "attention_mask": rt2},
        ]

        # Act
        result = extract_layouts_from_container(traj_list)

        # Assert
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], RTensor)
        assert isinstance(result[1], RTensor)
        assert result[0].shard.seqlens == [5, 12]
        assert result[1].shard.seqlens == [8]

    def test_extract_layouts_from_container_no_rtensor(self):
        """Test that extract_layouts_from_container returns None for non-traj inputs."""
        # Scalars
        assert extract_layouts_from_container(42) is None
        # Dict of Tensor (NOT RTensor) — not a container of dicts
        assert extract_layouts_from_container({"x": torch.zeros(3)}) is None
        # String
        assert extract_layouts_from_container("hello") is None

    def test_extract_layouts_from_container_empty_list(self):
        """Test that empty list input returns None."""
        assert extract_layouts_from_container([]) is None

    def test_auto_remotize_list_tensor_with_layouts(self, rpc_server):
        """Test that auto_remotize with layouts propagates seqlens from input RTensors.

        This is the core test: compute_logp returns list[Tensor], and layouts
        from the input traj-list should give each output RTensor the correct seqlens
        (valid token counts), NOT tensor shape.
        """
        # Arrange: output tensors (like compute_logp results)
        tensor1 = torch.randn(2, 20).cpu()  # shape[1]=20, but valid tokens are [5, 12]
        tensor2 = torch.randn(1, 20).cpu()  # shape[1]=20, but valid tokens are [8]

        # Input layouts: RTensors with correct seqlens from attention_mask
        layout1 = RTensor(
            shard=TensorShardInfo(
                size=2, seqlens=[5, 12], shard_id="l1", node_addr=rpc_server
            ),
            data=torch.empty(2, 20, device="meta"),
        )
        layout2 = RTensor(
            shard=TensorShardInfo(
                size=1, seqlens=[8], shard_id="l2", node_addr=rpc_server
            ),
            data=torch.empty(1, 20, device="meta"),
        )

        # Act: auto_remotize with layouts
        result = RTensor.auto_remotize(
            [tensor1, tensor2], node_addr=rpc_server, layouts=[layout1, layout2]
        )

        # Assert: output RTensors have seqlens from layouts, NOT from tensor shape
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], RTensor)
        assert isinstance(result[1], RTensor)
        # This is the CRITICAL assertion: seqlens should be [5, 12] not [20, 20]
        assert result[0].shard.seqlens == [5, 12]
        # This is the CRITICAL assertion: seqlens should be [8] not [20]
        assert result[1].shard.seqlens == [8]

    def test_auto_remotize_list_tensor_without_layouts(self, rpc_server):
        """Test that auto_remotize without layouts uses tensor shape (existing behavior)."""
        tensor1 = torch.randn(2, 20).cpu()
        tensor2 = torch.randn(1, 15).cpu()

        result = RTensor.auto_remotize([tensor1, tensor2], node_addr=rpc_server)

        assert isinstance(result, list)
        assert len(result) == 2
        # Without layouts: seqlens = [shape[1]] * shape[0]
        assert result[0].shard.seqlens == [20, 20]
        assert result[1].shard.seqlens == [15]

    def test_auto_remotize_layouts_length_mismatch(self, rpc_server):
        """Test that mismatched layout length falls back to shape-based seqlens."""
        tensor1 = torch.randn(2, 20).cpu()
        tensor2 = torch.randn(1, 15).cpu()

        # Layout has 3 items but obj has 2 — mismatch
        layout1 = RTensor(
            shard=TensorShardInfo(
                size=2, seqlens=[5, 12], shard_id="m1", node_addr=rpc_server
            ),
            data=torch.empty(2, 20, device="meta"),
        )
        layout2 = RTensor(
            shard=TensorShardInfo(
                size=1, seqlens=[8], shard_id="m2", node_addr=rpc_server
            ),
            data=torch.empty(1, 20, device="meta"),
        )
        layout3 = RTensor(
            shard=TensorShardInfo(
                size=1, seqlens=[3], shard_id="m3", node_addr=rpc_server
            ),
            data=torch.empty(1, 20, device="meta"),
        )

        # Act: layouts has 3 items, obj has 2 — should fall back
        result = RTensor.auto_remotize(
            [tensor1, tensor2],
            node_addr=rpc_server,
            layouts=[layout1, layout2, layout3],
        )

        # Assert: falls back to shape-based seqlens
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shard.seqlens == [20, 20]
        assert result[1].shard.seqlens == [15]


class TestTensorShardInfoDocumentation:
    """Tests verifying TensorShardInfo construction and field semantics."""

    def test_construction_with_all_fields(self):
        """TensorShardInfo can be constructed with all required fields."""
        from areal.infra.rpc.rtensor import TensorShardInfo

        shard = TensorShardInfo(
            size=4,
            seqlens=[10, 20, 15, 8],
            shard_id="test-shard-001",
            node_addr="localhost:8080",
        )
        assert shard.size == 4
        assert shard.seqlens == [10, 20, 15, 8]
        assert shard.shard_id == "test-shard-001"
        assert shard.node_addr == "localhost:8080"

    def test_seqlens_length_matches_size(self):
        """seqlens list length should equal size (one entry per sequence)."""
        from areal.infra.rpc.rtensor import TensorShardInfo

        size = 3
        seqlens = [100, 200, 150]
        shard = TensorShardInfo(
            size=size,
            seqlens=seqlens,
            shard_id="",
            node_addr="",
        )
        assert len(shard.seqlens) == shard.size

    def test_ray_backend_empty_node_addr(self):
        """Ray backend uses empty string for node_addr."""
        from areal.infra.rpc.rtensor import TensorShardInfo

        shard = TensorShardInfo(
            size=1,
            seqlens=[50],
            shard_id="",  # Will be filled by Ray ObjectRef
            node_addr="",  # Empty for Ray backend
        )
        assert shard.node_addr == ""

    def test_http_backend_node_addr(self):
        """HTTP backend uses host:port for node_addr."""
        from areal.infra.rpc.rtensor import TensorShardInfo

        shard = TensorShardInfo(
            size=2,
            seqlens=[100, 200],
            shard_id="some-uuid",
            node_addr="192.168.1.1:8080",
        )
        assert ":" in shard.node_addr
