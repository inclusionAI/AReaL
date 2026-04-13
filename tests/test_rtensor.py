"""Integration tests for RTensor with RPC server."""

import asyncio
import subprocess
import sys
import time
import uuid

import orjson
import pytest
import requests
import torch

from areal.infra.rpc.rtensor import (
    HttpRTensorBackend,
    RTensor,
    TensorShardInfo,
)
from areal.infra.rpc.serialization import deserialize_value, serialize_value
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
            ),
            data=tensor.to("meta"),
        )

        # Verify RTensor structure
        assert rtensor.shard.shard_id == shard_id
        assert rtensor.shard.node_addr == rpc_server
        assert rtensor.shape[0] == tensor.shape[0]

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
                ),
                data=torch.empty(tensor1.shape, device="meta"),
            ),
            "metadata": {"count": 3},
            "values": RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_id2,
                    node_addr=rpc_server,
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
        """Test remotize and localize roundtrip."""
        # Simulate output with tensors
        output = {
            "logits": torch.randn(4, 10).cpu(),
            "score": 0.95,
        }

        # remotize using new 2-arg signature
        remotized = RTensor.remotize(output, node_addr=rpc_server)

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

    def test_batch_shard_retrieval(self, rpc_server):
        """Retrieve multiple shards with one HTTP request."""
        tensors = [torch.randn(2, 3).cpu(), torch.randn(4, 5).cpu()]
        shard_ids = [str(uuid.uuid4()) for _ in tensors]

        for shard_id, tensor in zip(shard_ids, tensors):
            serialized = serialize_value(tensor)
            resp = requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )
            assert resp.status_code == 200

        resp = requests.post(
            f"http://{rpc_server}/data/batch",
            json={"shard_ids": shard_ids},
        )
        assert resp.status_code == 200
        serialized_batch = orjson.loads(resp.content)
        localized = deserialize_value(serialized_batch)
        assert len(localized) == len(tensors)
        for actual, expected in zip(localized, tensors):
            assert torch.allclose(actual, expected)

    def test_batch_shard_retrieval_reports_missing_shards(self, rpc_server):
        """Missing shards return a structured client error instead of a compatibility 404."""
        tensor = torch.randn(2, 3).cpu()
        present_shard_id = str(uuid.uuid4())
        missing_shard_id = str(uuid.uuid4())

        resp = requests.put(
            f"http://{rpc_server}/data/{present_shard_id}",
            data=orjson.dumps(serialize_value(tensor)),
        )
        assert resp.status_code == 200

        resp = requests.post(
            f"http://{rpc_server}/data/batch",
            json={"shard_ids": [present_shard_id, missing_shard_id]},
        )
        assert resp.status_code == 400
        payload = resp.json()
        assert payload["status"] == "error"
        assert payload["missing_shard_ids"] == [missing_shard_id]


class TestHttpRTensorBackendBatching:
    """Unit tests for HTTP batch fetching behavior."""

    def test_fetch_chunks_large_requests(self, monkeypatch):
        """Large same-node fetches are split into bounded batch requests."""
        backend = HttpRTensorBackend(max_shards_per_request=2)
        shards = [
            TensorShardInfo(shard_id=f"s{i}", node_addr="node-a") for i in range(5)
        ]
        requested_chunks = []

        class _FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        async def fake_fetch_shard_group(self, session, node_addr, grouped):
            requested_chunks.append(
                (node_addr, [shard.shard_id for _, shard in grouped])
            )
            return [torch.tensor([int(shard.shard_id[1:])]) for _, shard in grouped]

        monkeypatch.setattr(
            backend,
            "_create_session",
            lambda: _FakeSession(),
        )
        monkeypatch.setattr(
            backend,
            "_fetch_shard_group",
            fake_fetch_shard_group.__get__(backend, HttpRTensorBackend),
        )

        results = backend.fetch(shards)

        assert requested_chunks == [
            ("node-a", ["s0", "s1"]),
            ("node-a", ["s2", "s3"]),
            ("node-a", ["s4"]),
        ]
        assert [int(tensor.item()) for tensor in results] == [0, 1, 2, 3, 4]

    def test_fetch_shard_group_raises_on_missing_batch_endpoint(self):
        """404 on /data/batch surfaces as an error."""
        backend = HttpRTensorBackend()
        grouped = [
            (0, TensorShardInfo(shard_id="s0", node_addr="node-a")),
            (1, TensorShardInfo(shard_id="s1", node_addr="node-a")),
        ]

        class _FakeResponse:
            status = 404

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def text(self):
                return "missing endpoint"

        class _FakeSession:
            def post(self, url, json):
                assert url == "http://node-a/data/batch"
                assert json == {"shard_ids": ["s0", "s1"]}
                return _FakeResponse()

        with pytest.raises(
            RuntimeError,
            match="Failed to fetch shard batch from http://node-a/data/batch: 404 body=missing endpoint",
        ):
            asyncio.run(backend._fetch_shard_group(_FakeSession(), "node-a", grouped))


class TestRTensorErrorHandling:
    """Test error handling for network and storage failures."""

    def test_to_local_with_missing_shard(self, rpc_server):
        """RuntimeError on HTTP 404."""
        rtensor = RTensor(
            shard=TensorShardInfo(
                shard_id="nonexistent-shard-id",
                node_addr=rpc_server,
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
            shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
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
                ),
                data=torch.empty(2, 5, 16, device="meta"),
            ),
            "decoder": RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_id2,
                    node_addr=rpc_server,
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
        obj = {"logits": torch.randn(4, 10).cpu(), "mask": None, "score": 0.95}

        remotized = RTensor.remotize(obj, node_addr="node1")

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
                ),
                data=torch.empty(3, 5, device="meta"),
            ),
            "batch2": {
                "inner": RTensor(
                    shard=TensorShardInfo(
                        shard_id=shard_ids[1],
                        node_addr=rpc_server,
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
            ),
            data=torch.empty(4, 6, device="meta"),
        )

        localized = rtensor.to_local()
        assert torch.allclose(localized, tensor)

        # Verify shard still on server (not auto-deleted)
        resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
        assert resp.status_code == 200


class TestRemotize:
    """Test remotize method with various input types."""

    def test_remotize_list_of_dicts(self, rpc_server):
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

        result = RTensor.remotize([traj1, traj2], node_addr=rpc_server)

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)
        assert isinstance(result[0]["logits"], RTensor)
        assert isinstance(result[1]["logits"], RTensor)
        # Verify different shard_ids (per-trajectory isolation)
        assert result[0]["logits"].shard.shard_id != result[1]["logits"].shard.shard_id
        # Verify size matches batch dimension
        assert result[0]["logits"].shape[0] == 2
        assert result[1]["logits"].shape[0] == 3

    def test_remotize_list_of_tensors(self, rpc_server):
        """Test remotizing list of standalone tensors."""
        tensors = [torch.randn(2, 5).cpu(), torch.randn(3, 7).cpu()]

        result = RTensor.remotize(tensors, node_addr=rpc_server)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, RTensor) for r in result)
        assert result[0].shape[0] == 2
        assert result[0].data.shape == torch.Size([2, 5])
        assert result[1].shape[0] == 3
        assert result[1].data.shape == torch.Size([3, 7])

    def test_remotize_list_with_none(self, rpc_server):
        """Test remotizing list with None values interspersed."""
        traj_dict = {
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "logits": torch.randn(1, 4).cpu(),
        }

        result = RTensor.remotize([traj_dict, None, traj_dict], node_addr=rpc_server)

        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], dict)
        assert result[1] is None
        assert isinstance(result[2], dict)
        assert isinstance(result[0]["logits"], RTensor)
        assert isinstance(result[2]["logits"], RTensor)

    def test_remotize_single_dict(self, rpc_server):
        """Test remotizing single dict (not wrapped in list)."""
        traj_dict = {
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "logits": torch.randn(1, 4).cpu(),
        }

        result = RTensor.remotize(traj_dict, node_addr=rpc_server)

        assert isinstance(result, dict)
        assert isinstance(result["logits"], RTensor)
        assert result["logits"].shape[0] == 1

    def test_remotize_standalone_tensor(self, rpc_server):
        """Test remotizing standalone tensor (not in dict or list)."""
        tensor = torch.randn(2, 5).cpu()

        result = RTensor.remotize(tensor, node_addr=rpc_server)

        assert isinstance(result, RTensor)
        assert result.shape[0] == 2
        assert result.data.shape == torch.Size([2, 5])

    def test_remotize_none(self):
        """Test that None input returns None."""
        result = RTensor.remotize(None, node_addr="localhost:8080")
        assert result is None

    def test_remotize_scalar(self):
        """Test that scalar values pass through unchanged."""
        result_int = RTensor.remotize(42, node_addr="localhost:8080")
        assert result_int == 42

        result_bool = RTensor.remotize(True, node_addr="localhost:8080")
        assert result_bool is True

        result_float = RTensor.remotize(3.14, node_addr="localhost:8080")
        assert result_float == 3.14

    def test_remotize_float_dict(self):
        """Test that dict without tensors returns with values unchanged."""
        obj = {"lr": 0.001, "grad_norm": 1.5}
        result = RTensor.remotize(obj, node_addr="localhost:8080")
        assert result["lr"] == 0.001
        assert result["grad_norm"] == 1.5

    def test_remotize_empty_list(self):
        """Test that empty list returns empty list."""
        result = RTensor.remotize([], node_addr="localhost:8080")
        assert result == []

    def test_remotize_roundtrip(self, rpc_server):
        """Test remotize->localize roundtrip for trajectory dict."""
        original_traj = {
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "logits": torch.randn(2, 4).cpu(),
        }
        original_logits = original_traj["logits"].clone()

        # Remotize
        remotized = RTensor.remotize(original_traj, node_addr=rpc_server)

        # Store tensors on server using the NEW shard_ids created by remotize
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
        # After unpadding, logits are trimmed to max seqlen=3 (from attention_mask)
        assert localized["logits"].shape == (2, 3)
        assert torch.allclose(localized["logits"], original_logits[:, :3], atol=1e-5)

    def test_remotize_trims_padding_from_attention_mask(self, rpc_server):
        """Verify remotize trims padding when dict has attention_mask.

        Create a dict with attention_mask [[1,1,1,0,0], [1,1,0,0,0]] (seqlen 5,
        actual max 3). Remotize and localize. Assert tensors trimmed to seqlen 3.
        """
        traj = {
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]),
            "input_ids": torch.randn(2, 5),
            "logits": torch.randn(2, 5),
        }

        remotized = RTensor.remotize(traj, node_addr=rpc_server)

        # All tensor values should be RTensors
        assert isinstance(remotized["attention_mask"], RTensor)
        assert isinstance(remotized["input_ids"], RTensor)
        assert isinstance(remotized["logits"], RTensor)

        # The data (meta tensor) should reflect the compacted shape
        assert remotized["attention_mask"].data.shape == torch.Size([2, 3])
        assert remotized["input_ids"].data.shape == torch.Size([2, 3])
        assert remotized["logits"].data.shape == torch.Size([2, 3])

        # Verify via localize roundtrip
        from areal.infra.rpc.rtensor import fetch

        for key in ["attention_mask", "input_ids", "logits"]:
            actual_shard_id = remotized[key].shard.shard_id
            tensor_from_local = fetch(actual_shard_id)
            serialized = serialize_value(tensor_from_local)
            resp = requests.put(
                f"http://{rpc_server}/data/{actual_shard_id}",
                data=orjson.dumps(serialized),
            )
            assert resp.status_code == 200

        localized = RTensor.localize(remotized)
        assert localized["attention_mask"].shape == (2, 3)
        assert localized["input_ids"].shape == (2, 3)
        assert localized["logits"].shape == (2, 3)
        # attention_mask should be trimmed to [[1,1,1],[1,1,0]]
        expected_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        assert torch.equal(localized["attention_mask"], expected_mask)


class TestFetchBuffer:
    """Test client-side fetch buffer for RTensor caching.

    The fetch buffer avoids redundant network fetches when the same
    rollout_batch is sent to multiple engine calls across RPC boundaries.
    """

    def setup_method(self):
        """Clear fetch buffer before each test."""
        from areal.infra.rpc.rtensor import _fetch_buffer, _fetch_buffer_lock

        with _fetch_buffer_lock:
            _fetch_buffer.clear()

    def test_to_local_populates_buffer(self, rpc_server):
        """to_local() should populate the fetch buffer on first access."""
        from areal.infra.rpc.rtensor import _fetch_buffer

        tensor = torch.randn(3, 5).cpu()
        shard_id = str(uuid.uuid4())

        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )

        rtensor = RTensor(
            shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
            data=torch.empty(3, 5, device="meta"),
        )

        result = rtensor.to_local()
        assert torch.allclose(result, tensor)
        assert shard_id in _fetch_buffer

    def test_to_local_serves_from_buffer(self, rpc_server):
        """Second to_local() with a fresh RTensor (same shard_id) should
        hit the buffer without making a network request."""
        tensor = torch.randn(4, 6).cpu()
        shard_id = str(uuid.uuid4())

        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )

        # First access: populates buffer
        rt1 = RTensor(
            shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
            data=torch.empty(4, 6, device="meta"),
        )
        result1 = rt1.to_local()

        # Delete shard from server so a real fetch would fail
        requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": [shard_id]},
        )

        # Second access with a new RTensor object (simulates RPC boundary)
        rt2 = RTensor(
            shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
            data=torch.empty(4, 6, device="meta"),
        )
        result2 = rt2.to_local()
        assert torch.allclose(result1, result2)

    def test_localize_populates_buffer(self, rpc_server):
        """localize() should populate the fetch buffer for all fetched shards."""
        from areal.infra.rpc.rtensor import _fetch_buffer

        tensor1 = torch.randn(2, 3).cpu()
        tensor2 = torch.randn(4, 5).cpu()
        shard_id1 = str(uuid.uuid4())
        shard_id2 = str(uuid.uuid4())

        for sid, t in [(shard_id1, tensor1), (shard_id2, tensor2)]:
            serialized = serialize_value(t)
            requests.put(
                f"http://{rpc_server}/data/{sid}",
                data=orjson.dumps(serialized),
            )

        nested = {
            "a": RTensor(
                shard=TensorShardInfo(shard_id=shard_id1, node_addr=rpc_server),
                data=torch.empty(2, 3, device="meta"),
            ),
            "b": RTensor(
                shard=TensorShardInfo(shard_id=shard_id2, node_addr=rpc_server),
                data=torch.empty(4, 5, device="meta"),
            ),
        }

        localized = RTensor.localize(nested)
        assert torch.allclose(localized["a"], tensor1)
        assert torch.allclose(localized["b"], tensor2)
        assert shard_id1 in _fetch_buffer
        assert shard_id2 in _fetch_buffer

    def test_localize_serves_from_buffer(self, rpc_server):
        """Second localize() with fresh meta RTensors (same shard_ids) should
        resolve entirely from the buffer."""
        tensor = torch.randn(3, 4).cpu()
        shard_id = str(uuid.uuid4())

        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )

        def _make_rtensor():
            return RTensor(
                shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
                data=torch.empty(3, 4, device="meta"),
            )

        # First localize: populates buffer
        result1 = RTensor.localize({"x": _make_rtensor()})

        # Remove from server
        requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": [shard_id]},
        )

        # Second localize with fresh meta RTensor: should hit buffer
        result2 = RTensor.localize({"x": _make_rtensor()})
        assert torch.allclose(result1["x"], result2["x"])

    def test_localize_partial_buffer_hit(self, rpc_server):
        """When some shards are in the buffer and others are not, only the
        misses should be fetched from the backend."""
        from areal.infra.rpc.rtensor import _fetch_buffer

        tensor_a = torch.randn(2, 3).cpu()
        tensor_b = torch.randn(4, 5).cpu()
        shard_a = str(uuid.uuid4())
        shard_b = str(uuid.uuid4())

        for sid, t in [(shard_a, tensor_a), (shard_b, tensor_b)]:
            serialized = serialize_value(t)
            requests.put(
                f"http://{rpc_server}/data/{sid}",
                data=orjson.dumps(serialized),
            )

        # Warm buffer with shard_a only
        rt_a = RTensor(
            shard=TensorShardInfo(shard_id=shard_a, node_addr=rpc_server),
            data=torch.empty(2, 3, device="meta"),
        )
        RTensor.localize(rt_a)
        assert shard_a in _fetch_buffer
        assert shard_b not in _fetch_buffer

        # Delete shard_a from server; shard_b remains
        requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": [shard_a]},
        )

        # Localize both: shard_a from buffer, shard_b from backend
        nested = {
            "a": RTensor(
                shard=TensorShardInfo(shard_id=shard_a, node_addr=rpc_server),
                data=torch.empty(2, 3, device="meta"),
            ),
            "b": RTensor(
                shard=TensorShardInfo(shard_id=shard_b, node_addr=rpc_server),
                data=torch.empty(4, 5, device="meta"),
            ),
        }
        result = RTensor.localize(nested)
        assert torch.allclose(result["a"], tensor_a)
        assert torch.allclose(result["b"], tensor_b)

    def test_clear_node_evicts_from_buffer(self, rpc_server):
        """clear_node() should remove entries from the fetch buffer."""
        from areal.infra.rpc.rtensor import _fetch_buffer

        tensor = torch.randn(2, 3).cpu()
        shard_id = str(uuid.uuid4())

        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )

        # Populate buffer
        rt = RTensor(
            shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
            data=torch.empty(2, 3, device="meta"),
        )
        rt.to_local()
        assert shard_id in _fetch_buffer

        # clear_node evicts from buffer
        asyncio.run(RTensor.clear_node(rpc_server, [shard_id]))
        assert shard_id not in _fetch_buffer

    def test_buffer_thread_safety(self, rpc_server):
        """Concurrent to_local() calls with the same shard_id should not crash."""
        import threading

        tensor = torch.randn(5, 8).cpu()
        shard_id = str(uuid.uuid4())

        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )

        results = [None] * 10

        def fetch_shard(idx):
            rt = RTensor(
                shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
                data=torch.empty(5, 8, device="meta"),
            )
            results[idx] = rt.to_local()

        threads = [threading.Thread(target=fetch_shard, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for result in results:
            assert result is not None
            assert torch.allclose(result, tensor)


class TestTensorShardInfoDocumentation:
    """Tests verifying TensorShardInfo construction and field semantics."""

    def test_construction_with_all_fields(self):
        """TensorShardInfo can be constructed with required fields."""
        from areal.infra.rpc.rtensor import TensorShardInfo

        shard = TensorShardInfo(
            shard_id="test-shard-001",
            node_addr="localhost:8080",
        )
        assert shard.shard_id == "test-shard-001"
        assert shard.node_addr == "localhost:8080"

    def test_ray_backend_empty_node_addr(self):
        """Ray backend uses empty string for node_addr."""
        from areal.infra.rpc.rtensor import TensorShardInfo

        shard = TensorShardInfo(
            shard_id="",  # Will be filled by Ray ObjectRef
            node_addr="",  # Empty for Ray backend
        )
        assert shard.node_addr == ""

    def test_http_backend_node_addr(self):
        """HTTP backend uses host:port for node_addr."""
        from areal.infra.rpc.rtensor import TensorShardInfo

        shard = TensorShardInfo(
            shard_id="some-uuid",
            node_addr="192.168.1.1:8080",
        )
        assert ":" in shard.node_addr


# =============================================================================
# SharedMemory IPC backend tests
# =============================================================================


class TestRTensorShmPool:
    def setup_method(self):
        from areal.infra.rpc.rtensor import (
            _fetch_buffer,
            _fetch_buffer_lock,
            _storage,
            _storage_lock,
            _storage_stats,
            set_backend,
        )

        with _storage_lock:
            _storage.clear()
            _storage_stats.clear()
        with _fetch_buffer_lock:
            _fetch_buffer.clear()
        set_backend(None)
        self._pools = []

    def teardown_method(self):
        from areal.infra.rpc.rtensor import set_backend

        for pool in self._pools:
            pool.close()
        self._pools.clear()
        set_backend(None)

    def _make_pool(self, pool_size_bytes: int = 1024 * 1024):
        from areal.infra.rpc.rtensor import RTensorShmPool

        pool = RTensorShmPool(
            job_token="test",
            role="actor",
            worker_index=0,
            pool_size_bytes=pool_size_bytes,
        )
        pool.init_writer()
        self._pools.append(pool)
        return pool

    @staticmethod
    def _assert_roundtrip_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
        if actual.dtype in {
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        }:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            return
        assert torch.equal(actual, expected)

    @staticmethod
    def _write_tensor_to_pool(pool, shard_id: str, tensor: torch.Tensor) -> None:
        from areal.infra.rpc.rtensor import _DTYPE_TO_ENUM

        t = tensor.contiguous()
        try:
            raw = t.numpy().view("uint8").ravel()
        except TypeError:
            raw = t.view(torch.uint8).numpy().ravel()

        nbytes = raw.nbytes
        dtype_enum = _DTYPE_TO_ENUM[t.dtype]

        with pool._lock:
            aligned = (pool._next_offset + 63) & ~63
            assert aligned + nbytes <= pool._pool_size
            pool._next_offset = aligned + nbytes
            pool._occupied[shard_id] = (aligned, nbytes, dtype_enum, list(t.shape))

        pool._shm.buf[aligned : aligned + nbytes] = raw

    def test_pool_name_generation(self):
        from areal.infra.rpc.rtensor import RTensorShmPool

        name1 = RTensorShmPool._make_pool_name("job", "actor", 1)
        name2 = RTensorShmPool._make_pool_name("job", "actor", 1)
        assert name1 == name2
        assert name1.startswith("rt_")

    def test_pool_name_different_params(self):
        from areal.infra.rpc.rtensor import RTensorShmPool

        name1 = RTensorShmPool._make_pool_name("job", "actor", 0)
        name2 = RTensorShmPool._make_pool_name("job", "master", 0)
        name3 = RTensorShmPool._make_pool_name("job", "actor", 1)
        assert name1 != name2
        assert name1 != name3

    def test_init_writer_creates_shm(self):
        pool = self._make_pool()
        assert pool._shm is not None
        assert pool._is_writer is True

    def test_init_writer_retry_on_exists(self):
        pool1 = self._make_pool()
        pool2 = self._make_pool()
        assert pool1._shm is not None
        assert pool2._shm is not None
        assert pool1._pool_name != pool2._pool_name

    def test_init_writer_failure_disables_pool(self):
        from unittest.mock import patch

        from areal.infra.rpc.rtensor import RTensorShmPool

        pool = RTensorShmPool(
            job_token="test",
            role="actor",
            worker_index=0,
            pool_size_bytes=1024,
        )
        self._pools.append(pool)

        with patch(
            "areal.infra.rpc.rtensor._SharedMemory", side_effect=OSError("boom")
        ):
            pool.init_writer()

        assert pool._enabled is False
        assert pool._shm is None

    def test_allocate_and_write_basic(self):
        pool = self._make_pool()
        shard_id = str(uuid.uuid4())
        tensor = torch.arange(12, dtype=torch.uint8).reshape(3, 4)

        ok = pool.allocate_and_write(shard_id, tensor)
        assert ok is True

        meta = pool.get_meta(shard_id)
        assert meta is not None
        pool_name, offset, nbytes, dtype_enum, shape = meta
        assert pool_name == pool._pool_name
        assert offset >= 0
        assert nbytes == tensor.nbytes
        assert isinstance(dtype_enum, int)
        assert shape == [3, 4]

    def test_allocate_and_write_alignment(self):
        pool = self._make_pool()
        shard_id1 = str(uuid.uuid4())
        shard_id2 = str(uuid.uuid4())

        t1 = torch.ones(3, dtype=torch.uint8)
        t2 = torch.ones(7, dtype=torch.uint8)

        assert pool.allocate_and_write(shard_id1, t1) is True
        assert pool.allocate_and_write(shard_id2, t2) is True

        meta1 = pool.get_meta(shard_id1)
        meta2 = pool.get_meta(shard_id2)
        assert meta1 is not None
        assert meta2 is not None
        assert meta1[1] % 64 == 0
        assert meta2[1] % 64 == 0

    def test_allocate_and_write_pool_full(self):
        pool = self._make_pool(pool_size_bytes=100)
        first_id = str(uuid.uuid4())
        second_id = str(uuid.uuid4())

        assert (
            pool.allocate_and_write(first_id, torch.ones(16, dtype=torch.uint8)) is True
        )
        assert (
            pool.allocate_and_write(second_id, torch.ones(128, dtype=torch.uint8))
            is False
        )

    def test_allocate_unsupported_dtype(self):
        pool = self._make_pool()
        shard_id = str(uuid.uuid4())
        tensor = torch.ones(4, dtype=torch.complex64)

        result = pool.allocate_and_write(shard_id, tensor)
        if result:
            pytest.skip("complex64 is supported in this build; no unsupported dtype")
        assert result is False

    def test_release_removes_metadata(self):
        pool = self._make_pool()
        shard_id = str(uuid.uuid4())

        assert (
            pool.allocate_and_write(shard_id, torch.ones(4, dtype=torch.uint8)) is True
        )
        assert pool.get_meta(shard_id) is not None

        pool.release(shard_id)
        assert pool.get_meta(shard_id) is None

    def test_reset_zeroes_offset(self):
        pool = self._make_pool()
        sid1 = str(uuid.uuid4())
        sid2 = str(uuid.uuid4())

        assert pool.allocate_and_write(sid1, torch.ones(11, dtype=torch.uint8)) is True
        meta1 = pool.get_meta(sid1)
        assert meta1 is not None
        assert meta1[1] == 0

        pool.release(sid1)
        pool.reset()

        assert pool.allocate_and_write(sid2, torch.ones(5, dtype=torch.uint8)) is True
        meta2 = pool.get_meta(sid2)
        assert meta2 is not None
        assert meta2[1] == 0

    def test_reset_asserts_on_live_tensors(self):
        pool = self._make_pool()
        shard_id = str(uuid.uuid4())

        assert (
            pool.allocate_and_write(shard_id, torch.ones(3, dtype=torch.uint8)) is True
        )
        with pytest.raises(AssertionError, match="live tensors"):
            pool.reset()

    def test_read_tensor_roundtrip(self):
        pool = self._make_pool()
        tensors = [
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(2, 3, dtype=torch.bfloat16),
            torch.randint(-100, 100, (2, 3), dtype=torch.int64),
            torch.randint(0, 2, (2, 3), dtype=torch.bool),
            torch.randint(0, 255, (2, 3), dtype=torch.uint8),
        ]

        for tensor in tensors:
            shard_id = str(uuid.uuid4())
            self._write_tensor_to_pool(pool, shard_id, tensor)
            meta = pool.get_meta(shard_id)
            assert meta is not None
            loaded = pool.read_tensor(
                pool_name=meta[0],
                offset=meta[1],
                nbytes=meta[2],
                dtype_enum=meta[3],
                shape=meta[4],
            )
            self._assert_roundtrip_equal(loaded, tensor)
            pool.release(shard_id)

    def test_read_tensor_independent_clone(self):
        pool = self._make_pool()
        shard_id = str(uuid.uuid4())
        tensor = torch.arange(12, dtype=torch.uint8).reshape(3, 4)

        assert pool.allocate_and_write(shard_id, tensor) is True
        meta = pool.get_meta(shard_id)
        assert meta is not None

        first = pool.read_tensor(meta[0], meta[1], meta[2], meta[3], meta[4])
        first.fill_(0)
        second = pool.read_tensor(meta[0], meta[1], meta[2], meta[3], meta[4])
        assert torch.equal(second, tensor)

    def test_read_tensor_bounds_check(self):
        pool = self._make_pool()
        with pytest.raises(ValueError, match="out of bounds"):
            pool.read_tensor(
                pool_name=pool._pool_name,
                offset=2 * 1024 * 1024,
                nbytes=32,
                dtype_enum=1,
                shape=[8],
            )

    def test_read_tensor_meta_mismatch(self):
        pool = self._make_pool()
        shard_id = str(uuid.uuid4())
        tensor = torch.arange(10, dtype=torch.uint8)

        assert pool.allocate_and_write(shard_id, tensor) is True
        meta = pool.get_meta(shard_id)
        assert meta is not None

        with pytest.raises(ValueError, match="Pool meta mismatch"):
            pool.read_tensor(
                pool_name=meta[0],
                offset=meta[1],
                nbytes=meta[2] + 1,
                dtype_enum=meta[3],
                shape=meta[4],
            )

    def test_close_cleans_up(self):
        from multiprocessing.shared_memory import SharedMemory

        pool = self._make_pool()
        pool_name = pool._pool_name
        pool.close()

        with pytest.raises(FileNotFoundError):
            SharedMemory(name=pool_name, create=False)

    def test_various_dtypes(self):
        pool = self._make_pool(pool_size_bytes=2 * 1024 * 1024)
        test_cases = [
            torch.randn(3, 4).half(),
            torch.randn(3, 4),
            torch.randn(3, 4).double(),
            torch.randint(-128, 127, (3, 4), dtype=torch.int8),
            torch.randint(-100, 100, (3, 4), dtype=torch.int16),
            torch.randint(-100, 100, (3, 4), dtype=torch.int32),
            torch.randint(-100, 100, (3, 4), dtype=torch.int64),
            torch.randint(0, 2, (3, 4), dtype=torch.bool),
            torch.randn(3, 4).bfloat16(),
            torch.randint(0, 255, (3, 4), dtype=torch.uint8),
        ]

        for tensor in test_cases:
            shard_id = str(uuid.uuid4())
            self._write_tensor_to_pool(pool, shard_id, tensor)
            meta = pool.get_meta(shard_id)
            assert meta is not None
            loaded = pool.read_tensor(meta[0], meta[1], meta[2], meta[3], meta[4])
            self._assert_roundtrip_equal(loaded, tensor)
            pool.release(shard_id)

    def test_various_shapes(self):
        pool = self._make_pool(pool_size_bytes=2 * 1024 * 1024)
        shapes = [
            torch.Size([10]),
            torch.Size([3, 4]),
            torch.Size([2, 3, 4]),
            torch.Size([2, 3, 4, 5]),
        ]

        for shape in shapes:
            shard_id = str(uuid.uuid4())
            tensor = torch.randint(0, 255, shape, dtype=torch.uint8)
            assert pool.allocate_and_write(shard_id, tensor) is True
            meta = pool.get_meta(shard_id)
            assert meta is not None
            loaded = pool.read_tensor(meta[0], meta[1], meta[2], meta[3], meta[4])
            assert torch.equal(loaded, tensor)
            pool.release(shard_id)

    def test_disabled_pool_returns_false(self):
        pool = self._make_pool()
        pool._enabled = False

        shard_id = str(uuid.uuid4())
        assert (
            pool.allocate_and_write(shard_id, torch.ones(4, dtype=torch.float32))
            is False
        )
        assert pool.get_meta(shard_id) is None


class TestShmPoolIntegration:
    def setup_method(self):
        from areal.infra.rpc.rtensor import (
            _fetch_buffer,
            _fetch_buffer_lock,
            _storage,
            _storage_lock,
            _storage_stats,
            set_backend,
        )

        with _storage_lock:
            _storage.clear()
            _storage_stats.clear()
        with _fetch_buffer_lock:
            _fetch_buffer.clear()
        set_backend(None)
        self._pools = []

    def teardown_method(self):
        from areal.infra.rpc.rtensor import set_backend

        for pool in self._pools:
            pool.close()
        self._pools.clear()
        set_backend(None)

    def _make_pool(self, pool_size_bytes: int = 1024 * 1024):
        from areal.infra.rpc.rtensor import RTensorShmPool

        pool = RTensorShmPool(
            job_token="test_integ",
            role="actor",
            worker_index=0,
            pool_size_bytes=pool_size_bytes,
        )
        pool.init_writer()
        self._pools.append(pool)
        return pool

    @staticmethod
    def _make_pool_shard(
        shard_id: str, node_addr: str, meta: tuple[str, int, int, int, list[int]]
    ) -> TensorShardInfo:
        return TensorShardInfo(
            shard_id=shard_id,
            node_addr=node_addr,
            pool_name=meta[0],
            pool_offset=meta[1],
            pool_nbytes=meta[2],
            pool_dtype=meta[3],
            pool_shape=meta[4],
        )

    def test_backend_store_writes_to_pool(self):
        pool = self._make_pool()
        backend = HttpRTensorBackend(shm_pool=pool)
        tensor = torch.arange(12, dtype=torch.uint8).reshape(3, 4)

        shard_id = backend.store(tensor)
        meta = pool.get_meta(shard_id)
        assert meta is not None
        assert meta[2] == tensor.nbytes
        assert meta[4] == [3, 4]

    def test_backend_fetch_uses_pool_for_local(self):
        pool = self._make_pool()
        backend = HttpRTensorBackend(shm_pool=pool)
        tensor = torch.randint(0, 255, (4, 5), dtype=torch.uint8)

        shard_id = backend.store(tensor)
        meta = pool.get_meta(shard_id)
        assert meta is not None

        shard = self._make_pool_shard(shard_id, "localhost:7000", meta)
        fetched = backend.fetch([shard])[0]
        assert torch.equal(fetched, tensor)

    def test_backend_fetch_falls_back_without_pool_meta(self):
        from unittest.mock import patch

        from areal.infra.rpc.rtensor import fetch

        pool = self._make_pool()
        backend = HttpRTensorBackend(shm_pool=pool)
        tensor = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        shard_id = backend.store(tensor)

        calls: list[str] = []

        async def fake_fetch_shard_group(
            session: object,
            node_addr: str,
            grouped: list[tuple[int, TensorShardInfo]],
            max_retries: int = 3,
            retry_delay: float = 1.0,
        ) -> list[torch.Tensor]:
            del session, max_retries, retry_delay
            calls.append(node_addr)
            return [fetch(shard.shard_id) for _, shard in grouped]

        backend._fetch_shard_group = fake_fetch_shard_group

        with patch.object(
            pool, "read_tensor", side_effect=AssertionError("unexpected")
        ):
            shard = TensorShardInfo(shard_id=shard_id, node_addr="localhost:7001")
            fetched = backend.fetch([shard])[0]

        assert calls == ["localhost:7001"]
        assert torch.equal(fetched, tensor)

    def test_remotize_localize_roundtrip_via_pool(self):
        from areal.infra.rpc.rtensor import set_backend

        pool = self._make_pool()
        backend = HttpRTensorBackend(shm_pool=pool)
        set_backend(backend)

        tensor = torch.randint(0, 255, (5, 6), dtype=torch.uint8)
        obj = {"data": tensor}

        remotized = RTensor.remotize(obj, node_addr="localhost:7010")
        assert isinstance(remotized["data"], RTensor)
        assert remotized["data"].shard.has_pool_meta is True

        localized = RTensor.localize(remotized)
        assert torch.equal(localized["data"], tensor)

    def test_remove_releases_from_pool(self):
        from areal.infra.rpc.rtensor import remove, set_backend

        pool = self._make_pool()
        backend = HttpRTensorBackend(shm_pool=pool)
        set_backend(backend)

        shard_id = backend.store(torch.randint(0, 255, (2, 2), dtype=torch.uint8))
        assert pool.get_meta(shard_id) is not None

        remove(shard_id)
        assert pool.get_meta(shard_id) is None

    def test_step_lifecycle(self):
        from areal.infra.rpc.rtensor import remove, set_backend

        pool = self._make_pool(pool_size_bytes=2 * 1024 * 1024)
        backend = HttpRTensorBackend(shm_pool=pool)
        set_backend(backend)

        tensors = [
            torch.randint(0, 255, (3, 4), dtype=torch.uint8),
            torch.randint(0, 255, (2, 5), dtype=torch.uint8),
            torch.randint(0, 255, (2, 2, 2), dtype=torch.uint8),
        ]
        shard_ids = [backend.store(tensor) for tensor in tensors]

        shards: list[TensorShardInfo] = []
        for shard_id in shard_ids:
            meta = pool.get_meta(shard_id)
            assert meta is not None
            shards.append(self._make_pool_shard(shard_id, "localhost:7020", meta))

        fetched = backend.fetch(shards)
        for actual, expected in zip(fetched, tensors, strict=True):
            assert torch.equal(actual, expected)

        for shard_id in shard_ids:
            remove(shard_id)

        pool.reset()

        tensor2 = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
        shard2 = backend.store(tensor2)
        meta2 = pool.get_meta(shard2)
        assert meta2 is not None
        assert meta2[1] == 0

        remove(shard2)

    def test_pool_full_falls_back(self):
        from areal.infra.rpc.rtensor import fetch

        pool = self._make_pool(pool_size_bytes=128)
        backend = HttpRTensorBackend(shm_pool=pool)
        tensor = torch.randn(256, dtype=torch.float32)

        shard_id = backend.store(tensor)
        assert pool.get_meta(shard_id) is None

        async def fake_fetch_shard_group(
            session: object,
            node_addr: str,
            grouped: list[tuple[int, TensorShardInfo]],
            max_retries: int = 3,
            retry_delay: float = 1.0,
        ) -> list[torch.Tensor]:
            del session, node_addr, max_retries, retry_delay
            return [fetch(shard.shard_id) for _, shard in grouped]

        backend._fetch_shard_group = fake_fetch_shard_group
        shard = TensorShardInfo(shard_id=shard_id, node_addr="localhost:7030")
        fetched = backend.fetch([shard])[0]
        torch.testing.assert_close(fetched, tensor, rtol=1e-5, atol=1e-5)

    def test_concurrent_fetch_via_pool(self):
        import threading

        pool = self._make_pool()
        backend = HttpRTensorBackend(shm_pool=pool)
        tensor = torch.randint(0, 255, (8, 8), dtype=torch.uint8)
        shard_id = backend.store(tensor)

        meta = pool.get_meta(shard_id)
        assert meta is not None
        shard = self._make_pool_shard(shard_id, "localhost:7040", meta)

        results: list[torch.Tensor | None] = [None] * 8

        def fetch_one(index: int) -> None:
            results[index] = backend.fetch([shard])[0]

        threads = [threading.Thread(target=fetch_one, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for result in results:
            assert result is not None
            assert torch.equal(result, tensor)

    def test_backend_without_pool_works(self):
        from areal.infra.rpc.rtensor import fetch

        backend = HttpRTensorBackend(shm_pool=None)
        tensor = torch.randn(4, 4, dtype=torch.float32)
        shard_id = backend.store(tensor)

        calls: list[str] = []

        async def fake_fetch_shard_group(
            session: object,
            node_addr: str,
            grouped: list[tuple[int, TensorShardInfo]],
            max_retries: int = 3,
            retry_delay: float = 1.0,
        ) -> list[torch.Tensor]:
            del session, max_retries, retry_delay
            calls.append(node_addr)
            return [fetch(shard.shard_id) for _, shard in grouped]

        backend._fetch_shard_group = fake_fetch_shard_group

        shard = TensorShardInfo(shard_id=shard_id, node_addr="localhost:7050")
        fetched = backend.fetch([shard])[0]

        assert calls == ["localhost:7050"]
        torch.testing.assert_close(fetched, tensor, rtol=1e-5, atol=1e-5)

    def test_storage_stats_without_shm_fields(self):
        from areal.infra.rpc.rtensor import remove, storage_stats

        backend = HttpRTensorBackend(shm_pool=None)
        shard_id = backend.store(torch.randn(4, 4, dtype=torch.float32))

        stats = storage_stats()
        assert "num_tensors" in stats
        assert "total_bytes" in stats
        assert "shm_segments" not in stats
        assert "shm_bytes" not in stats

        remove(shard_id)


class TestIsLocalAddr:
    """Tests for the is_local_addr utility."""

    def test_localhost_is_local(self):
        from areal.utils.network import is_local_addr

        assert is_local_addr("localhost:8080") is True

    def test_127_0_0_1_is_local(self):
        from areal.utils.network import is_local_addr

        assert is_local_addr("127.0.0.1:8080") is True

    def test_loopback_ipv6_is_local(self):
        from areal.utils.network import is_local_addr

        assert is_local_addr("[::1]:8080") is True

    def test_hostname_is_local(self):
        import socket

        from areal.utils.network import is_local_addr

        hostname = socket.gethostname()
        assert is_local_addr(f"{hostname}:8080") is True

    def test_remote_addr_is_not_local(self):
        from areal.utils.network import is_local_addr

        # Use an address that is very unlikely to be local
        assert is_local_addr("203.0.113.1:8080") is False

    def test_bare_localhost(self):
        from areal.utils.network import is_local_addr

        assert is_local_addr("localhost") is True
