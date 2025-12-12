"""Integration tests for RTensor with RPC server."""

import subprocess
import sys
import time

import orjson
import pytest
import requests
import torch

from areal.scheduler.rpc.rtensor import RTensor, TensorShardId, TensorShardInfo
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
        shard_id = TensorShardId(task_id="task_001", tensor_name="logits")

        # Create RTensor using single shard constructor
        rtensor = RTensor.from_single(tensor, shard_id=shard_id, node_addr=rpc_server)

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
        rtensor_meta = RTensor(
            shards=rtensor.shards,
            data=torch.empty(tensor.shape, device="meta"),
        )
        localized = rtensor_meta.to_local()

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
                    shard_id=TensorShardId(task_id="t1", tensor_name="input"),
                    node_addr=rpc_server,
                    size=3,
                ),
                TensorShardInfo(
                    shard_id=TensorShardId(task_id="t2", tensor_name="input"),
                    node_addr=rpc_server,
                    size=5,
                ),
                TensorShardInfo(
                    shard_id=TensorShardId(task_id="t3", tensor_name="input"),
                    node_addr=rpc_server,
                    size=2,
                ),
            ],
            data=torch.empty(0, device="meta"),
        )

        # Create batched RTensor (this stores shards locally)
        rtensor = RTensor.from_batched(
            batch_tensor,
            layout=layout,
            tensor_name="output",
            node_addr=rpc_server,
        )

        # Verify shards were created correctly
        assert len(rtensor.shards) == 3
        assert rtensor.shards[0].shard_id.task_id == "t1"
        assert rtensor.shards[0].shard_id.tensor_name == "output"
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
        rtensor_meta = RTensor(
            shards=rtensor.shards,
            data=torch.empty(batch_tensor.shape, device="meta"),
        )
        reconstructed = rtensor_meta.to_local()

        assert reconstructed.shape == batch_tensor.shape
        assert torch.allclose(reconstructed, batch_tensor)

    def test_localize_nested_structure(self, rpc_server):
        """Test localizing nested structures containing RTensors."""
        # Create tensors
        tensor1 = torch.randn(3, 4).cpu()
        tensor2 = torch.randn(2, 6).cpu()

        # Store on server
        shard_id1 = TensorShardId(task_id="nested_1", tensor_name="logits")
        shard_id2 = TensorShardId(task_id="nested_2", tensor_name="values")

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

    def test_rtensorize_and_localize_roundtrip(self, rpc_server):
        """Test full roundtrip: rtensorize outputs, store, then localize."""
        # Simulate InferenceEngine output (single tensor)
        output = {"logits": torch.randn(4, 10).cpu(), "score": 0.95}

        # Rtensorize (converts tensors to RTensors)
        rtensorized = RTensor.rtensorize(
            output,
            layouts={"args": None, "kwargs": None},
            node_addr=rpc_server,
            task_id="task_roundtrip",
        )

        # Verify RTensor was created
        assert isinstance(rtensorized["logits"], RTensor)
        assert rtensorized["score"] == 0.95

        # Store tensor on server
        shard_id = rtensorized["logits"].shards[0].shard_id
        serialized = serialize_value(output["logits"])
        resp = requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )
        assert resp.status_code == 200

        # Localize (fetches remote tensors)
        localized = RTensor.localize(rtensorized)

        assert isinstance(localized["logits"], torch.Tensor)
        assert torch.allclose(localized["logits"], output["logits"])
        assert localized["score"] == 0.95

    def test_clear_batch_data(self, rpc_server):
        """Test clearing stored tensor shards."""
        # Store some tensors
        shard_ids = []
        for i in range(3):
            tensor = torch.randn(2, 3).cpu()
            shard_id = TensorShardId(task_id=f"clear_{i}", tensor_name="data")
            shard_ids.append(str(shard_id))

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
            shard_id = TensorShardId(task_id=f"pad_{i}", tensor_name="hidden")
            shards.append(
                TensorShardInfo(
                    shard_id=shard_id,
                    node_addr=rpc_server,
                    size=tensor.shape[0],
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
