import importlib
import io
import pickle
import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from areal.api.engine_api import TrainEngine
from areal.scheduler.rpc.serialization import serialize_value


@dataclass
class _ClusterConfig:
    name_resolve: dict[str, str]


@dataclass
class _ExperimentConfig:
    cluster: _ClusterConfig
    seed: int


class _DummyTrainEngine(TrainEngine):
    def __init__(self, *args, **kwargs):
        self._destroy_called = False

    def initialize(self, **kwargs):
        self._initialized_with = kwargs

    def generate(self, *args, **kwargs):
        return {"text": "mocked"}

    def destroy(self):
        self._destroy_called = True

    def current_data_parallel_head(self) -> int:
        return 0

    @property
    def data_parallel_group(self):
        return "dp-group"

    @property
    def context_and_model_parallel_group(self):
        return "mp-group"


@pytest.fixture(autouse=True)
def rpc_server(monkeypatch):
    module_name = "areal.scheduler.rpc.rpc_server"
    engine_module_name = "areal.engine.fsdp_engine"

    stub_module = types.SimpleNamespace(FSDPEngine=_DummyTrainEngine)
    monkeypatch.setitem(sys.modules, engine_module_name, stub_module)

    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    monkeypatch.setattr(module, "tensor_container_to", lambda data, device: data)
    monkeypatch.setattr(
        module,
        "broadcast_tensor_container",
        lambda data, **kwargs: data,
    )
    monkeypatch.setattr(module.current_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(module.name_resolve, "reconfigure", MagicMock())
    monkeypatch.setattr(module.seeding, "set_random_seed", MagicMock())
    module._engine = None
    # Clear batch storage before each test
    with module._batch_storage_lock:
        module._batch_storage.clear()
        module._batch_storage_stats.clear()
    yield module
    module._engine = None
    # Clear batch storage after each test
    with module._batch_storage_lock:
        module._batch_storage.clear()
        module._batch_storage_stats.clear()


@pytest.fixture
def client(rpc_server):
    return rpc_server.app.test_client()


class TestSyncRPCServer:
    def test_lifecycle_endpoints(self, rpc_server, client):
        create_resp = client.post(
            "/create_engine",
            json={
                "engine": "areal.engine.fsdp_engine.FSDPEngine",
                "init_args": [],
                "init_kwargs": {
                    "addr": None,
                    "ft_spec": {"total_train_epochs": 1},
                },
            },
        )
        assert create_resp.status_code == 200
        create_data = create_resp.get_json()
        assert create_data["status"] == "success"

        call_resp = client.post(
            "/call",
            json={
                "method": "generate",
                "args": ["hello"],
                "kwargs": {
                    "max_tokens": 10,
                    "should_broadcast": False,
                },
            },
        )
        assert call_resp.status_code == 200
        call_data = call_resp.get_json()
        assert call_data["status"] == "success"
        assert call_data["result"]["text"] == "mocked"

        config_payload = serialize_value(
            _ExperimentConfig(
                cluster=_ClusterConfig(name_resolve={"type": "nfs"}),
                seed=42,
            )
        )
        cfg_resp = client.post(
            "/configure",
            json={
                "config": config_payload,
                "role": "trainer",
                "rank": 0,
            },
        )
        assert cfg_resp.status_code == 200
        assert cfg_resp.get_json()["status"] == "success"

        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.get_json()["engine_initialized"] is True

    def test_set_env_endpoint(self, client):
        resp = client.post("/set_env", json={"env": {"RANK": 0, "WORLD_SIZE": 1}})
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "success"

        bad_resp = client.post("/set_env", json={})
        assert bad_resp.status_code == 400

    def test_store_batch_data(self, client):
        """Test storing batch data shard."""
        shard_id = "test-shard-1"
        data = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([0, 1]),
        }

        # Serialize data
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        data_bytes = buffer.getvalue()

        # Store data
        resp = client.put(
            f"/data/{shard_id}?global_step=10",
            data=data_bytes,
            content_type="application/octet-stream",
        )

        assert resp.status_code == 200
        resp_data = resp.get_json()
        assert resp_data["status"] == "ok"
        assert resp_data["shard_id"] == shard_id

    def test_store_batch_data_default_step(self, client):
        """Test storing batch data with default global_step."""
        shard_id = "test-shard-2"
        data = {"input_ids": torch.tensor([[1, 2]])}

        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        data_bytes = buffer.getvalue()

        resp = client.put(f"/data/{shard_id}", data=data_bytes)
        assert resp.status_code == 200

    def test_retrieve_batch_data(self, client):
        """Test retrieving batch data shard."""
        shard_id = "test-shard-3"
        original_data = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "labels": torch.tensor([0, 1, 2]),
        }

        # Store data first
        buffer = io.BytesIO()
        pickle.dump(original_data, buffer)
        data_bytes = buffer.getvalue()

        store_resp = client.put(
            f"/data/{shard_id}?global_step=5",
            data=data_bytes,
            content_type="application/octet-stream",
        )
        assert store_resp.status_code == 200

        # Retrieve full data
        get_resp = client.get(f"/data/{shard_id}")
        assert get_resp.status_code == 200
        assert get_resp.content_type == "application/octet-stream"

        retrieved_data = pickle.loads(get_resp.data)
        assert "input_ids" in retrieved_data
        assert "labels" in retrieved_data
        assert torch.equal(retrieved_data["input_ids"], original_data["input_ids"])
        assert torch.equal(retrieved_data["labels"], original_data["labels"])

    def test_retrieve_batch_data_not_found(self, client):
        """Test retrieving non-existent shard returns 404."""
        resp = client.get("/data/non-existent-shard")
        assert resp.status_code == 404
        resp_data = resp.get_json()
        assert resp_data["status"] == "error"
        assert "not found" in resp_data["message"].lower()

    def test_retrieve_batch_data_with_offset(self, client):
        """Test retrieving batch data with offset parameter."""
        shard_id = "test-shard-4"
        original_data = {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "labels": torch.tensor([0, 1, 2, 3]),
        }

        # Store data
        buffer = io.BytesIO()
        pickle.dump(original_data, buffer)
        data_bytes = buffer.getvalue()

        client.put(f"/data/{shard_id}?global_step=1", data=data_bytes)

        # Retrieve with offset=2 (should get last 2 samples)
        resp = client.get(f"/data/{shard_id}?offset=2")
        assert resp.status_code == 200

        retrieved_data = pickle.loads(resp.data)
        assert retrieved_data["input_ids"].shape[0] == 2
        assert torch.equal(retrieved_data["input_ids"], original_data["input_ids"][2:])
        assert torch.equal(retrieved_data["labels"], original_data["labels"][2:])

    def test_retrieve_batch_data_with_batch_size(self, client):
        """Test retrieving batch data with batch_size parameter."""
        shard_id = "test-shard-5"
        original_data = {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "labels": torch.tensor([0, 1, 2, 3]),
        }

        # Store data
        buffer = io.BytesIO()
        pickle.dump(original_data, buffer)
        data_bytes = buffer.getvalue()

        client.put(f"/data/{shard_id}?global_step=1", data=data_bytes)

        # Retrieve with batch_size=2 (should get first 2 samples)
        resp = client.get(f"/data/{shard_id}?batch_size=2")
        assert resp.status_code == 200

        retrieved_data = pickle.loads(resp.data)
        assert retrieved_data["input_ids"].shape[0] == 2
        assert retrieved_data["labels"].shape[0] == 2
        assert torch.equal(retrieved_data["input_ids"], original_data["input_ids"][:2])
        assert torch.equal(retrieved_data["labels"], original_data["labels"][:2])

    def test_retrieve_batch_data_with_offset_and_batch_size(self, client):
        """Test retrieving batch data with both offset and batch_size."""
        shard_id = "test-shard-6"
        original_data = {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            "labels": torch.tensor([0, 1, 2, 3, 4]),
        }

        # Store data
        buffer = io.BytesIO()
        pickle.dump(original_data, buffer)
        data_bytes = buffer.getvalue()

        client.put(f"/data/{shard_id}?global_step=1", data=data_bytes)

        # Retrieve with offset=1 and batch_size=2 (should get samples at indices 1-2)
        resp = client.get(f"/data/{shard_id}?offset=1&batch_size=2")
        assert resp.status_code == 200

        retrieved_data = pickle.loads(resp.data)
        assert retrieved_data["input_ids"].shape[0] == 2
        assert retrieved_data["labels"].shape[0] == 2
        assert torch.equal(retrieved_data["input_ids"], original_data["input_ids"][1:3])
        assert torch.equal(retrieved_data["labels"], original_data["labels"][1:3])

    def test_clear_batch_data(self, client):
        """Test clearing old batch data."""
        # Store multiple shards with different global_steps
        shards = [
            ("shard-1", 5),
            ("shard-2", 10),
            ("shard-3", 15),
            ("shard-4", 20),
        ]

        data = {"input_ids": torch.tensor([[1, 2]])}
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        data_bytes = buffer.getvalue()

        for shard_id, step in shards:
            resp = client.put(
                f"/data/{shard_id}?global_step={step}",
                data=data_bytes,
                content_type="application/octet-stream",
            )
            assert resp.status_code == 200

        # Clear shards with step < 15 (should remove shard-1 and shard-2)
        clear_resp = client.delete("/data/clear?global_step=15")
        assert clear_resp.status_code == 200
        clear_data = clear_resp.get_json()
        assert clear_data["status"] == "ok"
        assert clear_data["cleared_count"] == 2

        # Verify shard-1 and shard-2 are removed
        assert client.get("/data/shard-1").status_code == 404
        assert client.get("/data/shard-2").status_code == 404

        # Verify shard-3 and shard-4 still exist
        assert client.get("/data/shard-3").status_code == 200
        assert client.get("/data/shard-4").status_code == 200

    def test_clear_batch_data_default_step(self, client):
        """Test clearing batch data with default global_step."""
        shard_id = "shard-clear-test"
        data = {"input_ids": torch.tensor([[1, 2]])}

        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        data_bytes = buffer.getvalue()

        # Store with step 10
        client.put(f"/data/{shard_id}?global_step=10", data=data_bytes)

        # Clear with default step (0) - should not clear anything with step >= 0
        clear_resp = client.delete("/data/clear")
        assert clear_resp.status_code == 200
        clear_data = clear_resp.get_json()
        assert clear_data["cleared_count"] == 0

        # Shard should still exist
        assert client.get(f"/data/{shard_id}").status_code == 200

    def test_batch_data_stats(self, client):
        """Test getting batch data storage statistics."""
        # Initially empty
        stats_resp = client.get("/data/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.get_json()
        assert stats_data["status"] == "ok"
        assert stats_data["total_shards"] == 0
        assert stats_data["total_size_bytes"] == 0

        # Store some data
        shard_id = "stats-test-shard"
        data = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([0, 1]),
        }

        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        data_bytes = buffer.getvalue()

        client.put(
            f"/data/{shard_id}?global_step=1",
            data=data_bytes,
            content_type="application/octet-stream",
        )

        # Check stats again
        stats_resp = client.get("/data/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.get_json()
        assert stats_data["status"] == "ok"
        assert stats_data["total_shards"] == 1
        assert stats_data["total_size_bytes"] > 0

    def test_batch_data_stats_multiple_shards(self, client):
        """Test stats with multiple shards."""
        data1 = {"input_ids": torch.tensor([[1, 2]])}
        data2 = {"input_ids": torch.tensor([[3, 4], [5, 6]])}

        buffer1 = io.BytesIO()
        pickle.dump(data1, buffer1)
        data_bytes1 = buffer1.getvalue()

        buffer2 = io.BytesIO()
        pickle.dump(data2, buffer2)
        data_bytes2 = buffer2.getvalue()

        client.put("/data/stats-shard-1?global_step=1", data=data_bytes1)
        client.put("/data/stats-shard-2?global_step=1", data=data_bytes2)

        stats_resp = client.get("/data/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.get_json()
        assert stats_data["total_shards"] == 2
        assert stats_data["total_size_bytes"] == len(data_bytes1) + len(data_bytes2)
