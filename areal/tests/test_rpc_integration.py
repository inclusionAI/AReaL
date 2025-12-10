import importlib
import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

import orjson
import pytest
import torch

from areal.api.engine_api import TrainEngine
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value


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
        data_bytes = orjson.dumps(serialize_value(data))

        # Store data
        resp = client.put(
            f"/data/{shard_id}",
            data=data_bytes,
            content_type="application/octet-stream",
        )

        assert resp.status_code == 200
        resp_data = resp.get_json()
        assert resp_data["status"] == "ok"
        assert resp_data["shard_id"] == shard_id

    def test_store_batch_data_simple(self, client):
        """Test storing batch data without query parameters."""
        shard_id = "test-shard-2"
        data = {"input_ids": torch.tensor([[1, 2]])}

        data_bytes = orjson.dumps(serialize_value(data))

        resp = client.put(f"/data/{shard_id}", data=data_bytes)
        assert resp.status_code == 200

    def test_retrieve_batch_data(self, client):
        """Test retrieving batch data shard."""
        shard_id = "test-shard-3:input_ids"
        original_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Store data first (single tensor)
        data_bytes = orjson.dumps(serialize_value(original_tensor))

        store_resp = client.put(
            f"/data/{shard_id}",
            data=data_bytes,
            content_type="application/octet-stream",
        )
        assert store_resp.status_code == 200

        # Retrieve full data
        get_resp = client.get(f"/data/{shard_id}")
        assert get_resp.status_code == 200
        assert get_resp.content_type == "application/octet-stream"

        retrieved_data = deserialize_value(orjson.loads(get_resp.data))
        # _batch_storage stores the data as-is, so a stored tensor returns as a tensor
        assert isinstance(retrieved_data, torch.Tensor)
        assert torch.equal(retrieved_data, original_tensor)

    def test_retrieve_batch_data_not_found(self, client):
        """Test retrieving non-existent shard returns 404."""
        resp = client.get("/data/non-existent-shard")
        assert resp.status_code == 404
        resp_data = resp.get_json()
        assert resp_data["status"] == "error"
        assert "not found" in resp_data["message"].lower()

    def test_retrieve_batch_data_complete(self, client):
        """Test retrieving complete batch data."""
        # Store a single tensor (new format: single tensor per shard)
        shard_id = "test-shard-4:input_ids"
        original_tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

        # Store data as single tensor
        data_bytes = orjson.dumps(serialize_value(original_tensor))

        client.put(f"/data/{shard_id}", data=data_bytes)

        # Retrieve complete data
        resp = client.get(f"/data/{shard_id}")
        assert resp.status_code == 200

        retrieved_data = deserialize_value(orjson.loads(resp.data))
        # _batch_storage stores the data as-is, so a stored tensor returns as a tensor
        assert isinstance(retrieved_data, torch.Tensor)
        # The data should match what was stored
        assert torch.equal(retrieved_data, original_tensor)

    def test_clear_batch_data(self, client):
        """Test clearing batch data by shard_ids."""
        # Store multiple shards
        shard_ids = ["shard-1", "shard-2", "shard-3", "shard-4"]

        data = {"input_ids": torch.tensor([[1, 2]])}
        data_bytes = orjson.dumps(serialize_value(data))

        for shard_id in shard_ids:
            resp = client.put(
                f"/data/{shard_id}",
                data=data_bytes,
                content_type="application/octet-stream",
            )
            assert resp.status_code == 200

        # Clear shard-1 and shard-2
        clear_resp = client.delete(
            "/data/clear",
            json={"shard_ids": ["shard-1", "shard-2"]},
        )
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

    def test_clear_batch_data_empty_list(self, client):
        """Test clearing batch data with empty shard_ids list."""
        shard_id = "shard-clear-test"
        data = {"input_ids": torch.tensor([[1, 2]])}

        data_bytes = orjson.dumps(serialize_value(data))

        # Store shard
        client.put(f"/data/{shard_id}", data=data_bytes)

        # Clear with empty list - should not clear anything
        clear_resp = client.delete("/data/clear", json={"shard_ids": []})
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

        data_bytes = orjson.dumps(serialize_value(data))

        client.put(
            f"/data/{shard_id}",
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

        data_bytes1 = orjson.dumps(serialize_value(data1))

        data_bytes2 = orjson.dumps(serialize_value(data2))

        client.put("/data/stats-shard-1", data=data_bytes1)
        client.put("/data/stats-shard-2", data=data_bytes2)

        stats_resp = client.get("/data/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.get_json()
        assert stats_data["total_shards"] == 2
        assert stats_data["total_size_bytes"] == len(data_bytes1) + len(data_bytes2)
