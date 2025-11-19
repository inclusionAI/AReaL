import importlib
import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

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
    monkeypatch.setattr(
        module.stats_tracker, "export", MagicMock(return_value={"loss": 0.1})
    )
    module._engine = None
    yield module
    module._engine = None


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
                    "_should_bcast": False,
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

        stats_resp = client.post("/export_stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.get_json()
        assert stats_data["status"] == "success"
        assert stats_data["result"] == {"loss": 0.1}
