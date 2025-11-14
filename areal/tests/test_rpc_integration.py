import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ConfigDict

from areal.api.cli_args import (
    NameResolveConfig,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec
from areal.scheduler.rpc.api import (
    CallEnginePayload,
    ConfigurePayload,
    CreateEnginePayload,
    EngineNameEnum,
)
from areal.scheduler.rpc.server import create_app

ConfigDict.use_enum_values = True


@pytest.fixture
def app():
    import os

    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"
    yield create_app()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True, scope="module")
def mock_engines():
    from areal.engine.fsdp_engine import FSDPEngine
    from areal.engine.megatron_engine import MegatronEngine
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.engine.vllm_remote import RemotevLLMEngine

    fsdp_cls = MagicMock(spec=FSDPEngine)
    megatron_cls = MagicMock(spec=MegatronEngine)
    sglang_cls = MagicMock(spec=RemoteSGLangEngine)
    vllm_cls = MagicMock(spec=RemotevLLMEngine)

    fsdp_inst = fsdp_cls.return_value
    megatron_inst = megatron_cls.return_value
    sglang_inst = sglang_cls.return_value
    vllm_inst = vllm_cls.return_value

    fsdp_inst.initialize.return_value = None
    megatron_inst.initialize.return_value = None
    sglang_inst.initialize.return_value = None
    vllm_inst.initialize.return_value = None

    fsdp_inst.generate.return_value = {"text": "mocked"}
    megatron_inst.generate.return_value = {"text": "mocked"}
    sglang_inst.generate.return_value = {"text": "mocked"}
    vllm_inst.generate.return_value = {"text": "mocked"}

    patchers = [
        patch("areal.scheduler.rpc.server.FSDPEngine", fsdp_cls),
        patch("areal.scheduler.rpc.server.MegatronEngine", megatron_cls),
        patch("areal.scheduler.rpc.server.RemoteSGLangEngine", sglang_cls),
        patch("areal.scheduler.rpc.server.RemotevLLMEngine", vllm_cls),
        patch("areal.scheduler.rpc.server.set_random_seed", MagicMock()),
    ]
    for p in patchers:
        p.start()
    yield
    for p in patchers:
        p.stop()


def rpc(client, method: str, params=None):
    resp = client.post(
        "/api",
        data=json.dumps(
            {"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1}
        ),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["jsonrpc"] == "2.0"
    assert "error" not in data, data.get("error")
    return data["result"]


class TestEngineRPC:
    def test_lifecycle(self, client):
        create_result = rpc(
            client,
            "areal.create_engine",
            {
                "payload": CreateEnginePayload(
                    class_name=EngineNameEnum.FSDP,
                    config=TrainEngineConfig(
                        experiment_name="test",
                        trial_name="test",
                        path="tiny-random-gpt2",
                        attn_impl="flash_attention_2",
                    ),
                    initial_args={
                        "addr": None,
                        "ft_spec": FinetuneSpec(
                            total_train_epochs=1,
                            dataset_size=1,
                            train_batch_size=1,
                        ),
                    },
                ).model_dump()
            },
        )
        assert create_result["success"] is True
        assert create_result["message"] == "ok"

        call_result = rpc(
            client,
            "areal.call_engine",
            {
                "payload": CallEnginePayload(
                    method="generate", args=("hello",), kwargs={"max_tokens": 10}
                ).model_dump()
            },
        )
        assert call_result["success"] is True

        cfg_result = rpc(
            client,
            "areal.configure",
            {
                "payload": ConfigurePayload(
                    seed_cfg={"base_seed": 42, "key": "test"},
                    name_resolve=NameResolveConfig(type="nfs"),
                ).model_dump()
            },
        )
        assert cfg_result["success"] is True

        health = rpc(client, "areal.health")
        assert health["success"] is True

        stats = rpc(client, "areal.export_stats", {"reset": True})
        assert stats["success"] is True
        assert isinstance(stats["data"], dict)
