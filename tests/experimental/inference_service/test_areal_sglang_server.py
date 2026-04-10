from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import types

import httpx
import pytest
import requests
from fastapi.testclient import TestClient

from tests.experimental.inference_service.integration_utils import (
    check_server_health,
    get_test_model_path,
    has_gpu,
)

from areal.engine.sglang_ext.areal_sglang_server import (
    _AWEX_SCHEDULER_LAUNCHER,
    _parse_args,
    create_app,
)
from areal.engine.sglang_ext.sglang_awex_adapter import AwexSGLangServerAdapter
from areal.engine.sglang_ext.sglang_worker_extension import _patch_awex_sglang_converter


class _FakeTokenizerManager:
    def __init__(self):
        self.paused = False

    async def pause_generation(self, *_args, **_kwargs):
        self.paused = True

    async def continue_generation(self, *_args, **_kwargs):
        self.paused = False


class _FakeEngine:
    def __init__(self):
        self.tokenizer_manager = _FakeTokenizerManager()
        self.server_args = type("ServerArgs", (), {"model_path": "fake-model"})()

    async def async_generate(self, **payload):
        return {"ok": True, "payload": payload}

    def encode(self, **payload):
        return {"encoded": payload}

    def flush_cache(self):
        return True


def test_create_app_exposes_basic_routes():
    app = create_app(_FakeEngine())
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "fake-model"

    resp = client.post("/generate", json={"input_ids": [1, 2, 3]})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    pause = client.post("/pause_generation", json={})
    assert pause.status_code == 200
    assert pause.json()["success"] is True

    resume = client.post("/continue_generation", json={})
    assert resume.status_code == 200
    assert resume.json()["success"] is True


def test_create_app_reports_starting_when_engine_not_ready():
    app = create_app(None)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["status"] == "starting"

    generate = client.post("/generate", json={"input_ids": [1]})
    assert generate.status_code == 503


def test_awex_routes_use_adapter(monkeypatch):
    class _FakeAdapter:
        def __init__(self, **_kwargs):
            self.initialized = False
            self.updated_steps: list[int] = []

        def initialize(self):
            self.initialized = True

        def update_weights(self, step_id: int, **_kwargs):
            self.updated_steps.append(step_id)

    import areal.engine.sglang_ext.sglang_awex_adapter as adapter_module

    monkeypatch.setattr(adapter_module, "AwexSGLangServerAdapter", _FakeAdapter)

    app = create_app(_FakeEngine())
    client = TestClient(app)

    init_resp = client.post(
        "/areal_awex_init",
        json={"meta_server_addr": "http://127.0.0.1:7081"},
    )
    assert init_resp.status_code == 200
    assert init_resp.json()["success"] is True

    update_resp = client.post("/areal_awex_update", json={"step_id": 7})
    assert update_resp.status_code == 200
    assert update_resp.json()["success"] is True


def test_parse_args_passes_namespace_to_server_args(monkeypatch):
    class _FakeServerArgs:
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--tensor-parallel-size", type=int, default=1)
            parser.add_argument("--model-path", type=str, default="fake-model")

        @classmethod
        def from_cli_args(cls, args):
            assert hasattr(args, "tensor_parallel_size")
            return types.SimpleNamespace(
                model_path=args.model_path,
                tp_size=args.tensor_parallel_size,
            )

    fake_sglang = types.ModuleType("sglang")
    fake_srt = types.ModuleType("sglang.srt")
    fake_server_args_mod = types.ModuleType("sglang.srt.server_args")
    setattr(fake_server_args_mod, "ServerArgs", _FakeServerArgs)

    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", fake_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", fake_server_args_mod)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "areal_sglang_server",
            "--host",
            "127.0.0.1",
            "--port",
            "31001",
            "--tensor-parallel-size",
            "2",
            "--model-path",
            "stub-model",
        ],
    )

    server_args, host, port = _parse_args()
    assert host == "127.0.0.1"
    assert port == 31001
    assert server_args.tp_size == 2
    assert server_args.model_path == "stub-model"


def test_awex_scheduler_launcher_is_pickle_safe():
    dumped = pickle.dumps(_AWEX_SCHEDULER_LAUNCHER)
    loaded = pickle.loads(dumped)
    assert callable(loaded)


def test_awex_adapter_exposes_hf_config_from_engine_model_config():
    pytest.importorskip("awex")

    marker = object()

    class _FakeServerArgs:
        tp_size = 1
        dp_size = 1
        pp_size = 1
        nnodes = 1
        node_rank = 0
        model_path = "unused-model"

    class _FakeModelConfig:
        hf_config = marker

    class _FakeSglEngine:
        server_args = _FakeServerArgs()
        model_config = _FakeModelConfig()

    adapter = AwexSGLangServerAdapter(
        sgl_engine=_FakeSglEngine(),
        meta_server_addr="127.0.0.1:7081",
        comm_backend="file",
    )
    assert adapter.hf_config is marker
    assert adapter.engine_name == "sglang"


def test_worker_patch_handles_qnorm_name(monkeypatch):
    pytest.importorskip("awex")

    import types

    class _FakeConverter:
        @staticmethod
        def _convert_layer_norm_param(converter, name, parameter, layer_number):
            if "query_layernorm" in name or "key_layernorm" in name:
                return [(name, parameter)]
            raise NotImplementedError(f"Unsupported layer norm parameter name: {name}")

    fake_module = types.ModuleType("awex.converter.sglang_converter")
    setattr(fake_module, "SGlangToHFWeightConverter", _FakeConverter)
    monkeypatch.setitem(sys.modules, "awex.converter.sglang_converter", fake_module)

    _patch_awex_sglang_converter()

    out = _FakeConverter._convert_layer_norm_param(
        _FakeConverter,
        "self_attn.q_norm.weight",
        object(),
        "0",
    )
    assert out[0][0] == "self_attn.query_layernorm.weight"


class _MockMegatronEngineForAwexFileWriter:
    """Mocked training engine that satisfies awex file-writer contract."""

    def __init__(self, source_model_path: str):
        self.comm_backend = "file"
        self.enable_debug_mode = False
        self.enable_colocate_mode = False
        self.source_model_path = source_model_path

    def save_hf_checkpoint(self, path: str):
        if os.path.exists(path):
            shutil.rmtree(path)
        shutil.copytree(self.source_model_path, path)

    def release_memory_occupation(self, tags=None):
        del tags

    def resume_memory_occupation(self, tags=None):
        del tags


@pytest.mark.slow
def test_awex_writer_reader_roundtrip_with_real_sglang_and_mocked_megatron_engine():
    """End-to-end file-backend awex update:

    - Real AReaL SGLang server process (reader path)
    - awex FileWeightExchangeWriter (writer path)
    - mocked Megatron engine as training source
    """

    if not has_gpu():
        pytest.skip("GPU required for real SGLang worker test")

    pytest.importorskip("sglang")
    pytest.importorskip("awex")

    from awex.writer.weights_writer import get_weights_exchange_writer

    from areal.api.cli_args import SGLangConfig
    from areal.infra.utils.proc import kill_process_tree
    from areal.utils import network

    host = network.gethostip()
    port, dist_port = network.find_free_ports(2)

    model_path = get_test_model_path()
    args = SGLangConfig.build_args(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=model_path,
            mem_fraction_static=0.15,
        ),
        host=host,
        port=port,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{host}:{dist_port}",
    )

    cmd = [sys.executable, "-m", "areal.engine.sglang_ext.areal_sglang_server"]
    for k, v in args.items():
        if v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])

    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
    base_url = f"http://{host}:{port}"

    temp_dir_obj = tempfile.TemporaryDirectory()
    ckpt_path = os.path.join(temp_dir_obj.name, "hf_ckpt")

    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            if check_server_health(base_url):
                break
            time.sleep(1)
        else:
            pytest.fail("AReaL SGLang server did not become healthy within timeout")

        init_resp = requests.post(
            f"{base_url}/areal_awex_init",
            json={
                "meta_server_addr": "127.0.0.1:7081",
                "comm_backend": "file",
                "engine_rank": 0,
                "num_engines": 1,
            },
            timeout=60,
        )
        assert init_resp.status_code == 200, init_resp.text
        assert init_resp.json()["success"] is True

        mock_train_engine = _MockMegatronEngineForAwexFileWriter(model_path)
        writer = get_weights_exchange_writer(mock_train_engine)
        writer.initialize()
        writer.write_weights(step_id=1, path=ckpt_path)

        update_resp = requests.post(
            f"{base_url}/areal_awex_update",
            json={"step_id": 1, "kwargs": {"path": ckpt_path}},
            timeout=300,
        )
        assert update_resp.status_code == 200, update_resp.text
        assert update_resp.json()["success"] is True
    finally:
        temp_dir_obj.cleanup()
        kill_process_tree(proc.pid, graceful=True)


@pytest.mark.slow
def test_awex_routes_with_real_sglang_worker_and_mocked_megatron_side():
    """Smoke-test awex routes on a real SGLang worker process.

    This test uses a mock awex adapter inside the worker process
    (AREAL_AWEX_USE_MOCK_ADAPTER=1), which represents a mocked training side.
    """

    if not has_gpu():
        pytest.skip("GPU required for real SGLang worker test")

    sglang_spec = pytest.importorskip("sglang")
    del sglang_spec

    from areal.api.cli_args import SGLangConfig
    from areal.infra.utils.proc import kill_process_tree
    from areal.utils import network

    host = network.gethostip()
    port, dist_port = network.find_free_ports(2)

    args = SGLangConfig.build_args(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=get_test_model_path(),
            mem_fraction_static=0.15,
        ),
        host=host,
        port=port,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{host}:{dist_port}",
    )
    cmd = [
        sys.executable,
        "-m",
        "areal.engine.sglang_ext.areal_sglang_server",
    ]
    for k, v in args.items():
        if v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])

    env = os.environ.copy()
    env["AREAL_AWEX_USE_MOCK_ADAPTER"] = "1"

    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout, env=env)
    base_url = f"http://{host}:{port}"

    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            if check_server_health(base_url):
                break
            time.sleep(1)
        else:
            pytest.fail("AReaL SGLang server did not become healthy within timeout")

        init_resp = httpx.post(
            f"{base_url}/areal_awex_init",
            json={"meta_server_addr": "http://127.0.0.1:7081"},
            timeout=30,
        )
        assert init_resp.status_code == 200, init_resp.text
        assert init_resp.json()["success"] is True

        update_resp = httpx.post(
            f"{base_url}/areal_awex_update",
            json={"step_id": 1},
            timeout=30,
        )
        assert update_resp.status_code == 200, update_resp.text
        assert update_resp.json()["success"] is True
    finally:
        kill_process_tree(proc.pid, graceful=True)
