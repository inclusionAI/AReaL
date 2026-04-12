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
from tests.experimental.inference_service.torchrun.run_awex_megatron_sglang_nccl import (
    _build_reader_server_env,
    _configure_single_gpu_runtime_env,
    _early_pin_visible_device_from_local_rank,
    _force_shutdown_on_signal,
    _global_pg_init_method,
    _sanitize_server_env,
)

from areal.engine.sglang_ext.areal_sglang_server import (
    _AWEX_SCHEDULER_LAUNCHER,
    _parse_args,
    create_app,
)
from areal.engine.sglang_ext.sglang_awex_adapter import (
    AwexSGLangServerAdapter,
    _AwexHFConfigProxy,
    _safe_rpc_error_message,
)
from areal.engine.sglang_ext.sglang_worker_extension import (
    _build_fallback_infer_engine_config,
    _inject_awex_parameter_aliases,
    _normalize_awex_param_meta_keys,
    _patch_awex_nccl_barrier_device_ids,
    _patch_awex_sglang_converter,
    _run_with_barrier_device_ids_stripped,
    _safe_exc_message,
)


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

    class _FakeHFConfig:
        def to_dict(self):
            return {"model_type": "qwen3"}

    marker = _FakeHFConfig()

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
    assert callable(getattr(adapter.hf_config, "to_dict", None))
    assert adapter.engine_name == "sglang"
    assert adapter.config.enable_dp_lm_head is True


def test_awex_adapter_unwraps_tokenizer_model_config_wrapper():
    pytest.importorskip("awex")

    class _FakeHFConfig:
        def to_dict(self):
            return {"model_type": "qwen3"}

    class _Wrapper:
        def __init__(self):
            self.hf_config = _FakeHFConfig()

    class _FakeTokenizerManager:
        model_config = _Wrapper()

    class _FakeServerArgs:
        tp_size = 1
        dp_size = 1
        pp_size = 1
        nnodes = 1
        node_rank = 0
        model_path = "unused-model"

    class _FakeSglEngine:
        server_args = _FakeServerArgs()
        tokenizer_manager = _FakeTokenizerManager()

    adapter = AwexSGLangServerAdapter(
        sgl_engine=_FakeSglEngine(),
        meta_server_addr="127.0.0.1:7081",
        comm_backend="file",
    )
    assert callable(getattr(adapter.hf_config, "to_dict", None))


def test_awex_adapter_prefers_nested_hf_config_over_wrapper_to_dict():
    pytest.importorskip("awex")

    class _FakeHFConfig:
        def to_dict(self):
            return {
                "architectures": ["Qwen3ForCausalLM"],
                "num_hidden_layers": 28,
            }

    class _WrapperWithToDict:
        def __init__(self):
            self.hf_config = _FakeHFConfig()

        def to_dict(self):
            return {"wrapper_only": True}

    class _FakeTokenizerManager:
        model_config = _WrapperWithToDict()

    class _FakeServerArgs:
        tp_size = 1
        dp_size = 1
        pp_size = 1
        nnodes = 1
        node_rank = 0
        model_path = "unused-model"

    class _FakeSglEngine:
        server_args = _FakeServerArgs()
        tokenizer_manager = _FakeTokenizerManager()

    adapter = AwexSGLangServerAdapter(
        sgl_engine=_FakeSglEngine(),
        meta_server_addr="127.0.0.1:7081",
        comm_backend="file",
    )
    cfg = adapter.hf_config.to_dict()
    assert cfg.get("architectures") == ["Qwen2ForCausalLM"]
    assert "wrapper_only" not in cfg


def test_awex_hf_config_proxy_maps_qwen3_arch_to_qwen2_for_awex():
    class _Cfg:
        def to_dict(self):
            return {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
            }

    proxied = _AwexHFConfigProxy(_Cfg())
    d = proxied.to_dict()
    assert d["architectures"] == ["Qwen2ForCausalLM"]
    assert proxied.architectures == ["Qwen2ForCausalLM"]


def test_awex_adapter_initialize_cleans_reader_on_failure(monkeypatch):
    pytest.importorskip("awex")

    class _FakeHFConfig:
        def to_dict(self):
            return {"model_type": "qwen3"}

    class _FakeServerArgs:
        tp_size = 1
        dp_size = 1
        pp_size = 1
        nnodes = 1
        node_rank = 0
        model_path = "unused-model"

    class _FakeModelConfig:
        hf_config = _FakeHFConfig()

    class _FakeSglEngine:
        server_args = _FakeServerArgs()
        model_config = _FakeModelConfig()

    cleanup_calls: list[str] = []

    class _FailingReader:
        def initialize(self):
            raise RuntimeError("boom")

        def close(self):
            cleanup_calls.append("close")

    import areal.engine.sglang_ext.sglang_awex_adapter as adapter_module

    monkeypatch.setattr(
        adapter_module,
        "logger",
        type(
            "_L",
            (),
            {
                "info": staticmethod(lambda *args, **kwargs: None),
                "warning": staticmethod(lambda *args, **kwargs: None),
            },
        )(),
    )

    import awex.reader.weights_reader as reader_mod

    monkeypatch.setattr(
        reader_mod, "get_weights_exchange_reader", lambda _adapter: _FailingReader()
    )

    adapter = AwexSGLangServerAdapter(
        sgl_engine=_FakeSglEngine(),
        meta_server_addr="127.0.0.1:7081",
        comm_backend="file",
    )
    with pytest.raises(RuntimeError, match="boom"):
        adapter.initialize()
    assert cleanup_calls == ["close"]
    assert adapter.weights_exchange_reader is None


def test_safe_rpc_error_message_handles_unprintable_message_obj():
    class _BadMessage:
        def __str__(self):
            raise UnicodeDecodeError("utf-8", b"\xf0", 0, 1, "invalid continuation")

    msg = _safe_rpc_error_message(_BadMessage())
    assert "BadMessage" in msg


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


def test_normalize_awex_param_meta_keys_maps_attention_aliases():
    src = {
        "model.layers.0.self_attn.o_proj.weight": {"shape": [1]},
        "model.layers.0.self_attn.qkv_proj.weight": {"shape": [2]},
        "model.layers.0.self_attn.query_layernorm.weight": {"shape": [3]},
    }
    out = _normalize_awex_param_meta_keys(src)
    assert "model.layers.0.attention.dense.weight" in out
    assert "model.layers.0.attention.query_key_value_proj.weight" in out
    assert "model.layers.0.attention.query_layernorm.weight" in out


def test_normalize_awex_param_meta_keys_adds_lm_head_alias_when_missing():
    src = {
        "model.embed_tokens.weight": {"shape": [151936, 1024]},
    }
    out = _normalize_awex_param_meta_keys(src)
    assert "lm_head.weight" in out
    assert out["lm_head.weight"] == out["model.embed_tokens.weight"]


def test_normalize_awex_param_meta_keys_maps_self_attn_even_without_qkv_signature():
    src = {
        "model.layers.0.self_attn.some_other_proj.weight": {"shape": [1]},
    }
    out = _normalize_awex_param_meta_keys(src)
    assert "model.layers.0.attention.some_other_proj.weight" in out


def test_normalize_awex_param_meta_keys_handles_nested_param_meta_container():
    src = {
        0: {
            "model.layers.0.self_attn.o_proj.weight": {"shape": [1]},
            "model.layers.0.self_attn.qkv_proj.weight": {"shape": [2]},
        }
    }
    out = _normalize_awex_param_meta_keys(src)
    assert isinstance(out, dict)
    assert "model.layers.0.attention.dense.weight" in out[0]
    assert "model.layers.0.attention.query_key_value_proj.weight" in out[0]


def test_normalize_awex_param_meta_keys_maps_name_field_values():
    src = [
        {
            "name": "model.layers.0.self_attn.o_proj.weight",
            "shape": [1024, 1024],
        }
    ]
    out = _normalize_awex_param_meta_keys(src)
    assert out[0]["name"] == "model.layers.0.attention.dense.weight"


def test_normalize_awex_param_meta_keys_adds_lm_head_alias_for_name_entries():
    src = [
        {
            "name": "model.embed_tokens.weight",
            "shape": [151936, 1024],
        }
    ]
    out = _normalize_awex_param_meta_keys(src)
    names = {item.get("name") for item in out if isinstance(item, dict)}
    assert "model.embed_tokens.weight" in names
    assert "lm_head.weight" in names


def test_normalize_awex_param_meta_keys_handles_list_results():
    src = [
        {
            "model.layers.0.self_attn.o_proj.weight": {"shape": [1]},
            "model.layers.0.self_attn.qkv_proj.weight": {"shape": [2]},
        }
    ]
    out = _normalize_awex_param_meta_keys(src)
    assert isinstance(out, list)
    assert "model.layers.0.attention.dense.weight" in out[0]
    assert "model.layers.0.attention.query_key_value_proj.weight" in out[0]


def test_inject_awex_parameter_aliases_adds_attention_aliases_for_self_attn_keys():
    dense = object()
    qkv = object()
    params = {
        "model.layers.0.self_attn.o_proj.weight": dense,
        "model.layers.0.self_attn.qkv_proj.weight": qkv,
    }

    added = _inject_awex_parameter_aliases(params)

    assert added >= 2
    assert params["model.layers.0.attention.dense.weight"] is dense
    assert params["model.layers.0.attention.query_key_value_proj.weight"] is qkv


def test_inject_awex_parameter_aliases_adds_self_attn_aliases_for_attention_keys():
    dense = object()
    qkv = object()
    params = {
        "model.layers.0.attention.dense.weight": dense,
        "model.layers.0.attention.query_key_value_proj.weight": qkv,
    }

    added = _inject_awex_parameter_aliases(params)

    assert added >= 2
    assert params["model.layers.0.self_attn.o_proj.weight"] is dense
    assert params["model.layers.0.self_attn.qkv_proj.weight"] is qkv


def test_build_fallback_infer_engine_config_from_scheduler_server_args():
    class _ServerArgs:
        tp_size = 2
        pp_size = 3
        dp_size = 4
        nnodes = 1
        node_rank = 0

    class _Scheduler:
        server_args = _ServerArgs()

    cfg = _build_fallback_infer_engine_config(_Scheduler())
    assert cfg.tp_size == 2
    assert cfg.pp_size == 3
    assert cfg.dp_size == 4
    assert cfg.enable_dp_lm_head is True


def test_patch_awex_nccl_barrier_device_ids_drops_device_ids(monkeypatch):
    import types

    calls = []

    def _barrier(*_args, **kwargs):
        calls.append(kwargs)

    fake_torch = types.ModuleType("torch")
    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.barrier = _barrier
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)

    class _FakeReader:
        def initialize(self):
            import torch.distributed as dist

            dist.barrier(group="g", device_ids=[0])

    class _FakeWriter:
        def initialize(self):
            import torch.distributed as dist

            dist.barrier(group="g", device_ids=[1])

    fake_reader_mod = types.ModuleType("awex.reader.nccl_reader")
    fake_writer_mod = types.ModuleType("awex.writer.nccl_writer")
    setattr(fake_reader_mod, "NCCLWeightsReader", _FakeReader)
    setattr(fake_writer_mod, "NCCLWeightsWriter", _FakeWriter)
    monkeypatch.setitem(sys.modules, "awex.reader.nccl_reader", fake_reader_mod)
    monkeypatch.setitem(sys.modules, "awex.writer.nccl_writer", fake_writer_mod)

    _patch_awex_nccl_barrier_device_ids()
    _FakeReader().initialize()
    _FakeWriter().initialize()

    assert len(calls) == 2
    assert all("device_ids" not in kwargs for kwargs in calls)


def test_run_with_barrier_device_ids_stripped(monkeypatch):
    import types

    calls = []

    def _barrier(*_args, **kwargs):
        calls.append(kwargs)

    fake_torch = types.ModuleType("torch")
    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.barrier = _barrier
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)

    def _fn():
        import torch.distributed as dist

        dist.barrier(group="g", device_ids=[0])
        return 7

    out = _run_with_barrier_device_ids_stripped(_fn)
    assert out == 7
    assert len(calls) == 1
    assert "device_ids" not in calls[0]


def test_force_shutdown_on_signal_calls_cleanup_and_exits():
    calls: list[str] = []
    exit_codes: list[int] = []

    def _destroy(_rank: int):
        calls.append("destroy")

    def _stop():
        calls.append("stop")

    def _exit(code: int):
        exit_codes.append(code)

    _force_shutdown_on_signal(
        rank=3,
        signum=2,
        destroy_dist_fn=_destroy,
        stop_server_and_meta_fn=_stop,
        exit_fn=_exit,
    )

    assert calls == ["destroy", "stop"]
    assert exit_codes == [130]


def test_sanitize_server_env_removes_torchrun_vars_but_keeps_nccl_vars():
    env = {
        "RANK": "2",
        "WORLD_SIZE": "8",
        "LOCAL_RANK": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "TORCHELASTIC_USE_AGENT_STORE": "True",
        "NCCL_DEBUG": "INFO",
    }
    out = _sanitize_server_env(env)
    assert "RANK" not in out
    assert "WORLD_SIZE" not in out
    assert "LOCAL_RANK" not in out
    assert "MASTER_ADDR" not in out
    assert "MASTER_PORT" not in out
    assert "TORCHELASTIC_USE_AGENT_STORE" not in out
    assert out["NCCL_DEBUG"] == "INFO"


def test_build_reader_server_env_sets_single_process_dist_env():
    out = _build_reader_server_env(
        parent_env={"NCCL_DEBUG": "INFO", "RANK": "7"},
        host="10.0.0.1",
        dist_port=29999,
        gpu_id="5",
    )
    assert out["CUDA_VISIBLE_DEVICES"] == "5"
    assert out["DEVICE"] == "0"
    assert out["RANK"] == "0"
    assert out["WORLD_SIZE"] == "1"
    assert out["LOCAL_RANK"] == "0"
    assert out["LOCAL_WORLD_SIZE"] == "1"
    assert out["MASTER_ADDR"] == "10.0.0.1"
    assert out["MASTER_PORT"] == "29999"
    assert out["TORCHELASTIC_USE_AGENT_STORE"] == "False"
    assert out["NCCL_DEBUG"] == "INFO"


def test_configure_single_gpu_runtime_env_normalizes_local_rank(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    selected = _configure_single_gpu_runtime_env(
        rank=1,
        world_size=2,
        local_rank=1,
        all_visible=["0", "1"],
    )
    assert selected == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["LOCAL_WORLD_SIZE"] == "1"
    assert os.environ["DEVICE"] == "0"
    assert os.environ["TORCHELASTIC_USE_AGENT_STORE"] == "False"


def test_global_pg_init_method_builds_tcp_endpoint():
    assert _global_pg_init_method("localhost", 23456) == "tcp://localhost:23456"


def test_early_pin_visible_device_from_local_rank(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.delenv("AREAL_ORIG_CUDA_VISIBLE_DEVICES", raising=False)

    _early_pin_visible_device_from_local_rank()

    assert os.environ["AREAL_ORIG_CUDA_VISIBLE_DEVICES"] == "0,1,2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["LOCAL_WORLD_SIZE"] == "1"
    assert os.environ["DEVICE"] == "0"


def test_safe_exc_message_handles_unprintable_exception():
    class _BadExc(Exception):
        def __str__(self):
            raise UnicodeDecodeError("utf-8", b"\xf0", 0, 1, "invalid continuation")

    msg = _safe_exc_message(_BadExc())
    assert "BadExc" in msg


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
