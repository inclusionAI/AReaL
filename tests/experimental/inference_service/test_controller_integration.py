"""Integration tests for GatewayInferenceController with real SGLang servers.

Requires GPU and a model. Run with:
    uv run pytest tests/experimental/inference_service/test_controller_integration.py -v -s

The test launches:
  1. A real SGLang server (GPU subprocess)
  2. A LocalScheduler (module-scoped)
  3. A GatewayInferenceController that spins up Gateway, Router, and Data Proxy
     micro-services in background threads.
"""

from __future__ import annotations

import subprocess
import sys
import time

import httpx
import pytest
import torch

from tests.experimental.inference_service.integration_utils import (
    EXPR_NAME,
    TRIAL_NAME,
    check_server_health,
    get_test_model_path,
    has_gpu,
)

pytestmark = pytest.mark.sglang
SERVER_STARTUP_TIMEOUT = 180  # seconds


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def sglang_server():
    """Launch an SGLang server and yield its (host, port, base_url)."""
    if not has_gpu():
        pytest.skip("GPU required for SGLang server")

    from areal.api.cli_args import SGLangConfig
    from areal.infra.utils.proc import kill_process_tree
    from areal.utils import network

    host = network.gethostip()
    port, dist_port = network.find_free_ports(2)

    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=get_test_model_path(),
            mem_fraction_static=0.3,
        ),
        host=host,
        port=port,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{host}:{dist_port}",
    )

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
    base_url = f"http://{host}:{port}"

    t0 = time.time()
    while time.time() - t0 < SERVER_STARTUP_TIMEOUT:
        if check_server_health(base_url):
            break
        time.sleep(1)

    if time.time() - t0 >= SERVER_STARTUP_TIMEOUT:
        kill_process_tree(process.pid, graceful=True)
        pytest.fail("SGLang server did not become healthy within timeout")

    yield {"host": host, "port": port, "base_url": base_url, "process": process}

    kill_process_tree(process.pid, graceful=True)


@pytest.fixture(scope="module")
def model_path() -> str:
    """Return the test model path."""
    return get_test_model_path()


@pytest.fixture(scope="module")
def local_scheduler(tmp_path_factory):
    """Create a LocalScheduler for testing."""
    if not has_gpu():
        pytest.skip("GPU required for LocalScheduler")

    from areal.infra.scheduler.local import LocalScheduler

    tmp_path = tmp_path_factory.mktemp("local_scheduler")
    fileroot = tmp_path / "fileroot"
    fileroot.mkdir()
    name_resolve_root = tmp_path / "name_resolve"
    name_resolve_root.mkdir()

    scheduler = LocalScheduler(
        gpu_devices=[0],
        log_dir=str(tmp_path),
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        fileroot=str(fileroot),
        nfs_record_root=str(name_resolve_root),
    )
    yield scheduler
    scheduler.delete_workers(None)


@pytest.fixture(scope="module")
def gateway_controller(sglang_server, local_scheduler, model_path):
    """Create and initialize a GatewayInferenceController, yield it, then destroy."""
    if not has_gpu():
        pytest.skip("GPU required")
    from areal.api.cli_args import SchedulingSpec
    from areal.api.io_struct import LocalInfServerInfo
    from areal.experimental.inference_service.controller.config import (
        GatewayControllerConfig,
    )
    from areal.experimental.inference_service.controller.controller import (
        GatewayInferenceController,
    )

    config = GatewayControllerConfig(
        tokenizer_path=model_path,
        model_path=model_path,
        scheduling_spec=(
            SchedulingSpec(
                gpu=0,
                cpu=1,
                mem=4,
                cmd="python -m areal.experimental.inference_service.guard",
            ),
        ),
        admin_api_key="test-admin",
        consumer_batch_size=4,
        max_head_offpolicyness=1024,
        setup_timeout=180.0,
    )

    ctrl = GatewayInferenceController(config=config, scheduler=local_scheduler)

    server_info = LocalInfServerInfo(
        process=None,
        host=sglang_server["host"],
        port=sglang_server["port"],
    )

    ctrl.initialize(
        role="rollout",
        server_infos=[server_info],
    )

    try:
        yield ctrl
    finally:
        ctrl.destroy()


# =============================================================================
# TestControllerLifecycle
# =============================================================================


@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerLifecycle:
    """Verify controller lifecycle: init starts services, properties set, destroy cleans up."""

    def test_gateway_services_started(self, gateway_controller):
        """After initialization, gateway services should be running."""
        # Verify addresses were resolved by the scheduler
        assert gateway_controller._gateway_addr != ""
        assert gateway_controller._router_addr != ""
        assert len(gateway_controller._data_proxy_addrs) > 0

    def test_gateway_health(self, gateway_controller):
        """The gateway HTTP service should respond healthy."""
        addr = gateway_controller._gateway_addr
        resp = httpx.get(f"{addr}/health", timeout=10.0)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_router_health(self, gateway_controller):
        """The router HTTP service should respond healthy with 1 worker."""
        resp = httpx.get(f"{gateway_controller._router_addr}/health", timeout=10.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["workers"] >= 1

    def test_data_proxy_health(self, gateway_controller):
        """The data proxy HTTP service should respond healthy."""
        dp_addr = gateway_controller._data_proxy_addrs[0]
        resp = httpx.get(f"{dp_addr}/health", timeout=10.0)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_proxy_gateway_addr_set(self, gateway_controller):
        """proxy_gateway_addr should point to the gateway port."""
        addr = gateway_controller.proxy_gateway_addr
        # proxy_gateway_addr should be a valid http URL
        assert addr.startswith("http://")
        assert addr == gateway_controller._gateway_addr


# =============================================================================
# TestControllerVersioning
# =============================================================================


@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerVersioning:
    """Verify version management on the controller."""

    def test_default_version_is_zero(self, gateway_controller):
        """Controller should start at version 0."""
        assert gateway_controller.get_version() == 0

    def test_set_version_updates_local(self, gateway_controller):
        """set_version should update the local version."""
        gateway_controller.set_version(5)
        assert gateway_controller.get_version() == 5
        # Reset for other tests
        gateway_controller.set_version(0)

    def test_set_version_does_not_raise_without_broadcast(self, gateway_controller):
        """set_version updates local state without broadcasting (broadcast removed)."""
        # Weight-update forwarding (including /set_version broadcast) has been
        # removed from the gateway HTTP stack.  This test verifies that
        # set_version still works for local version tracking.
        gateway_controller.set_version(10)
        assert gateway_controller.get_version() == 10
        # Verify gateway is still healthy (no stale broadcast attempted)
        addr = gateway_controller.proxy_gateway_addr
        resp = httpx.get(f"{addr}/health", timeout=10.0)
        assert resp.status_code == 200
        # Reset
        gateway_controller.set_version(0)


# =============================================================================
# TestControllerPauseResume
# =============================================================================


@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerPauseResume:
    """Verify pause/resume broadcasts to workers."""

    def test_pause_broadcasts_to_workers(self, gateway_controller):
        """pause() should broadcast pause to all data proxy workers."""
        gateway_controller.pause()
        # Verify data proxy reports paused
        dp_addr = gateway_controller._data_proxy_addrs[0]
        resp = httpx.get(f"{dp_addr}/health", timeout=10.0)
        assert resp.status_code == 200
        assert resp.json().get("paused") is True
        # Clean up: resume
        gateway_controller.resume()

    def test_resume_broadcasts_to_workers(self, gateway_controller):
        """resume() should broadcast resume to all data proxy workers."""
        gateway_controller.pause()
        gateway_controller.resume()
        # Verify data proxy is no longer paused
        dp_addr = gateway_controller._data_proxy_addrs[0]
        resp = httpx.get(f"{dp_addr}/health", timeout=10.0)
        assert resp.status_code == 200
        assert resp.json().get("paused") is False

    def test_pause_resume_roundtrip_keeps_services_healthy(self, gateway_controller):
        """After pause → resume, all services should remain healthy."""
        gateway_controller.pause()
        time.sleep(0.5)
        gateway_controller.resume()
        time.sleep(0.5)

        # Gateway still healthy
        addr = gateway_controller.proxy_gateway_addr
        resp = httpx.get(f"{addr}/health", timeout=10.0)
        assert resp.status_code == 200

        # Router still healthy
        resp = httpx.get(f"{gateway_controller._router_addr}/health", timeout=10.0)
        assert resp.status_code == 200


# =============================================================================
# TestControllerRolloutBatch
# =============================================================================


@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerRolloutBatch:
    """Test rollout_batch through the controller with SimpleAgent workflow."""

    def test_rollout_batch_with_simple_agent(self, gateway_controller):
        """rollout_batch with SimpleAgent should return list of trajectory dicts."""
        data = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            }
        ]

        result = gateway_controller.rollout_batch(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        traj = result[0]
        assert isinstance(traj, dict)
        assert "input_ids" in traj
        # Values should be RTensor (matching RolloutController API)
        from areal.infra.rpc.rtensor import RTensor

        assert isinstance(traj["input_ids"], RTensor)
        assert traj["input_ids"].ndim == 2


# =============================================================================
# TestControllerPrepareBatch
# =============================================================================


class _FakeDataLoader:
    """Minimal dataloader stub for prepare_batch tests.

    Yields one batch of dicts per iteration with a `.batch_size` attribute,
    which is all that `workflow_executor.prepare_batch` requires.
    """

    def __init__(self, items: list[dict], batch_size: int = 1) -> None:
        self._items = items
        self.batch_size = batch_size

    def __iter__(self):
        # Yield a single batch containing all items, matching StatefulDataLoader
        # semantics where each iteration yields a batch (list of dicts).
        yield self._items


@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerPrepareBatch:
    """Test prepare_batch through the controller with SimpleAgent workflow."""

    def test_prepare_batch_returns_results(self, gateway_controller):
        """prepare_batch should return a list of trajectory dicts."""
        items = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            },
            {
                "messages": [{"role": "user", "content": "What is 3+3?"}],
                "answer": "6",
            },
        ]
        dataloader = _FakeDataLoader(items, batch_size=len(items))

        result = gateway_controller.prepare_batch(
            dataloader=dataloader,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert isinstance(result, list)
        assert len(result) > 0
        traj = result[0]
        assert isinstance(traj, dict)
        assert "input_ids" in traj
        from areal.infra.rpc.rtensor import RTensor

        assert isinstance(traj["input_ids"], RTensor)
        assert traj["input_ids"].ndim == 2


# =============================================================================
# TestControllerSubmitWait
# =============================================================================


@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerSubmitWait:
    """Test submit/wait API on the controller."""

    def test_submit_returns_task_id(self, gateway_controller):
        """submit() should return an integer task ID."""
        data = {
            "messages": [{"role": "user", "content": "What is 1+1?"}],
            "answer": "2",
        }

        task_id = gateway_controller.submit(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert isinstance(task_id, int)
        assert task_id >= 0

        # Wait for the submitted task to finish so it doesn't leak
        gateway_controller.wait(count=1, timeout=120.0)

    def test_submit_wait_roundtrip(self, gateway_controller):
        """submit + wait should complete a full roundtrip."""
        data = {
            "messages": [{"role": "user", "content": "Say hello."}],
            "answer": "hello",
        }

        task_id = gateway_controller.submit(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert isinstance(task_id, int)

        results = gateway_controller.wait(count=1, timeout=120.0)

        assert results is not None
        assert len(results) == 1
        # Each result should be a dict (interaction data) or None
        result = results[0]
        assert result is None or isinstance(result, dict)


# =============================================================================
# TestControllerFullInitialization (no pre-existing server_infos)
# =============================================================================


@pytest.fixture(scope="module")
def gateway_controller_full_init(local_scheduler, model_path):
    """Create a GatewayInferenceController that launches SGLang via the full init path.

    Unlike ``gateway_controller`` which passes pre-existing ``server_infos``,
    this fixture lets the controller create RPC workers, create
    RPCGuard on them, and fork SGLang servers internally.
    """
    if not has_gpu():
        pytest.skip("GPU required")

    from areal.api.cli_args import SchedulingSpec
    from areal.experimental.inference_service.controller.config import (
        GatewayControllerConfig,
    )
    from areal.experimental.inference_service.controller.controller import (
        GatewayInferenceController,
    )

    config = GatewayControllerConfig(
        tokenizer_path=model_path,
        model_path=model_path,
        backend="sglang:d1",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1, cmd="python -m areal.experimental.inference_service.guard"
            ),
        ),
        admin_api_key="test-admin",
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
    )

    server_args = {
        "skip_tokenizer_init": True,
        "mem_fraction_static": 0.3,
    }

    ctrl = GatewayInferenceController(config=config, scheduler=local_scheduler)
    ctrl.initialize(
        role="rollout-full",
        server_args=server_args,
    )

    try:
        yield ctrl
    finally:
        ctrl.destroy()


@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerFullInitialization:
    """Test the full initialization path where the controller launches SGLang itself.

    This covers the code path where ``server_infos`` is **not** provided, so the
    controller creates RPC workers, creates RPCGuard on each, forks
    ``launch_server``, then forks data proxies from the workers.
    """

    def test_server_infos_populated(self, gateway_controller_full_init):
        """server_infos should be populated after full init."""
        ctrl = gateway_controller_full_init
        assert len(ctrl.server_infos) > 0
        info = ctrl.server_infos[0]
        assert info.host
        assert info.port > 0

    def test_inf_server_health(self, gateway_controller_full_init):
        """The inference server launched by the controller should be healthy."""
        ctrl = gateway_controller_full_init
        for addr in ctrl._inf_addrs:
            resp = httpx.get(f"{addr}/health", timeout=30.0)
            assert resp.status_code == 200

    def test_gateway_health(self, gateway_controller_full_init):
        """Gateway should be healthy after full init."""
        ctrl = gateway_controller_full_init
        resp = httpx.get(f"{ctrl._gateway_addr}/health", timeout=10.0)
        assert resp.status_code == 200

    def test_data_proxy_health(self, gateway_controller_full_init):
        """Data proxies should be healthy after full init."""
        ctrl = gateway_controller_full_init
        for dp_addr in ctrl._data_proxy_addrs:
            resp = httpx.get(f"{dp_addr}/health", timeout=10.0)
            assert resp.status_code == 200

    def test_data_proxy_forked_from_inf_workers(self, gateway_controller_full_init):
        """Data proxies should have been forked via RPCGuard in full init path."""
        ctrl = gateway_controller_full_init
        inf_role = f"rollout-full{ctrl._INF_SUFFIX}"
        # RPCGuard worker role should be in service_roles
        assert inf_role in ctrl._service_roles
        # Data proxies are forked via RPCGuard /fork, tracked in _forked_services
        dp_entries = [
            (addr, role, idx)
            for addr, role, idx in ctrl._forked_services
            if role == "data-proxy"
        ]
        assert len(dp_entries) > 0, "No data-proxy entries in _forked_services"

    def test_chat_completion_via_gateway(self, gateway_controller_full_init):
        """Full e2e: start_session → /chat/completions → validate → end_session."""
        ctrl = gateway_controller_full_init
        gw = ctrl._gateway_addr
        admin_key = "test-admin"

        # --- grant capacity (required before start_session) ---
        resp = httpx.post(
            f"{gw}/grant_capacity",
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=10.0,
        )
        assert resp.status_code == 200, resp.text

        # --- start session ---
        resp = httpx.post(
            f"{gw}/rl/start_session",
            json={"task_id": "full-init-chat-test"},
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=30.0,
        )
        assert resp.status_code == 201, resp.text
        session = resp.json()
        session_api_key = session["api_key"]

        # --- non-streaming chat completion ---
        resp = httpx.post(
            f"{gw}/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_completion_tokens": 64,
                "temperature": 0.0,
                "stream": False,
            },
            headers={"Authorization": f"Bearer {session_api_key}"},
            timeout=60.0,
        )
        assert resp.status_code == 200, resp.text
        completion = resp.json()

        # Validate OpenAI-compatible structure
        assert completion["object"] == "chat.completion"
        assert "id" in completion
        assert "choices" in completion
        assert len(completion["choices"]) == 1

        choice = completion["choices"][0]
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert len(choice["message"]["content"]) > 0
        assert choice["finish_reason"] in ("stop", "length")

        # Validate usage
        assert "usage" in completion
        usage = completion["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )

        # --- end session ---
        resp = httpx.post(
            f"{gw}/rl/end_session",
            headers={"Authorization": f"Bearer {session_api_key}"},
            timeout=10.0,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["interaction_count"] == 1

    def test_rollout_batch_with_simple_agent(self, gateway_controller_full_init):
        """rollout_batch with SimpleAgent should return list of trajectory dicts."""
        ctrl = gateway_controller_full_init
        data = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            }
        ]

        result = ctrl.rollout_batch(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        traj = result[0]
        assert isinstance(traj, dict)
        assert "input_ids" in traj
        from areal.infra.rpc.rtensor import RTensor

        assert isinstance(traj["input_ids"], RTensor)
        assert traj["input_ids"].ndim == 2

    def test_rtensor_localize_on_rollout_result(self, gateway_controller_full_init):
        """RTensor.localize() should successfully fetch tensors from data proxy."""
        ctrl = gateway_controller_full_init
        data = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            }
        ]

        result = ctrl.rollout_batch(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        traj = result[0]
        assert isinstance(traj, dict)
        assert "input_ids" in traj

        from areal.infra.rpc.rtensor import RTensor

        # The result values should be RTensors with meta data (not yet fetched)
        rtensor_input_ids = traj["input_ids"]
        assert isinstance(rtensor_input_ids, RTensor)
        assert rtensor_input_ids.data.is_meta

        # Verify shard points to a data proxy address (not just a bare IP)
        assert ":" in rtensor_input_ids.shard.node_addr

        # Localize the trajectory — this fetches tensors from the data proxy
        local_traj = RTensor.localize(traj)

        # After localization, values should be real tensors (not RTensor)
        assert isinstance(local_traj, dict)
        assert "input_ids" in local_traj
        assert isinstance(local_traj["input_ids"], torch.Tensor)
        assert not local_traj["input_ids"].is_meta
        assert local_traj["input_ids"].ndim == 2
        assert local_traj["input_ids"].shape[0] >= 1  # at least 1 sample

        # Check other expected keys are also localized
        if "attention_mask" in local_traj:
            assert isinstance(local_traj["attention_mask"], torch.Tensor)
            assert not local_traj["attention_mask"].is_meta

    def test_rtensor_localize_batch4(self, gateway_controller_full_init):
        """RTensor.localize() on a batch of 4 should produce 4 trajectory dicts."""
        ctrl = gateway_controller_full_init
        batch_size = 4
        data = [
            {
                "messages": [{"role": "user", "content": f"What is {i}+{i}?"}],
                "answer": str(i * 2),
            }
            for i in range(batch_size)
        ]

        result = ctrl.rollout_batch(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == batch_size

        from areal.infra.rpc.rtensor import RTensor

        # Localize each trajectory and verify tensors
        for i, traj in enumerate(result):
            assert isinstance(traj, dict), f"Trajectory {i} is not a dict"
            assert "input_ids" in traj, f"Trajectory {i} missing input_ids"

            local_traj = RTensor.localize(traj)
            assert isinstance(local_traj["input_ids"], torch.Tensor)
            assert not local_traj["input_ids"].is_meta
            assert local_traj["input_ids"].ndim == 2
