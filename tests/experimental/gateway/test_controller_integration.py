"""Integration tests for GatewayRolloutController with real SGLang servers.

Requires GPU and a model. Marked @pytest.mark.slow to exclude from default CI.
Run manually:
    uv run pytest tests/experimental/gateway/test_controller_integration.py -v -s

The test launches:
  1. A real SGLang server (GPU subprocess)
  2. A LocalScheduler (function-scoped)
  3. A GatewayRolloutController that spins up Gateway, Router, and Data Proxy
     micro-services in background threads.
"""

from __future__ import annotations

import subprocess
import sys
import time

import httpx
import pytest
import torch

from tests.experimental.gateway.integration_utils import (
    EXPR_NAME,
    TRIAL_NAME,
    check_server_health,
    get_test_model_path,
    has_gpu,
)

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


@pytest.fixture
def local_scheduler(tmp_path):
    """Create a LocalScheduler for testing."""
    if not has_gpu():
        pytest.skip("GPU required for LocalScheduler")

    from areal.infra.scheduler.local import LocalScheduler

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


@pytest.fixture
def gateway_controller(sglang_server, local_scheduler, model_path, tmp_path):
    """Create and initialize a GatewayRolloutController, yield it, then destroy."""
    if not has_gpu():
        pytest.skip("GPU required")

    from areal.api.alloc_mode import AllocationMode
    from areal.api.cli_args import SchedulingSpec
    from areal.api.io_struct import LocalInfServerInfo
    from areal.experimental.gateway.controller.config import GatewayControllerConfig
    from areal.experimental.gateway.controller.controller import (
        GatewayRolloutController,
    )

    config = GatewayControllerConfig(
        tokenizer_path=model_path,
        model_path=model_path,
        scheduling_spec=(SchedulingSpec(),),
        admin_api_key="test-admin",
        consumer_batch_size=2,
        setup_timeout=180.0,
    )

    ctrl = GatewayRolloutController(config=config, scheduler=local_scheduler)

    server_info = LocalInfServerInfo(
        process=None,
        host=sglang_server["host"],
        port=sglang_server["port"],
    )

    ctrl.initialize(
        role="rollout",
        alloc_mode=AllocationMode.from_str("sglang:d1"),
        server_infos=[server_info],
    )

    try:
        yield ctrl
    finally:
        ctrl.destroy()


# =============================================================================
# TestControllerLifecycle
# =============================================================================


@pytest.mark.slow
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

    def test_callback_addr_set(self, gateway_controller):
        """callback_addr should be a valid host:port string after init."""
        addr = gateway_controller.callback_addr
        assert ":" in addr
        host, port_str = addr.rsplit(":", 1)
        assert len(host) > 0
        assert int(port_str) > 0

    def test_proxy_gateway_addr_set(self, gateway_controller):
        """proxy_gateway_addr should point to the gateway port."""
        addr = gateway_controller.proxy_gateway_addr
        # proxy_gateway_addr should be a valid http URL
        assert addr.startswith("http://")
        assert addr == gateway_controller._gateway_addr


# =============================================================================
# TestControllerVersioning
# =============================================================================


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerRolloutBatch:
    """Test rollout_batch through the controller with SimpleAgent workflow."""

    def test_rollout_batch_with_simple_agent(self, gateway_controller):
        """rollout_batch with SimpleAgent should return concatenated batch dict."""
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
        assert isinstance(result, dict)
        assert len(result) > 0
        assert "input_ids" in result
        # Values should be RTensor (matching RolloutController API)
        from areal.infra.rpc.rtensor import RTensor

        assert isinstance(result["input_ids"], RTensor)
        assert result["input_ids"].ndim == 2
        assert hasattr(result["input_ids"], "shards")
        assert len(result["input_ids"].shards) > 0

    def test_rollout_batch_with_should_accept_fn_rejects(self, gateway_controller):
        """rollout_batch with a rejecting should_accept_fn returns empty dict."""

        def reject_all(trajectory: dict) -> bool:
            return False

        data = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            }
        ]

        result = gateway_controller.rollout_batch(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
            should_accept_fn=reject_all,
        )

        # All trajectories should be rejected, so the result is an empty dict
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_rollout_batch_with_should_accept_fn_accepts(self, gateway_controller):
        """rollout_batch with an accepting should_accept_fn returns concatenated batch dict."""

        def accept_all(trajectory: dict) -> bool:
            return True

        data = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            }
        ]

        result = gateway_controller.rollout_batch(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
            should_accept_fn=accept_all,
        )

        assert isinstance(result, dict)
        assert len(result) > 0
        assert "input_ids" in result
        # Values should be RTensor (matching RolloutController API)
        from areal.infra.rpc.rtensor import RTensor

        assert isinstance(result["input_ids"], RTensor)
        assert result["input_ids"].ndim == 2
        assert hasattr(result["input_ids"], "shards")
        assert len(result["input_ids"].shards) > 0


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


@pytest.mark.slow
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerPrepareBatch:
    """Test prepare_batch through the controller with SimpleAgent workflow."""

    def test_prepare_batch_returns_results(self, gateway_controller):
        """prepare_batch should return a concatenated batch dict."""
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

        assert isinstance(result, dict)
        assert len(result) > 0
        assert "input_ids" in result
        # Values should be RTensor (matching RolloutController API)
        from areal.infra.rpc.rtensor import RTensor

        assert isinstance(result["input_ids"], RTensor)
        assert result["input_ids"].ndim == 2
        assert hasattr(result["input_ids"], "shards")
        assert len(result["input_ids"].shards) > 0

    def test_prepare_batch_with_should_accept_fn_rejects(self, gateway_controller):
        """prepare_batch with a rejecting should_accept_fn returns empty dict."""

        def reject_all(trajectory: dict) -> bool:
            return False

        items = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            },
        ]
        dataloader = _FakeDataLoader(items, batch_size=len(items))

        result = gateway_controller.prepare_batch(
            dataloader=dataloader,
            workflow="tests.experimental.openai.utils.SimpleAgent",
            should_accept_fn=reject_all,
            dynamic_bs=True,
        )

        # All trajectories should be rejected, so the result is an empty dict
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_prepare_batch_with_should_accept_fn_accepts(self, gateway_controller):
        """prepare_batch with an accepting should_accept_fn returns concatenated batch dict."""

        def accept_all(trajectory: dict) -> bool:
            return True

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
            should_accept_fn=accept_all,
        )

        assert isinstance(result, dict)
        assert len(result) > 0
        assert "input_ids" in result
        # Values should be RTensor (matching RolloutController API)
        from areal.infra.rpc.rtensor import RTensor

        assert isinstance(result["input_ids"], RTensor)
        assert result["input_ids"].ndim == 2
        assert hasattr(result["input_ids"], "shards")
        assert len(result["input_ids"].shards) > 0


# =============================================================================
# TestControllerSubmitWait
# =============================================================================


@pytest.mark.slow
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


@pytest.fixture
def gateway_controller_full_init(local_scheduler, model_path, tmp_path):
    """Create a GatewayRolloutController that launches SGLang via the full init path.

    Unlike ``gateway_controller`` which passes pre-existing ``server_infos``,
    this fixture lets the controller create RPC workers, create
    RPCGuard on them, and fork SGLang servers internally.
    """
    if not has_gpu():
        pytest.skip("GPU required")

    from areal.api.alloc_mode import AllocationMode
    from areal.api.cli_args import SchedulingSpec
    from areal.experimental.gateway.controller.config import GatewayControllerConfig
    from areal.experimental.gateway.controller.controller import (
        GatewayRolloutController,
    )

    config = GatewayControllerConfig(
        tokenizer_path=model_path,
        model_path=model_path,
        scheduling_spec=(
            SchedulingSpec(gpu=1, cmd="python -m areal.experimental.gateway.guard"),
        ),
        admin_api_key="test-admin",
        consumer_batch_size=2,
        setup_timeout=300.0,
    )

    alloc_mode = AllocationMode.from_str("sglang:d1")
    server_args = {
        "skip_tokenizer_init": True,
        "mem_fraction_static": 0.3,
    }

    ctrl = GatewayRolloutController(config=config, scheduler=local_scheduler)
    ctrl.initialize(
        role="rollout-full",
        alloc_mode=alloc_mode,
        server_args=server_args,
    )

    try:
        yield ctrl
    finally:
        ctrl.destroy()


@pytest.mark.slow
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
        """Data proxies should have been forked (not standalone) in full init path."""
        ctrl = gateway_controller_full_init
        inf_role = f"rollout-full{ctrl._INF_SUFFIX}"
        dp_role = f"rollout-full{ctrl._DATA_PROXY_SUFFIX}"
        # Both roles should be in service_roles
        assert inf_role in ctrl._service_roles
        assert dp_role in ctrl._service_roles
