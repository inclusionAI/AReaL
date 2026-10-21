# SPDX-License-Identifier: Apache-2.0

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import psutil
import pytest
from fastapi import FastAPI

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.autopilot.elastic.runtime.training.service import TrainingFrontend
from areal.autopilot.elastic.runtime.training.service import (
    create_app as standby_app,
)
from areal.autopilot.elastic.runtime.training.workers import (
    LocalV2TrainController,
    TrainGroup,
)
from areal.v2.training_service.gateway.app import create_app as gateway_app
from areal.v2.training_service.gateway.config import GatewayConfig
from areal.v2.training_service.router.app import create_app as router_app
from areal.v2.training_service.router.config import RouterConfig


def make_frontend():
    config = TrainEngineConfig(
        experiment_name="elastic-test",
        trial_name="frontend",
        backend="fsdp:d1",
        _version="v2",
        admin_api_key="test-admin",
        scheduling_spec=(SchedulingSpec(gpu=1),),
    )
    return TrainingFrontend("not_imported.Engine", config, Mock())


@pytest.mark.asyncio
async def test_stable_key_only_routes_to_model_after_publication():
    """Real gateway/router apps keep staging private and preserve the key on park."""
    admin = "test-admin"
    router = router_app(RouterConfig(admin_api_key=admin))
    gateway = gateway_app(
        GatewayConfig(admin_api_key=admin, router_addr="http://router")
    )
    standby = standby_app()
    model = FastAPI()

    @model.get("/get_version")
    async def version():
        return {"result": 2}

    transports = {
        "standby": httpx.ASGITransport(app=standby),
        "model": httpx.ASGITransport(app=model),
    }

    async def dispatch(request):
        return await transports[request.url.host].handle_async_request(request)

    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=router), base_url="http://router"
        ) as routing,
        httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as upstream,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway), base_url="http://gateway"
        ) as public,
    ):
        gateway.state.router_client = routing
        gateway.state.upstream_client = upstream
        admin_headers = {"Authorization": f"Bearer {admin}"}

        async def register(address, key):
            response = await routing.post(
                "/register",
                headers=admin_headers,
                json={"model_addr": address, "api_key": key},
            )
            assert response.status_code == 200

        async def check_suspended():
            headers = {"Authorization": "Bearer stable"}
            assert (await public.get("/health")).status_code == 200
            assert (await routing.get("/health")).status_code == 200
            assert (
                await public.get("/get_version", headers=headers)
            ).status_code == 503
            response = await public.post("/sft/train", headers=headers, json={})
            assert response.status_code == 503
            assert response.json() == {
                "error": "training_unavailable",
                "retryable": True,
            }

        await register("http://standby", "stable")
        await check_suspended()
        # Workers are initialized/restored using their private staging key.
        await register("http://model", "staging")
        await check_suspended()
        staged = await public.get(
            "/get_version", headers={"Authorization": "Bearer staging"}
        )
        assert staged.status_code == 200 and staged.json() == {"result": 2}
        await register("http://model", "stable")
        ready = await public.get(
            "/get_version", headers={"Authorization": "Bearer stable"}
        )
        assert ready.status_code == 200 and ready.json() == {"result": 2}
        assert (
            await public.get(
                "/get_version", headers={"Authorization": "Bearer staging"}
            )
        ).status_code == 401
        await register("http://standby", "stable")
        await routing.post(
            "/unregister", headers=admin_headers, json={"model_addr": "http://model"}
        )
        await check_suspended()


def test_cpu_frontend_role_has_zero_gpus_and_no_process_group():
    """CPU guard shares the scheduler without borrowing the training role or PG."""
    frontend = make_frontend()
    frontend.scheduler.get_workers.return_value = [
        SimpleNamespace(ip="127.0.0.1", worker_ports=["19001"])
    ]
    frontend.control._async_fork_on_guard = AsyncMock(
        side_effect=[("127.0.0.1", port) for port in (19002, 19003, 19004)]
    )
    frontend._register = Mock()
    frontend.probe = Mock(return_value={"training_status": 503})
    frontend.start()
    job = frontend.scheduler.create_workers.call_args.args[0]
    assert job.role == frontend.role and job.role != "actor-guard"
    assert job.replicas == 1
    assert job.tasks[0].gpu == 0
    assert job.tasks[0].env_vars["CUDA_VISIBLE_DEVICES"] == ""
    assert frontend.control._service_roles == [frontend.role]
    assert not frontend.control._own_process_group


@pytest.mark.parametrize("role", ["router", "gateway"])
@pytest.mark.asyncio
async def test_generation_borrows_frontend_without_owning_its_cleanup(role):
    """Destroying a GPU generation must not kill persistent frontend processes."""
    frontend = make_frontend()
    frontend.endpoints = {role: ("127.0.0.1", 19002)}
    actor = LocalV2TrainController(
        "not_imported.Engine",
        frontend.config,
        Mock(),
        gpu_uuids=["GPU-test"],
        frontend=frontend,
    )
    with patch.object(
        type(frontend.control),
        "_async_fork_on_guard",
        new_callable=AsyncMock,
    ) as fork:
        result = await actor._async_fork_on_guard("http://gpu-guard", role, 0, [])
    assert result == frontend.endpoints[role]
    fork.assert_not_called()
    assert actor._forked_services == actor._service_roles == []


def test_retired_generation_cannot_park_the_current_generation():
    """Late cleanup from an old generation must not withdraw a newer route."""
    frontend = make_frontend()
    frontend.active_generation = "current"
    frontend._register = Mock()
    assert not frontend.park("retired")
    assert frontend.active_generation == "current"
    frontend._register.assert_not_called()


def test_lost_publication_response_retains_generation_cleanup_ownership():
    """A committed registration can lose its response and still require parking."""
    frontend = make_frontend()
    frontend.endpoints = {"standby": ("127.0.0.1", 19002)}
    frontend._register = Mock(side_effect=TimeoutError)
    actor = SimpleNamespace(launch_id="new", _model_addr="http://model")
    with pytest.raises(TimeoutError):
        frontend.publish(actor, 2)
    assert frontend.active_generation == "new"
    frontend._register.side_effect = None
    frontend.probe = Mock()
    assert frontend.park("new")
    frontend._register.assert_called_with(frontend.address("standby"))
    assert frontend.active_generation is None


def test_failed_parking_still_releases_gpus_without_claiming_suspension(
    tmp_path, monkeypatch
):
    """Routing failure cannot skip teardown or be reported as a safe suspension."""
    group = object.__new__(TrainGroup)
    group.root, group.step, group.events = tmp_path, 2, []
    group.is_v2, group.worker_role = False, "actor"
    group.devices, group.uuids = [4], ["GPU-test"]
    group.scheduler = SimpleNamespace(_workers={})
    group.actor = actor = Mock()
    group.frontend = Mock()
    group.frontend.park.side_effect = RuntimeError("router unavailable")
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.gpu_processes",
        lambda: {},
    )
    probe = Mock()
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.subprocess.run", probe
    )
    with pytest.raises(RuntimeError, match="GPUs released but frontend"):
        group.stop()
    actor.destroy.assert_called_once_with()
    probe.assert_called_once()
    assert group.actor is None
    assert group.events[-1]["state"] == "FAILED"
    assert not any(e["state"] == "SUSPENDED" for e in group.events)
    assert group._gpu_release_verified
    group.frontend.close.return_value = {"process_exit_verified": True}
    with pytest.raises(RuntimeError, match="GPUs released but frontend"):
        group.close()
    group.frontend.close.assert_called_once()


@pytest.mark.parametrize("load_fails", [False, True])
def test_training_route_publishes_only_after_restore_verification(
    tmp_path, monkeypatch, load_fails
):
    """Failed restore cannot expose initial weights through the stable credential."""
    state = {
        "format_version": 1,
        "rank": 0,
        "world_size": 1,
        "policy_version": 2,
        "lr_scheduler": {},
    }
    (tmp_path / "runtime-0.json").write_text(json.dumps(state))
    group = object.__new__(TrainGroup)
    group.root, group.step, group.events = tmp_path, 0, []
    group.actor = None
    group._release_pending = False
    group.config, group.ft_spec = Mock(), Mock()
    group.is_v2, group.worker_role = True, "actor-guard"
    group.devices, group.uuids = [4], ["GPU-test"]
    group.engine_class, group.compatibility = "unused.Engine", {}
    group._require_free_gpus = Mock()
    group._started_processes = set()
    group.store = Mock()
    group.store.latest.return_value = (
        tmp_path,
        {"global_step": 2, "world_size": 1, "compatibility": {}},
    )
    group.scheduler = SimpleNamespace(
        _workers={
            "actor-guard": [SimpleNamespace(process=SimpleNamespace(pid=os.getpid()))]
        }
    )
    group.frontend = Mock()
    group.frontend.endpoints = {"gateway": ("127.0.0.1", 19002)}
    group.frontend.probe.return_value = {"training_status": 503}
    group.frontend.publish.return_value = {"training_status": 200}
    group.stop = Mock()
    actor = Mock()
    actor.service_endpoints.return_value = {}
    actor._gateway_get_result.return_value = 2

    def load(_):
        group.frontend.publish.assert_not_called()
        if load_fails:
            raise RuntimeError("restore failed")

    def records(kind):
        if kind == "restored":
            group.frontend.publish.assert_not_called()
            return [{"state": state}]
        return [{"rank": 0, "pid": os.getpid()}]

    actor.load.side_effect = load
    actor.rank_records.side_effect = records
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.LocalV2TrainController",
        Mock(return_value=actor),
    )
    if load_fails:
        with pytest.raises(RuntimeError, match="restore failed"):
            group.start(restore=True)
        group.frontend.publish.assert_not_called()
        group.stop.assert_called_once_with()
    else:
        group.start(restore=True)
        actor.set_version.assert_called_once_with(2)
        group.frontend.publish.assert_called_once_with(actor, 2)
        assert all(
            e["frontend"]["training_status"] == 503
            for e in group.events
            if e["state"] == "STAGING"
        )
        assert group.events[-1]["state"] == "ACTIVE"


def test_frontend_close_retry_keeps_unreleased_process_evidence(monkeypatch):
    """Deleting scheduler records cannot turn a failed close into verified success."""
    frontend = make_frontend()
    frontend.scheduler._workers = {
        frontend.role: [SimpleNamespace(process=SimpleNamespace(pid=os.getpid()))]
    }
    frontend.control.destroy = Mock(side_effect=frontend.scheduler._workers.clear)
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.service.time",
        SimpleNamespace(monotonic=Mock(side_effect=[0, 0, 31, 0, 0, 31]), sleep=Mock()),
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Frontend processes failed to exit"):
            frontend.close()
        assert not frontend.closed
    assert frontend.scheduler._workers == {}
    assert (os.getpid(), psutil.Process().create_time()) in frontend._closing_processes


def test_frontend_snapshot_traverses_logging_shell_wrappers(monkeypatch):
    """The scheduler PID is a shell; actual CPU services are deeper descendants."""
    frontend = make_frontend()
    frontend.scheduler._workers = {
        frontend.role: [
            SimpleNamespace(process=SimpleNamespace(pid=100), gpu_devices=[])
        ]
    }
    root = Mock()
    root.cmdline.return_value = ["/bin/bash", "-c", "python -m guard | tee log"]
    children = []
    for index, module in enumerate(
        [
            "areal.v2.training_service.guard",
            "areal.v2.training_service.gateway",
            "areal.v2.training_service.router",
            "areal.autopilot.elastic.runtime.training.service",
        ]
    ):
        p = Mock(pid=101 + index)
        p.cmdline.return_value = ["python", "-m", module]
        p.environ.return_value = {"CUDA_VISIBLE_DEVICES": ""}
        p.create_time.return_value = 10 + index
        p.status.return_value = psutil.STATUS_RUNNING
        children.append(p)
    root.children.return_value = children
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.service.psutil.Process",
        Mock(return_value=root),
    )
    records = frontend._process_snapshot()
    root.children.assert_called_once_with(recursive=True)
    assert {p["pid"] for p in records} == {101, 102, 103, 104}
