# SPDX-License-Identifier: Apache-2.0

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import psutil
import pytest

from examples.autopilot.elastic.fixtures.groups import LMGroup

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
from areal.autopilot.elastic.runtime.training.workers import (
    LocalV2TrainController,
    TrainGroup,
)


def make_controller(tmp_path, gpu_uuids=None):
    config = TrainEngineConfig(
        experiment_name="elastic-test",
        trial_name="v2",
        backend="fsdp:d2",
        _version="v2",
        scheduling_spec=(
            SchedulingSpec(env_vars={"AREAL_ELASTIC_DIAGNOSTICS_DIR": str(tmp_path)}),
        ),
    )
    return LocalV2TrainController(
        "not_imported.TrainEngine",
        config,
        Mock(),
        gpu_uuids=["GPU-a", "GPU-b"] if gpu_uuids is None else gpu_uuids,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role,index,visible",
    [
        ("train-worker", 0, "GPU-a"),
        ("train-worker", 1, "GPU-b"),
        ("gateway", 0, ""),
        ("router", 0, ""),
        ("data-proxy", 0, ""),
    ],
)
async def test_v2_fork_isolates_gpu_workers_from_cpu_services(
    tmp_path, role, index, visible
):
    """Only the selected rank's compute worker may see its assigned GPU."""
    controller = make_controller(tmp_path)
    original_env = {"CUSTOM": "preserved", "CUDA_VISIBLE_DEVICES": "wrong"}
    with patch(
        "areal.v2.training_service.controller.controller.GatewayTrainController._async_fork_on_guard",
        new_callable=AsyncMock,
        return_value=("127.0.0.1", 19001),
    ) as fork:
        await controller._async_fork_on_guard(
            "http://guard", role, index, ["python"], original_env
        )
    passed_env = fork.await_args.args[4]
    assert passed_env["CUDA_VISIBLE_DEVICES"] == visible
    assert passed_env["CUSTOM"] == "preserved"
    assert passed_env["AREAL_ELASTIC_LAUNCH_ID"] == controller.launch_id
    assert original_env["CUDA_VISIBLE_DEVICES"] == "wrong"


def test_v2_pool_size_must_match_distributed_world_size(tmp_path):
    """Reject missing GPU assignments before starting any guard."""
    with pytest.raises(ValueError, match="pool size"):
        make_controller(tmp_path, ["GPU-a"])


def test_v2_rank_records_reject_missing_or_wrong_rank(tmp_path):
    """The first DP head's response is insufficient evidence for other ranks."""
    controller = make_controller(tmp_path)
    paths = [
        tmp_path / f"{controller.launch_id}-process-rank-{rank}.json"
        for rank in range(2)
    ]
    atomic_json(paths[0], {"rank": 0})
    with pytest.raises(FileNotFoundError):
        controller.rank_records("process")
    atomic_json(paths[1], {"rank": 0})
    with pytest.raises(RuntimeError, match="inconsistent ranks"):
        controller.rank_records("process")


def make_stopping_group(tmp_path, controller):
    group = object.__new__(TrainGroup)
    group.actor = controller
    group.root = tmp_path
    group.step = 0
    group.events = []
    group.devices = [4, 5]
    group.uuids = ["GPU-a", "GPU-b"]
    group.is_v2 = True
    group.frontend = None
    group.worker_role = "actor-guard"
    group.scheduler = controller.scheduler
    return group


def test_v2_stop_waits_for_late_resource_creation_before_snapshot(
    tmp_path, monkeypatch
):
    """A timed-out init must settle before cleanup inspects owned processes."""
    controller = make_controller(tmp_path)
    order = []

    def finish_creation():
        time.sleep(0.03)
        controller._service_roles.append("actor-guard")
        order.append("created")

    class Workers(dict):
        def get(self, key, default=None):
            assert "created" in order
            order.append("snapshot")
            return []

    controller.scheduler._workers = Workers()
    controller._cleanup_runtime_state = Mock(
        side_effect=lambda: order.append("cleanup")
    )
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.gpu_processes",
        lambda: {},
    )
    probe = Mock()
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.subprocess.run", probe
    )
    group = make_stopping_group(tmp_path, controller)
    with ThreadPoolExecutor(max_workers=1) as executor:
        controller._init_future = executor.submit(finish_creation)
        group.stop()
    assert order == ["created", "snapshot", "cleanup"]
    assert group.events[-1]["state"] == "SUSPENDED"
    assert probe.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"


def test_v2_pending_initialization_cannot_claim_successful_release(
    tmp_path, monkeypatch
):
    """If initialization cannot settle, record failure and defer final cleanup."""
    controller = make_controller(tmp_path)
    controller.config.setup_timeout = 0.01
    future = Future()
    controller._init_future = future
    controller._cleanup_runtime_state = Mock()
    probe = Mock()
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.subprocess.run", probe
    )
    group = make_stopping_group(tmp_path, controller)
    with pytest.raises(RuntimeError, match="release cannot be verified"):
        group.stop()
    assert group.events[-1]["state"] == "FAILED"
    assert not any(event["state"] == "SUSPENDED" for event in group.events)
    probe.assert_not_called()
    future.set_result(None)
    controller._cleanup_runtime_state.assert_called_once_with()


def test_v2_repeated_stop_waits_for_deferred_cleanup_completion(tmp_path):
    """A completed init future does not imply its cleanup callback has finished."""
    controller = make_controller(tmp_path)
    controller.config.setup_timeout = 0.01
    controller._init_future = future = Future()
    entered, release, finished = Event(), Event(), Event()

    def cleanup():
        entered.set()
        assert release.wait(timeout=5)
        finished.set()

    controller._cleanup_runtime_state = Mock(side_effect=cleanup)
    with pytest.raises(RuntimeError, match="release cannot be verified"):
        controller.quiesce_initialization()
    with ThreadPoolExecutor(max_workers=1) as executor:
        completion = executor.submit(future.set_result, None)
        try:
            assert entered.wait(timeout=5)
            assert future.done() and not finished.is_set()
            with pytest.raises(RuntimeError, match="deferred cleanup is still running"):
                controller.destroy()
            controller._cleanup_runtime_state.assert_called_once_with()
        finally:
            release.set()
        completion.result(timeout=5)
    controller.quiesce_initialization()
    assert finished.is_set()
    assert controller._init_future is None


def test_v2_deferred_cleanup_preserves_process_identities(tmp_path):
    """Removed scheduler records must not hide a CPU service that failed to exit."""
    controller = make_controller(tmp_path)
    controller.config.setup_timeout = 0.01
    controller._service_roles.append("actor-guard")
    controller.scheduler._workers = {
        "actor-guard": [SimpleNamespace(process=SimpleNamespace(pid=os.getpid()))]
    }
    controller._cleanup_runtime_state = Mock(
        side_effect=controller.scheduler._workers.clear
    )
    controller._init_future = future = Future()
    with pytest.raises(RuntimeError, match="release cannot be verified"):
        controller.quiesce_initialization()
    future.set_result(None)
    controller.quiesce_initialization()
    assert controller.scheduler._workers == {}
    assert (
        os.getpid(),
        psutil.Process().create_time(),
    ) in controller.deferred_processes


@pytest.mark.parametrize("second_rank_success", [0, 1])
def test_v2_update_requires_success_from_every_rank(second_rank_success):
    """A successful gateway response cannot conceal another rank's skipped update."""
    group = object.__new__(LMGroup)
    group.is_v2 = True
    group.step = 0
    actor = SimpleNamespace(
        _gateway_post_result=Mock(return_value={"update_successful": 1}),
        rank_records=Mock(
            return_value=[
                {"rng_probe": {"rank": 0}, "stats": {"update_successful": 1}},
                {
                    "rng_probe": {"rank": 1},
                    "stats": {"update_successful": second_rank_success},
                },
            ]
        ),
        step_lr_scheduler=Mock(),
        version=0,
    )
    actor.set_version = lambda version: setattr(actor, "version", version)
    actor.get_version = lambda: actor.version
    group.actor = actor
    if second_rank_success:
        result = group.train_fixed_batch([{"input_ids": [1, 2]}])
        assert result["step"] == actor.version == 1
        assert len(result["rank_stats"]) == 2
    else:
        with pytest.raises(RuntimeError, match="Optimizer update failed"):
            group.train_fixed_batch([{"input_ids": [1, 2]}])
        assert group.step == actor.version == 0
        actor.step_lr_scheduler.assert_not_called()
