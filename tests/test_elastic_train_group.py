# SPDX-License-Identifier: Apache-2.0

import errno
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.autopilot.elastic.runtime.training.workers import TrainGroup


def test_named_actor_profile_is_rejected_before_worker_or_scheduler_mutation():
    """A hybrid profile cannot be silently executed as a pure-DP card count."""
    from areal.autopilot.contracts import TrainingConfig

    group = object.__new__(TrainGroup)
    group.actor = object()
    group._release_pending = False
    group._world_size = 32
    group.config = SimpleNamespace(backend="fsdp:d32")
    group.scheduler = SimpleNamespace(_gpu_counter=5)
    actor = group.actor
    with pytest.raises(ValueError, match="named actor profiles"):
        group.validate_config(TrainingConfig(32, "hybrid"))
    assert group.actor is actor
    assert group.config.backend == "fsdp:d32" and group._world_size == 32
    assert group.scheduler._gpu_counter == 5


def test_stop_releases_workers_even_when_lifecycle_disk_is_full(tmp_path, monkeypatch):
    """A failed event write must not bypass teardown after checkpoint failure."""
    group = object.__new__(TrainGroup)
    actor = Mock()
    group.actor = actor
    group.root = tmp_path
    group.step = 2
    group.events = []
    group.devices = [4]
    group.uuids = ["test-gpu"]
    group.worker_role = "actor"
    group.is_v2 = False
    group.frontend = None
    group.scheduler = SimpleNamespace(_workers={})
    allocation_probe = Mock()

    def full_disk(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.atomic_json", full_disk
    )
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.gpu_processes",
        lambda: {},
    )
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.subprocess.run",
        allocation_probe,
    )

    with pytest.raises(RuntimeError, match="resources released"):
        group.stop()

    actor.destroy.assert_called_once_with()
    allocation_probe.assert_called_once()
    assert group.actor is None
    assert group.events[-1]["state"] == "SUSPENDED"
    assert group.events[-1]["process_exit_verified"] is True


def test_repeated_stop_retains_gpu_evidence_until_release_is_verified(
    tmp_path, monkeypatch
):
    """actor=None is insufficient after a failed PID/GPU release check."""
    group = object.__new__(TrainGroup)
    group.actor = Mock()
    actor = group.actor
    group.root, group.step, group.events = tmp_path, 2, []
    group.devices, group.uuids = [4], ["test-gpu"]
    group.worker_role, group.is_v2, group.frontend = "actor", False, None
    group.scheduler = SimpleNamespace(_workers={})
    clock = iter(range(0, 2000, 100))
    occupied = {"test-gpu": [123]}
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.time.monotonic",
        lambda: next(clock),
    )
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.gpu_processes",
        lambda: occupied,
    )
    probe = Mock()
    monkeypatch.setattr(
        "areal.autopilot.elastic.runtime.training.workers.subprocess.run", probe
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="not released"):
            group.stop()
        assert group._release_pending
    actor.destroy.assert_called_once()
    with pytest.raises(RuntimeError, match="release is unverified"):
        group.start()
    occupied.clear()
    group.stop()
    assert not group._release_pending
    probe.assert_called_once()


@pytest.mark.parametrize("release_pending", [False, True])
def test_close_shuts_frontend_only_after_gpu_release_proof(release_pending):
    """An event error allows CPU cleanup; surviving GPU workers prevent it."""
    group = object.__new__(TrainGroup)
    group.actor = None
    group._release_pending = release_pending
    group.frontend = Mock()
    group.frontend.close.return_value = {"process_exit_verified": True}
    group._event = Mock()
    group.stop = Mock(side_effect=RuntimeError("stop failed"))
    with pytest.raises(RuntimeError, match="stop failed"):
        group.close()
    assert group.frontend.close.call_count == (0 if release_pending else 1)


def test_resize_keeps_pool_and_tracks_scheduler_rotation_after_partial_failure():
    """Scheduler cursor predicts the actual per-generation physical GPU order."""
    from dataclasses import dataclass
    from types import SimpleNamespace

    from areal.autopilot.contracts import TrainingConfig

    @dataclass
    class Config:
        backend: str
        scheduling_spec: tuple
        scheduling_strategy: object

    group = object.__new__(TrainGroup)
    group.actor, group._release_pending, group.is_v2 = None, False, True
    group.compatibility = {"elastic_dp": "fsdp-pure-dp-v1"}
    group.pool_devices = [4, 5, 6, 7]
    group.pool_uuids = [f"GPU-{device}" for device in group.pool_devices]
    group.config = Config(
        "fsdp:d4", (SimpleNamespace(gpu=1),), SimpleNamespace(type="separation")
    )
    group.scheduler = SimpleNamespace(_gpu_counter=3)
    group.configure(TrainingConfig(3))
    group._select_devices()
    assert group.devices == [7, 4, 5]
    assert group.uuids == ["GPU-7", "GPU-4", "GPU-5"]
    assert group.config.backend == "fsdp:d3" and group.pool_devices == [4, 5, 6, 7]
    group.scheduler._gpu_counter = 5  # two rank allocations before a failed launch
    group._select_devices()
    assert group.devices == [5, 6, 7]
    group._release_pending = True
    with pytest.raises(RuntimeError, match="verified GPU release"):
        group.configure(TrainingConfig(1))
    group._release_pending = False
    with pytest.raises(ValueError, match="exceeds"):
        group.configure(TrainingConfig(5))
