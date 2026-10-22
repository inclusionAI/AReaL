# SPDX-License-Identifier: Apache-2.0

"""Native PPO policy integration with real durable transactions; CPU protocol only."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from tests.test_elastic_checkpoint import (
    _complete_native_update,
    _ready_native_episodes,
    native_lifecycle,  # noqa: F401 -- shared native controller fixture
)

from areal.autopilot.domains.elastic.native import RevisionAdapter
from areal.autopilot.domains.elastic.policy import PolicyBoundaryBusy, PolicyRequest
from areal.elastic.contracts import BatchWorkload, HistorySnapshot, TrainingConfig
from areal.elastic.control.config_application import ConfigRequest


@pytest.fixture
def native_policy(request):
    fixture = request.getfixturevalue("native_lifecycle")
    fixture.manager.close()
    runtime = RevisionAdapter()
    manager = fixture.create(runtime)
    yield manager, runtime, fixture
    manager.close()


def update(manager):
    _ready_native_episodes(
        manager, manager.applied_version * 4, manager.applied_version
    )
    assert len(manager.next_batch()) == 4
    manager.actor.updates += 1
    _complete_native_update(manager, manager.applied_version + 1)


def accepted_native_bundle(runtime, root, *, native=True):
    """Synthetic held-out labels exercise real acceptance, not timing accuracy."""
    from examples.autopilot.revising import AFFINE_SOURCE

    from areal.autopilot.domains.elastic import RevisionEvaluator
    from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE
    from areal.autopilot.domains.elastic.online import LifecycleRevisionCycle
    from areal.autopilot.revision import Proposal
    from areal.elastic.contracts import Measurement

    profile = {
        **runtime.profile,
        "measurements": [
            asdict(Measurement(phase, 1, 100 if phase == "train" else 1, 16))
            for phase in (
                "start",
                "restore",
                "train",
                "checkpoint",
                "publish",
                "release",
            )
        ],
    }
    profile["measurements"] = [
        row for row in profile["measurements"] if row["phase"] != "train"
    ] + [asdict(Measurement("train", 1, 6, 10)), asdict(Measurement("train", 1, 7, 20))]
    if not native:
        for key in (
            "execution_scope",
            "native_execution_id",
            "native_actor_world_size",
        ):
            profile.pop(key)
    work = BatchWorkload(4, 8, 16)
    observation = replace(
        runtime.observe(),
        batch_ready=True,
        gate_available=True,
        training_active=True,
        next_batch=work,
    )
    case = {
        "observation": asdict(observation),
        "history": asdict(HistorySnapshot()),
        "phase": "train",
        "forecast": [{"workload": asdict(work), "wait_seconds": 0}],
        "plan": [{"config": asdict(TrainingConfig(1)), "release_before": False}],
        "observed": {"feasible": True, "seconds": 6.6, "total_gpu_seconds": 13.2},
    }
    from tests.test_autopilot_revision import runner

    model = RevisionEvaluator(
        "simulator",
        profile,
        [case],
        settings=runtime.settings,
        runner=runner.__wrapped__(),
    )
    cases = [{"observation": asdict(observation), "history": asdict(HistorySnapshot())}]

    class Agent:
        def propose(self, request):
            return Proposal(
                AFFINE_SOURCE
                if request.target == "simulator"
                else BASELINE_SOURCE["optimizer"]
            )

    result = LifecycleRevisionCycle(
        model, cases, root, idle_seconds=runtime.owner.options.idle_timeout
    ).run(Agent(), Agent(), max_model_rounds=1, max_optimizer_rounds=1)
    assert result["model"]["accepted"], result
    assert result["bundle"]
    return result["bundle"]


def test_native_policy_records_published_updates_and_restores_history(native_policy):
    manager, runtime, fixture = native_policy
    for _ in range(3):
        update(manager)
    history = runtime.runtime_history.snapshot(3)
    assert len(history["updates"]) == 3
    assert [row["weights_version"] for row in history["updates"]] == [1, 2, 3]
    assert all(row["workload"]["samples"] == 4 for row in history["updates"])
    assert runtime.observe().committed_samples == 12
    assert runtime.observe().committed_tokens == 24
    assert all(not row["missing_phases"] for row in history["updates"])
    assert {row["phase"] for row in history["updates"][0]["measurements"]} >= {
        "start",
        "train",
        "checkpoint",
        "publish",
    }
    assert manager.rollout is fixture.rollout
    manager.close()
    history = runtime.runtime_history.snapshot(3)
    recovered_runtime = RevisionAdapter()
    recovered = fixture.create(recovered_runtime)
    try:
        assert recovered.applied_version == 3 and recovered.actor is None
        assert recovered_runtime.runtime_history.snapshot(3) == history
        assert recovered_runtime.observe().committed_tokens == 24
        update(recovered)
        assert recovered.actor.updates == 4
    finally:
        recovered.close()


def test_native_policy_tracks_updates_between_periodic_checkpoints(native_policy):
    """Measured skipped saves are distinct from missing checkpoint measurements."""
    manager, runtime, _ = native_policy
    manager.checkpoint_frequency.step_ctl.frequency_steps = 3
    for _ in range(3):
        update(manager)
    rows = runtime.runtime_history.snapshot(3)["updates"]
    assert [row["checkpointed"] for row in rows] == [False, False, True]
    assert [row["checkpoint_id"] for row in rows[:2]] == [None, None]
    assert all(not row["missing_phases"] for row in rows)
    assert runtime.observe().committed_tokens == 24
    assert manager.inbox.status()["checkpoint_version"] == 3


def test_native_policy_rollback_removes_unsaved_history_before_replay(native_policy):
    """Abandoned weight versions cannot collide with replayed runtime records."""
    manager, runtime, _ = native_policy
    update(manager)
    manager.checkpoint_frequency.step_ctl.frequency_steps = 100
    update(manager)
    update(manager)
    assert len(runtime.runtime_history.snapshot(3)["updates"]) == 3
    manager.recover_uncertain(RuntimeError("lost unsaved GPU state"))
    assert manager.applied_version == 1 and runtime.observe().committed_tokens == 8
    assert len(runtime.runtime_history.snapshot(3)["updates"]) == 1
    for version in (2, 3):
        manager.next_batch()
        manager.actor.updates += 1
        _complete_native_update(manager, version)
    assert len(runtime.runtime_history.snapshot(3)["updates"]) == 3
    assert runtime.observe().committed_tokens == 24


def test_native_publication_failure_replays_unsaved_update_and_records_history_once(
    native_policy,
):
    manager, runtime, _ = native_policy
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    manager.actor.fail_publication = True
    with pytest.raises(TimeoutError):
        manager.publish_update(1)
    assert runtime.runtime_history.snapshot(1)["updates"] == []
    manager.recover_uncertain(TimeoutError("ack lost"))
    assert manager.applied_version == manager.actor.updates == 0
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    assert manager.applied_version == manager.actor.updates == 1
    assert len(runtime.runtime_history.snapshot(1)["updates"]) == 1
    manager.monitor.complete_update()
    assert len(runtime.runtime_history.snapshot(1)["updates"]) == 1
    assert manager.inbox.status()["active_credits"] == 0


def test_native_config_rejects_layout_and_placement_before_mutation(native_policy):
    manager, runtime, fixture = native_policy
    config = runtime.config_application.executor.config
    request = ConfigRequest("unchanged", config, 0, 0, "native")
    receipt = runtime.apply_config(request)
    assert receipt["status"] == "unchanged"
    assert runtime.apply_config(request) == receipt
    assert fixture.actors == [] and manager.actor is None
    for target in (
        replace(config, actor_backend="megatron:d1"),
        replace(config, actor_devices=(7,)),
        replace(config, rollout_backend="sglang:d2"),
        replace(config, execution_id="f" * 64),
    ):
        with pytest.raises(ValueError, match="fixed"):
            runtime.apply_config(replace(request, request_id="invalid", target=target))
    assert runtime.config_application.state()["pending"] is None
    assert manager.inbox.status()["admission_open"]


def test_native_policy_cannot_bypass_batch_or_release_guards(native_policy):
    manager, runtime, _ = native_policy
    update(manager)
    runtime.policy = SimpleNamespace(decide=lambda observation: "release")
    assert not manager.maybe_suspend(now=manager.ready_since + 59)
    assert manager.actor is not None
    assert manager.maybe_suspend(now=manager.ready_since + 121)
    assert manager.actor is None
    observation = runtime.observe()
    from areal.elastic.contracts import ResourceDecision

    with pytest.raises(ValueError, match="prewarm"):
        runtime._action(
            ResourceDecision("launch", TrainingConfig(1), True), observation
        )
    with pytest.raises(ValueError, match="fixed"):
        runtime._action(
            ResourceDecision("reconfigure", TrainingConfig(2), True), observation
        )


def test_native_timeout_adapter_matches_default_policy_clock(native_policy):
    """Explicit baseline instrumentation preserves default timeout decisions."""
    manager, runtime, _ = native_policy
    assert runtime is manager.policy_adapter
    update(manager)
    now = max(manager.ready_since + 120, manager.idle_since + 60)
    observation = manager.monitor.observe(now=now)
    assert runtime.observe(now=now) == replace(
        observation, committed_tokens=manager.monitor.completed_tokens
    )
    assert runtime.decide(None, now=now) == manager.policy_adapter.policy.decide(
        observation
    )
    assert manager.maybe_suspend(now=now)


def test_native_policy_rejects_legacy_bundle_and_activates_native_acceptance(
    native_policy, tmp_path
):
    manager, runtime, fixture = native_policy
    legacy = accepted_native_bundle(runtime, tmp_path / "legacy", native=False)
    with pytest.raises(ValueError, match="native execution scope"):
        runtime.prepare_policy(legacy)
    bundle = accepted_native_bundle(runtime, tmp_path / "native")
    assert json.loads(open(bundle).read())["kind"] == "native-async-policy-v1"
    prepared = runtime.prepare_policy(bundle)
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    request = PolicyRequest("activate", prepared.bundle_id, 0, 0)
    with pytest.raises(PolicyBoundaryBusy):
        runtime.apply_policy(request, prepared)
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    request = replace(request, expected_weights_version=1)
    before = manager.applied_version, manager.actor, fixture.actors[:]
    receipt = runtime.apply_policy(request, prepared)
    assert runtime.apply_policy(request, prepared) == receipt
    assert runtime.policy_generation == 1
    assert (manager.applied_version, manager.actor, fixture.actors) == before
    update(manager)
    assert manager.actor.updates == 2
    manager.close()
    restored_runtime = RevisionAdapter()
    restored = fixture.create(restored_runtime)
    try:
        assert restored_runtime.policy_generation == 1
        assert restored.applied_version == 2 and restored.actor is None
        assert len(restored_runtime.history.snapshot().workloads) == 2
    finally:
        restored.close()


def test_native_empty_wait_releases_lock_and_can_be_closed(native_policy, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    manager, runtime, _ = native_policy
    polled = threading.Event()
    original = runtime.poll

    def poll():
        polled.set()
        original()

    monkeypatch.setattr(runtime, "poll", poll)
    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(manager.next_batch)
        assert polled.wait(2)
        manager.close()
        with pytest.raises(RuntimeError, match="closed"):
            waiting.result(timeout=2)
    assert runtime.closed and manager.actor is None


def test_native_cleanup_closes_policy_and_consumer_after_admission_disk_failure(
    native_policy, monkeypatch
):
    manager, runtime, fixture = native_policy
    update(manager)
    with monkeypatch.context() as patch:
        patch.setattr(
            manager.inbox,
            "set_admission_open",
            lambda value: (_ for _ in ()).throw(OSError("disk full")),
        )
        with pytest.raises(ExceptionGroup, match="cleanup"):
            manager.close()
    assert runtime.closed and manager._closed and manager.actor is None
    recovered = fixture.create(RevisionAdapter())
    recovered.close()


def test_native_revision_attaches_to_ordinary_checkpoint_with_shared_monitor(request):
    """An ordinary run already has all measurement state required by revision."""
    fixture = request.getfixturevalue("native_lifecycle")
    update(fixture.manager)
    fixture.manager.close()
    context = fixture.manager.monitor.runtime_history.context
    resumed = fixture.create(RevisionAdapter())
    try:
        assert resumed.applied_version == 1
        assert resumed.monitor.completed_tokens == 8
        assert resumed.policy_adapter.runtime_history is resumed.monitor.runtime_history
        assert resumed.monitor.runtime_history.context == context
        assert len(resumed.policy_adapter.history.snapshot().workloads) == 1
    finally:
        resumed.close()


def test_native_uncertain_policy_pointer_recovers_without_closed_admission_deadlock(
    native_policy, tmp_path, monkeypatch
):
    import areal.elastic.control.policy_activation as store

    manager, runtime, _ = native_policy
    bundle = accepted_native_bundle(runtime, tmp_path / "revision")
    prepared = runtime.prepare_policy(bundle)
    original = store.atomic_json

    def lost_ack(path, value):
        original(path, value)
        if path.name == "active.json" and value["generation"] == 1:
            raise RuntimeError("lost policy acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(store, "atomic_json", lost_ack)
        with pytest.raises(RuntimeError, match="acknowledgement"):
            runtime.apply_policy(
                PolicyRequest("activate", prepared.bundle_id, 0, 0), prepared
            )
    assert runtime.failed and not manager.inbox.status()["admission_open"]
    with pytest.raises(RuntimeError, match="Recover pending policy"):
        manager._poll_batch()
    manager.recover_uncertain(RuntimeError("activation outcome unknown"))
    assert runtime.policy_generation == 1 and manager.applied_version == 0
    assert manager.inbox.status()["admission_open"]
    update(manager)
    assert manager.actor.updates == 1


def test_native_four_gpu_missing_measurement_keeps_fixed_layout_cold_fallback():
    from tests.test_elastic_basic_policy import model, observation

    from areal.elastic.policy.basic.optimizer import BasicOptimizer
    from areal.elastic.policy.basic.policy import BasicPolicy
    from areal.elastic.policy.basic.profile import FixedLayoutSimulator
    from areal.elastic.policy.basic.simulator import MeasuredComponent

    policy = BasicPolicy(
        FixedLayoutSimulator(replace(model(), train=MeasuredComponent("train")), 4),
        optimizer=BasicOptimizer(horizon=1, candidate_configs=(TrainingConfig(4),)),
    )
    decision = policy.decide(
        observation(
            training_active=False,
            training_config=TrainingConfig(4),
            has_checkpoint=False,
            max_train_gpus=4,
        )
    )
    assert decision.action == "launch" and decision.train
    assert decision.config == TrainingConfig(4)


def test_native_history_disk_failure_still_closes_policy_and_releases_consumer(
    native_policy, monkeypatch
):
    manager, runtime, fixture = native_policy
    update(manager)
    with monkeypatch.context() as patch:
        patch.setattr(
            runtime.runtime_history,
            "append_boundary",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("history disk full")),
        )
        manager.close()
        assert not runtime.measurement_ready
    assert runtime.closed and manager._closed
    assert manager.actor is None and manager.owner is None
    recovered = fixture.create(RevisionAdapter())
    recovered.close()


def test_native_background_revision_uses_committed_host_and_durable_window_budget(
    native_policy, tmp_path
):
    from tests.test_autopilot_revision import runner

    from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE
    from areal.autopilot.domains.elastic.online import RuntimeRevisionLoop
    from areal.autopilot.revision import Proposal
    from areal.elastic.contracts import Measurement

    manager, runtime, fixture = native_policy
    manager.close()
    profile = {
        **runtime.profile,
        "measurements": [
            asdict(Measurement(phase, 1, 1, 8))
            for phase in (
                "start",
                "restore",
                "train",
                "checkpoint",
                "publish",
                "release",
            )
        ],
    }
    fixture.config.actor_lifecycle.state_root = str(tmp_path / "measured-run")
    measured = RevisionAdapter(profile=profile)
    manager = fixture.create(measured)

    class Unchanged:
        def propose(self, request):
            return Proposal(BASELINE_SOURCE[request.target])

    measured.revision = RuntimeRevisionLoop(
        tmp_path / "revision",
        profile=profile,
        model_agent=Unchanged(),
        optimizer_agent=Unchanged(),
        settings=measured.settings,
        scope="native-async-v1",
        horizon=1,
        max_windows=1,
        max_model_rounds=1,
        max_optimizer_rounds=1,
        idle_seconds=manager.options.idle_timeout,
        runner=runner.__wrapped__(),
    )
    try:
        for _ in range(4):
            update(manager)
        measured.poll()
        assert measured.revision.future is not None
        measured.revision.future.result(timeout=30)
        measured.poll()
        outcome = json.loads(
            (tmp_path / "revision/window-0001/outcome.json").read_text()
        )
        assert outcome["status"] == "unchanged", outcome
        assert measured.policy_generation == 0 and manager.applied_version == 4
        for _ in range(4):
            update(manager)
        measured.poll()
        assert len(list((tmp_path / "revision").glob("window-*"))) == 1
    finally:
        manager.close()
    assert measured.closed


@pytest.mark.parametrize("enabled", [False, True])
def test_hermes_entry_preserves_config_arguments_and_optionally_attaches_native_policy(
    monkeypatch, enabled
):
    from examples.hermes import train

    config = object()
    seen = {}

    def load(args, config_type):
        seen["args"] = args
        return config, None

    class Trainer:
        def __init__(self, actual_config, **kwargs):
            assert actual_config is config
            seen.update(kwargs)

        def __enter__(self):
            return self

        def train(self):
            seen["trained"] = True

        def __exit__(self, *args):
            if "elastic_policy" in seen:
                seen["elastic_policy"].close()

    monkeypatch.setattr(train, "load_expr_config", load)
    monkeypatch.setattr(train, "PPOTrainer", Trainer)
    config_args = [
        "--config",
        "examples/hermes/config.yaml",
        "actor_lifecycle.mode=on_demand",
    ]
    train.main([*(["--elastic-policy"] if enabled else []), *config_args])
    assert seen["args"] == config_args and seen["trained"]
    assert ("elastic_policy" in seen) == enabled
    if enabled:
        assert isinstance(seen["elastic_policy"], RevisionAdapter)
        assert seen["elastic_policy"].closed


def test_hermes_measured_profile_selects_areal_adapter_without_autopilot(
    monkeypatch, tmp_path
):
    """Profile-only runs use the ordinary adapter and default Monitor."""
    import builtins

    from examples.hermes import train

    from areal.elastic.policy.adapter import PolicyAdapter

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"measurements": []}))
    seen = {}

    class Trainer:
        def __init__(self, config, **kwargs):
            seen.update(kwargs)

        def __enter__(self):
            return self

        def train(self):
            seen["trained"] = True

        def __exit__(self, *args):
            seen["elastic_policy"].close()

    original_import = builtins.__import__

    def without_autopilot(name, *args, **kwargs):
        if name == "areal.autopilot" or name.startswith("areal.autopilot."):
            raise AssertionError("Measured Basic entry must not import AutoPilot")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(train, "load_expr_config", lambda *args: (object(), None))
    monkeypatch.setattr(train, "PPOTrainer", Trainer)
    monkeypatch.setattr(builtins, "__import__", without_autopilot)
    train.main(["--elastic-profile", str(profile), "actor_lifecycle.mode=on_demand"])
    assert type(seen["elastic_policy"]) is PolicyAdapter
    assert seen["elastic_policy"].hooks is None and seen["trained"]


def test_duplicate_policy_owner_rejection_does_not_close_original_trainer(
    native_policy,
):
    from areal import PPOTrainer

    manager, runtime, fixture = native_policy
    with pytest.raises(ValueError, match="already bound"):
        PPOTrainer(fixture.config, elastic_policy=runtime)
    with pytest.raises(ValueError, match="already bound"):
        fixture.create(runtime)
    assert not runtime.closed and runtime.owner is manager
    update(manager)
    assert manager.applied_version == 1


def test_history_append_failure_preserves_update_and_pauses_revision(
    native_policy, monkeypatch
):
    """An optional history failure never replays a successfully published PPO."""
    from unittest.mock import MagicMock

    from tests.test_elastic_batch_queue import _receive_episode

    manager, runtime, fixture = native_policy
    update(manager)
    for number in range(4, 8):
        _receive_episode(manager.inbox, number, version=1, rows=2)
    manager.next_batch()
    manager.actor.updates += 1
    with monkeypatch.context() as patch:
        patch.setattr(
            runtime.runtime_history,
            "append",
            lambda *args: (_ for _ in ()).throw(OSError("history unavailable")),
        )
        _complete_native_update(manager, 2)
    assert manager.applied_version == manager.published_version == 2
    assert manager.actor.updates == 2 and manager.active is None
    assert not runtime.measurement_ready and not runtime.policy_boundary_ready
    revision = MagicMock()
    monkeypatch.setattr(runtime, "revision", revision)
    runtime.poll()
    revision.tick.assert_not_called()
    expected = manager.data_progress.copy()
    assert manager.monitor.completed_tokens == expected["effective_tokens"]
    assert manager.maybe_suspend(now=manager.ready_since + 121)
    assert manager.latest_manifest["data_progress"] == expected
    assert manager.consumer.cursor["applied_batches"] == 2
    manager.close()
    restored = fixture.create(RevisionAdapter())
    try:
        assert restored.applied_version == 2 and restored.data_progress == expected
        assert restored.monitor.runtime_history.snapshot(2)["updates"] == []
        restored.ensure_actor()
        assert restored.actor.updates == 2
    finally:
        restored.close()


def test_native_public_control_serializes_policy_activation_and_reports_boundary(
    native_policy,
):
    import threading

    from areal.elastic.control.policy_activation import PolicyBoundaryBusy

    manager, runtime, _ = native_policy
    entered, finish = threading.Event(), threading.Event()
    before = manager.control_snapshot()
    assert before.boundary_ready and not before.closed

    def lifecycle_operation():
        with manager.control_operation(blocking=True):
            entered.set()
            assert finish.wait(5)

    worker = threading.Thread(target=lifecycle_operation)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(PolicyBoundaryBusy, match="busy"):
            runtime.apply_policy(None, None)
    finally:
        finish.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert manager.control_snapshot() == before
    assert runtime.policy_generation == 0 and not runtime.failed
    manager.fence_admission()
    assert not manager.inbox.status()["admission_open"]


def test_accepted_bundle_still_rejects_protected_comment_changes(
    native_policy, tmp_path, monkeypatch
):
    """Measurement compatibility never relaxes accepted-bundle source proofs."""
    from pathlib import Path

    from areal.autopilot.domains.elastic.policy import validate_bundle
    from areal.elastic.identity import measurement_identity

    manager, runtime, _ = native_policy
    bundle = json.loads(
        Path(accepted_native_bundle(runtime, tmp_path / "proof")).read_text()
    )
    validate_bundle(bundle)
    original = Path.read_text
    identity = measurement_identity(manager)

    def changed_trace(path, *args, **kwargs):
        source = original(path, *args, **kwargs)
        if path.as_posix().endswith("elastic/monitor/trace.py"):
            source += "\n# a harmless offline parser comment\n"
        return source

    monkeypatch.setattr(Path, "read_text", changed_trace)
    assert measurement_identity(manager) == identity
    with pytest.raises(ValueError, match="code changed"):
        validate_bundle(bundle)


@pytest.mark.parametrize("fault", ["snapshot_io", "missing_metrics", "changed_series"])
def test_native_revision_pauses_on_unavailable_measurement_window(
    native_policy, tmp_path, monkeypatch, fault
):
    """Missing observations cannot launch a revision or fail the native PPO loop."""
    from unittest.mock import MagicMock

    from areal.autopilot.domains.elastic.online import RuntimeRevisionLoop
    from areal.elastic.contracts import Measurement
    from areal.utils.atomic import atomic_json

    initial, adapter, fixture = native_policy
    initial.close()
    profile = {
        **adapter.profile,
        "measurements": [
            asdict(Measurement(phase, 1, 1, 8))
            for phase in (
                "start",
                "restore",
                "train",
                "checkpoint",
                "publish",
                "release",
            )
        ],
    }
    fixture.config.actor_lifecycle.state_root = str(tmp_path / "measured-fault")
    host = RevisionAdapter(profile=profile)
    manager = fixture.create(host)
    loop = host.revision = RuntimeRevisionLoop(
        tmp_path / "revision-fault",
        profile=profile,
        model_agent=MagicMock(),
        optimizer_agent=MagicMock(),
        settings=host.settings,
        scope="native-async-v1",
        horizon=1,
        max_windows=1,
        max_model_rounds=1,
        max_optimizer_rounds=1,
        idle_seconds=manager.options.idle_timeout,
    )
    try:
        for _ in range(4):
            update(manager)
        snapshot = MagicMock(side_effect=OSError("measurement window unavailable"))
        if fault == "changed_series":
            atomic_json(
                loop.root / "execution.json",
                {
                    "policy_context": host.policy_application.context,
                    "history_context": "previous-measurement-series",
                },
            )
        else:
            monkeypatch.setattr(host.runtime_history, "snapshot", snapshot)
            if fault == "missing_metrics":
                manager.monitor.mark_unavailable(OSError("earlier metrics failure"))
        loop.tick(host)
        assert loop.future is None and not list(loop.root.glob("window-*"))
        assert not host.measurement_ready
        assert manager.applied_version == manager.published_version == 4
        assert manager.actor.updates == 4
        if fault == "missing_metrics":
            snapshot.assert_not_called()
    finally:
        manager.close()
