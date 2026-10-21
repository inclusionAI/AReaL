# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, replace

import pytest

from tests.test_elastic_runtime_state import make_engine

from areal.elastic.contracts import TrainingConfig
from areal.elastic.control.config import (
    ConfigApplication,
    ConfigRequest,
    ConfigSnapshot,
)
from areal.elastic.policy.basic.cost_model.rollout import RolloutConfig
from areal.elastic.policy.basic.joint import JointConfig
from areal.elastic.state.checkpoint import (
    artifact_manifest,
    atomic_json,
    capture_runtime_state,
    recovery_view,
    restore_recovery_runtime,
)


def source(tmp_path):
    engine = make_engine("cpu")
    engine.world_size = 2
    path = tmp_path / "step-00000001-source"
    (path / "dcp").mkdir(parents=True)
    (path / "dcp/.metadata").write_text("immutable")
    for rank in range(2):
        engine.rank = rank
        atomic_json(path / f"runtime-{rank}.json", capture_runtime_state(engine))
    manifest = {
        "world_size": 2,
        "global_step": 1,
        "compatibility": {"elastic_dp": "fsdp-pure-dp-v1", "seed": 17},
    }
    atomic_json(path / "manifest.json", manifest)
    return path, manifest


def test_resize_recovery_view_is_immutable_and_retries_are_deterministic(tmp_path):
    path, manifest = source(tmp_path)
    before = artifact_manifest(path)
    views = [
        recovery_view(path, manifest, target_world_size=3, root=tmp_path / "recovery")
        for _ in range(2)
    ]
    states = []
    for view in views:
        assert (view / "dcp").is_symlink()
        engine = make_engine("cpu")
        engine.world_size = 3
        ranks = []
        for rank in range(3):
            engine.rank = rank
            plan = json.loads((view / f"runtime-{rank}.json").read_text())
            restored = restore_recovery_runtime(engine, plan)
            ranks.append(restored)
            assert restored["world_size"] == 3 and restored["policy_version"] == 1
        states.append(ranks)
    assert states[0] == states[1]
    assert states[0][0]["torch_rng"] != states[0][1]["torch_rng"]
    assert artifact_manifest(path) == before


def test_same_size_recovery_preserves_exact_rng_and_legacy_resize_is_rejected(tmp_path):
    path, manifest = source(tmp_path)
    view = recovery_view(
        path, manifest, target_world_size=2, root=tmp_path / "recovery"
    )
    original = json.loads((path / "runtime-0.json").read_text())
    engine = make_engine("cpu")
    engine.world_size = 2
    actual = restore_recovery_runtime(
        engine, json.loads((view / "runtime-0.json").read_text())
    )
    assert json.loads(json.dumps(actual)) == original
    manifest["compatibility"] = {"backend": "fsdp:d2"}
    with pytest.raises(ValueError, match="does not authorize"):
        recovery_view(path, manifest, target_world_size=3, root=tmp_path / "recovery")


def rollout_config():
    return RolloutConfig("model", "cpu", "fixture", "fixed", 1, 1, 8, 64, 128, 1)


class ConfigFixture:
    """Joint execution fake: no training or inference backend internals are mocked."""

    context_id = "joint-config-fixture-v1"

    def __init__(self):
        self.current = ConfigSnapshot(
            JointConfig(TrainingConfig(1), rollout_config()), 5, "checkpoint-5", True
        )
        self.applies = []
        self.recoveries = []
        self.failure = None
        self.fail_recovery = False

    def observe(self):
        return self.current

    def validate(self, target):
        if target.world_size > 8:
            raise ValueError("Insufficient resources")

    def apply(self, target):
        self.applies.append(target)
        self.current = replace(self.current, config=target)
        if self.failure is not None:
            raise self.failure

    def recover(self, source):
        self.recoveries.append(source)
        if self.fail_recovery:
            raise RuntimeError("Cleanup unverified")
        self.current = source


def config_request(executor, generation=0, request_id="request-one", target=None):
    current = executor.observe()
    return ConfigRequest(
        request_id,
        target or replace(current.config, actor=TrainingConfig(2)),
        generation,
        current.weights_version,
        "policy-v0",
    )


def test_config_application_commits_joint_target_and_retries_without_execution(
    tmp_path,
):
    executor = ConfigFixture()
    application = ConfigApplication(executor, tmp_path)
    source = executor.observe()
    target = JointConfig(TrainingConfig(3), replace(rollout_config(), replicas=2))
    request = config_request(executor, target=target)
    receipt = application.apply(request)
    assert receipt["status"] == "applied" and receipt["generation"] == 1
    assert executor.observe() == replace(source, config=target)
    assert ConfigApplication(executor, tmp_path).apply(request) == receipt
    assert executor.applies == [target]
    second = config_request(executor, 1, "request-two", source.config)
    application.apply(second)
    assert application.apply(request) == receipt  # Historical, not a new transition.
    assert executor.current.config == source.config and len(executor.applies) == 2
    with pytest.raises(ValueError, match="identity was reused"):
        application.apply(replace(request, target=source.config))


@pytest.mark.parametrize("field", ["expected_generation", "expected_weights_version"])
def test_config_application_rejects_stale_requests_before_mutation(tmp_path, field):
    executor = ConfigFixture()
    application = ConfigApplication(executor, tmp_path)
    request = config_request(executor)
    with pytest.raises(ValueError, match="Stale"):
        application.apply(replace(request, **{field: getattr(request, field) + 1}))
    assert executor.applies == [] and application.state()["pending"] is None


def test_config_application_noop_and_context_mismatch_are_explicit(tmp_path):
    executor = ConfigFixture()
    application = ConfigApplication(executor, tmp_path)
    request = config_request(executor, target=executor.current.config)
    receipt = application.apply(request)
    assert receipt["status"] == "unchanged" and receipt["generation"] == 0
    assert not executor.applies
    executor.context_id = "another-machine"
    with pytest.raises(ValueError, match="another execution context"):
        ConfigApplication(executor, tmp_path).state()


@pytest.mark.parametrize(
    "point",
    [
        "intent-before",
        "intent-after",
        "execute-after",
        "commit-before",
        "commit-after",
        "recovering-after",
    ],
)
def test_config_application_recovers_interruption_without_replaying_target(
    tmp_path, monkeypatch, point
):
    import areal.elastic.control.config as module

    class PowerLoss(BaseException):
        pass

    executor = ConfigFixture()
    source = executor.observe()
    application = ConfigApplication(executor, tmp_path)
    application.state()
    request = config_request(executor)
    write = module.atomic_json
    fired = False

    def interrupted(path, state):
        nonlocal fired
        phase = (state["pending"] or {}).get("phase")
        kind = (
            "commit"
            if phase is None
            else "intent"
            if phase == "applying"
            else "recovering"
        )
        before = point == kind + "-before" and not fired
        after = point == kind + "-after" and not fired
        if before:
            fired = True
            raise PowerLoss()
        write(path, state)
        if after:
            fired = True
            raise PowerLoss()

    if point == "execute-after":
        executor.failure = PowerLoss()
    elif point == "recovering-after":
        executor.failure = RuntimeError("Partial transition")
    monkeypatch.setattr(module, "atomic_json", interrupted)
    with pytest.raises(PowerLoss):
        application.apply(request)
    monkeypatch.setattr(module, "atomic_json", write)
    executor.failure = None
    calls = len(executor.applies)
    resumed = ConfigApplication(executor, tmp_path)
    resumed.recover()
    assert len(executor.applies) == calls
    assert resumed.state()["pending"] is None
    expected = (
        replace(source, config=request.target) if point == "commit-after" else source
    )
    assert executor.observe() == expected
    assert resumed.state()["generation"] == (point == "commit-after")


def test_config_application_cleanup_failure_keeps_intent_and_blocks_new_work(tmp_path):
    executor = ConfigFixture()
    source = executor.observe()
    application = ConfigApplication(executor, tmp_path)
    executor.failure = RuntimeError("Target failed")
    executor.fail_recovery = True
    with pytest.raises(RuntimeError, match="Cleanup unverified"):
        application.apply(config_request(executor))
    assert application.state()["pending"]["phase"] == "recovering"
    assert application.state()["config"] == json.loads(
        json.dumps(asdict(source.config))
    )
    with pytest.raises(RuntimeError, match="pending config"):
        application.apply(config_request(executor, request_id="another"))
    executor.fail_recovery = False
    receipt = application.recover()
    assert receipt["status"] == "rolled_back" and receipt["generation"] == 0
    assert executor.observe() == source


def test_old_receipt_does_not_bypass_pending_config_intent(tmp_path):
    executor = ConfigFixture()
    application = ConfigApplication(executor, tmp_path)
    original = config_request(executor)
    application.apply(original)
    executor.failure = RuntimeError("Target failed")
    executor.fail_recovery = True
    with pytest.raises(RuntimeError, match="Cleanup unverified"):
        application.apply(
            config_request(
                executor,
                1,
                "next",
                replace(executor.current.config, actor=TrainingConfig(3)),
            )
        )
    with pytest.raises(RuntimeError, match="pending config"):
        application.apply(original)
