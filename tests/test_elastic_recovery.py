# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, replace

import pytest

from tests.test_elastic_basic_policy import model
from tests.test_elastic_loop import ProtocolGroup, ProtocolPublisher
from tests.test_elastic_runtime_state import make_engine

from areal.autopilot.contracts import BatchWorkload, TrainingConfig
from areal.autopilot.elastic.loop import ElasticLoop
from areal.autopilot.elastic.policy import BasicPolicy
from areal.autopilot.elastic.policy.basic.cost_model.rollout import RolloutConfig
from areal.autopilot.elastic.policy.basic.joint import JointConfig
from areal.autopilot.elastic.policy.basic.optimizer import BasicOptimizer
from areal.autopilot.elastic.policy.basic.simulator import (
    MeasuredComponent,
    Measurement,
    WorkflowSimulator,
)
from areal.autopilot.elastic.runtime.config import (
    ActorConfigExecutor,
    ConfigApplication,
    ConfigRequest,
    ConfigSnapshot,
    JointConfigExecutor,
)
from areal.autopilot.elastic.runtime.rollout import RolloutGroup
from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue
from areal.autopilot.elastic.runtime.state.checkpoint import (
    artifact_manifest,
    atomic_json,
    capture_runtime_state,
    recovery_view,
    restore_recovery_runtime,
)
from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer


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


class ResizableGroup(ProtocolGroup):
    def __init__(self, root):
        super().__init__(root)
        self.pool_devices = list(range(5))
        self.training_config = TrainingConfig(5)
        self.configurations = []
        self.devices = []
        self.placement = None

    def configure(self, target):
        assert self.actor is None
        self.training_config = target
        self.configurations.append(target.world_size)

    def configure_placement(self, devices):
        assert self.actor is None
        assert len(devices) == self.training_config.world_size
        self.placement = list(devices)

    def start(self, *, restore=False):
        self.devices = list(
            self.placement or self.pool_devices[: self.training_config.world_size]
        )
        super().start(restore=restore)

    @staticmethod
    def workload(payload):
        return BatchWorkload(
            payload["samples"], payload["tokens"], payload["tokens"] * 2
        )


@pytest.mark.parametrize("durable", [False, True])
def test_basic_loop_uses_optimizer_config_and_recovers_committed_sample_counts(
    tmp_path,
    durable,
):
    """Real queue/checkpoint protocol, arbitrary world sizes, no rank-multiplied D."""
    queue = BatchQueue(tmp_path / "queue")
    group = ResizableGroup(tmp_path / "checkpoints")
    profiles = tuple(
        Measurement("train", n, duration, 100)
        for n, duration in enumerate((40, 22, 9, 8, 10), 1)
    )
    policy = BasicPolicy(
        WorkflowSimulator(
            replace(
                model(), train=MeasuredComponent("train", profiles, scale_tokens=True)
            )
        ),
        optimizer=BasicOptimizer(horizon=1),
    )
    publisher = ProtocolPublisher()
    trainer = QueuedTrainer(group, queue)
    application = (
        ConfigApplication(
            ActorConfigExecutor(trainer, rollout_config()), tmp_path / "applications"
        )
        if durable
        else None
    )
    loop = ElasticLoop(
        trainer, publisher, policy=policy, config_application=application
    )
    try:
        loop.start()
        for i in range(2):
            with loop.episode() as version:
                queue.enqueue(
                    str(i),
                    {
                        "policy_version": version,
                        "value": i + 1,
                        "samples": 2,
                        "tokens": 50,
                    },
                    policy_version=version,
                )
            loop.tick()
        assert group.configurations == [3]
        assert trainer.committed_samples == 4 and trainer.committed_tokens == 100
        assert len(policy.history.snapshot().workloads) == 2
        assert queue.status()["counts"]["committed"] == 2
        assert loop.observe().committed_samples == 4
        if application is not None:
            assert application.state()["generation"] == 1
            assert loop.last_config_receipt["actual"]["weights_version"] == 0
    finally:
        loop.close()
    recovered = QueuedTrainer(ResizableGroup(tmp_path / "checkpoints"), queue)
    try:
        recovered.prepare()
        assert (
            recovered.version,
            recovered.committed_samples,
            recovered.committed_tokens,
        ) == (2, 4, 100)
    finally:
        recovered.stop()


def test_uncommitted_replay_does_not_double_count_samples(tmp_path):
    queue = BatchQueue(tmp_path / "queue")
    group = ResizableGroup(tmp_path / "checkpoints")
    trainer = QueuedTrainer(group, queue)
    queue.enqueue(
        "one",
        {"policy_version": 0, "value": 1, "samples": 2, "tokens": 50},
        policy_version=0,
    )
    trainer.start()
    trainer.train_next()
    assert trainer.committed_samples == 0
    trainer.stop()
    restored = QueuedTrainer(ResizableGroup(tmp_path / "checkpoints"), queue)
    try:
        restored.start()
        restored.train_next()
        restored.checkpoint()
        assert restored.committed_samples == 2 and restored.committed_tokens == 50
    finally:
        restored.stop()


@pytest.mark.parametrize("durable", [False, True])
def test_failed_target_rolls_back_and_consumes_ready_batch_once(tmp_path, durable):
    """A verified rollback keeps admission usable and does not replay an update."""
    from areal.autopilot.contracts import ResourceDecision

    class FailedTarget(ResizableGroup):
        def start(self, *, restore=False):
            if self.training_config.world_size == 3:
                raise RuntimeError("target initialization failed")
            super().start(restore=restore)

    class ChooseThree:
        def decide(self, obs):
            return ResourceDecision("reconfigure", TrainingConfig(3), True)

    group = FailedTarget(tmp_path / "checkpoints")
    group.configure(TrainingConfig(2))
    queue = BatchQueue(tmp_path / "queue")
    trainer = QueuedTrainer(group, queue)
    publisher = ProtocolPublisher()
    application = (
        ConfigApplication(
            ActorConfigExecutor(trainer, rollout_config()), tmp_path / "applications"
        )
        if durable
        else None
    )
    loop = ElasticLoop(trainer, publisher, config_application=application)
    try:
        loop.start()
        for i in range(2):
            if i:
                loop._policy = ChooseThree()
            with loop.episode() as version:
                queue.enqueue(
                    str(i),
                    {
                        "policy_version": version,
                        "value": i + 1,
                        "samples": 2,
                        "tokens": 50,
                    },
                    policy_version=version,
                )
            assert loop.tick() == "ACTIVE"
        assert group.training_config.world_size == 2
        assert group.updates == 2 and group.history == ["episode-0", "episode-1"]
        assert trainer.is_committed and trainer.committed_samples == 4
        assert publisher.ready and publisher.version == 2
        assert any(event["event"] == "RECONFIGURE_ROLLED_BACK" for event in loop.events)
    finally:
        loop.close()


@pytest.mark.parametrize(
    "missing", ["source_sha256", "run_seed", "source_world_size", "lr_scheduler"]
)
def test_incomplete_resize_plan_is_rejected(tmp_path, missing):
    path, manifest = source(tmp_path)
    view = recovery_view(
        path, manifest, target_world_size=3, root=tmp_path / "recovery"
    )
    plan = json.loads((view / "runtime-0.json").read_text())
    plan.pop(missing)
    engine = make_engine("cpu")
    engine.world_size = 3
    with pytest.raises(ValueError, match="Incomplete or invalid"):
        restore_recovery_runtime(engine, plan)


def test_uncertain_target_cleanup_blocks_rollback_and_retains_consumer_fence(tmp_path):
    """An unverified target must not overlap another worker generation."""

    class UncleanTarget(ResizableGroup):
        allow_cleanup = False

        def start(self, *, restore=False):
            if self.training_config.world_size == 3:
                self.actor = object()
                raise RuntimeError("target failed")
            super().start(restore=restore)

        def stop(self):
            if (
                self.training_config.world_size == 3
                and self.actor is not None
                and not self.allow_cleanup
            ):
                raise RuntimeError("release unverified")
            super().stop()

    queue = BatchQueue(tmp_path / "queue")
    group = UncleanTarget(tmp_path / "checkpoints")
    group.configure(TrainingConfig(2))
    trainer = QueuedTrainer(group, queue)
    queue.enqueue(
        "initial",
        {"policy_version": 0, "value": 1, "samples": 2, "tokens": 50},
        policy_version=0,
    )
    trainer.start()
    trainer.train_next()
    trainer.checkpoint()
    try:
        with pytest.raises(RuntimeError, match="release unverified"):
            trainer.configure(TrainingConfig(3))
        assert group.training_config.world_size == 3
        assert not trainer.is_committed
        with pytest.raises(BlockingIOError):
            queue.consumer()
    finally:
        group.allow_cleanup = True
        trainer.stop()


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
    import areal.autopilot.elastic.runtime.config as module

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


def test_loop_applies_actor_config_without_replaying_training_and_rejects_rollout(
    tmp_path,
):
    group = ResizableGroup(tmp_path / "checkpoints")
    queue = BatchQueue(tmp_path / "queue")
    trainer = QueuedTrainer(group, queue)
    executor = ActorConfigExecutor(trainer, rollout_config())
    application = ConfigApplication(executor, tmp_path / "applications")
    publisher = ProtocolPublisher()
    loop = ElasticLoop(trainer, publisher, config_application=application)
    try:
        loop.start()
        with loop.episode() as version:
            queue.enqueue(
                "one",
                {"policy_version": version, "value": 1, "samples": 2, "tokens": 50},
                policy_version=version,
            )
        loop.tick()
        source = executor.observe()
        request = config_request(executor)
        receipt = loop.apply_config(request)
        assert receipt["status"] == "applied" and receipt["generation"] == 1
        assert trainer.training_config == TrainingConfig(2)
        assert trainer.version == 1 and group.updates == 1
        assert executor.observe().boundary_id == source.boundary_id
        assert publisher.ready and publisher.version == 1
        configurations = group.configurations.copy()
        assert loop.apply_config(request) == receipt
        assert group.configurations == configurations
        changed_rollout = replace(
            request.target, rollout=replace(rollout_config(), replicas=2)
        )
        with pytest.raises(ValueError, match="Rollout reconfiguration"):
            loop.apply_config(config_request(executor, 1, "rollout", changed_rollout))
        assert publisher.ready and group.configurations == configurations
    finally:
        loop.close()


def test_actor_config_journal_reopens_same_run_and_rejects_another_run(tmp_path):
    queue = BatchQueue(tmp_path / "queue")
    journal = tmp_path / "applications"

    def executor(queue, store):
        return ActorConfigExecutor(
            QueuedTrainer(ResizableGroup(store), queue), rollout_config()
        )

    original = executor(queue, tmp_path / "checkpoints")
    original.trainer.prepare()
    ConfigApplication(original, journal).state()
    original.trainer.stop()
    same_run = executor(BatchQueue(queue.root), tmp_path / "checkpoints")
    assert same_run.context_id == original.context_id
    assert ConfigApplication(same_run, journal).state()["generation"] == 0
    for other in (
        executor(BatchQueue(tmp_path / "other-queue"), tmp_path / "checkpoints"),
        executor(queue, tmp_path / "other-checkpoints"),
    ):
        with pytest.raises(ValueError, match="another execution context"):
            ConfigApplication(other, journal).state()


def test_automatic_config_preflight_rejects_before_closing_admission(
    tmp_path, monkeypatch
):
    from areal.autopilot.contracts import ResourceDecision

    class ChooseTwo:
        def decide(self, obs):
            return ResourceDecision("launch", TrainingConfig(2), True)

    queue = BatchQueue(tmp_path / "queue")
    group = ResizableGroup(tmp_path / "checkpoints")
    trainer = QueuedTrainer(group, queue)
    executor = ActorConfigExecutor(trainer, rollout_config())
    application = ConfigApplication(executor, tmp_path / "applications")
    publisher = ProtocolPublisher()
    loop = ElasticLoop(
        trainer, publisher, policy=ChooseTwo(), config_application=application
    )
    try:
        loop.start()
        with loop.episode() as version:
            queue.enqueue(
                "one",
                {"policy_version": version, "value": 1, "samples": 2, "tokens": 50},
                policy_version=version,
            )

        def reject(target):
            assert publisher.ready
            raise ValueError("Executor capability unavailable")

        monkeypatch.setattr(executor, "validate", reject)
        assert loop.tick() == "SUSPENDED"
        assert publisher.ready and group.updates == 0 and not group.configurations
        assert application.state()["pending"] is None
        assert loop.events[-1]["event"] == "DECISION_REJECTED"
    finally:
        loop.close()


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


def test_v0_interrupted_config_restart_can_release_recovered_actor(
    tmp_path, monkeypatch
):
    import areal.autopilot.elastic.runtime.config as module

    class PowerLoss(BaseException):
        pass

    queue = BatchQueue(tmp_path / "queue")
    trainer = QueuedTrainer(ResizableGroup(tmp_path / "checkpoints"), queue)
    executor = ActorConfigExecutor(trainer, rollout_config())
    application = ConfigApplication(executor, tmp_path / "applications")
    loop = ElasticLoop(trainer, ProtocolPublisher(), config_application=application)
    write = module.atomic_json

    def interrupt(path, state):
        write(path, state)
        if state["pending"] is not None:
            raise PowerLoss()

    loop.start()
    monkeypatch.setattr(module, "atomic_json", interrupt)
    try:
        with pytest.raises(PowerLoss):
            loop.apply_config(config_request(executor))
        assert not loop.publisher.ready
    finally:
        monkeypatch.setattr(module, "atomic_json", write)
        loop.close()
    restored = QueuedTrainer(ResizableGroup(tmp_path / "checkpoints"), queue)
    application = ConfigApplication(
        ActorConfigExecutor(restored, rollout_config()), application.root
    )
    clock = [0.0]
    loop = ElasticLoop(
        restored,
        ProtocolPublisher(),
        clock=lambda: clock[0],
        config_application=application,
    )
    try:
        loop.start()
        assert loop.state == "ACTIVE" and restored.is_active
        assert restored.version == 0 and restored.latest_checkpoint is None
        clock[0] = 10_000
        assert loop.tick() == "SUSPENDED" and not restored.is_active
        assert loop.publisher.ready
    finally:
        loop.close()


@pytest.fixture
def joint_execution(tmp_path):
    from types import SimpleNamespace

    group = ResizableGroup(tmp_path / "checkpoints")
    group.pool_devices = list(range(4))
    group.is_v2 = True
    group.compatibility = {**group.compatibility, "elastic_dp": "fsdp-pure-dp-v1"}
    group.configure(TrainingConfig(2))
    queue = BatchQueue(tmp_path / "queue")
    trainer = QueuedTrainer(group, queue)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"num_attention_heads": 4, "num_key_value_heads": 2})
    )
    group.compatibility.update(
        model=str(model.resolve()), model_revision="fixture-revision"
    )
    failures = {"start": False, "close": False, "resume": False}
    incarnations = []

    class Service:
        def __init__(self, **kwargs):
            self.devices, self.layout = kwargs["devices"], kwargs["rollout_config"]
            self.owner = str(len(incarnations))
            self.alive, self.sessions = False, 0
            stats = SimpleNamespace(accepted=0)
            stats.on_version_recovered = lambda version: setattr(
                stats, "accepted", version
            )
            self.controller = SimpleNamespace(
                version=0,
                staleness_manager=stats,
                pause=lambda: None,
                pause_generation=lambda: None,
                continue_generation=lambda: None,
                resume=lambda: None,
            )
            incarnations.append(self)

        def start(self):
            assert not trainer.is_active or not set(self.devices).intersection(
                group.devices
            )
            self.alive = True
            if failures["start"]:
                failures["start"] = False
                raise RuntimeError("Target Rollout startup failed")

        def close(self):
            if failures["close"]:
                raise RuntimeError("Rollout cleanup unverified")
            self.alive = False

        def assert_drained(self):
            if self.sessions:
                raise ValueError("Rollout sessions have not drained")

        def verify_version(self, version):
            return self.alive and self.controller.version == version

        def resume(self):
            if failures["resume"]:
                raise RuntimeError("A Rollout replica is still paused")

    rollout = RolloutGroup(
        config=replace(rollout_config(), backend="sglang"),
        model=model,
        model_name="fixture",
        model_revision="fixture-revision",
        root=tmp_path / "rollout",
        lock_root=tmp_path / "locks",
        service_factory=Service,
        context_length=64,
    )

    class Publisher(ProtocolPublisher):
        def __init__(self):
            super().__init__()
            self.service = rollout

        def initialize(self, checkpoint=None, *, admit=True):
            self.version = (
                0
                if checkpoint is None
                else json.loads((checkpoint / "manifest.json").read_text())[
                    "global_step"
                ]
            )
            rollout.controller.version = self.version
            rollout.mark_weights_verified(self.version)
            self.ready = admit
            return {"version": self.version}

        def publish(self, checkpoint, *, admit=True):
            return self.initialize(checkpoint, admit=admit)

    publisher = Publisher()
    executor = JointConfigExecutor(trainer, rollout, publisher)
    application = ConfigApplication(executor, tmp_path / "applications")
    loop = ElasticLoop(trainer, publisher, config_application=application)
    loop.start()
    with loop.episode() as version:
        queue.enqueue(
            "one",
            {"policy_version": version, "value": 1, "samples": 2, "tokens": 50},
            policy_version=version,
        )
    loop.tick()
    yield loop, executor, application, failures, incarnations
    failures["close"] = False
    failures["resume"] = False
    loop.close()


def test_joint_basic_policy_uses_capacity_remaining_after_rollout(joint_execution):
    loop, executor, application, failures, incarnations = joint_execution
    profiles = tuple(
        Measurement("train", n, duration, 100)
        for n, duration in enumerate((1000, 100, 1, 0.001), 1)
    )
    policy = BasicPolicy(
        WorkflowSimulator(
            replace(
                model(), train=MeasuredComponent("train", profiles, scale_tokens=True)
            )
        ),
        optimizer=BasicOptimizer(horizon=1),
    )
    loop._policy = policy
    assert loop.observe().max_train_gpus == 3

    def consume(identity):
        with loop.episode() as version:
            executor.trainer.queue.enqueue(
                identity,
                {"policy_version": version, "value": 2, "samples": 2, "tokens": 50},
                policy_version=version,
            )
        assert loop.tick() == "ACTIVE" and policy.last_search.prediction is not None

    consume("two")
    assert executor.trainer.training_config == TrainingConfig(3)
    current = executor.observe().config
    target = replace(
        current, actor=TrainingConfig(2), rollout=replace(current.rollout, replicas=2)
    )
    loop.apply_config(
        config_request(executor, application.state()["generation"], "replicas", target)
    )
    assert loop.observe().max_train_gpus == 2
    consume("three")
    assert executor.trainer.training_config == TrainingConfig(2)
    assert executor.trainer.group.updates == 3


def test_joint_apply_changes_both_roles_and_moves_gpus_without_replaying_training(
    joint_execution,
):
    loop, executor, application, failures, incarnations = joint_execution
    source = executor.observe()
    initial_owner = executor.rollout.owner
    target = replace(
        source.config,
        actor=TrainingConfig(1),
        rollout=replace(
            source.config.rollout,
            replicas=2,
            max_batch_sequences=4,
            kv_capacity_tokens=256,
        ),
    )
    request = config_request(executor, target=target)
    result = loop.apply_config(request)
    assert result["status"] == "applied"
    assert executor.observe() == replace(source, config=target)
    assert executor.rollout.devices == (0, 1) and executor.trainer.group.devices == [2]
    assert executor.rollout.owner != initial_owner and not incarnations[0].alive
    assert executor.trainer.group.updates == 1 and loop.publisher.ready
    count = len(incarnations)
    assert loop.apply_config(request) == result and len(incarnations) == count
    owner = executor.rollout.owner
    actor_only = replace(target, actor=TrainingConfig(2))
    loop.apply_config(config_request(executor, 1, "actor-only", actor_only))
    assert executor.rollout.owner == owner  # Unchanged Rollout is retained.
    assert executor.trainer.group.devices == [2, 3]
    tp = replace(actor_only, rollout=replace(target.rollout, replicas=1, tp=2))
    loop.apply_config(config_request(executor, 2, "tp", tp))
    assert executor.observe() == replace(source, config=tp)
    assert (
        executor.trainer.group.updates == 1 and application.state()["generation"] == 3
    )


def test_joint_preflight_rejects_budget_and_live_sessions_before_mutation(
    joint_execution,
):
    loop, executor, application, failures, incarnations = joint_execution
    source = executor.observe()
    target = replace(source.config, rollout=replace(source.config.rollout, replicas=4))
    with pytest.raises(ValueError, match="shared GPU pool"):
        loop.apply_config(config_request(executor, target=target))
    executor.rollout.service.sessions = 1
    with pytest.raises(ValueError, match="not drained"):
        loop.apply_config(config_request(executor))
    executor.rollout.service.sessions = 0
    assert loop.publisher.ready and application.state()["pending"] is None
    assert len(incarnations) == 1 and executor.observe() == source


def test_joint_rollout_failure_recovers_both_roles_at_the_same_boundary(
    joint_execution,
):
    loop, executor, application, failures, incarnations = joint_execution
    source = executor.observe()
    failures["start"] = True
    target = replace(
        source.config,
        actor=TrainingConfig(1),
        rollout=replace(source.config.rollout, replicas=2),
    )
    receipt = loop.apply_config(config_request(executor, target=target))
    assert receipt["status"] == "rolled_back" and receipt["generation"] == 0
    assert executor.observe() == source and executor.trainer.group.updates == 1
    assert loop.publisher.ready and len(incarnations) == 3
    assert (
        executor.rollout.controller.staleness_manager.accepted == source.weights_version
    )


def test_joint_uncertain_cleanup_retains_queue_fence_and_admission(joint_execution):
    loop, executor, application, failures, incarnations = joint_execution
    source = executor.observe()
    failures["close"] = True
    target = replace(source.config, rollout=replace(source.config.rollout, replicas=2))
    with pytest.raises(RuntimeError, match="cleanup unverified"):
        loop.apply_config(config_request(executor, target=target))
    assert loop.state == "FAILED" and not loop.publisher.ready
    assert (
        executor.trainer.consumer is not None
        and application.state()["pending"] is not None
    )
    failures["close"] = False
    assert loop.recover_config()["status"] == "rolled_back"
    assert executor.observe() == source and loop.publisher.ready


def test_joint_receipt_failure_keeps_new_service_closed_until_recovery(
    joint_execution, monkeypatch
):
    import areal.autopilot.elastic.runtime.config as module

    loop, executor, application, failures, incarnations = joint_execution
    source = executor.observe()
    write = module.atomic_json
    fired = False

    def interrupt(path, state):
        nonlocal fired
        assert not loop.publisher.ready
        if not fired and state["pending"] is None:
            fired = True
            raise OSError("Lost joint receipt")
        write(path, state)

    request = config_request(
        executor,
        target=replace(
            source.config, rollout=replace(source.config.rollout, replicas=2)
        ),
    )
    monkeypatch.setattr(module, "atomic_json", interrupt)
    with pytest.raises(OSError, match="Lost joint receipt"):
        loop.apply_config(request)
    assert not loop.publisher.ready and loop.state == "FAILED"
    monkeypatch.setattr(module, "atomic_json", write)
    loop.recover_config()
    assert executor.observe() == source and executor.trainer.group.updates == 1


def test_joint_mismatched_initial_models_are_rejected(joint_execution):
    loop, executor, application, failures, incarnations = joint_execution
    executor.rollout.model_revision = "another-revision"
    with pytest.raises(ValueError, match="immutable base model and revision"):
        JointConfigExecutor(executor.trainer, executor.rollout, loop.publisher)
    executor.rollout.model_revision = "fixture-revision"


def test_joint_partial_resume_failure_stays_closed_after_durable_commit(
    joint_execution,
):
    loop, executor, application, failures, incarnations = joint_execution
    source = executor.observe()
    target = replace(source.config, rollout=replace(source.config.rollout, replicas=2))
    failures["resume"] = True
    with pytest.raises(RuntimeError, match="still paused"):
        loop.apply_config(config_request(executor, target=target))
    assert loop.state == "FAILED" and not loop.publisher.ready
    assert (
        application.state()["generation"] == 1
        and application.state()["pending"] is None
    )
    assert executor.observe() == replace(source, config=target)
    failures["resume"] = False
    assert loop.recover_config() is None
    assert loop.publisher.ready and executor.trainer.group.updates == 1


def install_online_joint(loop, executor, tmp_path):
    from tests.test_elastic_basic_policy import serial_joint_policy

    from areal.autopilot.elastic.runtime.history import RuntimeHistory

    policy, current, target = serial_joint_policy(seconds=100)
    # The initial fixture uses the same complete Rollout identity.
    assert executor.observe().config == current
    policy.simulator.episodes = tuple(
        replace(row, seconds=1000 if row.rollout == current.rollout else 0.1)
        for row in policy.simulator.episodes
    )
    loop._policy = policy
    loop.history = policy.history
    loop.runtime_history = RuntimeHistory(
        tmp_path / "online-history.json",
        context={
            "queue_id": loop.trainer.queue.queue_id,
            "checkpoint_root": str(loop.trainer.group.store.root.resolve()),
            "compatibility": loop.trainer.group.compatibility,
        },
    )
    return policy, current, target


def online_episode(loop, executor, *, duration=100):
    with loop.episode() as version:
        # Independent clock makes the whole-episode measurement observable.
        clock = loop.clock
        loop.clock = lambda: clock() + duration
        executor.trainer.queue.enqueue(
            f"auto-{version}",
            {"policy_version": version, "value": 2, "samples": 2, "tokens": 50},
            policy_version=version,
        )


def test_online_joint_loop_automatically_switches_both_roles_and_commits_actual_feedback(
    joint_execution,
    tmp_path,
):
    """No manual ConfigRequest: causal history triggers a real protocol transition."""
    loop, executor, application, failures, incarnations = joint_execution
    policy, current, target = install_online_joint(loop, executor, tmp_path)
    online_episode(loop, executor)
    assert policy.through_version == 0 and not policy.episodes
    loop.tick()  # Cold bootstrap, then commit current-config episode.
    assert executor.observe().config == current and policy.through_version == 2
    assert len(policy.episodes) == 1
    online_episode(loop, executor)
    loop.tick()
    assert executor.observe().config == target
    assert application.state()["generation"] == 1
    assert policy.last_window.observation.model_version == 2
    assert [row.source_id for row in policy.last_window.episodes] == [
        "committed-episode-v2"
    ]
    assert policy.through_version == 3 and len(policy.episodes) == 2
    assert all(row.rollout == current.rollout for row in policy.episodes)
    rows = loop.runtime_history.snapshot(3)["updates"]
    assert rows[-1]["joint_config"] == asdict(target)
    assert rows[-1]["episode"]["rollout"] == asdict(current.rollout)
    assert rows[-1]["observation"]["model_version"] == 2
    assert executor.trainer.group.updates == 3 and loop.publisher.ready
    assert executor.rollout.devices == (0, 1) and executor.trainer.group.devices == [2]
    online_episode(loop, executor, duration=1)
    loop.tick()
    assert policy.episodes[-1].rollout == target.rollout
    assert policy.last_window.observation.model_version == 3
    assert loop.trainer.version == loop.published_version == 4


def test_online_joint_failure_feedback_uses_actual_source_and_no_duplicate_training(
    joint_execution,
    tmp_path,
):
    loop, executor, application, failures, incarnations = joint_execution
    policy, current, target = install_online_joint(loop, executor, tmp_path)
    online_episode(loop, executor)
    loop.tick()
    failures["start"] = True
    online_episode(loop, executor)
    loop.tick()
    assert loop.trainer.version == 3 and executor.observe().config == current
    assert application.state()["generation"] == 0
    assert loop.runtime_history.snapshot(3)["updates"][-1]["joint_config"] == asdict(
        current
    )
    assert loop.last_config_receipt["status"] == "rolled_back" and loop.publisher.ready
    online_episode(loop, executor)
    loop.tick()
    assert executor.observe().config == target and loop.trainer.version == 4
    assert loop.trainer.last_reconfiguration_error is None
    assert loop.last_config_receipt["status"] == "applied"


def test_online_joint_can_automatically_change_rollout_with_unchanged_actor_count(
    joint_execution,
    tmp_path,
):
    loop, executor, application, failures, incarnations = joint_execution
    policy, current, target = install_online_joint(loop, executor, tmp_path)
    rollout_only = replace(target, actor=current.actor)
    policy.optimizer = replace(policy.optimizer, candidates=(current, rollout_only))
    policy.simulator.phases = tuple(
        replace(row, config=rollout_only) if row.config == target else row
        for row in policy.simulator.phases
    )
    policy.simulator.transitions = replace(
        policy.simulator.transitions,
        profiles=tuple(
            replace(
                row,
                source=rollout_only if row.source == target else row.source,
                target=rollout_only if row.target == target else row.target,
            )
            for row in policy.simulator.transitions.profiles
        ),
    )
    online_episode(loop, executor)
    loop.tick()
    online_episode(loop, executor)
    loop.tick()
    assert executor.observe().config == rollout_only
    assert loop.trainer.training_config == current.actor
    assert application.state()["generation"] == 1 and loop.trainer.version == 3
    assert executor.rollout.devices == (0, 1) and loop.trainer.group.devices == [2, 3]


def test_online_joint_durable_history_restores_without_replaying_config_or_updates(
    joint_execution,
    tmp_path,
):
    loop, executor, application, failures, incarnations = joint_execution
    policy, current, target = install_online_joint(loop, executor, tmp_path)
    online_episode(loop, executor)
    loop.tick()
    online_episode(loop, executor)
    loop.tick()
    rows = loop.runtime_history.snapshot(3)["updates"]
    from tests.test_elastic_basic_policy import serial_joint_policy

    recovered, _, _ = serial_joint_policy()
    # Restoring twice is idempotent, as when publication recovery rechecks history.
    for row in (*rows, *rows):
        recovered.record_committed(row)
    assert recovered.through_version == 3 and recovered.episodes == policy.episodes
    assert recovered.history.snapshot() == policy.history.snapshot()
    assert application.state()["generation"] == 1 and loop.trainer.version == 3
    history = loop.runtime_history
    loop.close()
    trainer = QueuedTrainer(executor.trainer.group, executor.trainer.queue)
    replacement_executor = JointConfigExecutor(
        trainer, executor.rollout, loop.publisher
    )
    replacement = ElasticLoop(
        trainer,
        loop.publisher,
        policy=recovered,
        config_application=ConfigApplication(replacement_executor, application.root),
        runtime_history=history,
    )
    try:
        replacement.start()
        assert recovered.through_version == replacement.trainer.version == 3
        assert len(recovered.episodes) == 2
        assert replacement_executor.observe().config == target
        assert replacement.config_application.state()["generation"] == 1
        online_episode(replacement, replacement_executor, duration=1)
        replacement.tick()
        assert replacement.trainer.version == recovered.through_version == 4
        assert replacement.trainer.group.updates == 4
    finally:
        replacement.close()


def test_online_joint_planning_does_not_run_under_live_episode_gate(
    joint_execution,
    tmp_path,
):
    from concurrent.futures import ThreadPoolExecutor

    loop, executor, application, failures, incarnations = joint_execution
    policy, current, target = install_online_joint(loop, executor, tmp_path)
    online_episode(loop, executor)
    loop.tick()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with loop.episode() as version:
            executor.trainer.queue.enqueue(
                "held",
                {"policy_version": version, "value": 3, "samples": 2, "tokens": 50},
                policy_version=version,
            )
            pool.submit(loop.tick).result(timeout=5)
            assert policy.last_search is None and application.state()["generation"] == 0
    loop.tick()
    assert executor.observe().config == target and loop.trainer.version == 3


def test_online_joint_idle_release_is_durable_boundary_measurement_not_an_update(
    joint_execution,
    tmp_path,
):
    from tests.test_elastic_basic_policy import serial_joint_policy

    loop, executor, application, failures, incarnations = joint_execution
    policy, current, target = install_online_joint(loop, executor, tmp_path)
    online_episode(loop, executor)
    loop.tick()
    loop.clock = lambda: loop.last_activity + 10
    assert loop.tick() == "SUSPENDED"
    snapshot = loop.runtime_history.snapshot(2)
    assert len(snapshot["updates"]) == 1 and len(snapshot["boundaries"]) == 1
    boundary = snapshot["boundaries"][0]
    assert boundary["phase"] == "release" and boundary["joint_config"] == asdict(
        current
    )
    assert policy.phases[-1].source_id == "boundary-1"
    recovered, _, _ = serial_joint_policy()
    for row in snapshot["updates"]:
        recovered.record_committed(row)
    recovered.record_boundary(boundary)
    recovered.record_boundary(boundary)
    assert recovered.phases == policy.phases
    assert recovered.history.snapshot() == policy.history.snapshot()
    assert loop.trainer.version == 2 and executor.trainer.group.updates == 2
