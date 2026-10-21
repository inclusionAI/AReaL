# SPDX-License-Identifier: Apache-2.0

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.test_elastic_queued_trainer import StateMachineGroup

from areal.autopilot.elastic.loop import ElasticLoop
from areal.autopilot.elastic.policy import IdleTimeoutPolicy
from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue
from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer


class ProtocolGroup(StateMachineGroup):
    """Order-sensitive CPU state machine, with observable resource ownership."""

    def __init__(self, root):
        super().__init__(root)
        self.starts, self.releases, self.saves, self.updates = [], 0, 0, 0

    def start(self, *, restore=False):
        self.starts.append(restore)
        super().start(restore=restore)

    def stop(self):
        self.releases += int(self.actor is not None)
        super().stop()

    def validate_trajectory(self, payload):
        if not payload.get("valid", True):
            raise ValueError("Invalid trajectory")
        if payload["policy_version"] != self.step:
            raise ValueError("Not the current policy")

    def train_trajectory(self, payload):
        self.validate_trajectory(payload)
        self.updates += 1
        return self.update(
            SimpleNamespace(
                batch_id=f"episode-{payload['policy_version']}",
                payload=payload,
            )
        )

    def checkpoint(self, *, data_cursor):
        self.saves += 1
        return super().checkpoint(data_cursor=data_cursor)


class ProtocolPublisher:
    def __init__(self):
        self.lock = threading.RLock()
        self.ready, self.version = False, 0
        self.failures, self.attempts, self.receipt = 0, [], None

    def initialize(self):
        self.ready = True

    @contextmanager
    def episode(self):
        with self.lock:
            if not self.ready:
                raise RuntimeError("Rollout admission is closed")
            yield self.version

    def publish(self, checkpoint):
        self.ready = False
        self.attempts.append(checkpoint)
        self.version = json.loads((checkpoint / "manifest.json").read_text())[
            "global_step"
        ]
        self.receipt = {"version": self.version}
        if self.failures:
            self.failures -= 1
            raise RuntimeError("Resume failed after writing receipt")
        self.ready = True
        return self.receipt


@pytest.fixture
def setup(tmp_path, request):
    now = [0.0]
    group = ProtocolGroup(tmp_path / "checkpoints")
    queue = BatchQueue(tmp_path / "queue")
    publisher = ProtocolPublisher()
    loop = ElasticLoop(
        QueuedTrainer(group, queue),
        publisher,
        policy=getattr(request, "param", IdleTimeoutPolicy(idle_seconds=5)),
        clock=lambda: now[0],
    )
    loop.start()
    yield loop, group, queue, publisher, now
    loop.close()


def enqueue(loop, queue):
    with loop.episode() as version:
        queue.enqueue(
            f"episode-{version}",
            {"policy_version": version, "value": version + 1},
            policy_version=version,
        )


def test_empty_queue_keeps_gpu_off_and_retains_consumer_ownership(setup):
    loop, group, queue, publisher, now = setup
    now[0] = 100
    assert loop.tick() == "SUSPENDED"
    assert group.starts == [] and group.saves == group.releases == 0
    with pytest.raises(BlockingIOError):
        queue.consumer()
    loop.close()
    with queue.consumer():
        pass


def test_trainer_preview_checks_readiness_without_leasing_or_allocating(setup):
    """Scheduling can inspect a ready trajectory while keeping training suspended."""
    loop, group, queue, publisher, now = setup
    trainer = loop.trainer
    assert trainer.version == 0 and not trainer.is_active
    assert not trainer.has_pending_batch() and trainer.next_batch_id() is None

    enqueue(loop, queue)

    assert trainer.has_pending_batch()
    assert trainer.next_batch_id() == trainer.next_batch_id() == "episode-0"
    assert not trainer.is_active and group.starts == []
    assert queue.status()["counts"] == dict(pending=1, leased=0, applied=0, committed=0)


def test_default_consumption_rechecks_claimed_envelope_before_update(
    setup, monkeypatch
):
    """A validated preview must not authorize a different version at dispatch."""
    loop, group, queue, publisher, now = setup
    trainer = loop.trainer
    enqueue(loop, queue)
    assert trainer.next_batch_id() == "episode-0"
    trainer.activate()
    claim = trainer.consumer.claim
    monkeypatch.setattr(
        trainer.consumer, "claim", lambda: replace(claim(), policy_version=1)
    )

    with pytest.raises(ValueError, match="stale policy version"):
        trainer.train_next()

    assert group.updates == trainer.version == 0
    assert queue.status()["counts"]["applied"] == 0
    with pytest.raises(RuntimeError, match="uncertain"):
        trainer.train_next()


def test_arrival_wakes_and_clean_idle_releases_without_another_checkpoint(setup):
    loop, group, queue, publisher, now = setup
    enqueue(loop, queue)
    assert loop.tick() == "ACTIVE"
    assert group.step == 1 and group.starts == [False]
    now[0] = 4.9
    assert loop.tick() == "ACTIVE"
    now[0] = 5
    assert loop.tick() == "SUSPENDED"
    assert group.releases == group.saves == 1
    now[0] = 500
    loop.tick()
    assert group.releases == group.saves == 1
    enqueue(loop, queue)
    loop.tick()
    assert group.starts == [False, True] and group.step == 2
    assert group.history == ["episode-0", "episode-1"]
    assert queue.status()["counts"]["committed"] == 2


def test_receipt_before_failed_resume_retries_same_checkpoint_without_release(setup):
    loop, group, queue, publisher, now = setup
    publisher.failures = 2
    enqueue(loop, queue)
    assert loop.tick() == "PUBLISHING"
    target = loop.pending_checkpoint
    assert publisher.receipt == {"version": 1} and not publisher.ready
    now[0] = 100
    assert loop.tick() == "PUBLISHING"
    assert group.actor is not None and group.releases == 0
    assert loop.tick() == "ACTIVE"
    assert group.updates == group.saves == 1
    assert publisher.attempts == [target] * 3
    # Idle starts after successful publication, not the earlier optimizer call.
    assert loop.tick() == "ACTIVE"
    now[0] = 105
    assert loop.tick() == "SUSPENDED"


@pytest.mark.parametrize("envelope,payload", [(2, 2), (0, 1), (0, 0)])
def test_invalid_fifo_head_fails_before_gpu_activation(setup, envelope, payload):
    loop, group, queue, publisher, now = setup
    queue.enqueue(
        "invalid",
        {"policy_version": payload, "value": 1, "valid": payload != 0},
        policy_version=envelope,
    )
    with pytest.raises(ValueError):
        loop.tick()
    assert loop.state == "FAILED" and not publisher.ready
    assert group.starts == [] and group.updates == 0
    assert queue.status()["counts"]["pending"] == 1


def test_second_producer_waits_for_durable_enqueue_then_receives_backpressure(setup):
    loop, group, queue, publisher, now = setup
    attempted = threading.Event()

    def second():
        attempted.set()
        enqueue(loop, queue)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with loop.episode() as version:
            future = pool.submit(second)
            assert attempted.wait(timeout=5) and not future.done()
            queue.enqueue(
                "first", {"policy_version": version, "value": 1}, policy_version=version
            )
        with pytest.raises(RuntimeError, match="awaiting"):
            future.result(timeout=5)
    assert queue.status()["counts"]["pending"] == 1
    assert group.starts == []


def test_slow_episode_does_not_pin_clean_training_gpus_or_allow_early_updates(setup):
    loop, group, queue, publisher, now = setup
    enqueue(loop, queue)
    loop.tick()
    now[0] = 10
    with ThreadPoolExecutor(max_workers=1) as pool:
        with loop.episode() as version:
            assert version == 1
            assert pool.submit(loop.tick).result(timeout=5) == "SUSPENDED"
            assert group.actor is None and group.releases == 1
            queue.enqueue(
                "next", {"policy_version": version, "value": 2}, policy_version=version
            )
            assert pool.submit(loop.tick).result(timeout=5) == "SUSPENDED"
            assert publisher.version == 1 and group.updates == 1
        assert pool.submit(loop.tick).result(timeout=5) == "ACTIVE"
    assert publisher.version == group.step == 2


def test_checkpoint_ack_failure_is_fatal_and_prepare_repairs_without_gpu(
    setup, monkeypatch
):
    loop, group, queue, publisher, now = setup
    enqueue(loop, queue)
    monkeypatch.setattr(
        loop.trainer.consumer,
        "acknowledge",
        lambda _: (_ for _ in ()).throw(OSError("lost ack")),
    )
    with pytest.raises(OSError, match="lost ack"):
        loop.tick()
    assert loop.state == "FAILED" and group.updates == 1
    loop.close()
    replacement = ElasticLoop(QueuedTrainer(group, queue), ProtocolPublisher())
    try:
        replacement.start()
        assert replacement.published_version == group.step == 1
        assert group.actor is None and group.starts == [False]
        assert queue.status()["counts"]["committed"] == 1
        assert replacement.tick() == "SUSPENDED"
        assert group.updates == 1
    finally:
        replacement.close()


def test_failed_release_keeps_consumer_fenced_and_cannot_resume(setup, monkeypatch):
    loop, group, queue, publisher, now = setup
    enqueue(loop, queue)
    loop.tick()
    now[0] = 10
    with monkeypatch.context() as patch:
        patch.setattr(
            group, "stop", lambda: (_ for _ in ()).throw(RuntimeError("worker alive"))
        )
        with pytest.raises(RuntimeError, match="alive"):
            loop.tick()
        with pytest.raises(BlockingIOError):
            queue.consumer()
        with pytest.raises(RuntimeError, match="uncertain"):
            loop.trainer.train_next(group.update)
    loop.close()
    with queue.consumer(
        loop.trainer.latest_checkpoint
        and json.loads((loop.trainer.latest_checkpoint / "manifest.json").read_text())
    ):
        pass


def test_shutdown_waits_for_an_inflight_tick_before_teardown(setup, monkeypatch):
    loop, group, queue, publisher, now = setup
    enqueue(loop, queue)
    entered, finish, stop = threading.Event(), threading.Event(), threading.Event()
    start = group.start

    def blocked_start(**kwargs):
        entered.set()
        assert finish.wait(timeout=5)
        start(**kwargs)

    monkeypatch.setattr(group, "start", blocked_start)
    errors = []

    def serve():
        try:
            loop.run(stop, poll_seconds=0.01)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        stop.set()
        assert thread.is_alive() and group.releases == 0
        with pytest.raises(BlockingIOError):
            queue.consumer()
    finally:
        finish.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert group.updates == group.saves == 1 and group.releases == 0
    loop.close()
    assert group.releases == 1


@pytest.mark.parametrize(
    "setup", [SimpleNamespace(decide=lambda _: "train")], indirect=True
)
def test_resource_train_decision_cannot_allocate_without_data(setup):
    loop, group, queue, publisher, now = setup

    assert loop.tick() == "SUSPENDED"

    assert loop.events[-1]["event"] == "DECISION_REJECTED"
    assert group.starts == [] and group.updates == 0


@pytest.mark.parametrize(
    "setup",
    [
        SimpleNamespace(
            decide=lambda sample: "train" if sample.model_version == 0 else "release"
        )
    ],
    indirect=True,
)
def test_resource_release_decision_cannot_skip_pending_data(setup):
    loop, group, queue, publisher, now = setup
    enqueue(loop, queue)
    loop.tick()
    enqueue(loop, queue)

    assert loop.tick() == "ACTIVE"

    assert loop.events[-1]["event"] == "DECISION_REJECTED"
    assert group.updates == 1 and group.releases == 0


@pytest.mark.parametrize(
    "setup", [SimpleNamespace(decide=lambda _: "train")], indirect=True
)
def test_resource_train_decision_cannot_bypass_episode_gate(setup):
    loop, group, queue, publisher, now = setup
    with ThreadPoolExecutor(max_workers=1) as pool:
        with loop.episode() as version:
            queue.enqueue(
                "pending",
                {"policy_version": version, "value": 1},
                policy_version=version,
            )
            assert pool.submit(loop.tick).result(timeout=5) == "SUSPENDED"
    assert group.starts == [] and group.updates == 0
    assert loop.events[-1]["event"] == "DECISION_REJECTED"


@pytest.mark.parametrize(
    "setup", [SimpleNamespace(decide=lambda _: "unknown")], indirect=True
)
def test_invalid_resource_action_fails_closed(setup):
    loop, group, queue, publisher, now = setup
    with pytest.raises(ValueError, match="Invalid resource decision"):
        loop.tick()
    assert loop.state == "FAILED" and not publisher.ready
    assert group.starts == []


@pytest.mark.parametrize("operation", ["train_next", "publish"])
def test_observation_reflects_current_stage_before_blocking_operation(
    setup, monkeypatch, operation
):
    loop, group, queue, publisher, now = setup
    target = loop.trainer if operation == "train_next" else publisher
    original = getattr(target, operation)
    observed = []

    def capture(*args, **kwargs):
        observed.append(loop.observe())
        return original(*args, **kwargs)

    monkeypatch.setattr(target, operation, capture)
    enqueue(loop, queue)
    loop.tick()

    (sample,) = observed
    assert sample.state == ("ACTIVE" if operation == "train_next" else "PUBLISHING")
    assert sample.pending_publication == (operation == "publish")
    assert sample.training_active and sample.committed
    assert sample.model_version == (0 if operation == "train_next" else 1)
