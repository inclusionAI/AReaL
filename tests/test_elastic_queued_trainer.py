# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from areal.autopilot.elastic.runtime.state.batch_queue import BatchQueue
from areal.autopilot.elastic.runtime.state.checkpoint import (
    CheckpointStore,
    atomic_json,
)
from areal.autopilot.elastic.runtime.training.trainer import QueuedTrainer


class StateMachineGroup:
    """Protocol fixture; order-sensitive updates, no mocked FSDP internals."""

    def __init__(self, root):
        self.store = CheckpointStore(root, free_space_floor=0)
        self.compatibility = {"fixture": "ordered-updates"}
        self.actor = None
        self.step = 0
        self.history = []
        self.value = 0

    def start(self, *, restore=False):
        self.actor = object()
        self.step, self.history, self.value = 0, [], 0
        if restore:
            path, manifest = self.store.latest(self.compatibility)
            self.step = manifest["global_step"]
            state = json.loads((path / "dcp" / ".metadata").read_text())
            self.history, self.value = state["history"], state["value"]

    def update(self, batch):
        self.step += 1
        self.history.append(batch.batch_id)
        self.value = self.value * 3 + batch.payload["value"]
        return {"step": self.step}

    def checkpoint(self, *, data_cursor):
        def write(path):
            (path / "dcp").mkdir()
            atomic_json(
                path / "dcp" / ".metadata",
                {"history": self.history, "value": self.value},
            )
            atomic_json(
                path / "runtime-0.json",
                {"rank": 0, "world_size": 1, "policy_version": self.step},
            )

        return self.store.commit(
            step=self.step,
            world_size=1,
            compatibility=self.compatibility,
            write=write,
            estimated_bytes=0,
            data_cursor=data_cursor,
        )

    def stop(self):
        self.actor = None


def setup_trainer(tmp_path):
    queue = BatchQueue(tmp_path / "queue")
    for i in range(6):
        queue.enqueue(str(i), {"value": i + 1}, policy_version=0)
    group = StateMachineGroup(tmp_path / "checkpoints")
    trainer = QueuedTrainer(group, queue)
    trainer.start()
    return group, queue, trainer


@pytest.mark.parametrize("window", ["after_update", "after_publish", "after_ack"])
def test_restart_reconciles_crash_windows_without_skipping_or_double_applying(
    tmp_path, monkeypatch, window
):
    """Only the checkpoint's data cursor decides which updates survive."""
    group, queue, trainer = setup_trainer(tmp_path)
    for _ in range(2):
        trainer.train_next(group.update)
    trainer.checkpoint()
    trainer.train_next(group.update)
    if window == "after_publish":

        def lost_ack(_):
            raise OSError("injected lost acknowledgement")

        monkeypatch.setattr(trainer.consumer, "acknowledge", lost_ack)
        with pytest.raises(OSError, match="lost acknowledgement"):
            trainer.checkpoint()
    elif window == "after_ack":
        trainer.checkpoint()
    trainer.stop()

    recovered = QueuedTrainer(group, BatchQueue(queue.root))
    recovered.start()
    assert group.step == (2 if window == "after_update" else 3)
    while recovered.train_next(group.update) is not None:
        pass
    recovered.checkpoint()
    recovered.stop()

    assert group.history == [str(i) for i in range(6)]
    assert group.value == sum((i + 1) * 3 ** (5 - i) for i in range(6))
    assert queue.status()["counts"]["committed"] == 6


def test_uncertain_update_cannot_retry_on_live_group(tmp_path):
    """An RPC may have updated weights before its response was lost."""
    group, queue, trainer = setup_trainer(tmp_path)

    def lost_response(batch):
        group.update(batch)
        raise TimeoutError("response lost")

    with pytest.raises(TimeoutError):
        trainer.train_next(lost_response)
    with pytest.raises(RuntimeError, match="uncertain"):
        trainer.train_next(group.update)
    with pytest.raises(RuntimeError, match="uncertain"):
        trainer.checkpoint()
    trainer.stop()
    trainer.start()
    assert group.step == 0
    assert trainer.train_next(group.update)["batch_id"] == "0"
    trainer.stop()


def test_failed_checkpoint_never_acknowledges_uncommitted_updates(
    tmp_path, monkeypatch
):
    """Disk exhaustion preserves the old cursor and forces recovery."""
    group, queue, trainer = setup_trainer(tmp_path)
    trainer.train_next(group.update)

    def full_disk(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(group, "checkpoint", full_disk)
    with pytest.raises(OSError, match="disk full"):
        trainer.checkpoint()
    assert queue.status()["counts"]["committed"] == 0
    with pytest.raises(RuntimeError, match="uncertain"):
        trainer.train_next(group.update)
    trainer.stop()


def test_failed_teardown_retains_consumer_fence(tmp_path, monkeypatch):
    """Never transfer queue ownership while an old GPU worker may be alive."""
    group, queue, trainer = setup_trainer(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(
            group, "stop", lambda: (_ for _ in ()).throw(RuntimeError("alive"))
        )
        with pytest.raises(RuntimeError, match="alive"):
            trainer.stop()
        with pytest.raises(RuntimeError, match="uncertain"):
            trainer.train_next(group.update)
        with pytest.raises(BlockingIOError):
            queue.consumer()
    trainer.stop()
    with queue.consumer():
        pass


def test_checkpoint_is_read_only_after_consumer_ownership_is_acquired(
    tmp_path, monkeypatch
):
    """Another trainer cannot publish between observing latest and taking ownership."""
    group, queue, trainer = setup_trainer(tmp_path)
    trainer.train_next(group.update)
    trainer.checkpoint()
    trainer.stop()
    original = group.store.latest

    def guarded_read(compatibility):
        with pytest.raises(BlockingIOError):
            queue.consumer()
        return original(compatibility)

    monkeypatch.setattr(group.store, "latest", guarded_read)
    trainer.start()
    assert group.step == 1
    trainer.stop()
