# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from areal.elastic.state.checkpoint import (
    CheckpointStore,
    atomic_json,
)


@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
def test_native_full_checkpoint_restores_optimizer_scheduler_and_rng(
    tmp_path, monkeypatch, backend
):
    import random
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import numpy as np
    import torch

    from areal.v2.training_service import checkpoint as native

    class Engine:
        rank, world_size, device = 0, 1, torch.device("cpu")

        def __init__(self):
            self.model = torch.nn.Linear(2, 1)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=0.5
            )
            self.version = 1
            self.checkpointer = SimpleNamespace(wait_async_saves=MagicMock())
            self.mcore_config = SimpleNamespace(use_checkpoint_opt_param_scheduler=True)

        def get_version(self):
            return self.version

        def set_version(self, version):
            self.version = version

        def save(self, meta):
            root = Path(meta.path)
            root.mkdir(exist_ok=True)
            (root / ".metadata").write_text("test native tensor payload")
            if backend == "megatron":
                (root / "metadata.json").write_text(
                    json.dumps(
                        {
                            "sharded_backend": "torch_dist",
                            "sharded_backend_version": 1,
                            "common_backend": "torch",
                            "common_backend_version": 1,
                        }
                    )
                )
                (root / "common.pt").write_text("test mcore common state")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                root / "state.pt",
            )
            torch.rand(1)

        def load(self, meta):
            saved = torch.load(Path(meta.path) / "state.pt", weights_only=True)
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])

    source = Engine()
    source.model(torch.ones(1, 2)).sum().backward()
    source.optimizer.step()
    source.lr_scheduler.step()
    tracker = SimpleNamespace(
        get_states=lambda: {"model": torch.tensor([1, 2, 3], dtype=torch.uint8)},
        set_states=MagicMock(),
    )
    monkeypatch.setattr(native, "_tracker", lambda: tracker)
    identity = "a" * 64
    metadata = dict(
        path=str(tmp_path), backend=backend, identity_sha256=identity, version=1
    )
    saved = native.save_full_checkpoint(source, **metadata)
    expected = (random.random(), np.random.rand(), torch.rand(4))
    restored = Engine()
    restored.version = 0
    if backend == "megatron":
        scheduler = restored.lr_scheduler
        restored.lr_scheduler = SimpleNamespace(
            state_dict=scheduler.state_dict, load_state_dict=scheduler.load_state_dict
        )
    assert native.inspect_full_checkpoint(restored, loading=True, **metadata) == saved
    assert native.load_full_checkpoint(restored, **metadata) == saved
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert restored.get_version() == 1
    assert restored.optimizer.param_groups[0]["lr"] == 0.005
    for source_param, restored_param in zip(
        source.model.parameters(), restored.model.parameters()
    ):
        torch.testing.assert_close(source_param, restored_param, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                source.optimizer.state[source_param][key],
                restored.optimizer.state[restored_param][key],
                rtol=0,
                atol=0,
            )
    if backend == "megatron":
        source.checkpointer.wait_async_saves.assert_called_once()
        tracker.set_states.assert_called()
    with pytest.raises(ValueError, match="identity"):
        native.inspect_full_checkpoint(
            restored, **{**metadata, "identity_sha256": "b" * 64}, loading=True
        )


def write_generation(path, *, step, ranks=2):
    (path / "dcp").mkdir()
    (path / "dcp" / ".metadata").write_bytes(b"test metadata")
    for rank in range(ranks):
        atomic_json(
            path / f"runtime-{rank}.json",
            {"rank": rank, "world_size": 2, "policy_version": step},
        )


def commit(store, step):
    return store.commit(
        step=step,
        world_size=2,
        compatibility={"backend": "d2"},
        estimated_bytes=0,
        write=lambda path: write_generation(path, step=step),
    )


def test_checkpoint_incomplete_write_preserves_previous_generation(tmp_path):
    """A missing rank must never replace the last recoverable pointer."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    previous = commit(store, 1)

    with pytest.raises(FileNotFoundError):
        store.commit(
            step=2,
            world_size=2,
            compatibility={"backend": "d2"},
            estimated_bytes=0,
            write=lambda path: write_generation(path, step=2, ranks=1),
        )

    path, manifest = store.latest({"backend": "d2"})
    assert path == previous
    assert manifest["global_step"] == 1
    assert previous.is_dir()


def test_checkpoint_retention_prunes_only_after_successful_commit(tmp_path):
    """Bound storage while retaining the latest two completed generations."""
    store = CheckpointStore(tmp_path, keep=2, free_space_floor=0)
    first = commit(store, 1)
    second = commit(store, 2)
    third = commit(store, 3)

    assert not first.exists()
    assert second.is_dir() and third.is_dir()
    assert store.latest({"backend": "d2"})[0] == third


def test_checkpoint_incompatible_config_rejected(tmp_path):
    """Changing the parallel layout requires an explicit migration."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    commit(store, 1)

    with pytest.raises(ValueError, match="incompatible"):
        store.latest({"backend": "d4"})


def test_checkpoint_corrupted_progress_rejected(tmp_path):
    """A changed manifest must not skip batches while loading older rank states."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    path = commit(store, 1)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(global_step=5, next_batch_id=5)
    atomic_json(path / "manifest.json", manifest)

    with pytest.raises(ValueError, match="rank state"):
        store.latest({"backend": "d2"})


def test_checkpoint_missing_payload_rejected(tmp_path):
    """A manifest alone does not make a checkpoint recoverable."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    path = commit(store, 1)
    (path / "dcp" / ".metadata").unlink()

    with pytest.raises(ValueError, match="payload"):
        store.latest({"backend": "d2"})


def test_checkpoint_pointer_cannot_escape_store(tmp_path):
    """Reject a malformed pointer before reading a different directory."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    (tmp_path / "latest.json").write_text(
        json.dumps({"generation": "../step-00000001"})
    )

    with pytest.raises(ValueError, match="generation"):
        store.latest({"backend": "d2"})


def test_checkpoint_low_space_keeps_existing_checkpoint(tmp_path, monkeypatch):
    """Do not start a large write that would consume the free-space floor."""
    from types import SimpleNamespace

    store = CheckpointStore(tmp_path, free_space_floor=0)
    previous = commit(store, 1)
    monkeypatch.setattr(
        "areal.elastic.state.checkpoint.shutil.disk_usage",
        lambda path: SimpleNamespace(free=0),
    )

    with pytest.raises(OSError, match="space"):
        store.commit(
            step=2,
            world_size=2,
            compatibility={},
            estimated_bytes=1,
            write=lambda path: pytest.fail("writer must not run"),
        )

    assert previous.is_dir()
    assert store.latest({"backend": "d2"})[0] == previous


def test_checkpoint_directory_published_without_pointer_can_be_replayed(
    tmp_path, monkeypatch
):
    """An orphan at the same step must not block recovery and a fresh commit."""
    store = CheckpointStore(tmp_path, keep=1, free_space_floor=0)
    previous = commit(store, 1)
    original = atomic_json

    def fail_pointer(path, value):
        if path.name == "latest.json":
            raise OSError("injected pointer failure")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr("areal.elastic.state.checkpoint.atomic_json", fail_pointer)
        with pytest.raises(OSError, match="pointer failure"):
            commit(store, 2)
    assert store.latest({"backend": "d2"})[0] == previous
    replacement = commit(store, 2)
    assert store.latest({"backend": "d2"})[0] == replacement
    assert list(tmp_path.glob("step-*")) == [replacement]


def test_checkpoint_queue_cursor_is_bound_to_optimizer_progress(tmp_path):
    """A queue manifest uses an explicit cursor rather than step as batch ID."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    cursor = {
        "format_version": 1,
        "queue_id": "queue",
        "applied_batches": 2,
        "through_sequence": 3,
        "chain_sha256": "a" * 64,
    }
    store.commit(
        step=2,
        world_size=2,
        compatibility={},
        estimated_bytes=0,
        data_cursor=cursor,
        write=lambda path: write_generation(path, step=2),
    )
    path, manifest = store.latest({})
    assert manifest["data_cursor"] == cursor
    assert "next_batch_id" not in manifest
    manifest["data_cursor"]["applied_batches"] = 3
    atomic_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="data cursor"):
        store.latest({})


def test_checkpoint_retention_keeps_previous_commit_instead_of_newer_orphan(
    tmp_path, monkeypatch
):
    """Only generations that reached latest publication count toward retention."""
    store = CheckpointStore(tmp_path, keep=2, free_space_floor=0)
    previous = commit(store, 1)
    original = atomic_json

    def fail_pointer(path, value):
        if path.name == "latest.json":
            raise OSError("pointer failure")
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr("areal.elastic.state.checkpoint.atomic_json", fail_pointer)
        with pytest.raises(OSError):
            commit(store, 2)
    current = commit(store, 2)
    assert set(tmp_path.glob("step-*")) == {previous, current}


def test_checkpoint_cannot_publish_older_step(tmp_path):
    """A stale coordinator must not replace latest with earlier training state."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    current = commit(store, 3)
    with pytest.raises(ValueError, match="backwards"):
        commit(store, 2)
    assert store.latest({"backend": "d2"})[0] == current


@pytest.fixture
def native_lifecycle(tmp_path, monkeypatch):
    """Protocol fixture: real inbox/checkpoint transactions, no GPU evidence."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from areal.api import FinetuneSpec, WeightUpdateMeta
    from areal.api.cli_args import ActorLifecycleConfig, PPOConfig
    from areal.trainer import on_demand

    config = PPOConfig()
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"protocol fixture")
    config.actor.path = str(base)
    config.actor.backend = "fsdp:d1"
    config.train_dataset.batch_size = config.rollout.consumer_batch_size = 4
    config.rollout.max_head_offpolicyness = 2
    config.actor_lifecycle = ActorLifecycleConfig(
        mode="on_demand",
        state_root=str(tmp_path / "state"),
        actor_devices=(0,),
        rollout_devices=(1,),
        checkpoint_estimated_bytes=1,
        checkpoint_free_space_floor=0,
    )
    monkeypatch.setattr(
        on_demand,
        "resolve_gpu_uuids",
        lambda devices: [f"GPU-{device}" for device in devices],
    )
    monkeypatch.setattr(on_demand, "gpu_processes", lambda: {})
    monkeypatch.setattr(on_demand, "owned_processes", lambda owner: [])
    stopped = []
    monkeypatch.setattr(on_demand, "stop_owned_processes", stopped.append)
    rollout = SimpleNamespace(version=0, _ensure_initialized=lambda: None)
    rollout.serving_versions = lambda: [rollout.version]
    actors = []

    class Actor:
        def __init__(self, cfg):
            self.config = cfg
            self.version = self.updates = 0
            self.create_process_group = MagicMock()
            self.initialize = MagicMock()
            self.config_perf_tracer = MagicMock()
            self.save_perf_tracer = MagicMock()
            self.quiesce_initialization = MagicMock()
            self.connect_engine = MagicMock()
            self.destroy = MagicMock()
            self.fail_publication = False
            actors.append(self)

        def set_version(self, version):
            self.version = version

        def full_checkpoint(
            self, operation, *, path, backend, identity_sha256, version
        ):
            root = Path(path)
            if operation == "save":
                (root / "dcp").mkdir()
                (root / "dcp/.metadata").write_text("protocol")
                atomic_json(root / "dcp/state.json", {"updates": self.updates})
                atomic_json(
                    root / "runtime-0.json",
                    {
                        "rank": 0,
                        "world_size": 1,
                        "policy_version": version,
                    },
                )
            else:
                self.updates = json.loads((root / "dcp/state.json").read_text())[
                    "updates"
                ]
                self.version = version

        def update_weights(self, meta):
            rollout.version = meta.version
            if self.fail_publication:
                raise TimeoutError("lost publication response")

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=48, train_batch_size=4)
    coordinator = {"step": 0}

    def create(policy_runtime=None):
        manager = on_demand.OnDemandActor(
            config, ft_spec, Actor, policy_runtime=policy_runtime
        )
        try:
            manager.bind(
                rollout,
                WeightUpdateMeta.from_awex(),
                capture_cpu=lambda: dict(coordinator),
                restore_cpu=lambda state: coordinator.update(state),
            )
        except BaseException:
            manager.close()
            raise
        return manager

    manager = create()
    yield SimpleNamespace(
        manager=manager,
        create=create,
        actors=actors,
        rollout=rollout,
        coordinator=coordinator,
        stopped=stopped,
        config=config,
    )
    manager.close()


def _ready_native_episodes(manager, first, version):
    from tests.test_elastic_batch_queue import _receive_episode

    for number in range(first, first + 4):
        _receive_episode(manager.inbox, number, version=version)


def test_native_lifecycle_three_cycles_restore_cursor_and_actor_state(native_lifecycle):
    fixture = native_lifecycle
    manager = fixture.manager
    assert manager.actor is None and fixture.actors == []
    rollout = manager.rollout
    for version in range(3):
        _ready_native_episodes(manager, version * 4, version)
        trajectories = manager.next_batch()
        assert len(trajectories) == 4
        assert manager.actor.updates == version
        manager.actor.initialize.assert_called_once_with(
            role="actor",
            addr=None,
            ft_spec=manager.ft_spec,
            wait=True,
            worker_seed=fixture.config.seed,
        )
        manager.actor.updates += 1
        fixture.coordinator["step"] = version + 1
        manager.commit_and_publish(version + 1)
        assert manager.inbox.status()["active_credits"] == 0
        assert not manager.maybe_suspend(now=manager.ready_since + 59)
        assert manager.maybe_suspend(now=manager.ready_since + 121)
        assert manager.actor is None and manager.owner is None
        assert manager.rollout is rollout
    manager.close()
    recovered = fixture.create()
    try:
        assert recovered.actor is None and recovered.version == 3
        assert recovered.consumer.cursor["applied_batches"] == 3
        _ready_native_episodes(recovered, 12, 3)
        recovered.next_batch()
        assert recovered.actor.updates == 3
    finally:
        recovered.close()


def test_native_lost_publication_ack_never_replays_committed_update(native_lifecycle):
    manager = native_lifecycle.manager
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    old = manager.actor
    old.updates = 1
    old.fail_publication = True
    with pytest.raises(TimeoutError):
        manager.commit_and_publish(1)
    assert manager.latest_manifest["global_step"] == 1
    assert manager.inbox.status()["published_version"] == 0
    manager.recover_uncertain(TimeoutError("lost response"))
    assert manager.version == manager.actor.updates == 1
    assert manager.actor is not old
    assert manager.active is None and manager.consumer.claim() is None
    assert manager.inbox.status()["active_credits"] == 0


def test_native_uncertain_uncommitted_update_replays_original_batch(native_lifecycle):
    manager = native_lifecycle.manager
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    original_id = manager.active.batch_id
    manager.actor.updates = 1
    manager.recover_uncertain(RuntimeError("worker died during update"))
    assert manager.version == 0 and manager.actor is None
    manager.next_batch()
    assert manager.active.batch_id == original_id
    assert manager.actor.updates == 0


def test_native_trainer_defers_actor_initialization_and_legacy_recover(
    native_lifecycle, monkeypatch
):
    from unittest.mock import MagicMock

    from areal.trainer import on_demand, rl_trainer

    fixture = native_lifecycle
    config = fixture.config
    config.actor._version = config.rollout._version = "v2"
    config.rollout.backend = "sglang:d1"
    config.sglang.model_path = config.actor.path
    config.actor.scheduling_spec[0].gpu = config.rollout.scheduling_spec[0].gpu = 1
    config.total_train_epochs, config.total_train_steps = 1, 12
    config.actor.adv_norm = config.actor.reward_norm = None
    config.gconfig.n_samples = 1
    config.gconfig.reward_normalization = False
    config.rollout.agent.mode = "online"
    config.scheduler.type = "local"
    monkeypatch.setenv("AREAL_SPMD_MODE", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(rl_trainer, "LocalScheduler", MagicMock())
    monkeypatch.setattr(
        rl_trainer, "load_hf_processor_and_tokenizer", lambda path: (None, object())
    )
    monkeypatch.setattr(rl_trainer.logging, "setup_file_logging", MagicMock())
    monkeypatch.setattr(
        on_demand, "OnDemandActor", MagicMock(return_value=fixture.manager)
    )
    for name in ("Saver", "Evaluator", "StatsLogger"):
        instance = MagicMock()
        instance.state_dict.return_value = {}
        monkeypatch.setattr(rl_trainer, name, MagicMock(return_value=instance))
    legacy_recover = MagicMock()
    monkeypatch.setattr(rl_trainer, "RecoverHandler", legacy_recover)
    serving = MagicMock()
    serving.serving_versions.return_value = [0]
    monkeypatch.setattr(
        rl_trainer.PPOTrainer, "_init_rollout", MagicMock(return_value=serving)
    )
    factory = MagicMock()
    monkeypatch.setattr(rl_trainer.PPOTrainer, "_create_train_engine", factory)
    trainer = rl_trainer.PPOTrainer(config)
    try:
        assert trainer.actor is None and fixture.actors == []
        factory.assert_not_called()
        legacy_recover.return_value.load.assert_not_called()
        assert trainer.scheduler is not None and trainer.rollout_scheduler is not None
        assert trainer._on_demand.inbox.status()["active_credits"] == 0
    finally:
        trainer.close()


def test_native_failed_teardown_preserves_fence_and_retries_same_generation(
    native_lifecycle, monkeypatch
):
    from areal.trainer import on_demand

    manager = native_lifecycle.manager
    manager.ensure_actor()
    actor, owner = manager.actor, manager.owner
    actor.destroy.side_effect = RuntimeError("inference teardown unknown")
    actor.save_perf_tracer.side_effect = RuntimeError("worker unreachable")
    monkeypatch.setattr(
        manager,
        "_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(RuntimeError, match="teardown unknown"):
        manager._release()
    assert manager.actor is actor and manager.owner == owner
    assert native_lifecycle.stopped == [owner]
    assert actor.destroy.call_args.kwargs == {"training_stopped": True}
    with pytest.raises(BlockingIOError):
        manager.inbox.queue.consumer()
    actor.destroy.side_effect = None
    monkeypatch.setattr(on_demand, "gpu_processes", lambda: {"GPU-0": [123]})
    with pytest.raises(RuntimeError, match="empty"):
        manager._release()
    assert manager.owner == owner
    monkeypatch.setattr(on_demand, "gpu_processes", lambda: {})
    manager._release()
    assert manager.actor is None and manager.owner is None
