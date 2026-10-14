# SPDX-License-Identifier: Apache-2.0
"""Explicit trainer startup with CPU participants; no distributed backend emulation."""

from __future__ import annotations

import ast
import dataclasses
import json
import random
import subprocess
import sys
import textwrap
import types
from unittest.mock import Mock

import pytest

from tests.test_reconfiguration import (
    REPO_ROOT,
    M,
    _source_function,
    _trainer_fixture,
)

WORKFLOW = "test.workflow"


def _control():
    return M.TrainerReconfiguration(
        run_id="restore-run",
        runtime_revision="test-runtime",
        configuration_id="same-config",
    )


def _fixture(
    monkeypatch, *, awex=False, critic=False, colocated=False, evaluation=False
):
    return _trainer_fixture(
        monkeypatch,
        awex=awex,
        critic=critic,
        colocated=colocated,
        evaluation=evaluation,
        total_steps=3,
    )


def _save(monkeypatch, tmp_path, **options):
    source, events = _fixture(monkeypatch, **options)
    channel = _control()
    channel.request_snapshot(
        tmp_path / "source", state_id="step-0", stop_after_save=True
    )
    stopped = source.train(workflow=WORKFLOW, reconfiguration=channel)
    return source, stopped


def _restore(trainer, stopped):
    trainer._snapshot_restore = M.TrainerSnapshotRestore(
        M.SnapshotRestoreRequest(stopped.snapshot, stopped.context)
    )


@pytest.mark.parametrize(
    "awex,critic,colocated,evaluation",
    [
        (False, False, False, False),
        (False, True, False, True),
        (False, True, True, True),
        (True, True, False, True),
    ],
)
def test_restore_syncs_roles_then_owners_then_rng_before_next_batch(
    monkeypatch, tmp_path, awex, critic, colocated, evaluation
):
    options = dict(awex=awex, critic=critic, colocated=colocated, evaluation=evaluation)
    _, stopped = _save(monkeypatch, tmp_path, **options)
    target, events = _fixture(monkeypatch, **options)
    _restore(target, stopped)
    target._requires_proxy_workflow = lambda _: True
    target._ensure_proxy_started = lambda: (events.append("proxy"), random.random())
    channel = _control()
    channel.request_snapshot(tmp_path / "next", state_id="step-1", stop_after_save=True)
    result = target.train(workflow=WORKFLOW, reconfiguration=channel)
    assert result.context.step_info == dict(
        epoch=0, epoch_step=1, global_step=1, steps_per_epoch=2
    )
    assert result.context.policy_version == 2
    assert target.actor.value == {"weights": [2], "optimizer": {"moment": [2]}}
    assert target.actor.schedule_step == 2
    assert target.train_dataloader.value["position"] == 4
    assert target.stats_logger.value["last_commit_step"] == 1
    assert target.saver.value["count"] == target.evaluator.value["count"] == 2
    assert target.snapshot_restore_result.context == stopped.context
    before_batch = events[: events.index("batch")]
    assert before_batch.count("weight_sync") == 1
    assert before_batch.index("proxy") < before_batch.index("actor.load")
    assert before_batch.index("actor.load") < before_batch.index("weight_sync")
    assert before_batch.index("weight_sync") < before_batch.index("actor.schedule")
    assert before_batch.index("rollout.load") < before_batch.index("actor.rng")
    assert before_batch.index("actor.rng") < before_batch.index("host.rng")
    assert before_batch.index("host.rng") < before_batch.index("rollout.release")
    if critic:
        assert before_batch.index("critic.load") < before_batch.index("weight_sync")
        assert target.critic.schedule_step == 2
    if awex:
        assert before_batch.index("offload.weights") < before_batch.index("actor.load")
    if colocated:
        assert before_batch.index("rollout.onload") < before_batch.index("host.rng")
        assert before_batch.count("rollout.onload") == 1
    if evaluation:
        assert target.eval_rollout.version == 2


def test_restore_matches_uninterrupted_cpu_state_and_input_rng(monkeypatch, tmp_path):
    random.seed(71)
    whole, _ = _fixture(monkeypatch, critic=True)
    whole.train(workflow=WORKFLOW)
    random.seed(71)
    source, stopped = _save(monkeypatch, tmp_path, critic=True)
    target, _ = _fixture(monkeypatch, critic=True)
    random.seed(999)
    _restore(target, stopped)
    target.train(workflow=WORKFLOW)
    assert source.samples + target.samples == whole.samples
    for role in ("actor", "critic"):
        a, b = getattr(whole, role), getattr(target, role)
        assert a.value == b.value
        assert a.runtime_state_dict() == b.runtime_state_dict()
        assert a.version == b.version == 3
    assert target.train_dataloader.value == whole.train_dataloader.value
    assert target.stats_logger.value == whole.stats_logger.value
    assert target.saver.value == whole.saver.value


def test_policy_version_is_independent_of_completed_step(monkeypatch, tmp_path):
    source, _ = _fixture(monkeypatch)
    source.actor.set_version = lambda version: setattr(
        source.actor, "version", version + 90
    )
    source.rollout.set_version = lambda version: setattr(
        source.rollout, "version", version + 90
    )
    channel = _control()
    channel.request_snapshot(
        tmp_path / "source", state_id="independent", stop_after_save=True
    )
    stopped = source.train(workflow=WORKFLOW, reconfiguration=channel)
    assert stopped.context.policy_version == 91
    target, _ = _fixture(monkeypatch)
    _restore(target, stopped)
    target.train(workflow=WORKFLOW)
    assert target.actor.version == target.rollout.version == 93
    assert target.stats_logger.value["last_commit_step"] == 2


@pytest.mark.parametrize("limit", ["steps", "epochs"])
def test_completed_training_restores_but_never_releases_inventory(
    monkeypatch, tmp_path, limit
):
    _, stopped = _save(monkeypatch, tmp_path)
    target, events = _fixture(monkeypatch, colocated=True)
    if limit == "steps":
        target.config.total_train_steps = 1
    else:
        # Save at the last step of the first epoch using another complete boundary.
        source, _ = _fixture(monkeypatch)
        _restore(source, stopped)
        channel = _control()
        channel.request_snapshot(tmp_path / "end", state_id="end", stop_after_save=True)
        stopped = source.train(workflow=WORKFLOW, reconfiguration=channel)
        target.config.total_train_epochs = 1
    _restore(target, stopped)
    target.train(workflow=WORKFLOW)
    assert target.snapshot_restore_result is not None
    assert target.rollout.frozen
    assert "batch" not in events and "rollout.release" not in events
    assert "rollout.resume" not in events and "rollout.onload" not in events


@pytest.mark.parametrize(
    "stage", ["load", "sync", "version", "owner", "rng", "input", "resume"]
)
def test_restore_failure_keeps_fence_and_never_retries_or_starts_fresh(
    monkeypatch, tmp_path, stage
):
    _, stopped = _save(monkeypatch, tmp_path)
    target, events = _fixture(monkeypatch)
    _restore(target, stopped)

    def fail(*_, **__):
        raise RuntimeError("injected restore failure")

    workflow = WORKFLOW
    if stage == "load":
        target.actor.load = fail
    elif stage == "sync":
        target.actor.update_weights = fail
    elif stage == "version":
        target.actor.set_version = lambda _: None
    elif stage == "owner":
        target.saver.load_state_dict = fail
    elif stage == "rng":
        target.actor.restore_runtime_rng = fail
    elif stage == "input":
        workflow = "other.workflow"
    elif stage == "resume":
        target.rollout.resume = fail
    channel = _control()
    future = channel.request_snapshot(tmp_path / "unused", state_id="unused")
    with pytest.raises((M.SnapshotError, RuntimeError)) as error:
        target.train(workflow=workflow, reconfiguration=channel)
    assert future.exception() is error.value
    assert target.rollout.frozen
    assert "batch" not in events and "rollout.release" not in events
    assert "rollout.resume" not in events
    with pytest.raises(RuntimeError, match="already attempted"):
        target.train(workflow=WORKFLOW)
    assert "batch" not in events


def test_wrong_epoch_size_fails_before_model_loading(monkeypatch, tmp_path):
    _, stopped = _save(monkeypatch, tmp_path)
    target, events = _fixture(monkeypatch)
    _restore(target, stopped)
    target.train_dataloader = [1, 2, 3]
    with pytest.raises(M.SnapshotError, match="steps_per_epoch"):
        target.train(workflow=WORKFLOW)
    assert "actor.load" not in events and "batch" not in events


def _constructor(single=True):
    init = _source_function(
        "areal/trainer/rl_trainer.py",
        "PPOTrainer",
        "__init__",
        {
            "is_single_controller": lambda: single,
            "TrainerSnapshotRestore": M.TrainerSnapshotRestore,
            "logger": Mock(),
        },
    )
    return type(
        "Constructor", (), {"__init__": init, "_init_impl": Mock(), "close": Mock()}
    )


@pytest.mark.parametrize("invalid", ["digest", "context", "type", "version", "spmd"])
def test_constructor_preflight_rejects_before_creating_workers(
    monkeypatch, tmp_path, invalid
):
    _, stopped = _save(monkeypatch, tmp_path)
    config = types.SimpleNamespace(actor=types.SimpleNamespace(_version="v1"))
    cls = _constructor(single=invalid != "spmd")
    request = M.SnapshotRestoreRequest(stopped.snapshot, stopped.context)
    if invalid == "digest":
        request = dataclasses.replace(
            request,
            snapshot=dataclasses.replace(stopped.snapshot, manifest_sha256="0" * 64),
        )
    elif invalid == "context":
        request = dataclasses.replace(
            request,
            expected_context=dataclasses.replace(stopped.context, state_id="wrong"),
        )
    elif invalid == "type":
        request = stopped.snapshot
    elif invalid == "version":
        config.actor._version = "v2"
    with pytest.raises((M.SnapshotError, TypeError, NotImplementedError)):
        cls(config, restore=request)
    cls._init_impl.assert_not_called()
    cls.close.assert_not_called()


def test_explicit_source_skips_legacy_checkpoint_and_staleness_recovery():
    """Execute the production startup block without importing GPU dependencies."""
    path = REPO_ROOT / "areal/trainer/rl_trainer.py"
    tree = ast.parse(path.read_text())
    trainer = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PPOTrainer"
    )
    method = next(
        n
        for n in trainer.body
        if isinstance(n, ast.FunctionDef) and n.name == "_init_impl"
    )
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "recover_info" for t in n.targets
        )
    )
    end = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "_config_perf_tracer"
    )
    block = compile(
        ast.Module(body=method.body[start:end], type_ignores=[]), str(path), "exec"
    )
    for explicit in (False, True):
        fake = Mock()
        fake._snapshot_restore = object() if explicit else None
        fake.recover_handler.load.return_value.last_step_info.global_step = 11
        exec(
            block,
            {"self": fake, "config": object(), "is_single_controller": lambda: True},
        )
        if explicit:
            fake.recover_handler.load.assert_not_called()
            fake.rollout.staleness_manager.on_version_recovered.assert_not_called()
            assert fake.recover_info is None
        else:
            fake.recover_handler.load.assert_called_once()
            fake.rollout.staleness_manager.on_version_recovered.assert_called_once_with(
                12
            )


def test_save_then_restore_in_distinct_cpu_processes(tmp_path):
    """Exercise real trainer hooks across process lifetimes, with CPU engines."""
    script = textwrap.dedent("""
        import dataclasses, json, random, sys
        from pathlib import Path
        from pytest import MonkeyPatch
        from tests.test_reconfiguration_restore import M, WORKFLOW, _fixture, _control
        phase, directory = sys.argv[1:]
        root = Path(directory)
        with MonkeyPatch.context() as monkeypatch:
            trainer, events = _fixture(monkeypatch, critic=True)
            random.seed(123 if phase == "save" else 999)
            if phase == "save":
                channel = _control()
                channel.request_snapshot(root / "snapshot", state_id="boundary", stop_after_save=True)
                result = trainer.train(workflow=WORKFLOW, reconfiguration=channel)
                (root / "ref.json").write_text(json.dumps(dataclasses.asdict(result)))
            else:
                saved = json.loads((root / "ref.json").read_text())
                trainer._snapshot_restore = M.TrainerSnapshotRestore(M.SnapshotRestoreRequest(
                    M.SnapshotRef(**saved["snapshot"]), M.SnapshotContext(**saved["context"])))
                trainer.train(workflow=WORKFLOW)
                (root / "result.json").write_text(json.dumps(dict(
                    weights=trainer.actor.value["weights"], schedule=trainer.actor.schedule_step,
                    position=trainer.train_dataloader.value["position"], samples=trainer.samples,
                    last=trainer.stats_logger.value["last_commit_step"], version=trainer.rollout.version)))
    """)
    for phase in ("save", "restore"):
        result = subprocess.run(
            [sys.executable, "-c", script, phase, str(tmp_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
    restored = json.loads((tmp_path / "result.json").read_text())
    assert restored["weights"] == [3]
    assert restored["schedule"] == restored["version"] == 3
    assert restored["position"] == 6 and restored["last"] == 2
    assert [item[0] for item in restored["samples"]] == [2, 4]


@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
def test_native_disk_marker_is_not_reused_by_first_resumed_update(
    monkeypatch, tmp_path, backend
):
    """Execute native disk orchestration with in-memory I/O participants.

    No FSDP/DTensor or Megatron model internals are substituted here: the model
    writer and receiver represent the two sides of the rendezvous protocol.
    """
    from concurrent.futures import Future
    from datetime import datetime

    _, stopped = _save(monkeypatch, tmp_path)
    target, _ = _fixture(monkeypatch)
    _restore(target, stopped)
    markers = {}
    written = set()
    transfers = []
    key = _source_function(
        "areal/utils/names.py",
        None,
        "update_weights_from_disk",
        {"USER_NAMESPACE": "test"},
    )

    def add(name, value, **_):
        if name in markers:
            raise RuntimeError("Duplicate disk completion marker")
        markers[name] = value

    def receive(meta):
        live_version = target.rollout.version
        name = key("experiment", "trial", live_version)
        # Real receiver waits before requesting model files. A leftover marker
        # would satisfy that wait before the new writer has finished.
        assert name not in markers, "receiver consumed a stale completion marker"
        transfers.append((live_version, meta.version))

        def completed():
            assert name in markers and meta.path in written

        return types.SimpleNamespace(result=completed)

    owner = types.SimpleNamespace(
        config=types.SimpleNamespace(experiment_name="experiment", trial_name="trial"),
        get_version=target.actor.get_version,
        tokenizer=None,
        processor=None,
        cpu_group=None,
        _save_model_to_hf=lambda path, *_: written.add(path),
        rollout_engine=types.SimpleNamespace(
            pause_generation=lambda: None,
            continue_generation=lambda: None,
            update_weights_from_disk=receive,
        ),
    )
    native = _source_function(
        f"areal/engine/{backend}_engine.py",
        "FSDPEngine" if backend == "fsdp" else "MegatronEngine",
        "_update_weights_from_disk",
        dict(
            Future=Future,
            datetime=datetime,
            trace_perf=lambda *_, **__: lambda fn: fn,
            names=types.SimpleNamespace(update_weights_from_disk=key),
            name_resolve=types.SimpleNamespace(add=add, delete=markers.pop),
            dist=types.SimpleNamespace(get_rank=lambda: 0, barrier=lambda **_: None),
            current_platform=types.SimpleNamespace(synchronize=lambda: None),
        ),
    )
    sync = target.actor.update_weights

    def transfer(meta):
        native(owner, meta)
        sync(meta)

    target.actor.update_weights = transfer
    target.weight_update_meta.with_version = lambda v: types.SimpleNamespace(
        version=v, path=f"cpu-model-v{v}"
    )
    target.train(workflow=WORKFLOW)
    assert -(2**63) <= transfers[0][0] < 0
    assert transfers[0][1] == stopped.context.policy_version == 1
    assert transfers[1:] == [(1, 2), (2, 3)]
    assert target.actor.version == target.rollout.version == 3
