# SPDX-License-Identifier: Apache-2.0
"""Explicit trainer startup with CPU participants; no distributed backend emulation."""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import pickle
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
    trainer, events = _trainer_fixture(monkeypatch, awex=awex, critic=critic)
    trainer.config.total_train_epochs = 2
    trainer.config.total_train_steps = 3
    trainer._snapshot_restore = None
    trainer.snapshot_restore_result = None
    trainer.weight_update_meta.type = "awex" if awex else "disk"
    monkeypatch.setattr(
        M, "capture_rng_state", lambda **_: {"python": random.getstate()}
    )

    def host_rng(state, **_):
        events.append("host.rng")
        random.setstate(state["python"])

    monkeypatch.setattr(M, "restore_rng_state", host_rng)
    module = types.ModuleType("areal.infra.controller.rollout_controller")
    module._same_input_spec = _source_function(
        "areal/infra/controller/rollout_controller.py",
        None,
        "_same_input_spec",
        {"torch": types.SimpleNamespace(Tensor=type("Tensor", (), {}))},
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    rollout = trainer.rollout
    rollout._resolve_workflow_str = lambda workflow: workflow
    rollout._resolve_should_accept_fn = lambda value: value
    rollout.input_spec = None
    rollout.inventory = {"pending": [], "ready": [], "accepted": 0}
    rollout.paused = False
    rollout.weights = [0]
    rollout.kv_offloaded = rollout.weights_offloaded = False
    trainer.samples = []

    def pause():
        rollout.paused = True
        events.append("rollout.pause")

    def resume():
        rollout.paused = False
        events.append("rollout.resume")

    rollout.pause = pause
    rollout.resume = resume
    rollout.pause_generation_sync = lambda: events.append("generation.pause")

    def offload(tags):
        events.append(f"offload.{tags[0]}")
        setattr(rollout, f"{tags[0].replace('kv_cache', 'kv')}_offloaded", True)

    rollout.offload = offload

    def rollout_state():
        assert rollout.frozen
        return copy.deepcopy(
            dict(
                policy_version=rollout.version,
                input_spec=rollout.input_spec,
                dataloader=trainer.train_dataloader.state_dict(),
                inventory=rollout.inventory,
            )
        )

    def rollout_load(state):
        assert rollout.frozen
        assert state["policy_version"] == rollout.version
        events.append("rollout.load")
        rollout.input_spec = copy.deepcopy(state["input_spec"])
        rollout.inventory = copy.deepcopy(state["inventory"])
        trainer.train_dataloader.load_state_dict(state["dataloader"])

    rollout.state_dict = rollout_state
    rollout.load_state_dict = rollout_load

    def batch(loader, **kwargs):
        assert not rollout.frozen and not rollout.paused
        if trainer.actor.loads:
            assert rollout.weights == trainer.actor.value["weights"]
        events.append("batch")
        trainer.samples.append((loader.value["position"], random.random()))
        loader.value["position"] += 2
        rollout.input_spec = {
            key: value for key, value in kwargs.items() if key != "dynamic_bs"
        }
        if rollout.input_spec["workflow_kwargs"] is None:
            rollout.input_spec["workflow_kwargs"] = {}
        return [{}]

    trainer.actor.prepare_batch = batch
    for engine in (trainer.actor, trainer.critic):
        if engine is None:
            continue
        engine.schedule_step = 0
        engine.rng = random.Random(10)
        original_update = engine.ppo_update

        def update(batch, engine=engine, original=original_update):
            engine.rng.random()
            original(batch)

        engine.ppo_update = update

        def step_lr(engine=engine):
            engine.schedule_step += 1
            events.append(f"{engine.name}.lr")

        engine.step_lr_scheduler = step_lr

        def runtime_state(engine=engine):
            return pickle.dumps((engine.schedule_step, engine.rng.getstate())).hex()

        engine.runtime_state_dict = runtime_state

        def runtime_load(state, engine=engine):
            engine.schedule_step = pickle.loads(bytes.fromhex(state))[0]
            events.append(f"{engine.name}.schedule")

        engine.load_runtime_state_dict = runtime_load

        def rng_load(state, engine=engine):
            engine.rng.setstate(pickle.loads(bytes.fromhex(state))[1])
            events.append(f"{engine.name}.rng")

        engine.restore_runtime_rng = rng_load
        original_load = engine.load

        def load(meta, engine=engine, original=original_load):
            assert rollout.frozen
            if awex:
                assert rollout.kv_offloaded and rollout.weights_offloaded
            events.append(f"{engine.name}.load")
            original(meta)

        engine.load = load

    def sync(meta):
        assert trainer.actor.version == rollout.version
        rollout.weights = trainer.actor.value["weights"].copy()
        events.append("weight_sync")

    trainer.actor.update_weights = sync
    trainer._should_offload_rollout = colocated
    trainer._should_offload_actor = colocated
    trainer._offload_model = lambda engine, role: events.append(f"{role}.offload")
    trainer._onload_model = lambda engine, role: events.append(f"{role}.onload")
    trainer._offload_rollout = lambda **_: events.append("rollout.offload")

    def onload(is_eval=False):
        events.append("rollout.onload")
        # Deliberately consume RNG during initialization to check finalization.
        random.random()
        if not is_eval:
            rollout.resume()

    trainer._onload_rollout = onload
    if evaluation:
        ev = types.SimpleNamespace(version=0)
        ev.set_version = lambda version: setattr(ev, "version", version)
        ev.get_version = lambda: ev.version
        ev.pause = lambda: events.append("eval.pause")
        ev.resume = lambda: events.append("eval.resume")
        trainer.eval_rollout = ev
    return trainer, events


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
