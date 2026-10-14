# SPDX-License-Identifier: Apache-2.0
"""CPU tests for snapshot publication, strict restore and completion RPCs.

The snapshot module is loaded without importing the optional training stack.
Small engine/controller methods are compiled unchanged from their source files
and exercised with fake checkpoint owners and RPC transports. These tests cover
orchestration and failure handling, not real GPU I/O or distributed collectives.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import dataclasses
import hashlib
import json
import math
import pickle
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
source = REPO_ROOT / "areal/infra/reconfiguration.py"
M = types.ModuleType("snapshot_core_under_test")
with patch.dict(sys.modules, {M.__name__: M}):
    exec(compile(source.read_text(), str(source), "exec"), M.__dict__)


class Owner:
    def __init__(self, value):
        self.value = value
        self.loads = 0
        self.fail_load = False

    def state_dict(self):
        return copy.deepcopy(self.value)

    def load_state_dict(self, state):
        self.loads += 1
        self.value = state
        if self.fail_load:
            raise RuntimeError("state restore failed")


class Role:
    def __init__(self, value):
        self.value = value
        self.pending = None
        self.loads = 0
        self.waits = 0
        self.fail_save = False
        self.fail_wait = False
        self.fail_load = False
        self.empty = False

    def save(self, path):
        self.pending = (path, json.dumps(self.value).encode())
        if self.fail_save:
            raise RuntimeError("save failed after dispatch")

    def wait(self):
        self.waits += 1
        path, encoded = self.pending
        if not self.empty:
            (path / "checkpoint.bin").write_bytes(encoded)
        if self.fail_wait:
            raise RuntimeError("async finalization failed")

    def load(self, path):
        self.loads += 1
        self.value = json.loads((path / "checkpoint.bin").read_text())
        if self.fail_load:
            raise RuntimeError("model restore failed")

    def binding(self):
        return M.CheckpointParticipant("fixture-dcp", self.save, self.wait, self.load)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="areal-snapshot-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.actor = Role({"weights": [1, 2], "optimizer": {"moments": [3, 4]}})
        self.critic = Role({"weights": [5], "optimizer": {"moments": [6]}})
        self.loader = Owner({"position": 123})
        self.rng = Owner({"seed_state": (1, 3, 5)})
        self.ctx = M.SnapshotContext(
            run_id="trial-a",
            runtime_revision="fixed-test-revision",
            configuration_id="config-a",
            state_id="boundary-a",
            step_info={
                "epoch": 0,
                "epoch_step": 3,
                "global_step": 3,
                "steps_per_epoch": 10,
            },
            policy_version=91,
        )
        self.manager = self.make_manager()

    def make_manager(self, roles=None, states=None):
        return M.ReconfigurationManager(
            checkpoints=roles
            if roles is not None
            else {"actor": self.actor.binding(), "critic": self.critic.binding()},
            statefuls=states
            if states is not None
            else {"dataloader": self.loader, "rng": self.rng},
        )

    def save(self, name="snapshot"):
        return self.manager.save_snapshot(self.root / name, context=self.ctx)

    def test_roundtrip_new_objects_and_serialized_reference(self):
        reference = self.save()
        reference = M.SnapshotRef(
            **json.loads(json.dumps(dataclasses.asdict(reference)))
        )
        actor, critic = Role({}), Role({})
        loader, rng = Owner({}), Owner({})
        reader = self.make_manager(
            {"actor": actor.binding(), "critic": critic.binding()},
            {"dataloader": loader, "rng": rng},
        )
        restored = reader.restore_snapshot(reference, expected_context=self.ctx)
        self.assertEqual(actor.value, self.actor.value)
        self.assertEqual(critic.value, self.critic.value)
        self.assertEqual(loader.value, self.loader.value)
        self.assertEqual(rng.value, self.rng.value)
        self.assertEqual(restored.context.policy_version, 91)
        self.assertEqual(restored.context.step_info["global_step"], 3)
        self.assertEqual(set(restored.checkpoint_roles), {"actor", "critic"})
        self.assertEqual(
            M.inspect_snapshot(reference)["scope"], "registered_components"
        )

    def test_finalization_precedes_publication_for_every_role(self):
        binding = self.manager._checkpoints["actor"]

        def finish():
            self.assertFalse((self.root / "snapshot" / "manifest.json").exists())
            self.assertFalse(
                (
                    self.root / "snapshot" / "checkpoints" / "actor" / "checkpoint.bin"
                ).exists()
            )
            self.actor.wait()

        self.manager = self.make_manager(
            {
                "actor": dataclasses.replace(binding, wait_saved=finish),
                "critic": self.critic.binding(),
            }
        )
        ref = self.save()
        self.assertEqual((self.actor.waits, self.critic.waits), (1, 1))
        M.inspect_snapshot(ref)

    def test_existing_directory_never_overwritten(self):
        ref = self.save()
        before = (self.root / "snapshot" / "manifest.json").read_bytes()
        with self.assertRaises(M.SnapshotError):
            self.save()
        self.assertEqual(
            (self.root / "snapshot" / "manifest.json").read_bytes(), before
        )
        M.inspect_snapshot(ref)

    def test_save_failure_waits_and_does_not_publish(self):
        self.actor.fail_save = True
        with self.assertRaises(M.SnapshotError):
            self.save()
        self.assertEqual(self.actor.waits, 1)
        self.assertFalse((self.root / "snapshot" / "manifest.json").exists())
        self.actor.fail_save = False
        M.inspect_snapshot(self.save("retry"))

    def test_finalization_failure_never_publishes(self):
        self.critic.fail_wait = True
        with self.assertRaises(M.SnapshotError):
            self.save()
        self.assertFalse((self.root / "snapshot" / "manifest.json").exists())

    def test_empty_role_checkpoint_is_not_success(self):
        self.critic.empty = True
        with self.assertRaises(M.SnapshotError):
            self.save()
        self.assertFalse((self.root / "snapshot" / "manifest.json").exists())

    def test_corruption_and_missing_files_preflight_all_roles(self):
        for kind in ("modify", "remove", "extra", "state"):
            with self.subTest(kind=kind):
                ref = self.save(kind)
                p = Path(ref.path) / "checkpoints" / "critic" / "checkpoint.bin"
                if kind == "modify":
                    p.write_bytes(b"corrupt")
                elif kind == "remove":
                    p.unlink()
                elif kind == "extra":
                    (Path(ref.path) / "extra.bin").write_bytes(b"extra")
                else:
                    (Path(ref.path) / "state.pkl").write_bytes(b"corrupt")
                with self.assertRaises(M.SnapshotError):
                    self.manager.restore_snapshot(ref, expected_context=self.ctx)
                self.assertEqual(
                    (self.actor.loads, self.critic.loads, self.loader.loads), (0, 0, 0)
                )

    def test_manifest_hash_and_identity_mismatch(self):
        ref = self.save()
        for bad in (
            dataclasses.replace(ref, manifest_sha256="0" * 64),
            dataclasses.replace(ref, snapshot_id="wrong"),
        ):
            with self.assertRaises(M.SnapshotError):
                self.manager.restore_snapshot(bad, expected_context=self.ctx)
        self.assertEqual(self.actor.loads, 0)

    def test_context_mismatch_does_not_load(self):
        ref = self.save()
        for field in ("run_id", "runtime_revision", "configuration_id", "state_id"):
            with self.subTest(field=field), self.assertRaises(M.SnapshotError):
                self.manager.restore_snapshot(
                    ref,
                    expected_context=dataclasses.replace(self.ctx, **{field: "other"}),
                )
        self.assertEqual(self.actor.loads, 0)

    def test_all_roles_and_state_owners_must_match(self):
        ref = self.save()
        readers = [
            self.make_manager({"actor": self.actor.binding()}),
            self.make_manager(states={"dataloader": self.loader}),
            self.make_manager(
                {
                    "actor": dataclasses.replace(
                        self.actor.binding(), format_id="other"
                    ),
                    "critic": self.critic.binding(),
                }
            ),
        ]
        for reader in readers:
            with self.assertRaises(M.SnapshotError):
                reader.restore_snapshot(ref, expected_context=self.ctx)
        self.assertEqual(self.actor.loads, 0)

    def test_missing_snapshot_does_not_start_fresh(self):
        ref = M.SnapshotRef(str(self.root / "missing"), "missing", "0" * 64)
        with self.assertRaises(M.SnapshotError):
            self.manager.restore_snapshot(ref, expected_context=self.ctx)
        self.assertEqual(self.actor.loads, 0)

    def test_partial_load_failure_poison_manager(self):
        ref = self.save()
        self.critic.fail_load = True
        with self.assertRaisesRegex(M.SnapshotError, "partially restored"):
            self.manager.restore_snapshot(ref, expected_context=self.ctx)
        self.assertEqual((self.actor.loads, self.critic.loads), (1, 1))
        self.assertEqual(self.loader.loads, 0)
        self.critic.fail_load = False
        with self.assertRaises(M.SnapshotError):
            self.manager.restore_snapshot(ref, expected_context=self.ctx)
        with self.assertRaises(M.SnapshotError):
            self.save("after-failure")
        self.assertFalse((self.root / "after-failure").exists())

    def test_state_load_failure_also_poisons_manager(self):
        ref = self.save()
        self.loader.fail_load = True
        with self.assertRaisesRegex(M.SnapshotError, "partially restored"):
            self.manager.restore_snapshot(ref, expected_context=self.ctx)
        with self.assertRaises(M.SnapshotError):
            self.save("after-failure")

    def test_invalid_capabilities_rejected_before_writes(self):
        with self.assertRaises(TypeError):
            self.make_manager(
                {"actor": dataclasses.replace(self.actor.binding(), wait_saved=None)}
            )
        with self.assertRaises(TypeError):
            self.make_manager(states={"dataloader": object()})

        async def async_wait():
            pass

        with self.assertRaises(TypeError):
            self.make_manager(
                {
                    "actor": dataclasses.replace(
                        self.actor.binding(), wait_saved=async_wait
                    )
                }
            )
        for name in ("../actor", ".", "/actor", "actor/a"):
            with self.assertRaises(ValueError):
                self.make_manager({name: self.actor.binding()})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_boundary_metadata_rejected_before_writes(self):
        for updates in (
            {"policy_version": True},
            {"run_id": ""},
            {"step_info": dict(self.ctx.step_info, epoch_step=10)},
            {"step_info": dict(self.ctx.step_info, global_step=-1)},
        ):
            with self.assertRaises(M.SnapshotError):
                self.manager.save_snapshot(
                    self.root / "snapshot",
                    context=dataclasses.replace(self.ctx, **updates),
                )
        self.assertFalse((self.root / "snapshot").exists())

    def test_symlink_checkpoint_rejected(self):
        binding = self.actor.binding()
        target = self.root / "outside"
        target.write_bytes(b"data")

        def wait():
            (
                self.root / "snapshot" / "checkpoints" / "actor" / "checkpoint.bin"
            ).symlink_to(target)

        self.manager = self.make_manager(
            {"actor": dataclasses.replace(binding, wait_saved=wait)}
        )
        with self.assertRaises(M.SnapshotError):
            self.save()
        self.assertFalse((self.root / "snapshot" / "manifest.json").exists())

    def test_interrupted_save_never_publishes_and_releases_lock(self):
        def interrupt():
            raise KeyboardInterrupt

        self.manager = self.make_manager(
            {"actor": dataclasses.replace(self.actor.binding(), wait_saved=interrupt)}
        )
        with self.assertRaises(KeyboardInterrupt):
            self.save()
        self.assertFalse((self.root / "snapshot" / "manifest.json").exists())
        self.assertTrue(self.manager._lock.acquire(blocking=False))
        self.manager._lock.release()

    def test_overlapping_operations_rejected(self):
        ref = self.save()
        self.manager._lock.acquire()
        try:
            with self.assertRaises(M.SnapshotError):
                self.save("concurrent")
            with self.assertRaises(M.SnapshotError):
                self.manager.restore_snapshot(ref, expected_context=self.ctx)
        finally:
            self.manager._lock.release()
        self.assertFalse((self.root / "concurrent").exists())

    def test_native_adapter_uses_dcp_with_optimizer(self):
        @dataclasses.dataclass
        class Meta:
            path: str
            weight_format: str
            with_optim: bool

        stub = types.ModuleType("areal.api.io_struct")
        stub.SaveLoadMeta = Meta
        calls = []
        engine = types.SimpleNamespace(
            save=lambda meta: calls.append(("save", meta)),
            load=lambda meta: calls.append(("load", meta)),
        )
        for format_id, expected_format in (
            ("native-dcp", "dcp"),
            ("areal-fsdp-dcp", "dcp"),
            ("areal-megatron-dcp", "dcp-reconfiguration"),
            ("areal-megatron-fully-reshardable-v1", "dcp-reconfiguration"),
        ):
            with self.subTest(format_id=format_id):
                calls.clear()
                callback = M.CheckpointParticipant.for_engine(
                    engine,
                    format_id=format_id,
                    wait_saved=lambda: calls.append(("wait", None)),
                )
                with patch.dict(sys.modules, {"areal.api.io_struct": stub}):
                    callback.save(self.root)
                    callback.wait_saved()
                    callback.load(self.root)
                self.assertEqual([kind for kind, _ in calls], ["save", "wait", "load"])
                self.assertTrue(
                    all(
                        meta.with_optim and meta.weight_format == expected_format
                        for _, meta in calls
                        if meta is not None
                    )
                )
                self.assertEqual(
                    callback.format_id,
                    "areal-megatron-fully-reshardable-v1"
                    if expected_format == "dcp-reconfiguration"
                    else format_id,
                )

    def test_invalid_path_does_not_leak_lock(self):
        with self.assertRaises(M.SnapshotError):
            self.manager.save_snapshot(None, context=self.ctx)
        M.inspect_snapshot(self.save())

    def test_unexpected_sidecar_cannot_be_published(self):
        def save(path):
            self.actor.save(path)
            (path.parent.parent / "sidecar.json").write_text("{}")

        self.manager = self.make_manager(
            {"actor": dataclasses.replace(self.actor.binding(), save=save)}
        )
        with self.assertRaises(M.SnapshotError):
            self.save()
        self.assertFalse((self.root / "snapshot" / "manifest.json").exists())

    def test_manifest_publication_failure_is_incomplete(self):
        with patch.object(M.os, "replace", side_effect=OSError("disk error")):
            with self.assertRaises(M.SnapshotError):
                self.save()
        self.assertFalse((self.root / "snapshot" / "manifest.json").exists())

    def test_post_publication_sync_failure_removes_marker(self):
        original = M._sync_directory

        def sync(path):
            if (self.root / "snapshot" / "manifest.json").exists():
                raise OSError("fsync failed")
            original(path)

        with patch.object(M, "_sync_directory", side_effect=sync):
            with self.assertRaises(M.SnapshotError):
                self.save()
        self.assertFalse((self.root / "snapshot" / "manifest.json").exists())

    def test_unserializable_state_never_starts_checkpoint(self):
        self.loader.value = lambda: None
        with self.assertRaises(M.SnapshotError):
            self.save()
        self.assertEqual(self.actor.waits, 0)
        self.assertFalse((self.root / "snapshot" / "manifest.json").exists())

    def test_payload_members_checked_even_with_valid_checksums(self):
        ref = self.save()
        root = Path(ref.path)
        (root / "state.pkl").write_bytes(pickle.dumps({"wrong": {}}))
        manifest = json.loads((root / "manifest.json").read_bytes())
        manifest["files"] = M._inventory(root)
        encoded = M._json_bytes(manifest)
        (root / "manifest.json").write_bytes(encoded)
        ref = dataclasses.replace(
            ref, manifest_sha256=hashlib.sha256(encoded).hexdigest()
        )
        with self.assertRaises(M.SnapshotError):
            self.manager.restore_snapshot(ref, expected_context=self.ctx)
        self.assertEqual((self.actor.loads, self.critic.loads), (0, 0))

    def test_no_load_side_effects_from_wrong_format_or_schema(self):
        ref = self.save()
        root = Path(ref.path)
        manifest = json.loads((root / "manifest.json").read_bytes())
        manifest["schema_version"] = 999
        encoded = M._json_bytes(manifest)
        (root / "manifest.json").write_bytes(encoded)
        ref = dataclasses.replace(
            ref, manifest_sha256=hashlib.sha256(encoded).hexdigest()
        )
        with self.assertRaises(M.SnapshotError):
            self.manager.restore_snapshot(ref, expected_context=self.ctx)
        self.assertEqual(self.actor.loads, 0)


def _source_function(relative_path, class_name, name, namespace=None):
    """Load a real function body without importing optional GPU dependencies."""
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    body = tree.body
    if class_name is not None:
        body = next(
            node
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    function = next(
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    scope = {} if namespace is None else dict(namespace)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope
    )
    return scope[name]


@pytest.fixture
def portable_optimizer(monkeypatch):
    """Exercise common-state mapping without emulating tensor resharding."""
    path = "areal/engine/megatron_utils/checkpointer.py"
    scope = {
        "copy": copy,
        "torch": types.SimpleNamespace(Tensor=type("TensorPlaceholder", (), {})),
    }
    tree = ast.parse((REPO_ROOT / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_GROUP_KEYS"
            for target in node.targets
        ):
            scope["_GROUP_KEYS"] = ast.literal_eval(node.value)
    for name in (
        "_common_equal",
        "_optimizer_leaves",
        "_merge_optimizer_common",
        "_portable_optimizer_template",
    ):
        scope[name] = _source_function(path, None, name, scope)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.dist_checkpointing.mapping",
        types.SimpleNamespace(
            LocalNonpersistentObject=lambda value: types.SimpleNamespace(data=value)
        ),
    )
    return types.SimpleNamespace(
        merge=scope["_merge_optimizer_common"],
        map=scope["_portable_optimizer_template"],
    )


def _optimizer_common_group(*, expert=False, wd=1.0, step=12):
    return {
        "wd_mult": wd,
        "lr_mult": 1.0,
        "is_decoupled_lr": False,
        "is_expert_parallel": expert,
        "step": step,
        "lr": 0.25,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }


def _optimizer_common_leaf(groups):
    return {
        "optimizer": {"param_groups": groups},
        "grad_scaler": {"scale": 1024, "growth_tracker": 3},
    }


def test_portable_optimizer_chain_changes_preserve_state(portable_optimizer):
    """Dense/expert ownership can merge and split without losing progress."""
    source = [
        _optimizer_common_leaf([_optimizer_common_group()]),
        _optimizer_common_leaf([_optimizer_common_group(expert=True, wd=0.0)]),
    ]
    original = copy.deepcopy(source)
    common = portable_optimizer.merge([source])
    # A new layout owns both groups in one optimizer, in the opposite order.
    target = _optimizer_common_leaf(
        [_optimizer_common_group(wd=0.0, step=0), _optimizer_common_group(step=0)]
    )
    tensors = {"native-sharded-key": object()}
    target["param_state"] = tensors
    portable_optimizer.map(target, common)
    assert target["param_state"] is tensors
    assert target["optimizer"].data["param_groups"] == [
        _optimizer_common_group(wd=0.0),
        _optimizer_common_group(),
    ]
    assert target["grad_scaler"].data == source[0]["grad_scaler"]
    # Restoring into separate dense/expert children retains target ownership.
    split = {
        index: dict(
            _optimizer_common_leaf([_optimizer_common_group(expert=expert, step=0)]),
            param_state=tensors,
        )
        for index, expert in enumerate((False, True))
    }
    portable_optimizer.map(split, common)
    for index, expert in enumerate((False, True)):
        assert split[index]["param_state"] is tensors
        assert split[index]["optimizer"].data["param_groups"] == [
            _optimizer_common_group(expert=expert)
        ]
    assert source == original


@pytest.mark.parametrize("field,value", [("step", 99), ("lr", 0.5), ("eps", 1e-6)])
def test_portable_optimizer_inconsistent_groups_rejected(
    portable_optimizer, field, value
):
    """Changing ownership cannot silently discard differing optimizer state."""
    changed = dict(_optimizer_common_group(expert=True), **{field: value})
    with pytest.raises(ValueError):
        portable_optimizer.merge(
            [[_optimizer_common_leaf([_optimizer_common_group(), changed])]]
        )


def test_portable_optimizer_empty_child_preserves_global_step(portable_optimizer):
    """MCore can omit the step for a child with no locally owned parameters."""
    empty_group = _optimizer_common_group(expert=True)
    del empty_group["step"]
    common = portable_optimizer.merge(
        [
            [_optimizer_common_leaf([_optimizer_common_group()])],
            [_optimizer_common_leaf([empty_group])],
        ]
    )
    assert common["optimizer"]["param_groups"][0]["step"] == 12


@pytest.mark.parametrize("portable", [False, True])
def test_checkpoint_loading_template_keeps_recovery_default(portable):
    """Portable loading requests moments; ordinary recovery keeps native RNG."""
    generate = _source_function(
        "areal/engine/megatron_utils/checkpointer.py",
        "MegatronCheckpointManager",
        "generate_state_dict",
        {
            "torch": types.SimpleNamespace(distributed=Mock()),
            "_mcore_version": lambda: "0.17.0",
            "_PORTABLE_METADATA": {"megatron_core": "0.17.0"},
        },
    )
    manager = types.SimpleNamespace(
        model=[types.SimpleNamespace(sharded_state_dict=Mock(return_value={}))],
        optimizer=types.SimpleNamespace(sharded_state_dict=Mock(return_value={})),
        lr_scheduler=None,
        get_rng_state=Mock(return_value="saved-native-rng"),
    )
    options = (
        {"with_rng": False, "optimizer_sharding_type": "fully_reshardable"}
        if portable
        else {}
    )
    state = generate(manager, is_loading=True, **options)
    optimizer_args = manager.optimizer.sharded_state_dict.call_args.kwargs
    assert optimizer_args["is_loading"] is True
    if portable:
        assert optimizer_args["metadata"] == {
            "distrib_optim_sharding_type": "fully_reshardable",
            "chained_optim_avoid_prefix": True,
            "singleton_local_shards": False,
            "distrib_optim_fully_reshardable_mem_efficient": True,
        }
        manager.model[0].sharded_state_dict.assert_called_once_with(
            metadata=optimizer_args["metadata"]
        )
        assert "rng_state" not in state
        manager.get_rng_state.assert_not_called()
    else:
        assert optimizer_args["metadata"] == {
            "distrib_optim_sharding_type": "dp_reshardable"
        }
        manager.model[0].sharded_state_dict.assert_called_once_with()
        assert state["rng_state"] == "saved-native-rng"


@pytest.fixture(scope="module")
def wait_methods():
    """Use the production completion methods while replacing only their owners."""
    run_async_task = _source_function(
        "areal/infra/utils/concurrent.py",
        None,
        "run_async_task",
        {"asyncio": asyncio},
    )
    return {
        "base": _source_function(
            "areal/api/engine_api.py", "TrainEngine", "wait_for_checkpoint"
        ),
        "fsdp": _source_function(
            "areal/engine/fsdp_engine.py", "FSDPEngine", "wait_for_checkpoint"
        ),
        "megatron": _source_function(
            "areal/engine/megatron_engine.py", "MegatronEngine", "wait_for_checkpoint"
        ),
        "controller": _source_function(
            "areal/infra/controller/train_controller.py",
            "TrainController",
            "wait_for_checkpoint",
            {"asyncio": asyncio, "math": math, "run_async_task": run_async_task},
        ),
        "engine_name": _source_function(
            "areal/infra/controller/train_controller.py",
            "TrainController",
            "_engine_name",
        ),
    }


def _controller(wait_methods, scheduler, count=3):
    """Create a controller owner with three ranks, only one of them a DP head."""
    cls = type(
        "ControllerUnderTest",
        (),
        {
            "wait_for_checkpoint": wait_methods["controller"],
            "_engine_name": wait_methods["engine_name"],
        },
    )
    controller = cls()
    controller._worker_role = "actor"
    controller.workers = [types.SimpleNamespace(id=f"worker/{i}") for i in range(count)]
    controller.workers_is_dp_head = [i == 0 for i in range(count)]
    controller.scheduler = scheduler
    return controller


class CompletionScheduler:
    """Fake network transport that records completion and injected failures."""

    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.calls = []
        self.finished = []

    async def async_call_engine(self, worker_id, method, engine_name, **kwargs):
        self.calls.append((worker_id, method, engine_name, kwargs))
        # A failed rank returns immediately, while peers are still in flight.
        outcome = self.outcomes.get(worker_id)
        if isinstance(outcome, BaseException):
            self.finished.append(worker_id)
            raise outcome
        if outcome == "hang":
            await asyncio.Event().wait()
        await asyncio.sleep(0)
        self.finished.append(worker_id)
        return outcome


def test_base_wait_without_backend_support_raises(wait_methods):
    """An unsupported backend cannot silently confirm a checkpoint."""
    with pytest.raises(NotImplementedError, match="does not support"):
        wait_methods["base"](object())
    assert not getattr(wait_methods["base"], "__isabstractmethod__", False)


def test_megatron_wait_delegates_and_propagates_failure(wait_methods):
    """Waits reach the checkpointer, including an empty local queue and errors."""
    checkpointer = Mock()
    checkpointer._async_queue.get_num_unfinalized_calls.return_value = 0
    owner = types.SimpleNamespace(checkpointer=checkpointer)
    assert wait_methods["megatron"](owner) is None
    checkpointer.wait_async_saves.assert_called_once_with()
    checkpointer.wait_async_saves.side_effect = OSError("write failed")
    with pytest.raises(OSError, match="write failed"):
        wait_methods["megatron"](owner)


def test_megatron_wait_without_checkpointer_raises(wait_methods):
    """Configurations that cannot save DCP do not report completion."""
    with pytest.raises(NotImplementedError, match="not available"):
        wait_methods["megatron"](types.SimpleNamespace(checkpointer=None))


def test_fsdp_wait_after_synchronous_save_returns_none(wait_methods):
    """The synchronous backend needs no extra engine state or collective."""
    assert wait_methods["fsdp"](object()) is None


def test_adapter_defaults_to_engine_completion_method():
    """The adapter binds the real completion interface rather than a no-op."""
    engine = types.SimpleNamespace(
        save=Mock(), load=Mock(), wait_for_checkpoint=Mock(return_value=None)
    )
    participant = M.CheckpointParticipant.for_engine(engine, format_id="fixture-dcp")
    participant.wait_saved()
    engine.wait_for_checkpoint.assert_called_once_with()


def test_adapter_allows_explicit_wait_override():
    """Callers can supply a controller wait with a custom deadline."""
    engine = types.SimpleNamespace(
        save=Mock(), load=Mock(), wait_for_checkpoint=Mock(return_value=None)
    )
    override = Mock(return_value=None)
    participant = M.CheckpointParticipant.for_engine(
        engine, format_id="fixture-dcp", wait_saved=override
    )
    participant.wait_saved()
    override.assert_called_once_with()
    engine.wait_for_checkpoint.assert_not_called()


def test_adapter_without_completion_interface_rejects_early():
    """Missing or invalid waits are rejected when constructing the participant."""
    engine = types.SimpleNamespace(save=Mock(), load=Mock())
    for override in (None, False):
        with pytest.raises(TypeError, match="wait_for_checkpoint"):
            M.CheckpointParticipant.for_engine(
                engine, format_id="fixture-dcp", wait_saved=override
            )


def test_controller_wait_contacts_every_rank_once(wait_methods):
    """Transport retry and timeout settings apply to DP heads and all peers."""
    scheduler = CompletionScheduler()
    controller = _controller(wait_methods, scheduler)
    assert controller.wait_for_checkpoint(timeout=1) is None
    assert len(scheduler.calls) == 3
    assert set(scheduler.finished) == {"worker/0", "worker/1", "worker/2"}
    for i, (worker_id, method, engine_name, options) in enumerate(scheduler.calls):
        assert worker_id == f"worker/{i}"
        assert engine_name == f"actor/{i}"
        assert method == "wait_for_checkpoint"
        assert options == {"max_retries": 1, "http_timeout": 1}


@pytest.mark.parametrize("failed_rank", [0, 1, 2])
def test_controller_failure_collects_peer_responses(wait_methods, failed_rank):
    """Failure on any rank propagates only after the other RPCs have settled."""
    worker_id = f"worker/{failed_rank}"
    scheduler = CompletionScheduler({worker_id: OSError("write failed")})
    controller = _controller(wait_methods, scheduler)
    with pytest.raises(RuntimeError, match=worker_id) as error:
        controller.wait_for_checkpoint(timeout=1)
    assert isinstance(error.value.__cause__, OSError)
    assert len(scheduler.calls) == 3
    assert set(scheduler.finished) == {"worker/0", "worker/1", "worker/2"}


def test_controller_wait_timeout_is_failure(wait_methods):
    """A transport that never responds is bounded by the controller deadline."""
    scheduler = CompletionScheduler({"worker/1": "hang"})
    controller = _controller(wait_methods, scheduler)
    with pytest.raises(RuntimeError, match="worker/1: TimeoutError"):
        controller.wait_for_checkpoint(timeout=0.02)
    assert len(scheduler.calls) == 3
    assert set(scheduler.finished) == {"worker/0", "worker/2"}


@pytest.mark.parametrize("outcome", [False, "unfinished", asyncio.CancelledError()])
def test_controller_wait_invalid_ack_is_failure(wait_methods, outcome):
    """Non-None replies and cancelled requests cannot confirm completion."""
    scheduler = CompletionScheduler({"worker/1": outcome})
    controller = _controller(wait_methods, scheduler)
    with pytest.raises(RuntimeError, match="worker/1"):
        controller.wait_for_checkpoint(timeout=1)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_controller_invalid_timeout_starts_no_rpc(wait_methods, timeout):
    """A completion request always has a finite positive deadline."""
    scheduler = CompletionScheduler()
    controller = _controller(wait_methods, scheduler)
    with pytest.raises(ValueError, match="finite and positive"):
        controller.wait_for_checkpoint(timeout=timeout)
    assert scheduler.calls == []


def test_controller_without_workers_cannot_confirm(wait_methods):
    """An uninitialized or torn-down controller cannot return vacuous success."""
    scheduler = CompletionScheduler()
    controller = _controller(wait_methods, scheduler, count=0)
    with pytest.raises(RuntimeError, match="without workers"):
        controller.wait_for_checkpoint()
    assert scheduler.calls == []


def _snapshot_manager(tmp_path, wait):
    """Prepare a minimal role whose wait callback is supplied by the controller."""
    role = Role({"weights": [1], "optimizer": {"moment": [2]}})
    participant = dataclasses.replace(role.binding(), wait_saved=wait)
    manager = M.ReconfigurationManager(checkpoints={"actor": participant}, statefuls={})
    context = M.SnapshotContext(
        run_id="test-run",
        runtime_revision="test-revision",
        configuration_id="test-config",
        state_id="test-boundary",
        step_info={
            "epoch": 0,
            "epoch_step": 1,
            "global_step": 1,
            "steps_per_epoch": 10,
        },
        policy_version=7,
    )
    return manager, context, role, tmp_path / "snapshot"


def test_snapshot_waits_for_all_worker_writes_before_publication(
    tmp_path, wait_methods
):
    """A live wait blocks publication until every fake worker has written its shard."""
    entered = threading.Event()
    release = threading.Event()
    manager, context, role, root = _snapshot_manager(tmp_path, lambda: None)

    class DelayedScheduler(CompletionScheduler):
        async def async_call_engine(self, worker_id, method, engine_name, **kwargs):
            self.calls.append((worker_id, method, engine_name, kwargs))
            if len(self.calls) == 3:
                entered.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            path, data = role.pending
            (path / f"rank-{worker_id.rsplit('/', 1)[1]}.bin").write_bytes(data)
            self.finished.append(worker_id)

    scheduler = DelayedScheduler()
    controller = _controller(wait_methods, scheduler)
    manager._checkpoints["actor"] = dataclasses.replace(
        role.binding(), wait_saved=partial(controller.wait_for_checkpoint, timeout=2)
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(manager.save_snapshot, root, context=context)
        try:
            assert entered.wait(timeout=2)
            assert not future.done()
            assert not (root / "manifest.json").exists()
        finally:
            release.set()
        reference = future.result(timeout=2)
    assert len(scheduler.finished) == 3
    assert len(M.inspect_snapshot(reference)["files"]) == 4


@pytest.mark.parametrize("failure", [OSError("disk failed"), "hang"])
def test_controller_failure_never_publishes_snapshot(tmp_path, wait_methods, failure):
    """Completed-looking files are insufficient when a worker fails to confirm."""
    scheduler = CompletionScheduler({"worker/1": failure})
    controller = _controller(wait_methods, scheduler)
    manager, context, role, root = _snapshot_manager(
        tmp_path, partial(controller.wait_for_checkpoint, timeout=0.02)
    )

    def save_with_files(path):
        role.save(path)
        role.wait()

    manager._checkpoints["actor"] = dataclasses.replace(
        manager._checkpoints["actor"], save=save_with_files
    )
    with pytest.raises(M.SnapshotError, match="save failed"):
        manager.save_snapshot(root, context=context)
    assert (root / "checkpoints" / "actor" / "checkpoint.bin").exists()
    assert not (root / "manifest.json").exists()


@pytest.fixture
def channel():
    """Close channels after each test so prepared snapshots cannot retain locks."""
    control = M.TrainerReconfiguration(
        run_id="trainer-run",
        runtime_revision="source-revision",
        configuration_id="source-config",
    )
    yield control
    control.close()


def _bind_channel(control, role=None, owner=None):
    role = role or Role({"weights": [1], "optimizer": {"moment": [2]}})
    owner = owner or Owner({"last_step": -1})
    control.bind(checkpoints={"actor": role.binding()}, statefuls={"stats": owner})
    return role, owner


STEP = {"epoch": 0, "epoch_step": 0, "global_step": 0, "steps_per_epoch": 2}


def test_channel_captures_host_state_only_at_step_end(channel, tmp_path):
    """Model staging precedes final state capture and never publishes on its own."""
    role, stats = _bind_channel(channel)
    root = tmp_path / "snapshot"
    future = channel.request_snapshot(root, state_id="boundary")
    step = dict(STEP)
    channel.begin_step_snapshot(step_info=step)
    step["global_step"] = 99
    assert role.waits == 1
    assert not future.done()
    assert not (root / "state.pkl").exists()
    assert not (root / "manifest.json").exists()
    # State changes between the native model-save point and stats commit.
    stats.value["last_step"] = 0
    channel.finish_step_snapshot(policy_version=73)
    manifest = M.inspect_snapshot(future.result())
    assert manifest["context"]["step_info"] == STEP
    assert manifest["context"]["policy_version"] == 73
    with (root / "state.pkl").open("rb") as stream:
        assert pickle.load(stream)["stats"] == {"last_step": 0}
    assert not channel.has_active_snapshot


def test_channel_late_request_waits_until_next_step(channel, tmp_path):
    """A request arriving after the save hook cannot publish mixed-step state."""
    role, _ = _bind_channel(channel)
    channel.begin_step_snapshot(step_info=STEP)
    future = channel.request_snapshot(tmp_path / "late", state_id="boundary")
    channel.finish_step_snapshot(policy_version=1)
    assert not future.done()
    assert role.waits == 0
    following = dict(STEP, epoch_step=1, global_step=1)
    channel.begin_step_snapshot(step_info=following)
    channel.finish_step_snapshot(policy_version=2)
    assert M.inspect_snapshot(future.result())["context"]["step_info"] == following


def test_channel_cancelled_request_does_no_io_and_can_be_replaced(channel, tmp_path):
    """Cancelling a queued request does not leave the channel occupied."""
    role, _ = _bind_channel(channel)
    cancelled = channel.request_snapshot(tmp_path / "cancelled", state_id="one")
    assert cancelled.cancel()
    replacement = channel.request_snapshot(tmp_path / "replacement", state_id="two")
    assert wait([cancelled], timeout=0).done == {cancelled}
    channel.begin_step_snapshot(step_info=STEP)
    assert not replacement.cancel()
    channel.finish_step_snapshot(policy_version=1)
    assert replacement.done()
    assert cancelled.cancelled()
    assert not (tmp_path / "cancelled").exists()
    assert role.waits == 1


def test_channel_rejects_duplicate_request_and_duplicate_binding(channel, tmp_path):
    """One channel has one trainer and at most one queued or active save."""
    role, _ = _bind_channel(channel)
    channel.request_snapshot(tmp_path / "snapshot", state_id="one")
    with pytest.raises(M.SnapshotError, match="already pending"):
        channel.request_snapshot(tmp_path / "other", state_id="two")
    channel.begin_step_snapshot(step_info=STEP)
    with pytest.raises(M.SnapshotError, match="already pending"):
        channel.request_snapshot(tmp_path / "other", state_id="two")
    with pytest.raises(M.SnapshotError, match="previous step"):
        channel.begin_step_snapshot(step_info=STEP)
    with pytest.raises(M.SnapshotError, match="new reconfiguration"):
        channel.bind(checkpoints={"actor": role.binding()}, statefuls={})


@pytest.mark.parametrize("phase", ["queued", "prepared"])
def test_channel_exit_ends_unfinished_request(channel, tmp_path, phase):
    """Normal early exit rejects unserviced requests and abandons staged files."""
    _bind_channel(channel)
    root = tmp_path / "snapshot"
    future = channel.request_snapshot(root, state_id="one")
    if phase == "prepared":
        channel.begin_step_snapshot(step_info=STEP)
    channel.close()
    with pytest.raises(M.SnapshotError, match="Training ended"):
        future.result()
    assert not (root / "manifest.json").exists()
    assert not channel._manager._lock.locked()
    with pytest.raises(M.SnapshotError, match="closed"):
        channel.request_snapshot(tmp_path / "after", state_id="two")


@pytest.mark.parametrize("phase", ["save", "wait", "state", "publish"])
def test_channel_errors_fail_future_and_release_manager(channel, tmp_path, phase):
    """No stage failure leaves a pending result or a completion marker."""
    role, owner = _bind_channel(channel)
    root = tmp_path / "snapshot"
    future = channel.request_snapshot(root, state_id="one")
    if phase == "save":
        role.fail_save = True
    elif phase == "wait":
        role.fail_wait = True
    with pytest.raises(M.SnapshotError):
        channel.begin_step_snapshot(step_info=STEP)
        if phase == "state":
            owner.value = lambda: None
        if phase == "publish":
            with patch.object(M.os, "replace", side_effect=OSError("disk failed")):
                channel.finish_step_snapshot(policy_version=1)
        else:
            channel.finish_step_snapshot(policy_version=1)
    with pytest.raises(M.SnapshotError):
        future.result()
    assert not (root / "manifest.json").exists()
    assert not channel._manager._lock.locked()


def test_channel_completion_callback_can_request_next_step(channel, tmp_path):
    """Future callbacks run outside the request lock."""
    _bind_channel(channel)
    first = channel.request_snapshot(tmp_path / "one", state_id="one")
    next_requests = []
    first.add_done_callback(
        lambda _: next_requests.append(
            channel.request_snapshot(tmp_path / "two", state_id="two")
        )
    )
    channel.begin_step_snapshot(step_info=STEP)
    channel.finish_step_snapshot(policy_version=1)
    assert len(next_requests) == 1
    channel.begin_step_snapshot(step_info=dict(STEP, epoch_step=1, global_step=1))
    channel.finish_step_snapshot(policy_version=2)
    assert (
        M.inspect_snapshot(next_requests[0].result())["context"]["policy_version"] == 2
    )


def _trainer_fixture(monkeypatch, *, awex=False, critic=False, single=True):
    """Exercise the actual trainer loop against fake engines and state owners."""
    events = []

    monkeypatch.setattr(M, "capture_rng_state", lambda **kwargs: {"host_seed": 3})

    class Loader(Owner):
        def __len__(self):
            return 2

    class Engine(Role):
        def __init__(self, name):
            super().__init__({"weights": [0], "optimizer": {"moment": [0]}})
            self.name = name
            self.version = 0
            self.get_version_calls = 0

        def prepare_batch(self, loader, **kwargs):
            loader.value["position"] += 2
            return [{}]

        def compute_values(self, batch):
            return [0] * len(batch)

        def compute_advantages(self, batch):
            return batch

        def ppo_update(self, batch):
            self.value["weights"][0] += 1
            self.value["optimizer"]["moment"][0] += 1
            events.append(f"{self.name}.update")

        def step_lr_scheduler(self):
            events.append(f"{self.name}.lr")

        def get_device_stats(self):
            return types.SimpleNamespace(log=lambda *args: None)

        def onload(self):
            events.append(f"{self.name}.onload")

        def update_weights(self, meta):
            events.append("weight_sync")

        def set_version(self, version):
            self.version = version

        def get_version(self):
            self.get_version_calls += 1
            return self.version

        def clear_batches(self, *args):
            pass

        def save(self, meta):
            events.append(f"{self.name}.save")
            expected_format = (
                "dcp-reconfiguration" if awex and self.name == "actor" else "dcp"
            )
            assert meta.weight_format == expected_format and meta.with_optim
            super().save(Path(meta.path))

        def wait_for_checkpoint(self):
            events.append(f"{self.name}.wait")
            super().wait()

        def load(self, meta):
            super().load(Path(meta.path))

        def runtime_state_dict(self):
            events.append(f"{self.name}.runtime")
            return f"runtime-{self.name}"

        def load_runtime_state_dict(self, state):
            assert state == f"runtime-{self.name}"

        def restore_runtime_rng(self, state):
            assert state == f"runtime-{self.name}"

    @dataclasses.dataclass
    class Meta:
        path: str
        weight_format: str
        with_optim: bool

    io_stub = types.ModuleType("areal.api.io_struct")
    io_stub.SaveLoadMeta = Meta
    monkeypatch.setitem(sys.modules, "areal.api.io_struct", io_stub)
    namespace = {
        "is_single_controller": lambda: single,
        "CheckpointParticipant": M.CheckpointParticipant,
        "EngineRuntimeState": M.EngineRuntimeState,
        "ProcessRngState": M.ProcessRngState,
        "ModelAllocation": types.SimpleNamespace(
            from_str=lambda spec: types.SimpleNamespace(backend=spec.split(":")[0])
        ),
        "stats_tracker": types.SimpleNamespace(
            record_timing=lambda *a, **k: nullcontext()
        ),
        "perf_tracer": types.SimpleNamespace(trace_scope=lambda *a, **k: nullcontext()),
        "Category": types.SimpleNamespace(
            COMPUTE="compute", COMM="comm", IO="io", INSTR="instr"
        ),
        "logger": Mock(),
    }
    cls = type(
        "TrainerUnderTest",
        (),
        {
            name: _source_function(
                "areal/trainer/rl_trainer.py", "PPOTrainer", name, namespace
            )
            for name in (
                "train",
                "_train_impl",
                "_save_training_state",
                "_bind_reconfiguration",
                "_reconfiguration_components",
            )
        },
    )
    trainer = cls()
    actor_config = types.SimpleNamespace(
        _version="v1",
        backend="megatron:d1" if awex else "fsdp:d1",
        should_compute_prox_logp=lambda: False,
    )
    trainer.config = types.SimpleNamespace(
        actor=actor_config,
        total_train_epochs=1,
        total_train_steps=1,
        critic=types.SimpleNamespace(_version="v1", backend="fsdp:d1")
        if critic
        else None,
        gconfig=types.SimpleNamespace(
            n_samples=1, reward_normalization=False, drop_incomplete_group=False
        ),
        dynamic_bs=False,
        memory_profiler=None,
    )
    trainer.actor_alloc = types.SimpleNamespace(
        backend=actor_config.backend.split(":")[0]
    )
    trainer.actor = Engine("actor")
    trainer.critic = Engine("critic") if critic else None
    trainer.ref = trainer.teacher = trainer.data_controller = None
    trainer.eval_rollout = None
    trainer.train_dataloader = Loader({"position": 0})

    class Rollout(Owner):
        def __init__(self):
            super().__init__({})
            self.version = 0
            self.frozen = False

        def bind_snapshot_dataloader(self, loader):
            self.loader = loader

        def freeze_and_drain(self, timeout):
            events.append("rollout.drain")
            self.frozen = True

        def release_snapshot(self):
            events.append("rollout.release")
            self.frozen = False

        def state_dict(self):
            assert self.frozen
            return {"dataloader": self.loader.state_dict()}

        def set_version(self, version):
            self.version = version

        def get_version(self):
            return self.version

    trainer.rollout = Rollout()
    trainer.rollout.pause = lambda: events.append("rollout.pause")
    trainer.rollout.resume = lambda: events.append("rollout.resume")
    trainer.rollout.pause_generation_sync = lambda: None
    trainer.rollout.offload = lambda **kwargs: None
    trainer.saver = Owner({"count": 0})
    trainer.saver.maybe_wait_for_staging = lambda: None
    trainer.evaluator = Owner({"count": 0})
    trainer.stats_logger = Owner({"last_commit_step": -1})
    trainer.recover_handler = types.SimpleNamespace(freq_ctl=Owner({"count": 0}))
    trainer.recover_info = None
    trainer.weight_update_meta = types.SimpleNamespace(
        with_version=lambda version: types.SimpleNamespace(version=version)
    )
    trainer._should_offload_rollout = False
    trainer._should_offload_actor = False
    trainer._should_offload_critic = False
    trainer._is_v1_awex_colocate = lambda config: awex
    trainer._requires_proxy_workflow = lambda workflow: False
    trainer._save_perf_tracer = lambda **kwargs: None

    def save_hf(**kwargs):
        events.append("native_save")
        trainer.saver.value["count"] += 1

    def save_recover(**kwargs):
        trainer.recover_handler.freq_ctl.value["count"] += 1

    def evaluate(**kwargs):
        events.append("evaluate")
        trainer.evaluator.value["count"] += 1

    def commit(**kwargs):
        events.append("stats")
        trainer.stats_logger.value["last_commit_step"] = kwargs["global_step"]

    trainer._save_hf = save_hf
    trainer._save_recover_checkpoint = save_recover
    trainer._evaluate = evaluate
    trainer._export_and_commit_stats = commit
    return trainer, events


@pytest.mark.parametrize("awex", [False, True])
@pytest.mark.parametrize("critic", [False, True])
def test_trainer_snapshot_preserves_native_order_and_final_state(
    monkeypatch, channel, tmp_path, awex, critic
):
    """AWEX stages before weight sync, ordinary training after it; both publish last."""
    trainer, events = _trainer_fixture(monkeypatch, awex=awex, critic=critic)
    root = tmp_path / "snapshot"
    future = channel.request_snapshot(root, state_id="actual-boundary")
    future.add_done_callback(lambda _: events.append("published"))
    trainer.train(workflow=object(), reconfiguration=channel)
    manifest = M.inspect_snapshot(future.result())
    if awex:
        assert events.index("actor.wait") < events.index("weight_sync")
    else:
        assert events.index("weight_sync") < events.index("actor.save")
    assert events.index("actor.lr") < events.index("actor.save")
    assert (
        events.index("stats")
        < events.index("actor.runtime")
        < events.index("published")
        < events.index("rollout.resume")
    )
    if critic:
        assert events.index("critic.lr") < events.index("critic.save")
        assert events.index("critic.wait") < events.index("published")
    assert set(manifest["checkpoints"]) == (
        {"actor", "critic"} if critic else {"actor"}
    )
    assert manifest["checkpoints"]["actor"] == (
        "areal-megatron-fully-reshardable-v1" if awex else "areal-fsdp-dcp"
    )
    if critic:
        assert manifest["checkpoints"]["critic"] == "areal-fsdp-dcp"
    assert manifest["context"]["step_info"] == STEP
    assert manifest["context"]["policy_version"] == trainer.actor.version == 1
    assert trainer.actor.get_version_calls == 1
    with (root / "state.pkl").open("rb") as stream:
        states = pickle.load(stream)
    assert states == {
        "actor_runtime": "runtime-actor",
        **({"critic_runtime": "runtime-critic"} if critic else {}),
        "host_rng": {"host_seed": 3},
        "rollout": {"dataloader": {"position": 2}},
        "saver": {"count": 1},
        "evaluator": {"count": 1},
        "stats_logger": {"last_commit_step": 0},
        "recover_frequency": {"count": 1},
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_without_request_performs_no_snapshot_io(
    monkeypatch, channel, tmp_path, enabled
):
    """Enabling an idle channel does not add checkpoint or version RPCs."""
    trainer, events = _trainer_fixture(monkeypatch)
    trainer.train(workflow=object(), reconfiguration=channel if enabled else None)
    assert not any(item.endswith((".save", ".wait")) for item in events)
    assert trainer.actor.get_version_calls == 0
    assert list(tmp_path.iterdir()) == []
    assert events[-1] == "rollout.resume"


@pytest.mark.parametrize("failure_point", ["weight_sync", "evaluate", "stats"])
def test_trainer_failure_between_hooks_aborts_publication(
    monkeypatch, channel, tmp_path, failure_point
):
    """A failure after staging rejects the Future and releases the snapshot lock."""
    trainer, events = _trainer_fixture(monkeypatch, awex=True)
    failure = RuntimeError(f"{failure_point} failed")

    def fail(*args, **kwargs):
        raise failure

    if failure_point == "weight_sync":
        trainer.actor.update_weights = fail
    elif failure_point == "evaluate":
        trainer._evaluate = fail
    else:
        trainer._export_and_commit_stats = fail
    root = tmp_path / "snapshot"
    future = channel.request_snapshot(root, state_id="boundary")
    with pytest.raises(RuntimeError) as raised:
        trainer.train(workflow=object(), reconfiguration=channel)
    assert raised.value is failure
    assert future.exception() is failure
    assert "actor.wait" in events
    assert not (root / "manifest.json").exists()
    assert not channel._manager._lock.locked()


@pytest.mark.parametrize("mode", ["spmd", "v2", "archon"])
def test_trainer_unsupported_mode_fails_before_save(
    monkeypatch, channel, tmp_path, mode
):
    """The single-coordinator integration cannot silently enter unsupported modes."""
    trainer, events = _trainer_fixture(monkeypatch, single=mode != "spmd")
    if mode == "v2":
        trainer.config.actor._version = "v2"
    if mode == "archon":
        trainer.actor_alloc.backend = "archon"
    future = channel.request_snapshot(tmp_path / "snapshot", state_id="boundary")
    with pytest.raises(NotImplementedError):
        trainer.train(workflow=object(), reconfiguration=channel)
    assert isinstance(future.exception(), NotImplementedError)
    assert events == []
    assert list(tmp_path.iterdir()) == []


def test_trainer_request_arriving_after_last_hook_is_reported(
    monkeypatch, channel, tmp_path
):
    """A late final-step request fails explicitly rather than waiting forever."""
    trainer, events = _trainer_fixture(monkeypatch)
    requests = []
    commit = trainer._export_and_commit_stats

    def late_request(**kwargs):
        commit(**kwargs)
        requests.append(channel.request_snapshot(tmp_path / "late", state_id="late"))

    trainer._export_and_commit_stats = late_request
    trainer.train(workflow=object(), reconfiguration=channel)
    assert len(requests) == 1
    with pytest.raises(M.SnapshotError, match="Training ended"):
        requests[0].result()
    assert "actor.save" not in events
    assert trainer.actor.get_version_calls == 0


@pytest.mark.parametrize("awex", [False, True])
def test_trainer_consecutive_snapshots_keep_models_and_steps_aligned(
    monkeypatch, channel, tmp_path, awex
):
    """A completion callback can request the next step without mixing its state."""
    trainer, _ = _trainer_fixture(monkeypatch, awex=awex, critic=True)
    trainer.config.total_train_steps = 2
    first = channel.request_snapshot(tmp_path / "one", state_id="one")
    following = []
    first.add_done_callback(
        lambda _: following.append(
            channel.request_snapshot(tmp_path / "two", state_id="two")
        )
    )
    trainer.train(workflow=object(), reconfiguration=channel)
    assert len(following) == 1
    for step, future in enumerate((first, following[0])):
        reference = future.result()
        manifest = M.inspect_snapshot(reference)
        assert manifest["context"]["step_info"]["global_step"] == step
        assert manifest["context"]["policy_version"] == step + 1
        for role in ("actor", "critic"):
            path = Path(reference.path) / "checkpoints" / role / "checkpoint.bin"
            assert json.loads(path.read_text()) == {
                "weights": [step + 1],
                "optimizer": {"moment": [step + 1]},
            }
        with (Path(reference.path) / "state.pkl").open("rb") as stream:
            states = pickle.load(stream)
        assert states["stats_logger"]["last_commit_step"] == step
        assert states["rollout"]["dataloader"]["position"] == 2 * (step + 1)


# The runner below is the real production thread/queue implementation. Only its
# uvloop factory is replaced by stdlib asyncio; no training/GPU package is loaded.
@pytest.fixture
def rollout_types(monkeypatch):
    import collections
    import logging
    import random
    import time
    import typing

    module = types.ModuleType("rollout_cpu_under_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(
        sys.modules,
        "uvloop",
        types.SimpleNamespace(new_event_loop=asyncio.new_event_loop),
    )
    path = REPO_ROOT / "areal/infra/async_task_runner.py"
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    scope = module.__dict__
    scope.update(
        math=math,
        random=random,
        copy=copy,
        time=time,
        deque=collections.deque,
        Lock=threading.Lock,
        Protocol=typing.Protocol,
        RolloutInputBuffer=M.RolloutInputBuffer,
        RolloutStat=dataclasses.make_dataclass(
            "RolloutStat",
            [
                (n, int, dataclasses.field(default=0))
                for n in ("accepted", "enqueued", "rejected", "running")
            ],
        ),
        fields=dataclasses.fields,
        logger=logging.getLogger("rollout-test"),
        trace_perf=lambda *a, **k: lambda f: f,
        ModelAllocation=types.SimpleNamespace(from_str=lambda x: x),
        RTensor=types.SimpleNamespace(localize=lambda value: value),
        torch=types.SimpleNamespace(Tensor=type("UnavailableTensor", (), {})),
        _MAX_FETCH_BATCH_SIZE=100,
        _SHUTDOWN_TIMEOUT_SECONDS=2,
        _DEFAULT_WAIT_TIMEOUT_SECONDS=3,
    )
    for relative, names in [
        ("areal/infra/staleness_manager.py", {"StalenessManager", "VersionProvider"}),
        (
            "areal/infra/workflow_executor.py",
            {
                "WithTaskID",
                "TInput",
                "TResult",
                "_select_results",
                "BatchTaskDispatcher",
                "TaskIdGenerator",
            },
        ),
        (
            "areal/infra/controller/rollout_controller.py",
            {
                "_RemoteRolloutTaskInput",
                "_RemoteRolloutResult",
                "_snapshot_cpu_copy",
                "_same_input_spec",
                "RolloutController",
            },
        ),
    ]:
        path = REPO_ROOT / relative
        body = [
            n
            for n in ast.parse(path.read_text()).body
            if getattr(n, "name", None) in names
            or isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
        ]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return module


@pytest.fixture
def make_dispatcher(rollout_types):
    dispatchers = []

    def make(factory=None, *, capacity=8, deterministic=True):
        provider = types.SimpleNamespace(get_version=lambda: 0)
        manager = rollout_types.StalenessManager(provider, capacity, 8, 1)
        if factory is None:

            def factory(item):
                async def run():
                    manager.on_rollout_accepted()
                    return {"id": item.task_id}

                return run

        dispatcher = rollout_types.BatchTaskDispatcher(
            16, factory, manager, deterministic_order=deterministic
        )
        dispatcher.runner.poll_sleep_time = 0.001
        dispatcher.runner.poll_wait_time = 0.001
        dispatcher.initialize(rollout_types.logger)
        dispatchers.append(dispatcher)
        return dispatcher

    yield make
    for dispatcher in dispatchers:
        dispatcher.destroy()
        assert not dispatcher._commit_thread.is_alive()
        assert not dispatcher._fetch_thread.is_alive()
        assert not dispatcher.runner.thread.is_alive()


def _until(predicate, timeout=3):
    import time

    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for test synchronization")
        threading.Event().wait(0.001)


def test_dispatcher_drains_admitted_work_preserves_unsubmitted_and_ready(
    make_dispatcher,
):
    running = threading.Event()
    complete = threading.Event()

    def factory(item):
        async def run():
            running.set()
            while not complete.is_set():
                await asyncio.sleep(0.001)
            dispatcher.staleness_manager.on_rollout_accepted()
            return {"id": item.task_id, "versions": [7]}

        return run

    dispatcher = make_dispatcher(factory, capacity=1)
    for i in range(3):
        dispatcher.submit_task_input(types.SimpleNamespace(task_id=i))
    assert running.wait(3)
    with ThreadPoolExecutor() as pool:
        drain = pool.submit(dispatcher.freeze_and_drain, 3)
        _until(lambda: dispatcher._snapshot_frozen)
        assert not drain.done()
        # Ordinary resume must never remove snapshot admission fencing.
        dispatcher.resume()
        with pytest.raises(RuntimeError, match="frozen"):
            dispatcher.submit_task_input(types.SimpleNamespace(task_id=10))
        with pytest.raises(RuntimeError, match="frozen"):
            dispatcher.wait_results(1)
        complete.set()
        drain.result(timeout=3)
    state = dispatcher.state_dict()
    assert [item.task_id for item in state["pending_inputs"]] == [1, 2]
    assert state["ready_results"] == [(0, {"id": 0, "versions": [7]})]
    assert dispatcher.staleness_manager.state_dict()["counters"] == dict(
        accepted=1, running=0, enqueued=2, rejected=0
    )
    dispatcher.release_snapshot()
    assert sorted(r["id"] for r in dispatcher.wait_results(3, timeout=3)) == [0, 1, 2]


def test_dispatcher_fence_waits_for_popped_input_handoff(make_dispatcher):
    popped = threading.Event()
    release_factory = threading.Event()

    def factory(item):
        popped.set()
        assert release_factory.wait(3)

        async def run():
            dispatcher.staleness_manager.on_rollout_accepted()
            return item.task_id

        return run

    dispatcher = make_dispatcher(factory)
    dispatcher.submit_task_input(types.SimpleNamespace(task_id=2))
    assert popped.wait(3)
    with ThreadPoolExecutor() as pool:
        draining = pool.submit(dispatcher.freeze_and_drain, 3)
        assert not draining.done()
        release_factory.set()
        draining.result(timeout=3)
    assert dispatcher.state_dict()["ready_results"] == [(2, 2)]


def test_dispatcher_timeout_never_claims_drained_state(make_dispatcher):
    release = threading.Event()
    started = threading.Event()

    def factory(item):
        async def run():
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            dispatcher.staleness_manager.on_rollout_rejected()
            return None

        return run

    dispatcher = make_dispatcher(factory)
    dispatcher.submit_task_input(types.SimpleNamespace(task_id=0))
    assert started.wait(3)
    try:
        with pytest.raises(TimeoutError):
            dispatcher.freeze_and_drain(0.02)
        with pytest.raises(RuntimeError, match="unsettled"):
            dispatcher.state_dict()
        assert dispatcher._snapshot_frozen
    finally:
        release.set()
    _until(lambda: 0 in dispatcher._pending_results)


def test_dispatcher_roundtrip_preserves_rejections_order_and_sampling_rng(
    make_dispatcher,
):
    import random

    source = make_dispatcher(deterministic=False)
    source.freeze_and_drain(1)
    state = source.state_dict()
    # Ready completion order differs from task order, as in asynchronous rollout.
    state["ready_results"] = [(8, {"versions": [2]}), (1, None), (5, {"versions": [3]})]
    source.load_state_dict(state)
    source.release_snapshot()
    expected = source.wait_results(2)
    target = make_dispatcher(deterministic=False)
    target.freeze_and_drain(1)
    target.load_state_dict(state)
    assert target._snapshot_frozen
    target.release_snapshot()
    assert target.wait_results(2) == expected
    assert target.wait_results(1) == [{"versions": [3]}]
    random.setstate(state["selection_rng"])


class CursorLoader:
    """Small stateful batch loader that also models sampler epoch resets."""

    def __init__(self):
        self.batch_size = 3
        self.position = 0
        self.epoch = 0
        self.sampler = types.SimpleNamespace(set_epoch=self.set_epoch)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.position = 0

    def __iter__(self):
        while self.position < 6:
            start = self.position
            self.position += 3
            yield [{"item": self.epoch * 6 + i} for i in range(start, start + 3)]

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


@pytest.mark.parametrize("consumed", [1, 3, 5, 6, 7, 12, 13])
def test_input_buffer_roundtrip_retains_half_batches_and_sampler_epoch(consumed):
    source = M.RolloutInputBuffer(CursorLoader())
    assert [next(source)["item"] for _ in range(consumed)] == list(range(consumed))
    state = copy.deepcopy(source.state_dict())
    target = M.RolloutInputBuffer(CursorLoader())
    target.load_state_dict(state)
    assert (
        [next(target)["item"] for _ in range(10)]
        == [next(source)["item"] for _ in range(10)]
        == list(range(consumed, consumed + 10))
    )


@pytest.mark.parametrize("damage", ["negative", "running", "limits"])
def test_staleness_restore_rejects_inconsistent_state(rollout_types, damage):
    manager = rollout_types.StalenessManager(
        types.SimpleNamespace(get_version=lambda: 3), 4, 2, 1
    )
    state = manager.state_dict()
    state["counters"]["accepted"] = 5
    if damage == "negative":
        state["counters"]["rejected"] = -1
    elif damage == "running":
        state["counters"]["running"] = 1
    else:
        state["limits"][1] = 8
    with pytest.raises(ValueError):
        manager.load_state_dict(state)
    assert manager.get_stats().accepted == 0


def _rollout_controller(rollout_types, make_dispatcher):
    controller = rollout_types.RolloutController(
        None, types.SimpleNamespace(backend="test", enable_rollout_tracing=False), None
    )
    controller._dispatcher = make_dispatcher()
    controller._staleness_manager = controller._dispatcher.staleness_manager
    controller._staleness_manager.version_provider = controller
    controller._collective_rpc = Mock()
    controller.bind_snapshot_dataloader(CursorLoader())
    return controller


def test_rollout_state_roundtrip_after_original_owner_disappears(
    rollout_types, make_dispatcher
):
    source = _rollout_controller(rollout_types, make_dispatcher)
    source.set_version(9)
    assert next(source._input_buffer) == {"item": 0}
    source._input_spec = dict(
        workflow="example.workflow", workflow_kwargs={}, should_accept_fn=None
    )
    # Populate both ready and pending through the real frozen import path.
    source.freeze_and_drain(1)
    inventory = source.dispatcher.state_dict()
    inventory["pending_inputs"] = [
        rollout_types._RemoteRolloutTaskInput(
            2, {"item": 1}, "example.workflow", {}, None
        )
    ]
    inventory["ready_results"] = [
        (
            0,
            rollout_types._RemoteRolloutResult(
                0, {"versions": [4], "tokens": [11, 12]}
            ),
        ),
        (1, None),
    ]
    source.dispatcher.load_state_dict(inventory)
    source._task_id_generator.load_state_dict({"next_id": 3})
    stats = source.staleness_manager.state_dict()
    stats["counters"].update(accepted=19, rejected=1, enqueued=1)
    source.staleness_manager.load_state_dict(stats)
    source.release_snapshot()
    source.dispatcher.pause()  # no admitted work, only unsubmitted inputs
    source.freeze_and_drain(1)
    state = pickle.loads(pickle.dumps(source.state_dict()))
    source.dispatcher.destroy()

    target = _rollout_controller(rollout_types, make_dispatcher)
    target.set_version(9)
    target.freeze_and_drain(1)
    target.load_state_dict(state)
    assert target.staleness_manager.get_stats().accepted == 19
    assert target._task_id_generator.next() == 3
    assert next(target._input_buffer) == {"item": 1}
    assert next(target._input_buffer) == {"item": 2}
    assert next(target._input_buffer) == {"item": 3}
    assert target.dispatcher._snapshot_frozen
    target.release_snapshot()
    assert target.wait(2) == [{"versions": [4], "tokens": [11, 12]}, None]


@pytest.mark.parametrize("reason", ["rpc", "gateway"])
def test_rollout_unconfirmed_work_cannot_be_snapshotted(
    rollout_types, make_dispatcher, reason
):
    controller = _rollout_controller(rollout_types, make_dispatcher)
    if reason == "rpc":
        controller._uncertain_tasks.add(3)
    else:
        controller._proxy_gateway_host = "test-gateway"
    with pytest.raises((RuntimeError, NotImplementedError)):
        controller.freeze_and_drain(1)
    with pytest.raises(RuntimeError, match="Drain"):
        controller.state_dict()


def test_rollout_rpc_timeout_is_not_a_normal_rejection(rollout_types, make_dispatcher):
    controller = _rollout_controller(rollout_types, make_dispatcher)
    controller.config.request_timeout = 0.01
    controller._choose_worker = lambda: (types.SimpleNamespace(id="worker"), 0)
    controller._worker_role = "rollout"
    controller._callback_host = "localhost"
    controller._callback_port = 1234

    async def fail(*args, **kwargs):
        raise TimeoutError("lost remote acknowledgement")

    controller.scheduler = types.SimpleNamespace(async_call_engine=fail)
    item = rollout_types._RemoteRolloutTaskInput(4, {}, "test.workflow", {}, None)
    controller.staleness_manager.on_rollout_enqueued()
    controller.staleness_manager.on_rollout_submitted()
    assert asyncio.run(controller._create_submit_callback(item)()) is None
    assert controller._uncertain_tasks == {4}
    with pytest.raises(RuntimeError, match="unconfirmed"):
        controller.freeze_and_drain(1)


def test_trainer_drains_before_offload_and_model_save(monkeypatch, channel, tmp_path):
    trainer, events = _trainer_fixture(monkeypatch, awex=True)
    trainer._should_offload_rollout = True
    trainer._onload_rollout = lambda **kwargs: events.append("rollout.onload")
    trainer._offload_rollout = lambda **kwargs: events.append("rollout.offload")
    saved = channel.request_snapshot(tmp_path / "snapshot", state_id="one")
    trainer.train(workflow=object(), reconfiguration=channel)
    saved.result()
    assert (
        events.index("rollout.drain")
        < events.index("rollout.offload")
        < events.index("actor.update")
    )
    assert (
        events.index("actor.wait")
        < events.index("rollout.release")
        < events.index("rollout.resume")
    )


def test_trainer_drain_failure_stops_before_offload_or_checkpoint(
    monkeypatch, channel, tmp_path
):
    trainer, events = _trainer_fixture(monkeypatch)
    trainer.rollout.freeze_and_drain = Mock(
        side_effect=TimeoutError("unfinished rollout")
    )
    root = tmp_path / "snapshot"
    saved = channel.request_snapshot(root, state_id="one")
    with pytest.raises(TimeoutError):
        trainer.train(workflow=object(), reconfiguration=channel)
    assert isinstance(saved.exception(), TimeoutError)
    assert "actor.save" not in events and "actor.update" not in events
    assert not root.exists()


class CpuPayloadTensor:
    """Tiny tensor transport double; never substitutes for real torch validation."""

    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return CpuPayloadTensor(self.values)

    def __eq__(self, other):
        raise ValueError("A multi-element tensor has no scalar truth value")


def test_rollout_copies_prefetch_and_remote_results_before_data_cleanup(
    rollout_types, make_dispatcher, monkeypatch
):
    class Remote:
        def __init__(self, key):
            self.key = key

    store = {"input": CpuPayloadTensor([10, 20]), "output": CpuPayloadTensor([1, 2])}

    def localize(value):
        if isinstance(value, Remote):
            return store[value.key]
        if isinstance(value, dict):
            return {k: localize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [localize(v) for v in value]
        if isinstance(value, tuple):
            return tuple(localize(v) for v in value)
        return value

    monkeypatch.setattr(
        rollout_types, "RTensor", types.SimpleNamespace(localize=localize)
    )
    monkeypatch.setattr(
        rollout_types, "torch", types.SimpleNamespace(Tensor=CpuPayloadTensor)
    )
    controller = _rollout_controller(rollout_types, make_dispatcher)
    next(controller._input_buffer)
    remote_input, remote_output = Remote("input"), Remote("output")
    controller._input_buffer._remaining[0] = {"image": remote_input}
    controller.freeze_and_drain(1)
    inventory = controller.dispatcher.state_dict()
    inventory["ready_results"] = [
        (
            0,
            rollout_types._RemoteRolloutResult(
                0, {"tokens": remote_output, "versions": [0]}
            ),
        )
    ]
    controller.dispatcher.load_state_dict(inventory)
    controller._task_id_generator.reserve(0)
    controller.release_snapshot()
    controller.freeze_and_drain(1)
    # Live ownership stays intact; snapshot copying must not clear remote shards.
    assert controller._input_buffer._remaining[0]["image"] is remote_input
    assert (
        controller.dispatcher._pending_results[0].data.trajectory["tokens"]
        is remote_output
    )
    store.clear()
    controller.set_version(1)
    saved = controller.state_dict()
    assert saved["input_buffer"]["remaining"][0]["image"].values == [10, 20]
    assert saved["inventory"]["ready_results"][0][1]["tokens"].values == [1, 2]
    assert saved["inventory"]["ready_results"][0][1]["versions"] == [0]
    assert saved["policy_version"] == 1


def test_prepare_batch_tensor_bindings_work_across_steps_and_restore(
    rollout_types, make_dispatcher, monkeypatch
):
    controller = _rollout_controller(rollout_types, make_dispatcher)
    controller._resolve_workflow_str = lambda value: value
    monkeypatch.setattr(
        rollout_types,
        "torch",
        types.SimpleNamespace(
            Tensor=CpuPayloadTensor, equal=lambda a, b: a.values == b.values
        ),
    )
    seen = []

    def consume(generator, **kwargs):
        item = next(generator)
        seen.append(item.data["item"])
        return [rollout_types._RemoteRolloutResult(item.task_id, {"versions": [0]})]

    controller.dispatcher.active_submit_and_wait = consume
    kwargs = {"vector": CpuPayloadTensor([1, 2])}
    for _ in range(2):
        controller.prepare_batch(
            controller._input_buffer.dataloader,
            workflow="example.Workflow",
            workflow_kwargs=kwargs,
        )
    assert seen == [0, 1]
    controller.freeze_and_drain(1)
    state = controller.state_dict()
    target = _rollout_controller(rollout_types, make_dispatcher)
    target._resolve_workflow_str = lambda value: value
    target.dispatcher.active_submit_and_wait = consume
    target.freeze_and_drain(1)
    target.load_state_dict(state)
    target.release_snapshot()
    target.prepare_batch(
        target._input_buffer.dataloader,
        workflow="example.Workflow",
        workflow_kwargs=kwargs,
    )
    assert seen == [0, 1, 2]


def test_explicit_task_id_reserves_counter_for_snapshot_restore(
    rollout_types, make_dispatcher
):
    source = _rollout_controller(rollout_types, make_dispatcher)
    source._resolve_workflow_str = lambda value: value
    source.dispatcher.pause()
    assert source.submit({}, "example.Workflow", task_id=99) == 99
    source.freeze_and_drain(1)
    state = source.state_dict()
    assert state["task_ids"]["next_id"] == 100
    target = _rollout_controller(rollout_types, make_dispatcher)
    target.freeze_and_drain(1)
    target.load_state_dict(state)
    assert target._task_id_generator.next() == 100
    assert target.dispatcher.state_dict()["pending_inputs"][0].task_id == 99


def test_rollout_partial_restore_stays_frozen_and_cannot_publish(
    rollout_types, make_dispatcher
):
    source = _rollout_controller(rollout_types, make_dispatcher)
    source.freeze_and_drain(1)
    state = source.state_dict()
    target = _rollout_controller(rollout_types, make_dispatcher)
    target.freeze_and_drain(1)
    target._input_buffer.dataloader.load_state_dict = Mock(
        side_effect=OSError("loader restore failed")
    )
    with pytest.raises(OSError):
        target.load_state_dict(state)
    for operation in (
        target.release_snapshot,
        target.state_dict,
        lambda: target.load_state_dict(state),
    ):
        with pytest.raises(RuntimeError, match="Recreate"):
            operation()
    assert target.dispatcher._snapshot_frozen


def test_real_rollout_inventory_roundtrips_through_published_snapshot(
    rollout_types, make_dispatcher, tmp_path
):
    source = _rollout_controller(rollout_types, make_dispatcher)
    source._resolve_workflow_str = lambda value: value
    source.dispatcher.pause()
    # This input has actually left the loader; the two remaining items must
    # remain in the feeder, not be regenerated or lost across restart.
    item = next(source._input_buffer)
    source.submit(item, "example.Workflow")
    source.freeze_and_drain(1)
    actor = Role({"weights": [4], "optimizer": {"moment": [8]}})
    context = M.SnapshotContext(
        "trial", "pinned-runtime", "same-config", "step", STEP, 0
    )
    writer = M.ReconfigurationManager(
        checkpoints={"actor": actor.binding()}, statefuls={"rollout": source}
    )
    ref = writer.save_snapshot(tmp_path / "snapshot", context=context)
    source.dispatcher.destroy()

    target = _rollout_controller(rollout_types, make_dispatcher)
    target.freeze_and_drain(1)
    restored_actor = Role({})
    reader = M.ReconfigurationManager(
        checkpoints={"actor": restored_actor.binding()}, statefuls={"rollout": target}
    )
    reader.restore_snapshot(ref, expected_context=context)
    assert restored_actor.value == actor.value
    assert target.dispatcher._snapshot_frozen
    inventory = target.dispatcher.state_dict()
    assert [i.data["item"] for i in inventory["pending_inputs"]] == [0]
    assert [next(target._input_buffer)["item"] for _ in range(8)] == list(range(1, 9))
    assert target.staleness_manager.get_stats().enqueued == 1


def test_request_after_drain_hook_waits_for_next_step(monkeypatch, channel, tmp_path):
    trainer, events = _trainer_fixture(monkeypatch)
    trainer.config.total_train_steps = 2
    update = trainer.actor.ppo_update
    requests = []

    def request_during_training(batch):
        if not requests:
            requests.append(
                channel.request_snapshot(tmp_path / "snapshot", state_id="late")
            )
        update(batch)

    trainer.actor.ppo_update = request_during_training
    trainer.train(workflow=object(), reconfiguration=channel)
    manifest = M.inspect_snapshot(requests[0].result())
    assert manifest["context"]["step_info"]["global_step"] == 1
    assert events.count("rollout.drain") == 1


def test_actor_rollout_version_mismatch_does_not_publish(
    monkeypatch, channel, tmp_path
):
    trainer, _ = _trainer_fixture(monkeypatch)
    trainer.rollout.set_version = lambda version: None
    root = tmp_path / "snapshot"
    saved = channel.request_snapshot(root, state_id="mismatch")
    with pytest.raises(M.SnapshotError, match="versions disagree"):
        trainer.train(workflow=object(), reconfiguration=channel)
    assert isinstance(saved.exception(), M.SnapshotError)
    assert not (root / "manifest.json").exists()


def test_trainer_restore_keeps_real_pending_inputs_fenced_until_resume(
    rollout_types, make_dispatcher, monkeypatch, tmp_path
):
    """Run the startup coordinator with the actual v1 controller and dispatcher."""
    monkeypatch.setitem(
        sys.modules, "areal.infra.controller.rollout_controller", rollout_types
    )
    monkeypatch.setattr(CursorLoader, "__len__", lambda _: 2, raising=False)
    source = _rollout_controller(rollout_types, make_dispatcher)
    source._resolve_workflow_str = lambda value: value
    spec = dict(
        workflow="example.Workflow",
        workflow_kwargs={},
        should_accept_fn=None,
        group_size=1,
        reward_normalization=False,
        drop_incomplete_group=False,
    )
    source._input_spec = spec
    source.dispatcher.pause()
    source.submit(next(source._input_buffer), **spec)
    source.freeze_and_drain(1)
    context = M.SnapshotContext("trial", "runtime", "config", "boundary", STEP, 0)
    original_actor = Role({"weights": [4], "optimizer": {"moment": [8]}})
    ref = M.ReconfigurationManager(
        checkpoints={"actor": original_actor.binding()}, statefuls={"rollout": source}
    ).save_snapshot(tmp_path / "snapshot", context=context)
    source.dispatcher.destroy()

    target = _rollout_controller(rollout_types, make_dispatcher)
    target._resolve_workflow_str = lambda value: value
    actor = Role({})
    actor.set_version = Mock()
    actor.get_version = lambda: 0
    actor.update_weights = Mock()
    trainer = types.SimpleNamespace(
        actor=actor,
        critic=None,
        rollout=target,
        eval_rollout=None,
        train_dataloader=target._input_buffer.dataloader,
        config=types.SimpleNamespace(
            gconfig=types.SimpleNamespace(
                n_samples=1, reward_normalization=False, drop_incomplete_group=False
            )
        ),
        _should_offload_actor=False,
        _should_offload_critic=False,
        _should_offload_rollout=False,
        _is_v1_awex_colocate=lambda _: False,
        _reconfiguration_components=lambda: ({"actor": actor.binding()}, {}),
        weight_update_meta=types.SimpleNamespace(type="disk", with_version=lambda v: v),
    )
    restore = M.TrainerSnapshotRestore(M.SnapshotRestoreRequest(ref, context))
    restore.restore(
        trainer,
        workflow="example.Workflow",
        workflow_kwargs=None,
        should_accept_fn=None,
        max_steps=3,
    )
    assert actor.value == original_actor.value
    actor.update_weights.assert_called_once_with(0)
    assert target.dispatcher._snapshot_frozen
    inventory = target.dispatcher.state_dict()
    assert [i.data["item"] for i in inventory["pending_inputs"]] == [0]
    assert target.staleness_manager.get_stats().enqueued == 1
    assert target.staleness_manager.get_stats().running == 0
    assert next(target._input_buffer) == {"item": 1}
    restore.resume(trainer)
    assert target.dispatcher.wait_results(1, timeout=3) == [{"id": 0}]
