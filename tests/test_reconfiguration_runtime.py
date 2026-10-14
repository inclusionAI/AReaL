# SPDX-License-Identifier: Apache-2.0
"""CPU checks for supplemental runtime state and staged restoration.

The collective and RPC transports are simulated, not FSDP/DTensor internals.
Real torch scheduler/RNG checks run when torch is installed; GPU continuity
and real distributed backend checkpoints still require integration testing.
"""

from __future__ import annotations

import copy
import dataclasses
import random
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from tests import test_reconfiguration as snapshot_tests
from tests.test_reconfiguration import (
    M,
    Owner,
    RpcTransport,
    _controller,
    _snapshot_manager,
    _source_function,
    _trainer_fixture,
)

wait_methods = snapshot_tests.wait_methods

IDENTITY = {"backend": "test", "parallelism": {"dp": 3}, "schedule": {"steps": 20}}


def _engine(rank=0, size=3):
    return types.SimpleNamespace(
        rank=rank,
        world_size=size,
        cpu_group=types.SimpleNamespace(rank=rank, size=size),
        lr_scheduler=Owner({"position": 5}),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 0.25}, {"lr": 0.125}]),
    )


def _packet(size=3, *, scheduler=None):
    return M._encode_runtime_state(
        {
            "version": 1,
            "ranks": [
                M._encode_runtime_state(
                    {
                        "rank": rank,
                        "identity": IDENTITY,
                        "scheduler": {"position": 5}
                        if scheduler is None
                        else scheduler,
                        "lrs": [0.25, 0.125],
                        "rng": ("independent-rng", rank),
                    }
                )
                for rank in range(size)
            ],
        }
    )


def _collect_on_three_ranks(monkeypatch, *, failure=None):
    """Run production capture code concurrently with one CPU collective stub."""
    arrivals = [None] * 3
    barrier = threading.Barrier(3, timeout=3)

    def gather(output, value, *, group):
        arrivals[group.rank] = value
        barrier.wait()
        output[:] = copy.deepcopy(arrivals)

    dist = types.ModuleType("torch.distributed")
    dist.get_rank = lambda group: group.rank
    dist.get_world_size = lambda group: group.size
    dist.all_gather_object = gather
    torch = types.ModuleType("torch")
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    def collect(rank):
        engine = _engine(rank)

        def rng():
            if rank == 1 and failure == "capture":
                raise ValueError("broken RNG")
            if rank == 1 and failure == "serialization":
                return lambda: None
            return ("independent-rng", rank)

        try:
            return M.collect_runtime_state(
                engine, identity=lambda: IDENTITY, capture_rng=rng
            )
        except Exception as error:
            return error

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(collect, range(3)))
    return results, arrivals


def test_collect_preserves_every_rank_and_returns_identical_opaque_packets(monkeypatch):
    """DP-head filtering cannot discard a non-head's independent RNG stream."""
    results, arrivals = _collect_on_three_ranks(monkeypatch)
    assert all(arrivals)
    assert isinstance(results[0], str)
    assert results[0] == results[1] == results[2]
    for rank in range(3):
        local = M.select_runtime_state(_engine(rank), results[0], identity=IDENTITY)
        assert local["rng"] == ("independent-rng", rank)


@pytest.mark.parametrize("failure", ["capture", "serialization"])
def test_capture_failure_still_enters_collective_and_fails_every_rank(
    monkeypatch, failure
):
    """A local exception must not strand the other ranks inside all-gather."""
    results, arrivals = _collect_on_three_ranks(monkeypatch, failure=failure)
    assert all(arrivals)
    assert all(isinstance(result, M.SnapshotError) for result in results)
    assert all("rank 1" in str(result) for result in results)


@pytest.mark.parametrize(
    "mismatch", ["size", "rank", "layout", "schedule", "duplicate"]
)
def test_runtime_identity_mismatch_rejected_before_scheduler_mutation(mismatch):
    """Same rank number is insufficient when layout or schedule differs."""
    engine, identity, packet = _engine(), copy.deepcopy(IDENTITY), _packet()
    if mismatch == "size":
        engine.world_size = 4
    elif mismatch == "rank":
        engine.rank = 3
    elif mismatch == "layout":
        identity["parallelism"]["dp"] = 4
    elif mismatch == "schedule":
        identity["schedule"]["steps"] = 40
    else:
        decoded = M._decode_runtime_state(packet)
        decoded["ranks"][1] = decoded["ranks"][0]
        packet = M._encode_runtime_state(decoded)
    with pytest.raises(M.SnapshotError, match="does not match"):
        M.load_runtime_scheduler(engine, packet, identity=identity)
    assert engine.lr_scheduler.loads == 0


def test_scheduler_load_preserves_progress_and_optimizer_lrs_without_step():
    """The saved boundary is restored directly, with no synthetic LR advance."""
    engine = _engine()
    engine.lr_scheduler.value = {"position": 0}
    engine.lr_scheduler.step = Mock(side_effect=AssertionError("extra scheduler step"))
    engine.optimizer.param_groups[0]["lr"] = 9.0
    M.load_runtime_scheduler(engine, _packet(), identity=IDENTITY)
    assert engine.lr_scheduler.value == {"position": 5}
    assert [group["lr"] for group in engine.optimizer.param_groups] == [0.25, 0.125]
    engine.lr_scheduler.step.assert_not_called()


def test_scheduler_group_mismatch_does_not_partially_load():
    engine = _engine()
    engine.optimizer.param_groups.pop()
    with pytest.raises(M.SnapshotError, match="parameter groups"):
        M.load_runtime_scheduler(engine, _packet(), identity=IDENTITY)
    assert engine.lr_scheduler.loads == 0


@pytest.mark.parametrize("native_loaded", [False, True])
def test_megatron_scheduler_restore_does_not_double_native_loaded_steps(native_loaded):
    """MCore loads an additive count; the final boundary must be absolute."""

    class AdditiveScheduler:
        num_steps = 5 if native_loaded else 0

        def step(self, increment=1):
            self.num_steps += increment

        def load_state_dict(self, state):
            self.step(increment=state["num_steps"])

    engine = _engine()
    engine._runtime_state_identity = lambda: IDENTITY
    engine.lr_scheduler = AdditiveScheduler()
    load = _source_function(
        "areal/engine/megatron_engine.py",
        "MegatronEngine",
        "load_runtime_state_dict",
        {"load_runtime_scheduler": M.load_runtime_scheduler},
    )
    load(engine, _packet(scheduler={"num_steps": 5}))
    assert engine.lr_scheduler.num_steps == 5
    engine.lr_scheduler.step()
    assert engine.lr_scheduler.num_steps == 6


class RngTensor:
    """Minimal CPU RNG value for dependency-free API tests, not a model tensor."""

    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def clone(self):
        return RngTensor(copy.deepcopy(self.value))


@pytest.fixture
def rng_dependencies(monkeypatch):
    """Independent deterministic streams behind the public numpy/torch APIs."""
    numpy_rng, torch_rng, device_rng = (random.Random(seed) for seed in (1, 2, 3))
    numpy = types.ModuleType("numpy")
    numpy.random = types.SimpleNamespace(
        get_state=numpy_rng.getstate, set_state=numpy_rng.setstate
    )
    torch = types.ModuleType("torch")
    torch.get_rng_state = lambda: RngTensor(torch_rng.getstate())
    torch.set_rng_state = lambda state: torch_rng.setstate(state.value)
    platform = types.SimpleNamespace(
        device_type="cuda",
        get_rng_state=Mock(side_effect=lambda: RngTensor(device_rng.getstate())),
        set_rng_state=Mock(side_effect=lambda state: device_rng.setstate(state.value)),
    )
    platforms = types.ModuleType("areal.infra.platforms")
    platforms.current_platform = platform
    for name, module in (
        ("numpy", numpy),
        ("torch", torch),
        ("areal.infra.platforms", platforms),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    original_python = random.getstate()
    yield platform, (numpy_rng, torch_rng, device_rng)
    random.setstate(original_python)


def test_rng_roundtrip_repeats_all_registered_streams(rng_dependencies):
    platform, generators = rng_dependencies
    saved = M.capture_rng_state()
    expected = [random.random(), *(rng.random() for rng in generators)]
    for rng in generators:
        rng.random()
    M.restore_rng_state(saved)
    actual = [random.random(), *(rng.random() for rng in generators)]
    assert actual == expected
    platform.get_rng_state.assert_called_once_with()
    platform.set_rng_state.assert_called_once()


def test_host_rng_is_deferred_and_does_not_touch_accelerator(rng_dependencies):
    platform, generators = rng_dependencies
    host = M.ProcessRngState()
    saved = host.state_dict()
    expected = [random.random(), generators[0].random(), generators[1].random()]
    host.load_state_dict(saved)
    random.random()  # e.g. rollout load and weight sync consume randomness
    for rng in generators:
        rng.random()
    device_before_finalize = generators[2].getstate()
    host.finalize_restore()
    assert [random.random(), generators[0].random(), generators[1].random()] == expected
    assert generators[2].getstate() == device_before_finalize
    platform.get_rng_state.assert_not_called()
    platform.set_rng_state.assert_not_called()


def test_device_rng_mismatch_fails_before_changing_host_streams(rng_dependencies):
    platform, _ = rng_dependencies
    saved = M.capture_rng_state()
    random.random()
    current_python = random.getstate()
    platform.device_type = "npu"
    with pytest.raises(M.SnapshotError, match="device type"):
        M.restore_rng_state(saved)
    assert random.getstate() == current_python
    platform.set_rng_state.assert_not_called()


@pytest.fixture
def runtime_controller(wait_methods):
    owner = _controller(wait_methods, RpcTransport())
    owner.workers_is_dp_head = [True, False, True]
    return owner


def test_runtime_rpc_retains_all_ranks_and_disables_individual_retries(
    runtime_controller,
):
    packet = _packet()
    transport = RpcTransport({"worker/0": packet, "worker/2": packet})
    runtime_controller.scheduler = transport
    assert runtime_controller.runtime_state_dict() == packet
    assert len(transport.calls) == 3
    assert all(
        call[-1]
        == {"http_timeout": 7200.0, "max_retries": 1, "rpc_meta": {"broadcast": False}}
        for call in transport.calls
    )
    transport = RpcTransport()
    runtime_controller.scheduler = transport
    runtime_controller.load_runtime_state_dict(packet)
    assert [call[3] for call in transport.calls] == [(packet,), (packet,), (packet,)]


def test_runtime_rpc_disagreeing_dp_heads_cannot_publish(runtime_controller):
    runtime_controller.scheduler = RpcTransport(
        {"worker/0": _packet(), "worker/2": "different"}
    )
    with pytest.raises(RuntimeError, match="inconsistent runtime state"):
        runtime_controller.runtime_state_dict()


@pytest.mark.parametrize(
    "backend,cls", [("fsdp", "FSDPEngine"), ("megatron", "MegatronEngine")]
)
def test_backend_runtime_hooks_defer_rng_and_select_local_rank(backend, cls):
    """Exercise actual engine entry points without importing GPU dependencies."""
    engine = _engine(rank=2)
    engine._runtime_state_identity = lambda: IDENTITY
    collector = Mock(return_value=_packet())
    rng_restore = Mock()
    native = types.SimpleNamespace(
        get_rng_state=Mock(return_value=[("rng", 2)]), load_rng_states=Mock()
    )
    engine.checkpointer = native
    capture = Mock(return_value=("rng", 2))
    namespace = {
        "collect_runtime_state": collector,
        "capture_rng_state": capture,
        "load_runtime_scheduler": M.load_runtime_scheduler,
        "select_runtime_state": M.select_runtime_state,
        "restore_rng_state": rng_restore,
    }
    methods = {
        name: _source_function(
            f"areal/engine/{backend}_engine.py", cls, name, namespace
        )
        for name in (
            "runtime_state_dict",
            "load_runtime_state_dict",
            "restore_runtime_rng",
        )
    }
    assert methods["runtime_state_dict"](engine) == _packet()
    assert collector.call_args.kwargs["capture_rng"]() == ("rng", 2)
    if backend == "megatron":
        native.get_rng_state.assert_called_once_with(use_dist_ckpt=False)
    methods["load_runtime_state_dict"](engine, _packet())
    rng_restore.assert_not_called()
    native.load_rng_states.assert_not_called()
    methods["restore_runtime_rng"](engine, _packet())
    if backend == "megatron":
        native.load_rng_states.assert_called_once_with([("independent-rng", 2)])
    else:
        rng_restore.assert_called_once_with(("independent-rng", 2))


def _staged_snapshot(tmp_path, monkeypatch):
    manager, context, role, root = _snapshot_manager(tmp_path, lambda: None)
    manager._checkpoints["actor"] = role.binding()
    events = []
    engine = types.SimpleNamespace(
        runtime_state_dict=lambda: "packet",
        load_runtime_state_dict=lambda state: events.append("scheduler"),
        restore_runtime_rng=lambda state: events.append("worker_rng"),
    )
    monkeypatch.setattr(M, "capture_rng_state", lambda **kwargs: {"rng": 1})
    monkeypatch.setattr(M, "restore_rng_state", lambda state: events.append("host_rng"))
    host = M.ProcessRngState()
    rollout = Owner({"cursor": 7})
    rollout.load_state_dict = lambda state: events.append("rollout")
    # Trainer.bind appends rollout after host; finalization must STILL be last.
    manager._statefuls = {
        "engine": M.EngineRuntimeState(engine),
        "host": host,
        "rollout": rollout,
    }
    ref = manager.save_snapshot(root, context=context)
    binding = manager._checkpoints["actor"]

    def load_model(path):
        binding.load(path)
        events.append("model")

    manager._checkpoints["actor"] = dataclasses.replace(binding, load=load_model)
    return manager, ref, context, events


def test_restore_syncs_weights_then_loads_owners_then_finalizes_rngs(
    tmp_path, monkeypatch
):
    manager, ref, context, events = _staged_snapshot(tmp_path, monkeypatch)

    def sync(saved):
        assert saved == context
        events.append("weight_sync")

    manager.restore_snapshot(ref, expected_context=context, after_checkpoints=sync)
    assert events == [
        "model",
        "weight_sync",
        "scheduler",
        "rollout",
        "worker_rng",
        "host_rng",
    ]


@pytest.mark.parametrize("stage", ["sync", "scheduler", "rng"])
def test_restore_stage_failure_poison_participants_and_skips_later_stages(
    tmp_path, monkeypatch, stage
):
    manager, ref, context, events = _staged_snapshot(tmp_path, monkeypatch)
    fail = Mock(side_effect=RuntimeError("injected restore failure"))
    sync = fail if stage == "sync" else None
    engine = manager._statefuls["engine"].engine
    if stage == "scheduler":
        engine.load_runtime_state_dict = fail
    if stage == "rng":
        engine.restore_runtime_rng = fail
    with pytest.raises(M.SnapshotError, match="partially restored"):
        manager.restore_snapshot(ref, expected_context=context, after_checkpoints=sync)
    assert "host_rng" not in events
    fail.assert_called_once()
    with pytest.raises(M.SnapshotError, match="preflight failed"):
        manager.restore_snapshot(ref, expected_context=context)
    fail.assert_called_once()


def test_async_sync_callback_rejected_before_loading_models(tmp_path, monkeypatch):
    manager, ref, context, events = _staged_snapshot(tmp_path, monkeypatch)

    async def async_sync(saved):
        pass

    with pytest.raises(M.SnapshotError, match="preflight failed"):
        manager.restore_snapshot(
            ref, expected_context=context, after_checkpoints=async_sync
        )
    assert events == []
    assert not manager._restore_failed


def test_trainer_runtime_capture_failure_never_publishes(monkeypatch, tmp_path):
    trainer, events = _trainer_fixture(monkeypatch)
    snapshot_channel = M.TrainerReconfiguration(
        run_id="trainer-run",
        runtime_revision="source-revision",
        configuration_id="source-config",
    )
    trainer.actor.runtime_state_dict = Mock(
        side_effect=M.SnapshotError("rank 1 capture failed")
    )
    root = tmp_path / "snapshot"
    future = snapshot_channel.request_snapshot(root, state_id="boundary")
    with pytest.raises(M.SnapshotError):
        trainer.train(workflow=object(), reconfiguration=snapshot_channel)
    assert future.exception() is not None
    assert not (root / "manifest.json").exists()
    assert "rollout.resume" not in events


def test_native_torch_scheduler_and_host_rng_continuity():
    """When torch is available, verify next LR and actual next random draws."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    original = M.capture_rng_state(include_device=False)
    try:
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        state = M._encode_runtime_state(
            {
                "version": 1,
                "ranks": [
                    M._encode_runtime_state(
                        {
                            "rank": 0,
                            "identity": IDENTITY,
                            "scheduler": scheduler.state_dict(),
                            "lrs": [optimizer.param_groups[0]["lr"]],
                            "rng": None,
                        }
                    )
                ],
            }
        )
        saved_rng = M.capture_rng_state(include_device=False)
        expected = (random.random(), np.random.rand(), torch.rand(4))
        optimizer.step()
        scheduler.step()
        expected_lr = optimizer.param_groups[0]["lr"]
        restored_optimizer = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(1))], lr=0.5
        )
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
            restored_optimizer, lambda step: 0.9**step
        )
        engine = types.SimpleNamespace(
            rank=0,
            world_size=1,
            optimizer=restored_optimizer,
            lr_scheduler=restored_scheduler,
        )
        M.load_runtime_scheduler(engine, state, identity=IDENTITY)
        restored_optimizer.step()
        restored_scheduler.step()
        assert restored_optimizer.param_groups[0]["lr"] == expected_lr
        M.restore_rng_state(saved_rng)
        assert random.random() == expected[0]
        assert np.random.rand() == expected[1]
        torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
    finally:
        M.restore_rng_state(original)
