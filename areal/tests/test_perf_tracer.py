import asyncio
import getpass
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from areal.api.cli_args import PerfTracerConfig
from areal.platforms import current_platform
from areal.utils import perf_tracer
from areal.utils.network import find_free_ports
from areal.utils.perf_tracer import Category


def _make_config(
    root: Path,
    *,
    enabled: bool = True,
    experiment: str = "test-exp",
    trial: str = "trial",
) -> PerfTracerConfig:
    return PerfTracerConfig(
        experiment_name=experiment,
        trial_name=trial,
        fileroot=str(root),
        enabled=enabled,
    )


def _expected_trace_path(config: PerfTracerConfig) -> Path:
    base_dir = Path(os.path.expanduser(config.fileroot))
    return (
        base_dir
        / "logs"
        / getpass.getuser()
        / config.experiment_name
        / config.trial_name
        / "traces.jsonl"
    )


def _load_trace_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(autouse=True)
def clean_global_tracer():
    perf_tracer.reset()
    yield
    perf_tracer.reset()


def test_module_level_helpers_require_configuration():
    with pytest.raises(RuntimeError):
        perf_tracer.get_tracer()


@pytest.mark.parametrize("override_rank", [None, 1])
def test_perf_tracer_records_events_and_save(tmp_path, override_rank):
    config = _make_config(tmp_path, experiment="unit", trial="scope")
    base_rank = 0
    tracer = perf_tracer.PerfTracer(config, rank=base_rank)
    if override_rank is None:
        expected_rank = base_rank
    else:
        tracer.set_rank(override_rank)
        expected_rank = override_rank

    assert tracer._rank == expected_rank  # noqa: SLF001

    with tracer.trace_scope(
        "unit-block",
        category=Category.INSTR,
        args={"step": 1},
    ):
        tracer.instant("inner-mark", args={"value": 42})
    tracer.instant("outer-mark")

    tracer.save()
    saved_path = _expected_trace_path(config)
    assert saved_path.exists()

    events = _load_trace_events(saved_path)
    event_names = {evt["name"] for evt in events if evt["ph"] != "M"}
    assert {"unit-block", "inner-mark", "outer-mark"}.issubset(event_names)


def test_perf_tracer_aggregate_combines_ranks(tmp_path):
    config0 = _make_config(
        tmp_path,
        enabled=True,
        experiment="agg",
        trial="shared",
    )
    tracer0 = perf_tracer.PerfTracer(config0, rank=0)
    with tracer0.trace_scope("rank0-step", args={"rank": 0}):
        pass
    tracer0.instant("rank0-mark", args={"rank": 0})
    tracer0.save()
    saved_path = _expected_trace_path(config0)
    assert saved_path.exists()

    config1 = _make_config(
        tmp_path,
        enabled=True,
        experiment="agg",
        trial="shared",
    )
    tracer1 = perf_tracer.PerfTracer(config1, rank=1)
    tracer1._pid = tracer0._pid + 1  # noqa: SLF001 - simulate distinct process id
    tracer1_thread = getattr(tracer1, "_thread_meta_emitted", set())
    tracer1_thread.clear()
    tracer1.instant("rank1-mark", args={"rank": 1})
    tracer1.save()
    saved_path_rank1 = _expected_trace_path(config1)
    assert saved_path_rank1 == saved_path

    events = _load_trace_events(saved_path)
    event_names = {evt["name"] for evt in events if evt["ph"] != "M"}
    assert {"rank0-step", "rank0-mark", "rank1-mark"}.issubset(event_names)
    pid_values = {evt["pid"] for evt in events if evt["ph"] != "M"}
    assert pid_values == {tracer0._pid, tracer1._pid}  # noqa: SLF001
    rank_values = {evt["args"].get("rank") for evt in events if evt["ph"] != "M"}
    assert {0, 1}.issubset(rank_values)
    meta_by_pid = {
        (evt["pid"], evt["args"].get("name"))
        for evt in events
        if evt["ph"] == "M" and evt["name"] == "process_name"
    }
    assert (
        tracer0._pid,
        "Rank 0, Process",
    ) in meta_by_pid  # noqa: SLF001
    assert (
        tracer1._pid,
        "Rank 1, Process",
    ) in meta_by_pid  # noqa: SLF001
    sort_meta = {
        evt["pid"]: evt["args"].get("sort_index")
        for evt in events
        if evt["ph"] == "M" and evt["name"] == "process_sort_index"
    }
    assert sort_meta[tracer0._pid] == 0  # noqa: SLF001
    assert sort_meta[tracer1._pid] == 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_global_tracer_configure_roundtrip(tmp_path):
    config = _make_config(tmp_path, experiment="global", trial="roundtrip")
    tracer = perf_tracer.configure(
        config,
        rank=0,
    )

    async with perf_tracer.atrace_scope(
        "async-step",
        category=Category.INSTR,
        args={"phase": "enter"},
    ):
        perf_tracer.instant("inside-async", args={"flag": True})

    with perf_tracer.trace_scope(
        "sync-step",
        category=Category.INSTR,
    ):
        pass

    tracer.save()
    saved_path = _expected_trace_path(config)
    assert saved_path.exists()
    events = _load_trace_events(saved_path)
    event_names = {evt["name"] for evt in events if evt["ph"] != "M"}
    assert {"async-step", "inside-async", "sync-step"}.issubset(event_names)


@pytest.mark.asyncio
async def test_async_multi_request_cross_phase_trace(tmp_path):
    config = _make_config(tmp_path, experiment="async", trial="requests")
    tracer = perf_tracer.configure(
        config,
        rank=0,
    )

    phase_schedules = {
        "req-0": [
            ("rollout", 0.1),
            ("train", 0.2),
            ("reward", 0.15),
        ],
        "req-1": [
            ("rollout", 0.12),
            ("train", 0.11),
            ("reward", 0.18),
        ],
    }

    async def run_request(
        req_id: str, phases: list[tuple[str, float]], offset: float
    ) -> None:
        if offset:
            await asyncio.sleep(offset)
        loop = asyncio.get_running_loop()
        async with perf_tracer.atrace_scope(
            "request", category=f"request.{req_id}", args={"request_id": req_id}
        ):
            for phase, duration in phases:
                with perf_tracer.trace_scope(
                    phase,
                    category=f"{phase}.{req_id}",
                    args={"request_id": req_id},
                ):
                    await loop.run_in_executor(None, time.sleep, duration)

    await asyncio.gather(
        run_request("req-0", phase_schedules["req-0"], 0.0),
        run_request("req-1", phase_schedules["req-1"], 0.05),
    )

    tracer.save()
    saved_path = _expected_trace_path(config)
    assert saved_path.exists()
    events = [evt for evt in _load_trace_events(saved_path) if evt.get("ph") != "M"]

    observed = {
        (evt["args"].get("request_id"), evt["name"])
        for evt in events
        if evt["ph"] == "X"
    }
    expected_phases = {
        ("req-0", "request"),
        ("req-0", "rollout"),
        ("req-0", "train"),
        ("req-0", "reward"),
        ("req-1", "request"),
        ("req-1", "rollout"),
        ("req-1", "train"),
        ("req-1", "reward"),
    }
    assert expected_phases.issubset(observed)

    request_timestamps = {
        req_id: [evt["ts"] for evt in events if evt["args"].get("request_id") == req_id]
        for req_id in phase_schedules
    }
    overlap = min(request_timestamps["req-1"]) < max(
        request_timestamps["req-0"]
    ) and min(request_timestamps["req-0"]) < max(request_timestamps["req-1"])
    assert overlap


def test_configure_preserves_output_path_when_rank_changes(tmp_path):
    config = _make_config(tmp_path, experiment="ranked", trial="zero")
    tracer = perf_tracer.configure(
        config,
        rank=0,
    )
    expected_path = _expected_trace_path(config)
    first_path = Path(tracer._output_path or "")  # noqa: SLF001
    assert first_path == expected_path

    perf_tracer.configure(
        config,
        rank=1,
    )
    second_path = Path(tracer._output_path or "")  # noqa: SLF001
    assert second_path == expected_path
    assert tracer._rank == 1  # noqa: SLF001


def test_module_level_save_helper(tmp_path):
    config = _make_config(tmp_path, experiment="module", trial="helper")
    perf_tracer.configure(
        config,
        rank=0,
    )
    perf_tracer.instant("module-level-mark", args={"flag": True})

    perf_tracer.save()
    saved_path = _expected_trace_path(config)
    assert saved_path.exists()
    assert saved_path == _expected_trace_path(config)
    events = _load_trace_events(saved_path)
    event_names = {evt["name"] for evt in events if evt.get("ph") != "M"}
    assert "module-level-mark" in event_names


def test_perf_tracer_respects_save_interval(tmp_path):
    config = _make_config(tmp_path, experiment="interval", trial="steps")
    config.save_interval_steps = 3
    tracer = perf_tracer.PerfTracer(config, rank=0)
    trace_path = _expected_trace_path(config)

    for step in (0, 1):
        tracer.instant(f"mark-{step}", args={"step": step})
        tracer.save(step=step)
        assert not trace_path.exists()

    tracer.instant("mark-2", args={"step": 2})
    tracer.save(step=2)
    assert trace_path.exists()
    events = _load_trace_events(trace_path)
    names = {evt["name"] for evt in events if evt.get("ph") != "M"}
    assert {"mark-0", "mark-1", "mark-2"}.issubset(names)

    tracer.instant("mark-3", args={"step": 3})
    tracer.save(step=3)
    tracer.instant("mark-4", args={"step": 4})
    tracer.save(step=4)
    tracer.save(force=True)

    events = _load_trace_events(trace_path)
    names = {evt["name"] for evt in events if evt.get("ph") != "M"}
    assert {"mark-3", "mark-4"}.issubset(names)


def _run_perf_tracer_torchrun(tmp_path: Path, world_size: int) -> None:
    port = find_free_ports(1)[0]
    env = {
        **os.environ,
        "AREAL_PERF_TRACE_BASE": str(tmp_path),
    }
    subprocess.run(
        [
            "torchrun",
            f"--nproc_per_node={world_size}",
            "--nnodes=1",
            "--master-addr=localhost",
            f"--master_port={port}",
            "areal/tests/torchrun/run_perf_tracer.py",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_perf_tracer_torchrun_multi_rank(tmp_path, world_size):
    device_count_fn = getattr(current_platform, "device_count", None)
    available_devices = device_count_fn() if callable(device_count_fn) else 0
    if available_devices < world_size:
        pytest.skip(f"This test requires {world_size} gpus")

    _run_perf_tracer_torchrun(tmp_path, world_size)

    config = PerfTracerConfig(
        experiment_name="torchrun",
        trial_name=f"world-{world_size}",
        fileroot=str(tmp_path),
        enabled=True,
    )
    trace_path = _expected_trace_path(config)
    assert trace_path.exists()
    payload = _load_trace_events(trace_path)
    ranks_seen = {
        evt["args"].get("rank") for evt in payload if evt["name"] == "torchrun-step"
    }
    assert ranks_seen == set(range(world_size))
    mark_ranks = {
        evt["args"].get("rank") for evt in payload if evt["name"] == "torchrun-mark"
    }
    assert mark_ranks == set(range(world_size))
