import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from areal.platforms import current_platform
from areal.utils import perf_tracer
from areal.utils.network import find_free_ports


@pytest.fixture(autouse=True)
def clean_global_tracer():
    tracer = perf_tracer.get_tracer()
    prev_config = {
        "enabled": tracer.enabled,
        "output_path": tracer._output_path,  # noqa: SLF001
        "rank": tracer._rank,  # noqa: SLF001
        "aggregate": tracer._aggregate,  # noqa: SLF001
        "user_output_path": tracer._user_output_path,  # noqa: SLF001
    }

    tracer.reset()
    yield tracer
    perf_tracer.configure(
        enabled=prev_config["enabled"],
        rank=prev_config["rank"],
        aggregate=prev_config["aggregate"],
    )
    tracer.reset()
    user_path = prev_config["user_output_path"]
    if user_path:
        tracer.set_output(user_path, rank=prev_config["rank"])
    else:
        tracer._user_output_path = None  # noqa: SLF001
        tracer._output_path = prev_config["output_path"]  # noqa: SLF001


@pytest.mark.parametrize("use_rank_suffix", [None, 0])
def test_perf_tracer_records_events_and_save(tmp_path, use_rank_suffix):
    tracer = perf_tracer.PerfTracer(enabled=True)
    output_file = tmp_path / "trace.json"
    tracer.set_output(str(output_file), rank=use_rank_suffix)

    with tracer.trace_scope("unit-block", category="unit", args={"step": 1}):
        tracer.instant("inner-mark", args={"value": 42})
    tracer.instant("outer-mark")

    saved = tracer.save(reset=False)
    assert saved is not None
    saved_path = Path(saved)
    expected_path = (
        output_file
        if use_rank_suffix is None
        else output_file.with_name("trace.rank0.json")
    )
    assert saved_path == expected_path

    payload = json.loads(saved_path.read_text())
    event_names = {evt["name"] for evt in payload["traceEvents"] if evt["ph"] != "M"}
    assert {"unit-block", "inner-mark", "outer-mark"}.issubset(event_names)

    # Ensure reset clears cached events and re-bases timestamps
    tracer.reset()
    assert tracer._events == []  # noqa: SLF001


def test_perf_tracer_aggregate_combines_ranks(tmp_path):
    output_path = tmp_path / "trace.json"

    tracer0 = perf_tracer.PerfTracer(enabled=True, rank=0, aggregate=True)
    tracer0.set_output(str(output_path))
    with tracer0.trace_scope("rank0-step", args={"rank": 0}):
        pass
    tracer0.instant("rank0-mark", args={"rank": 0})
    tracer0.save()

    tracer1 = perf_tracer.PerfTracer(enabled=True, rank=1, aggregate=True)
    tracer1.set_output(str(output_path))
    tracer1._pid = tracer0._pid + 1  # noqa: SLF001 - simulate distinct process id
    tracer1_thread = getattr(tracer1, "_thread_meta_emitted", set())
    tracer1_thread.clear()
    tracer1.instant("rank1-mark", args={"rank": 1})
    tracer1.save()

    payload = json.loads(output_path.read_text())
    event_names = {evt["name"] for evt in payload["traceEvents"] if evt["ph"] != "M"}
    assert {"rank0-step", "rank0-mark", "rank1-mark"}.issubset(event_names)
    pid_values = {evt["pid"] for evt in payload["traceEvents"] if evt["ph"] != "M"}
    assert pid_values == {tracer0._pid, tracer1._pid}  # noqa: SLF001
    rank_values = {
        evt["args"].get("rank") for evt in payload["traceEvents"] if evt["ph"] != "M"
    }
    assert {0, 1}.issubset(rank_values)
    meta_by_pid = {
        (evt["pid"], evt["args"].get("name"))
        for evt in payload["traceEvents"]
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
        for evt in payload["traceEvents"]
        if evt["ph"] == "M" and evt["name"] == "process_sort_index"
    }
    assert sort_meta[tracer0._pid] == 0  # noqa: SLF001
    assert sort_meta[tracer1._pid] == 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_global_tracer_configure_roundtrip(tmp_path):
    tracer = perf_tracer.get_tracer()

    tracer.reset()

    perf_tracer.configure(
        enabled=True,
        output_path=str(tmp_path / "global_trace.json"),
        rank=1,
    )

    async with perf_tracer.atrace_scope(
        "async-step", category="unit-test", args={"phase": "enter"}
    ):
        perf_tracer.instant("inside-async", args={"flag": True})

    with perf_tracer.trace_scope("sync-step", category="unit-test"):
        pass

    saved = tracer.save(reset=True)
    assert saved is not None
    saved_path = Path(saved)
    payload = json.loads(saved_path.read_text())
    event_names = {evt["name"] for evt in payload["traceEvents"] if evt["ph"] != "M"}
    assert {"async-step", "inside-async", "sync-step"}.issubset(event_names)


@pytest.mark.asyncio
async def test_async_multi_request_cross_phase_trace(tmp_path):
    tracer = perf_tracer.get_tracer()

    tracer.reset()

    output_path = tmp_path / "async_requests.json"
    perf_tracer.configure(
        enabled=True,
        output_path=str(output_path),
        rank=None,
        aggregate=False,
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

    saved = tracer.save(reset=True)
    assert saved is not None
    saved_path = Path(saved)
    payload = json.loads(saved_path.read_text())
    events = [evt for evt in payload["traceEvents"] if evt.get("ph") != "M"]

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


def test_configure_updates_output_path_when_rank_changes(tmp_path):
    tracer = perf_tracer.get_tracer()

    tracer.reset()

    base_path = tmp_path / "ranked.json"
    perf_tracer.configure(
        enabled=True,
        output_path=str(base_path),
        rank=0,
        aggregate=False,
    )
    first_path = Path(tracer._output_path or "")
    assert first_path.name == "ranked.rank0.json"

    perf_tracer.configure(rank=1)
    second_path = Path(tracer._output_path or "")
    assert second_path.name == "ranked.rank1.json"


def test_module_level_save_helper(tmp_path):
    tracer = perf_tracer.get_tracer()

    tracer.reset()
    tracer._rank = None  # noqa: SLF001

    output_path = tmp_path / "module_trace.json"
    perf_tracer.configure(
        enabled=True,
        output_path=str(output_path),
        rank=None,
        aggregate=False,
    )
    perf_tracer.instant("module-level-mark", args={"flag": True})

    saved = perf_tracer.save(reset=True)
    assert saved is not None
    saved_path = Path(saved)
    assert saved_path.exists()
    payload = json.loads(saved_path.read_text())
    event_names = {
        evt["name"] for evt in payload["traceEvents"] if evt.get("ph") != "M"
    }
    assert "module-level-mark" in event_names


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

    trace_path = tmp_path / "trace.json"
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text())
    ranks_seen = {
        evt["args"].get("rank")
        for evt in payload["traceEvents"]
        if evt["name"] == "torchrun-step"
    }
    assert ranks_seen == set(range(world_size))
    mark_ranks = {
        evt["args"].get("rank")
        for evt in payload["traceEvents"]
        if evt["name"] == "torchrun-mark"
    }
    assert mark_ranks == set(range(world_size))
