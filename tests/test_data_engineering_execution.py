# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, replace

import pytest

from areal.v2.data_engineering import (
    CalibrationConfig,
    DataJobConfig,
    ExecutionConfig,
    run_data_job,
)


def config(directory, **kwargs):
    return DataJobConfig(
        "tests.data_engineering_parallel_fixtures:create_recipe",
        8,
        params={"directory": str(directory)},
        execution=ExecutionConfig(batch_size=1),
        **kwargs,
    )


def calibration(directory, **params):
    return CalibrationConfig(
        "tests.data_engineering_parallel_fixtures:create_calibrator",
        params={"directory": str(directory), **params},
    )


def calls(directory):
    path = directory / "calls.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["thread", "process"])
async def test_parallel_workers_overlap_and_keep_output_order(tmp_path, backend):
    """Real local workers overlap; completion timing never chooses dataset order."""
    cfg = replace(
        config(tmp_path),
        params={"directory": str(tmp_path), "delay": 0.03},
        execution=ExecutionConfig(max_workers=2, batch_size=1, backend=backend),
    )
    snapshot = await run_data_job(cfg, tmp_path / "job")
    assert [row["task_id"] for row in snapshot.training_rows()] == [
        str(i) for i in range(8)
    ]
    active = peak = 0
    for call in calls(tmp_path):
        if call["phase"] == "clean.start":
            active += 1
            peak = max(peak, active)
        elif call["phase"] == "clean.end":
            active -= 1
    assert peak == 2
    if backend == "process":
        assert (
            len(
                {
                    call["pid"]
                    for call in calls(tmp_path)
                    if call["phase"] == "clean.start"
                }
            )
            == 2
        )
        assert all(
            call["pid"] != os.getpid()
            for call in calls(tmp_path)
            if call["phase"] == "clean.start"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["clean", "training", "calibration"])
async def test_resume_reuses_committed_chunks_and_matches_fresh_result(tmp_path, phase):
    """Storage/provider failures resume in place without repeating committed work."""
    cfg = replace(
        config(
            tmp_path,
            calibration=calibration(
                tmp_path, fail_task="3" if phase == "calibration" else "absent"
            ),
        ),
        params={"directory": str(tmp_path), "fail_phase": phase},
    )
    (tmp_path / "fail").touch()
    with pytest.raises(OSError, match="Injected"):
        await run_data_job(cfg, tmp_path / "job")
    assert not (tmp_path / "job/dataset").exists()
    (tmp_path / "fail").unlink()
    # Clean/training faults use their own phase; calibration only faults in that case.
    resumed = await run_data_job(cfg, tmp_path / "job", resume=True)
    done = Counter((c["phase"], c["task_id"]) for c in calls(tmp_path))
    assert done[phase + ".start", "0"] == 1
    assert done[phase + ".start", "3"] == 2
    before = calls(tmp_path)
    same = await run_data_job(cfg, tmp_path / "job", resume=True)
    assert same.dataset_id == resumed.dataset_id
    assert calls(tmp_path) == before
    fresh = await run_data_job(cfg, tmp_path / "fresh")
    assert fresh.dataset_id == resumed.dataset_id


@pytest.mark.asyncio
async def test_calibration_concurrency_is_bounded(tmp_path):
    """Async model work reaches the configured limit without unbounded task creation."""
    cfg = replace(
        config(tmp_path, calibration=calibration(tmp_path, delay=0.02)),
        execution=ExecutionConfig(max_workers=3, batch_size=1),
    )
    snapshot = await run_data_job(cfg, tmp_path / "job")
    assert len(snapshot.training_rows()) == 8
    active = peak = 0
    for call in calls(tmp_path):
        if call["phase"] == "calibration.start":
            active += 1
            peak = max(peak, active)
        elif call["phase"] == "calibration.end":
            active -= 1
    assert peak == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid, failures, expected", [(False, 2, 10), (False, 3, 3), (True, 0, 1)]
)
async def test_transient_retries_are_bounded_and_validation_is_not_retried(
    tmp_path, invalid, failures, expected
):
    """Only transient exceptions retry; exhaustion retains a resumable failed job."""
    cfg = replace(
        config(
            tmp_path,
            calibration=calibration(
                tmp_path, transient_failures=failures, invalid=invalid
            ),
        ),
        execution=ExecutionConfig(batch_size=1, max_retries=2, retry_delay=0),
    )
    if invalid or failures == 3:
        with pytest.raises(ValueError if invalid else OSError):
            await run_data_job(cfg, tmp_path / "job")
    else:
        await run_data_job(cfg, tmp_path / "job")
    assert sum(c["phase"] == "calibration.start" for c in calls(tmp_path)) == expected


@pytest.mark.asyncio
async def test_resume_rejects_changed_config_sources_and_corrupt_checkpoint(tmp_path):
    """Recovery validates job identity before modifying terminal state."""
    source = tmp_path / "source"
    source.write_text("original")
    cfg = config(tmp_path, inputs={"raw": str(source)})
    (tmp_path / "fail").touch()
    with pytest.raises(OSError):
        await run_data_job(cfg, tmp_path / "job")
    state = (tmp_path / "job/job.json").read_bytes()
    with pytest.raises(ValueError, match="config changed"):
        await run_data_job(replace(cfg, seed=1), tmp_path / "job", resume=True)
    source.write_text("changed")
    with pytest.raises(ValueError, match="sources changed"):
        await run_data_job(cfg, tmp_path / "job", resume=True)
    assert (tmp_path / "job/job.json").read_bytes() == state
    source.write_text("original")
    with sqlite3.connect(tmp_path / "job/work/checkpoints.sqlite3") as db:
        db.execute("UPDATE checkpoints SET value='{}' WHERE key='binding'")
    with pytest.raises(ValueError, match="digest mismatch"):
        await run_data_job(cfg, tmp_path / "job", resume=True)


@pytest.mark.asyncio
async def test_same_job_cannot_have_two_coordinators(tmp_path):
    """An active job cannot be resumed concurrently, and cancellation releases its lock."""
    cfg = config(tmp_path, calibration=calibration(tmp_path, fail_task="0"))
    (tmp_path / "block").touch()
    task = asyncio.create_task(run_data_job(cfg, tmp_path / "job"))
    try:
        async with asyncio.timeout(10):
            while not any(c["phase"] == "calibration.start" for c in calls(tmp_path)):
                await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match="already running"):
            await run_data_job(cfg, tmp_path / "job", resume=True)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        (tmp_path / "block").unlink()
    snapshot = await run_data_job(cfg, tmp_path / "job", resume=True)
    assert len(snapshot.training_rows()) == 8


@pytest.mark.asyncio
async def test_publication_failure_resumes_without_repeating_any_execution(
    tmp_path, monkeypatch
):
    """An interrupted snapshot write preserves completed projections and probes."""
    from areal.v2.data_engineering import snapshot as module

    original = module.write_file

    def fail(path, data):
        if path.name == "train.jsonl":
            raise OSError("Snapshot write failed")
        original(path, data)

    cfg = config(tmp_path, calibration=calibration(tmp_path))
    monkeypatch.setattr(module, "write_file", fail)
    with pytest.raises(OSError, match="Snapshot write"):
        await run_data_job(cfg, tmp_path / "job")
    before = calls(tmp_path)
    monkeypatch.setattr(module, "write_file", original)
    await run_data_job(cfg, tmp_path / "job", resume=True)
    assert calls(tmp_path) == before


def test_module_recovers_after_sigkill(tmp_path):
    """Actual process death leaves WAL checkpoints usable by a new coordinator."""
    cfg = config(tmp_path, calibration=calibration(tmp_path))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(cfg)))
    (tmp_path / "block").touch()
    command = [
        sys.executable,
        "-m",
        "areal.v2.data_engineering",
        "--config",
        str(path),
        "--output",
        str(tmp_path / "job"),
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 30
        while not any(
            c["phase"] == "calibration.start" and c["task_id"] == "3"
            for c in calls(tmp_path)
        ):
            if process.poll() is not None:
                pytest.fail(process.stderr.read().decode())
            if time.monotonic() > deadline:
                pytest.fail("Worker did not reach interruption point")
            time.sleep(0.02)
        process.kill()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stderr.close()
        (tmp_path / "block").unlink()
    result = subprocess.run(
        command + ["--resume"], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    counts = Counter((c["phase"], c["task_id"]) for c in calls(tmp_path))
    assert counts["calibration.start", "0"] == 1
    assert counts["calibration.start", "3"] == 2
    assert json.loads((tmp_path / "job/job.json").read_text())["status"] == "succeeded"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_workers": 0},
        {"batch_size": True},
        {"backend": "ray"},
        {"max_retries": -1},
        {"retry_delay": float("nan")},
        {"retry_delay": True},
    ],
)
def test_invalid_execution_config_fails_before_start(kwargs):
    """Worker limits and retry budgets cannot create invalid execution plans."""
    with pytest.raises(ValueError):
        ExecutionConfig(**kwargs)


@pytest.mark.asyncio
async def test_cleaning_retry_starts_from_original_raw_values(tmp_path):
    """A failed in-place normalization must not compound on the retry input."""
    (tmp_path / "fail").touch()
    cfg = replace(
        config(tmp_path),
        recipe="tests.data_engineering_parallel_fixtures:mutating_recipe",
        execution=ExecutionConfig(batch_size=2, max_retries=1, retry_delay=0),
    )
    snapshot = await run_data_job(cfg, tmp_path / "job")
    assert all(row["value"] == row["answer"] + 1 for row in snapshot.training_rows())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    [
        "generate",
        "score",
        "select",
        "checkpoint",
        "publish",
        "verify",
        "sources",
        "calibration",
    ],
)
async def test_sync_job_stages_leave_callers_event_loop_responsive(
    tmp_path, monkeypatch, phase
):
    """Blocking plugins and storage must not stall another task on the caller loop."""
    from tests.data_engineering_parallel_fixtures import (
        ParallelCalibrator,
        ParallelRecipe,
    )

    from areal.v2.data_engineering import runner
    from areal.v2.data_engineering.storage import WorkStore

    entered, release, stalled = threading.Event(), threading.Event(), threading.Event()
    cfg = config(tmp_path, calibration=calibration(tmp_path))
    output = tmp_path / "job"
    if phase == "verify":
        await run_data_job(cfg, output)
    owner, name = {
        "generate": (ParallelRecipe, "generate"),
        "score": (ParallelRecipe, "score"),
        "select": (ParallelRecipe, "select"),
        "checkpoint": (WorkStore, "retain"),
        "publish": (runner, "publish_snapshot"),
        "verify": (runner, "load_snapshot"),
        "sources": (runner, "_sources"),
        "calibration": (ParallelCalibrator, "__call__"),
    }[phase]
    original = getattr(owner, name)

    def gate():
        entered.set()
        if not release.wait(1):
            stalled.set()

    def blocking(*args, **kwargs):
        gate()
        return original(*args, **kwargs)

    async def blocking_calibration(*args, **kwargs):
        gate()
        return await original(*args, **kwargs)

    monkeypatch.setattr(
        owner, name, blocking_calibration if phase == "calibration" else blocking
    )
    job = asyncio.create_task(run_data_job(cfg, output, resume=phase == "verify"))
    try:
        assert await asyncio.to_thread(entered.wait, 5), (
            "Job never reached the blocked stage"
        )
        assert not stalled.is_set(), (
            "Caller loop could not release the blocked job stage"
        )
        assert not job.done()
        await asyncio.sleep(0)
    finally:
        release.set()
        await job
    assert not stalled.is_set()


@pytest.mark.asyncio
async def test_cancellation_during_sync_generation_waits_for_cleanup_and_resumes(
    tmp_path, monkeypatch
):
    """Cancellation stays responsive, survives repeated cancel(), and releases the job lock."""
    from tests.data_engineering_parallel_fixtures import ParallelRecipe

    entered, release = threading.Event(), threading.Event()
    original = ParallelRecipe.generate

    def blocked_generate(self, context):
        entered.set()
        assert release.wait(5), "Test did not release the synchronous plugin"
        return original(self, context)

    monkeypatch.setattr(ParallelRecipe, "generate", blocked_generate)
    cfg = config(tmp_path, calibration=calibration(tmp_path))
    job = asyncio.create_task(run_data_job(cfg, tmp_path / "job"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        job.cancel()
        await asyncio.sleep(0.01)
        assert not job.done(), (
            "Cancellation returned before the job released its resources"
        )
        job.cancel()
        await asyncio.sleep(0.01)
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await job
    assert json.loads((tmp_path / "job/job.json").read_text())["status"] == "cancelled"
    assert not (tmp_path / "job/dataset").exists()
    assert not any(c["phase"] == "calibration.start" for c in calls(tmp_path))
    snapshot = await run_data_job(cfg, tmp_path / "job", resume=True)
    assert len(snapshot.training_rows()) == 8
