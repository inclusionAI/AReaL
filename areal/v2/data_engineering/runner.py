# SPDX-License-Identifier: Apache-2.0

"""Bounded recipe execution with public-only calibration and atomic publication."""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import math
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from areal.utils import logging

from .snapshot import (
    DatasetSnapshot,
    atomic_write_json,
    digest,
    json_bytes,
    publish_snapshot,
    read_jsonl,
    write_file,
)
from .types import (
    Attempt,
    CalibrationBatch,
    CalibrationContext,
    Calibrator,
    DataJobConfig,
    DataRecipe,
    DataSample,
    InsufficientDataError,
    RecipeContext,
    Selection,
)

logger = logging.getLogger("DataEngineeringRunner")


def _load_factory(spec: str, params: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    module_name, separator, name = spec.partition(":")
    if not separator or not module_name or not name or ":" in name:
        raise ValueError(f"Expected module:callable, got {spec!r}")
    factory = getattr(importlib.import_module(module_name), name)
    source = inspect.getsourcefile(factory)
    if source is None:
        raise ValueError(f"Factory needs an auditable Python source file: {spec}")
    instance = factory(**copy.deepcopy(params))
    return instance, {
        "implementation": spec,
        "source_digest": digest(Path(source).read_bytes()),
    }


def _sources(config: DataJobConfig, recipe: Any, calibrator: Any) -> dict[str, str]:
    paths = list(config.inputs.values()) + list(config.exclude)
    specs = [config.recipe]
    if config.calibration is not None:
        specs.append(config.calibration.implementation)
    for spec in specs:
        source = inspect.getsourcefile(importlib.import_module(spec.partition(":")[0]))
        if source is not None:
            paths.append(source)
    for component in (recipe, calibrator):
        declared = getattr(component, "source_files", None)
        if declared is not None:
            paths.extend(str(path) for path in declared())
    return {
        str(Path(path).resolve()): digest(Path(path).read_bytes()) for path in paths
    }


def _validate_sample(sample: DataSample, *, serializable: bool = True) -> None:
    if not isinstance(sample, DataSample):
        raise TypeError("Recipe must yield DataSample instances")
    task = sample.task
    if not isinstance(task.task_id, str) or not task.task_id.strip():
        raise ValueError("Task ID must be nonempty")
    if not isinstance(task.input, dict) or not isinstance(task.metadata, dict):
        raise ValueError("Task input and metadata must be objects")
    if serializable:
        json_bytes(asdict(sample))


def _grade(
    samples: list[DataSample],
    attempts: list[Attempt],
    repeats: int,
    recipe: DataRecipe,
) -> dict[str, tuple[float | None, ...]]:
    expected = {
        (sample.task.task_id, index) for sample in samples for index in range(repeats)
    }
    actual: dict[tuple[str, int], Attempt] = {}
    for attempt in attempts:
        if not isinstance(attempt, Attempt) or type(attempt.repeat) is not int:
            raise ValueError(
                "Calibration must return Attempt instances with integer repeat indices"
            )
        key = attempt.task_id, attempt.repeat
        if key not in expected or key in actual:
            raise ValueError(
                "Calibration returned unknown or duplicate task/repeat identities"
            )
        if attempt.failure_reason is not None and (
            not isinstance(attempt.failure_reason, str) or not attempt.failure_reason
        ):
            raise ValueError("failure_reason must be a nonempty string or None")
        json_bytes(asdict(attempt))
        actual[key] = attempt
    if set(actual) != expected:
        raise ValueError("Calibration omitted task/repeat identities")
    scores = {}
    for sample in samples:
        values = []
        for index in range(repeats):
            attempt = actual[sample.task.task_id, index]
            if attempt.failure_reason is not None:
                values.append(None)
                continue
            value = recipe.score(copy.deepcopy(sample), copy.deepcopy(attempt.response))
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Recipe scores must be finite numbers in [0, 1]")
            values.append(float(value))
        scores[sample.task.task_id] = tuple(values)
    return scores


async def run_data_job(config: DataJobConfig, output: str | Path) -> DatasetSnapshot:
    """Run a foreground job in a fresh directory; failed jobs retain diagnostics.

    Each clean task is probed once per configured repeat, then retained for
    selection across batches. Completed snapshots are immutable by convention
    and content-verified when opened. Restarting requires a new output directory.
    """
    config = DataJobConfig.from_dict(asdict(config))
    config_bytes = json_bytes(asdict(config))
    output = Path(output).resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    work = output / "work"
    work.mkdir(mode=0o700)
    write_file(output / "config.json", config_bytes)
    started = time.time()
    state: dict[str, Any] = {
        "status": "running",
        "started_at": started,
        "config_digest": digest(config_bytes),
    }
    atomic_write_json(output / "job.json", state)
    try:
        recipe, recipe_identity = _load_factory(config.recipe, config.params)
        for method in ("generate", "clean", "score", "select", "training_row"):
            if not callable(getattr(recipe, method, None)):
                raise TypeError(f"Recipe is missing {method}()")
        calibrator: Calibrator | None = None
        calibration_identity = None
        if config.calibration is not None:
            calibrator, calibration_identity = _load_factory(
                config.calibration.implementation, config.calibration.params
            )
            if not callable(calibrator):
                raise TypeError("Calibration factory must return an async callable")
        sources = _sources(config, recipe, calibrator)
        excluded: set[str] = set()
        for path in config.exclude:
            for row in read_jsonl(Path(path)):
                if not isinstance(row.get("task_id"), str) or not row["task_id"]:
                    raise ValueError("Exclusion rows need a nonempty task_id")
                excluded.add(row["task_id"])
        context = RecipeContext(
            config.seed,
            config.target_count,
            config.purpose,
            {key: Path(value) for key, value in config.inputs.items()},
        )
        samples: list[DataSample] = []
        scores: dict[str, tuple[float | None, ...]] | None = (
            {} if calibrator is not None else None
        )
        seen_ids: dict[str, bytes] = {}
        observed_ids: dict[str, bytes] = {}
        seen_inputs: dict[str, bytes] = {}
        counts: Counter[str] = Counter()
        probe_rows: list[dict[str, Any]] = []
        calibration_batches: list[dict[str, Any]] = []
        selection: Selection | None = None
        last_shortage = "No eligible candidates"
        for batch in range(config.max_batches):
            context = replace(context, batch_index=batch)
            fresh = []
            for original in recipe.generate(context):
                counts["generated"] += 1
                if counts["generated"] > config.max_candidates:
                    raise InsufficientDataError("Candidate budget exhausted")
                _validate_sample(original, serializable=False)
                cleaned = recipe.clean(original)
                if cleaned is None:
                    counts["rejected_by_recipe"] += 1
                    continue
                _validate_sample(cleaned)
                task = cleaned.task
                payload = json_bytes(asdict(cleaned))
                if task.task_id in observed_ids:
                    if observed_ids[task.task_id] != payload:
                        raise ValueError(
                            "Same task ID refers to different candidate content"
                        )
                    counts["duplicates"] += 1
                    continue
                observed_ids[task.task_id] = payload
                if task.task_id in excluded:
                    counts["excluded"] += 1
                    continue
                key = digest(json_bytes(task.input))
                if key in seen_inputs:
                    if seen_inputs[key] != json_bytes(cleaned.reference):
                        raise ValueError(
                            "Duplicate task inputs have conflicting references"
                        )
                    counts["duplicates"] += 1
                    continue
                seen_ids[task.task_id] = payload
                seen_inputs[key] = json_bytes(cleaned.reference)
                fresh.append(cleaned)
            write_file(
                work / f"batch-{batch}-candidates.jsonl",
                b"".join(json_bytes(asdict(sample)) for sample in fresh),
            )
            write_file(
                work / f"batch-{batch}-public.jsonl",
                b"".join(json_bytes(asdict(sample.task)) for sample in fresh),
            )
            if calibrator is not None and fresh:
                assert config.calibration is not None and scores is not None
                calibration_context = CalibrationContext(
                    config.calibration.repeats,
                    batch,
                )
                # 校准器只接收公开输入；答案和评分逻辑留在执行器中。
                tasks = [copy.deepcopy(sample.task) for sample in fresh]
                result = await calibrator(tasks, calibration_context)
                if not isinstance(result, CalibrationBatch):
                    raise TypeError("Calibration must return CalibrationBatch")
                if not isinstance(result.provenance, dict):
                    raise ValueError("Calibration provenance must be an object")
                json_bytes(result.provenance)
                calibration_batches.append(
                    {
                        "batch_index": batch,
                        "provenance": copy.deepcopy(result.provenance),
                    }
                )
                attempts = list(result.attempts)
                batch_scores = _grade(
                    fresh, attempts, config.calibration.repeats, recipe
                )
                scores.update(batch_scores)
                rows = [
                    asdict(attempt)
                    | {"score": batch_scores[attempt.task_id][attempt.repeat]}
                    for attempt in attempts
                ]
                probe_rows.extend(rows)
                write_file(
                    work / f"batch-{batch}-probes.jsonl",
                    b"".join(json_bytes(row) for row in rows),
                )
            samples.extend(fresh)
            counts["batches"] += 1
            counts["clean_candidates"] = len(samples)
            atomic_write_json(output / "progress.json", dict(counts))
            try:
                selection = recipe.select(
                    copy.deepcopy(samples), copy.deepcopy(scores), context
                )
            except InsufficientDataError as error:
                last_shortage = str(error)
                logger.info("Batch %d needs more data: %s", batch, error)
                continue
            if not isinstance(selection, Selection):
                raise TypeError("Recipe must return Selection")
            ids = selection.task_ids
            if (
                len(ids) != config.target_count
                or len(set(ids)) != len(ids)
                or any(task_id not in seen_ids for task_id in ids)
            ):
                raise ValueError(
                    "Selection must contain target_count distinct candidate IDs"
                )
            json_bytes(selection.statistics)
            break
        else:
            raise InsufficientDataError(f"Data budget exhausted: {last_shortage}")
        assert selection is not None
        by_id = {sample.task.task_id: sample for sample in samples}
        chosen = [by_id[task_id] for task_id in selection.task_ids]
        if any(
            json_bytes(asdict(sample)) != seen_ids[sample.task.task_id]
            for sample in samples
        ):
            raise ValueError("Recipe mutated a retained candidate outside cleaning")
        training = [recipe.training_row(copy.deepcopy(sample)) for sample in chosen]
        if any(not isinstance(row, dict) for row in training):
            raise ValueError("Training rows must be objects")
        if any(
            json_bytes(asdict(sample)) != seen_ids[sample.task.task_id]
            for sample in samples
        ):
            raise ValueError("Recipe mutated a retained candidate during projection")
        if _sources(config, recipe, calibrator) != sources:
            raise ValueError("Input sources changed while the data job was running")
        report = dict(counts) | {
            "selected": len(chosen),
            "selection": selection.statistics,
            "calibrated": calibrator is not None,
        }
        core_sources = {
            name: digest((Path(__file__).parent / name).read_bytes())
            for name in ("runner.py", "snapshot.py", "types.py")
        }
        snapshot = publish_snapshot(
            output / "dataset",
            public=[asdict(sample.task) for sample in chosen],
            references=[
                {"task_id": sample.task.task_id, "reference": sample.reference}
                for sample in chosen
            ],
            training=training,
            probes=probe_rows,
            report=report,
            provenance={
                "config": asdict(config),
                "recipe": recipe_identity,
                "calibration": calibration_identity,
                "calibration_batches": calibration_batches,
                "input_digests": sources,
                "executor_digests": core_sources,
            },
        )
        state.update(
            status="succeeded",
            dataset_id=snapshot.dataset_id,
            dataset_path=str(snapshot.root),
        )
        logger.info("Published %s (%d tasks)", snapshot.dataset_id, len(chosen))
        return snapshot
    except BaseException as error:
        state.update(
            status="cancelled"
            if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            else "failed",
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        state["finished_at"] = time.time()
        atomic_write_json(output / "job.json", state)
