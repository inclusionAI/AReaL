# SPDX-License-Identifier: Apache-2.0

"""External recipes with observable calls and injected storage/provider failures."""

import asyncio
import json
import os
import time
from pathlib import Path

from areal.v2.data_engineering import (
    Attempt,
    CalibrationBatch,
    DataSample,
    GenerationShard,
    PublicTask,
    RankedCandidate,
    Selection,
    SelectionPolicy,
)


def record(directory, phase, task_id):
    if directory is None:
        return
    value = (
        json.dumps(
            {
                "phase": phase,
                "task_id": task_id,
                "pid": os.getpid(),
                "time": time.monotonic(),
            }
        )
        + "\n"
    )
    fd = os.open(
        str(Path(directory) / "calls.jsonl"),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(fd, value.encode())
    finally:
        os.close(fd)


class ParallelRecipe:
    def __init__(
        self, count=8, directory=None, delay=0, fail_task="3", fail_phase="clean"
    ):
        self.count, self.directory, self.delay = count, directory, delay
        self.fail_task, self.fail_phase = fail_task, fail_phase

    def generate(self, context):
        record(self.directory, "generate", context.batch_index)
        for index in range(self.count):
            yield DataSample(PublicTask(str(index), {"value": index}), index)

    def visit(self, phase, sample):
        record(self.directory, phase + ".start", sample.task.task_id)
        if self.delay:
            time.sleep(self.delay)
        if (
            self.directory
            and sample.task.task_id == self.fail_task
            and phase == self.fail_phase
            and (Path(self.directory) / "fail").exists()
        ):
            raise OSError("Injected transient storage failure")
        record(self.directory, phase + ".end", sample.task.task_id)

    def clean(self, sample):
        self.visit("clean", sample)
        return sample

    def score(self, sample, response):
        return float(response == sample.reference)

    def select(self, samples, scores, context):
        assert samples[-1].task.task_id == str(self.count - 1)
        assert samples[:0] == []
        if scores is not None:
            assert len(scores) == self.count
        return Selection(tuple(s.task.task_id for s in samples[: context.target_count]))

    def training_row(self, sample):
        self.visit("training", sample)
        return {
            "task_id": sample.task.task_id,
            "value": sample.task.input["value"],
            "answer": sample.reference,
        }


class ParallelCalibrator:
    def __init__(
        self,
        directory=None,
        delay=0,
        fail_task="3",
        transient_failures=0,
        invalid=False,
    ):
        self.directory, self.delay, self.fail_task = directory, delay, fail_task
        self.remaining, self.invalid = transient_failures, invalid

    async def __call__(self, tasks, context):
        task_id = tasks[0].task_id
        record(self.directory, "calibration.start", task_id)
        await asyncio.sleep(self.delay)
        if self.invalid:
            raise ValueError("Invalid provider input")
        if self.remaining:
            self.remaining -= 1
            raise OSError("Temporary provider failure")
        if self.directory and task_id == self.fail_task:
            if (Path(self.directory) / "fail").exists():
                raise OSError("Injected transient provider failure")
            while (Path(self.directory) / "block").exists():
                await asyncio.sleep(0.01)
        attempts = tuple(
            Attempt(task.task_id, repeat, task.input["value"])
            for task in tasks
            for repeat in range(context.repeats)
        )
        record(self.directory, "calibration.end", task_id)
        return CalibrationBatch(attempts, {"method": "fixture-provider"})


def create_recipe(**params):
    return ParallelRecipe(**params)


def create_calibrator(**params):
    return ParallelCalibrator(**params)


class MutatingRetryRecipe(ParallelRecipe):
    def clean(self, sample):
        sample.task.input["value"] += 1
        marker = Path(self.directory) / "fail"
        if marker.exists():
            marker.unlink()
            raise OSError("Retry after normalization")
        return sample


def mutating_recipe(**params):
    return MutatingRetryRecipe(**params)


class ShardedRankRecipe(ParallelRecipe):
    def generate(self, context):
        raise AssertionError("Parallel generation must use the shard protocol")

    def generation_shards(self, context, shard_size):
        for start in range(0, self.count, shard_size):
            yield GenerationShard(start, min(start + shard_size, self.count))

    def generate_shard(self, context, shard):
        for index in range(shard.start, shard.stop):
            sample = DataSample(PublicTask(str(index), {"value": index}), index)
            self.visit("generation", sample)
            yield sample

    def score(self, sample, response):
        self.visit("score", sample)
        return super().score(sample, response)

    def selection_policy(self, context, calibrated):
        return SelectionPolicy({"even": 2, "odd": 2}, {str(i): 2 for i in range(4)})

    def rank(self, sample, scores, context):
        self.visit("rank", sample)
        index = sample.task.input["value"]
        return RankedCandidate(
            sample.task.task_id,
            -index,
            sample.task.task_id,
            "even" if index % 2 == 0 else "odd",
            str(index % 4),
        )

    def finalize_selection(self, selected, eligible_count, context):
        return Selection(
            tuple(
                item.task_id
                for item in sorted(selected, key=lambda item: item.priority)
            ),
            {"eligible": eligible_count},
        )

    def select(self, samples, scores, context):
        raise AssertionError("Parallel selection must use ranking and a global merge")


def create_sharded_recipe(**params):
    return ShardedRankRecipe(**params)
