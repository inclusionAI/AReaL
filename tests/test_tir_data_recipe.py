# SPDX-License-Identifier: Apache-2.0

import json
import math
import re
from collections import Counter
from fractions import Fraction
from itertools import combinations

import pytest

from examples.data_engineering.tir.data_recipe import Family, TirDataRecipe

from areal.v2.data_engineering import (
    CalibrationConfig,
    DataJobConfig,
    InsufficientDataError,
    RecipeContext,
    run_data_job,
)
from areal.v2.data_engineering.snapshot import read_jsonl


def generated(count=256):
    return TirDataRecipe(candidate_count=count).generate(
        RecipeContext(42, 64, "joint", {})
    )


def independent_answer(question, recipe):
    """Reconstruct the mathematical problem from its public text, not its seed."""
    family, variant = recipe.rsplit("_", 1)
    if family == "algebra":
        if variant == "0":
            total, product = map(
                int, re.search(r"x\+y=(\d+) and xy=(\d+)", question).groups()
            )
            return total**3 - 3 * product * total
        points = [
            (int(x), int(y)) for x, y in re.findall(r"P\((-?\d+)\)=(\d+)", question)
        ]
        (x1, y1), (x2, y2) = points
        linear = Fraction(y2 - y1 - (x2**2 - x1**2), x2 - x1)
        constant = y1 - x1**2 - linear * x1
        target = int(re.search(r"Find P\((-?\d+)\)", question)[1])
        return target**2 + linear * target + constant
    if family == "number_theory":
        if variant == "0":
            a, p, b, q, modulus = map(int, re.findall(r"\d+", question))
            return (a**p + b**q) % modulus
        lower, rp, p, rq, q = map(int, re.findall(r"\d+", question))
        return next(
            value
            for value in range(lower + 1, lower + p * q + 1)
            if value % p == rp and value % q == rq
        )
    if family == "counting":
        if variant == "0":
            k = int(re.search(r"size (\d+)", question)[1])
            n = int(re.search(r"\.\.\.,(\d+)", question)[1])
            previous, current = [1] + [0] * k, [1, 1] + [0] * (k - 1)
            for _ in range(2, n + 1):
                previous, current = (
                    current,
                    [1] + [current[j] + previous[j - 1] for j in range(1, k + 1)],
                )
            return current[k]
        total, bound = map(int, re.findall(r"\d+", question))
        return sum(total - x + 1 for x in range(bound + 1))
    if family == "probability":
        numbers = list(map(int, re.findall(r"\d+", question)))
        red, blue = numbers[:2]
        if variant == "1":
            both = Fraction(red, red + blue) * Fraction(red - 1, red + blue - 1)
            neither = Fraction(blue, red + blue) * Fraction(blue - 1, red + blue - 1)
            return both / (1 - neither)
        k = numbers[2]
        result = Fraction(0)
        for red_positions in combinations(range(k), 2):
            probability = Fraction(1)
            reds, blues = red, blue
            for index in range(k):
                if index in red_positions:
                    probability *= Fraction(reds, reds + blues)
                    reds -= 1
                else:
                    probability *= Fraction(blues, reds + blues)
                    blues -= 1
            result += probability
        return result
    if family == "geometry":
        if variant == "1":
            width, height, cut = map(int, re.findall(r"\d+", question))
            return width * height - 4 * Fraction(cut**2, 2)
        vertices = [
            (int(x), int(y)) for x, y in re.findall(r"\((-?\d+),(-?\d+)\)", question)
        ]
        a, b, c = vertices
        double_area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        perimeter = sum(
            math.isqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
            for x, y in ((a, b), (b, c), (c, a))
        )
        return Fraction(double_area, perimeter)
    if family == "recurrence":
        first = int(re.search(r"u_1=(\d+)", question)[1])
        n = int(re.search(r"Find u_(\d+)", question)[1])
        if variant == "0":
            ratio, offset = map(
                int, re.search(r"u_\(n\+1\)=(\d+)u_n\+(\d+)", question).groups()
            )
            return first * ratio ** (n - 1) + offset * (ratio ** (n - 1) - 1) // (
                ratio - 1
            )
        second = int(re.search(r"u_2=(\d+)", question)[1])
        difference = int(re.search(r"u_n=(\d+)", question)[1])
        for _ in range(3, n + 1):
            first, second = second, 2 * second - first + difference
        return second
    if family == "optimization":
        if variant == "0":
            total = int(re.search(r"x\+y=(\d+)", question)[1])
            candidates = {(total + 1) // 2, (total + 2) // 2}
            return max(x * (total - x) + x for x in candidates)
        cost1, cost2, budget, reward1, reward2 = map(
            int, re.search(r"(\d+)x\+(\d+)y<=(\d+).*?(\d+)x\+(\d+)y", question).groups()
        )
        best = [0] * (budget + 1)
        for capacity in range(1, budget + 1):
            best[capacity] = max(
                [best[capacity - 1]]
                + [
                    best[capacity - cost] + reward
                    for cost, reward in ((cost1, reward1), (cost2, reward2))
                    if capacity >= cost
                ]
            )
        return best[-1]
    if family == "rates":
        numbers = list(map(int, re.findall(r"\d+", question)))
        if variant == "0":
            t1, t2, elapsed = numbers
            return t1 - Fraction(elapsed * t1, t2)
        speed, current = numbers[0], numbers[-1]
        return Fraction(speed**2 - current**2, speed)
    raise AssertionError(recipe)


def test_tir_generation_has_deterministic_balanced_recipes_and_independent_answers():
    """All 16 structures have answers checked from the actual question text."""
    samples = generated()
    assert samples == generated()
    assert len({item.task.task_id for item in samples}) == 256
    assert set(Counter(item.task.metadata["recipe"] for item in samples).values()) == {
        16
    }
    for item in samples:
        question = item.task.input["messages"][0]["content"]
        assert Fraction(item.reference) == independent_answer(
            question, item.task.metadata["recipe"]
        )


@pytest.mark.parametrize(
    "response,expected",
    [
        (r"result \boxed{3/2}", 1),
        (r"\boxed{\frac{3}{2}}", 1),
        (r"\boxed{1.5}", 1),
        ("3/2", 0),
        (r"\boxed{1}", 0),
        (r"\boxed{1/0}", 0),
        (r"\boxed{3/2", 0),
    ],
)
def test_tir_verifier_handles_exact_numeric_forms_and_rejects_missing_box(
    response, expected
):
    """The recipe's reward requires a final, well-formed exact numeric answer."""
    from dataclasses import replace

    sample = replace(generated(16)[0], reference="3/2")
    assert TirDataRecipe().score(sample, response) == expected


def record_reports(tmp_path, samples):
    paths = []
    for repeat in range(2):
        cases = []
        for index, item in enumerate(samples):
            category = (index // 16) % 4
            correct = category == 2 or (category == 0 and repeat == 0)
            cases.append(
                {
                    "task_id": item.task.task_id,
                    "response": f"\\boxed{{{item.reference}}}"
                    if correct
                    else r"\boxed{-999999999}",
                    "failure_reason": "timeout" if category == 3 else None,
                    "score": 1,
                }
            )
        report = {
            "model": "model-v1",
            "harness_id": "harness-v1",
            "backend_model_digest": "sha256:model",
            "verifier_digest": "sha256:verifier",
            "task_set_id": "probe-v1",
            "executable_digest": "sha256:runner",
            "trial_id": f"trial-{repeat}",
            "cases": cases,
        }
        path = tmp_path / f"report-{repeat}.json"
        path.write_text(json.dumps(report))
        paths.append(str(path))
    return paths


@pytest.mark.asyncio
async def test_tir_calibrated_job_preserves_512_case_policy_and_rescores_reports(
    tmp_path,
):
    """Completed/wrong/failing reports retain distinct selection semantics."""
    samples = generated(1024)
    reports = record_reports(tmp_path, samples)
    config = DataJobConfig(
        "examples.data_engineering.tir.data_recipe:create_recipe",
        512,
        seed=42,
        params={
            "candidate_count": 1024,
            "minimum_one_pass": 128,
            "maximum_per_recipe": 48,
        },
        calibration=CalibrationConfig(
            "examples.data_engineering.tir.data_calibration:recorded_responses",
            params={
                "model_ref": "model-v1",
                "harness_ref": "harness-v1",
                "reports": [reports],
                "backend_model_digest": "sha256:model",
                "verifier_digest": "sha256:verifier",
            },
        ),
    )
    snapshot = await run_data_job(config, tmp_path / "job")
    selected_ids = {row["task_id"] for row in read_jsonl(snapshot.public_path)}
    expected = {
        item.task.task_id
        for index, item in enumerate(samples)
        if (index // 16) % 4 in {0, 1}
    }
    assert selected_ids == expected
    statistics = json.loads((snapshot.root / "report.json").read_text())["selection"]
    assert statistics["one_pass"] == statistics["zero_pass"] == 256
    assert min(statistics["families"].values()) >= 32
    assert max(statistics["recipes"].values()) <= 48
    assert all("answer" not in row["input"] for row in read_jsonl(snapshot.public_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["model", "harness", "trial"])
async def test_recorded_calibration_rejects_wrong_bindings_and_reused_trial(
    tmp_path, change
):
    """Replay inputs must retain the model/Harness and independent-repeat identity."""
    paths = record_reports(tmp_path, generated(128))
    second = json.loads(open(paths[1]).read())
    key = {"model": "model", "harness": "harness_id", "trial": "trial_id"}[change]
    second[key] = "trial-0" if change == "trial" else "wrong"
    from pathlib import Path

    Path(paths[1]).write_text(json.dumps(second))
    config = DataJobConfig(
        "examples.data_engineering.tir.data_recipe:create_recipe",
        32,
        seed=42,
        params={"candidate_count": 128},
        calibration=CalibrationConfig(
            "examples.data_engineering.tir.data_calibration:recorded_responses",
            params={
                "model_ref": "model-v1",
                "harness_ref": "harness-v1",
                "reports": [paths],
                "backend_model_digest": "sha256:model",
                "verifier_digest": "sha256:verifier",
            },
        ),
    )
    with pytest.raises(ValueError):
        await run_data_job(config, tmp_path / "job")
    assert not (tmp_path / "job/dataset").exists()


@pytest.mark.asyncio
async def test_tir_can_exclude_all_previous_probe_candidates(tmp_path):
    """New rounds avoid the full previous pool, not just its selected subset."""
    config = DataJobConfig(
        "examples.data_engineering.tir.data_recipe:create_recipe",
        16,
        seed=42,
        params={"candidate_count": 64},
    )
    await run_data_job(config, tmp_path / "first")
    pool = tmp_path / "first/work/batch-0-public.jsonl"
    previous = {row["task_id"] for row in read_jsonl(pool)}
    from dataclasses import replace

    snapshot = await run_data_job(
        replace(config, exclude=(str(pool),), max_batches=2), tmp_path / "second"
    )
    assert previous.isdisjoint(
        row["task_id"] for row in read_jsonl(snapshot.public_path)
    )
    report = json.loads((snapshot.root / "report.json").read_text())
    assert report["excluded"] == 64
    assert report["batches"] == 2


@pytest.mark.parametrize("calibrated", [False, True])
def test_tir_smallest_selection_covers_every_family(calibrated):
    """Eight selected tasks retain one case per family, with or without probes."""
    samples = generated(128)
    scores = (
        {sample.task.task_id: (1.0, 0.0) for sample in samples} if calibrated else None
    )
    selection = TirDataRecipe().select(
        samples, scores, RecipeContext(42, 8, "joint", {})
    )
    by_id = {sample.task.task_id: sample for sample in samples}
    families = Counter(
        by_id[task_id].task.metadata["family"] for task_id in selection.task_ids
    )
    assert families == {family.value: 1 for family in Family}
    assert selection.statistics["families"] == dict(families)


@pytest.mark.parametrize("calibrated", [False, True])
def test_tir_smallest_selection_rejects_missing_eligible_families(calibrated):
    """Enough algebra cases cannot replace a missing family at target_count=8."""
    samples = generated(128)
    if calibrated:
        scores = {
            sample.task.task_id: (
                (1.0, 0.0)
                if sample.task.metadata["family"] == Family.ALGEBRA.value
                else (1.0, 1.0)
            )
            for sample in samples
        }
    else:
        samples = [
            sample
            for sample in samples
            if sample.task.metadata["family"] == Family.ALGEBRA.value
        ]
        scores = None
    with pytest.raises(
        InsufficientDataError, match="Too few eligible number_theory cases"
    ):
        TirDataRecipe().select(samples, scores, RecipeContext(42, 8, "joint", {}))
