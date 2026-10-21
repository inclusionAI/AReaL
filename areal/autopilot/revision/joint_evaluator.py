# SPDX-License-Identifier: Apache-2.0

"""Joint optimizer revisions against a frozen finite projection of one model."""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from areal.autopilot.elastic.policy.basic.joint import (
    JointConfig,
    JointPrediction,
    JointScenario,
)
from areal.autopilot.elastic.policy.basic.online import SerialJointWindow
from areal.autopilot.elastic.policy.basic.optimizer import JointModel, JointOptimizer
from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
from areal.autopilot.revision.contracts import EvaluationReport, digest, source_digest
from areal.autopilot.revision.evaluator import finite_nonnegative
from areal.autopilot.revision.execution import CandidateRunner
from areal.autopilot.revision.model_execution import (
    joint_config_from_dict,
    joint_scenario_from_dict,
)


@dataclass(frozen=True)
class JointOptimizerCriteria:
    min_relative_improvement: float = 0.005
    max_cost_regression: float = 0.0
    min_runtime_improvement: float = 0.1
    min_runtime_seconds: float = 0.01

    def __post_init__(self):
        for name in ("min_relative_improvement", "min_runtime_improvement"):
            value = getattr(self, name)
            if not finite_nonnegative(value) or value >= 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("max_cost_regression", "min_runtime_seconds"):
            if not finite_nonnegative(getattr(self, name)):
                raise ValueError(f"{name} must be finite and nonnegative")


class JointOptimizerEvaluator:
    """Fixed scenarios/configurations, model version, query budget and acceptance.

    Snapshot the supplied model on the complete finite configuration set once;
    workers see only queried predictions. The snapshot is the immutable model
    reference for this experiment, including construction identity and outputs.
    No Expert dependency or real measurement labels are required here.
    """

    target = "joint_optimizer"

    def __init__(
        self,
        model: JointModel,
        scenarios: tuple[JointScenario, ...],
        candidates: tuple[JointConfig, ...],
        *,
        model_version: str,
        max_queries: int = 128,
        max_job_seconds: float | None = None,
        switch_margin: float = 0.05,
        max_optimizer_seconds: float = 5,
        repetitions: int = 3,
        criteria: JointOptimizerCriteria | None = None,
        runner: CandidateRunner | None = None,
    ):
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("Explicit frozen model_version is required")
        scenarios = tuple(scenarios)
        search = JointOptimizer(candidates, max_queries, max_job_seconds, switch_margin)
        if not scenarios or any(
            not isinstance(s, JointScenario | SerialJointWindow) for s in scenarios
        ):
            raise ValueError("Nonempty typed joint scenarios are required")
        if not finite_nonnegative(max_optimizer_seconds) or max_optimizer_seconds == 0:
            raise ValueError("Optimizer time budget must be finite and positive")
        if type(repetitions) is not int or repetitions < 3 or repetitions % 2 != 1:
            raise ValueError("Timing requires an odd number of at least three runs")
        self._criteria = criteria or JointOptimizerCriteria()
        self._runner = runner or CandidateRunner()
        self._settings_json = json.dumps(
            asdict(search), sort_keys=True, allow_nan=False
        )
        self._scenarios_json = json.dumps(
            [asdict(s) for s in scenarios], sort_keys=True, allow_nan=False
        )
        self._max_seconds, self._repetitions = max_optimizer_seconds, repetitions
        configs, predictions = [], []
        for scenario in json.loads(self._scenarios_json):
            typed = joint_scenario_from_dict(scenario)
            options = tuple(
                dict.fromkeys(
                    (() if typed.current is None else (typed.current,))
                    + search.candidates
                )
            )
            identities = {
                (c.rollout.model_id, c.rollout.hardware_id, c.rollout.sampling_id)
                for c in options
            }
            if len(identities) != 1:
                raise ValueError(
                    "Joint candidates/current must preserve model/hardware/sampling"
                )
            rows = []
            for config in options:
                prediction = model.evaluate(joint_scenario_from_dict(scenario), config)
                if (
                    not isinstance(prediction, JointPrediction)
                    or prediction.config != config
                ):
                    raise ValueError(
                        "Frozen model returned a different configuration/type"
                    )
                if prediction.status == "complete":
                    expected = (
                        len(typed.batches),
                        sum(b.actor.samples for b in typed.batches),
                        sum(b.actor.effective_tokens for b in typed.batches),
                        sum(b.actor.sequence_tokens for b in typed.batches),
                    )
                    if (
                        prediction.completed_batches,
                        prediction.samples,
                        prediction.effective_tokens,
                        prediction.sequence_tokens,
                    ) != expected:
                        raise ValueError(
                            "Frozen model changed the identical workload window"
                        )
                    if not finite_nonnegative(prediction.total_gpu_seconds):
                        raise ValueError("Frozen model returned nonfinite total cost")
                rows.append(asdict(prediction))
            configs.append(options)
            predictions.append(rows)
        self._configs = tuple(configs)
        self._scenario_rows = tuple(
            json.dumps(s, sort_keys=True) for s in json.loads(self._scenarios_json)
        )
        self._prediction_rows = tuple(
            tuple(json.dumps(p, sort_keys=True) for p in rows) for rows in predictions
        )
        self._search = search
        self._predictions_json = json.dumps(
            predictions, sort_keys=True, allow_nan=False
        )
        self._oracles = []
        for index in range(len(configs)):
            eligible = [
                self._cost(p)
                for p in predictions[index]
                if self._eligible(index, joint_config_from_dict(p["config"]))
            ]
            if not eligible:
                raise ValueError(
                    f"Joint scenario {index} has no eligible reference configuration"
                )
            self._oracles.append(min(eligible))
        root = Path(__file__).parent
        self._reference_files = (
            tuple(
                root / n
                for n in (
                    "joint_evaluator.py",
                    "evaluator.py",
                    "model_execution.py",
                    "contracts.py",
                    "execution.py",
                    "worker.py",
                    "loop.py",
                    "serial.py",
                    "../elastic/runtime/serial_policy.py",
                )
            )
            + (root.parent / "contracts.py",)
            + tuple(sorted((root.parent / "elastic/policy/basic").rglob("*.py")))
        )
        self._code = self._code_fingerprint()
        self._context = {
            "api_version": "joint-optimizer-v1",
            "model_version": model_version,
            "model_snapshot_sha256": digest(json.loads(self._predictions_json)),
            "scenarios_sha256": digest(json.loads(self._scenarios_json)),
            "settings": self.settings,
            "criteria": asdict(self.criteria),
            "max_optimizer_seconds": max_optimizer_seconds,
            "repetitions": repetitions,
            "worker": {
                "command": list(
                    self.runner.command
                    or (sys.executable, "-m", "areal.autopilot.revision.worker")
                ),
                "timeout_seconds": self.runner.timeout_seconds,
                "max_output_bytes": self.runner.max_output_bytes,
            },
            "code": self._code,
        }
        self._benchmark_sha256 = digest(self._context)
        self._reports = {}

    @property
    def runner(self):
        return self._runner

    @property
    def criteria(self):
        return self._criteria

    @property
    def settings(self):
        return json.loads(self._settings_json)

    @property
    def context(self):
        return json.loads(json.dumps(self._context))

    @property
    def benchmark_sha256(self):
        return self._benchmark_sha256

    @property
    def calibration(self):
        return {"model_version": self._context["model_version"]}

    @property
    def problem_sha256(self) -> str:
        return digest(
            {
                k: v
                for k, v in self.context.items()
                if k not in {"model_version", "model_snapshot_sha256"}
            }
        )

    @property
    def model_requests(self) -> list[dict]:
        return [
            {"scenario": json.loads(scenario), "config": asdict(config)}
            for scenario, configs in zip(
                self._scenario_rows, self._configs, strict=True
            )
            for config in configs
        ]

    def with_model(
        self, model: JointModel, *, model_version: str
    ) -> JointOptimizerEvaluator:
        """Rebuild the same fixed search problem under a newly frozen model."""
        self.assert_unchanged()
        updated = JointOptimizerEvaluator(
            model,
            tuple(joint_scenario_from_dict(json.loads(s)) for s in self._scenario_rows),
            self._search.candidates,
            model_version=model_version,
            max_queries=self._search.max_evaluations,
            max_job_seconds=self._search.max_seconds,
            switch_margin=self._search.switch_margin,
            max_optimizer_seconds=self._max_seconds,
            repetitions=self._repetitions,
            criteria=self.criteria,
            runner=self.runner,
        )
        if updated.problem_sha256 != self.problem_sha256:
            raise ValueError("Rebinding changed the frozen optimizer problem")
        return updated

    def _code_fingerprint(self):
        root = Path(__file__).parent.parent
        return {
            str(p.relative_to(root)): source_digest(p.read_text())
            for p in self._reference_files
        }

    def assert_unchanged(self):
        if self._code != self._code_fingerprint():
            raise ValueError("Joint evaluation reference code changed")

    @staticmethod
    def _cost(prediction):
        return prediction["actor_gpu_seconds"] + prediction["rollout_gpu_seconds"]

    def _prediction(self, index, config):
        position = self._configs[index].index(config)
        return json.loads(self._prediction_rows[index][position])

    def _eligible(self, index, config):
        p = self._prediction(index, config)
        s = json.loads(self._scenario_rows[index])
        if (
            p["status"] != "complete"
            or config.actor.world_size > s.get("max_actor_gpus", s["max_total_gpus"])
            or config.rollout.world_size
            > s.get("max_rollout_gpus", s["max_total_gpus"])
            or config.world_size > s["max_total_gpus"]
            or (
                self._search.max_seconds is not None
                and p["seconds"] > self._search.max_seconds
            )
        ):
            return False
        current = joint_config_from_dict(s["current"])
        if current is not None and config != current:
            old = self._prediction(index, current)
            old_usable = (
                old["status"] == "complete"
                and (
                    self._search.max_seconds is None
                    or old["seconds"] <= self._search.max_seconds
                )
                and (
                    current.actor.world_size
                    <= s.get("max_actor_gpus", s["max_total_gpus"])
                    and current.rollout.world_size
                    <= s.get("max_rollout_gpus", s["max_total_gpus"])
                    and current.world_size <= s["max_total_gpus"]
                )
            )
            if old_usable and self._cost(p) >= self._cost(old) * (
                1 - self._search.switch_margin
            ):
                return False
        return True

    def evaluate(self, source: str, directory: Path) -> EvaluationReport:
        errors, metrics, score, runs = [], {}, None, []
        try:
            self.assert_unchanged()
            for repeat in range(self._repetitions):
                cases = json.loads(self._scenarios_json)
                calls, elapsed, selections = [0] * len(cases), [], []
                active, started, violations = None, None, []

                def simulate(message):
                    nonlocal active, started
                    try:
                        index, operation = message["case"], message["operation"]
                        if type(index) is not int or not 0 <= index < len(cases):
                            raise ValueError("Invalid joint scenario index")
                        if operation == "start":
                            if active is not None or index != len(selections):
                                raise ValueError(
                                    "Joint scenarios must run once in fixed order"
                                )
                            active, started = index, time.monotonic()
                            return {}
                        if active != index:
                            raise ValueError(
                                "Joint query/finish outside active scenario"
                            )
                        duration = time.monotonic() - started
                        if duration > self._max_seconds:
                            raise ValueError("Optimizer parent time budget exceeded")
                        if operation == "query":
                            calls[index] += 1
                            if calls[index] > self._search.max_evaluations:
                                raise ValueError("Joint query budget exceeded")
                            if digest(message["scenario"]) != digest(cases[index]):
                                raise ValueError(
                                    "Optimizer changed the frozen scenario"
                                )
                            config = joint_config_from_dict(message["config"])
                            if config not in self._configs[index]:
                                raise ValueError(
                                    "Query configuration outside fixed allowlist"
                                )
                            return {"result": self._prediction(index, config)}
                        if operation != "finish":
                            raise ValueError("Unknown joint optimizer operation")
                        config = joint_config_from_dict(message["config"])
                        if config not in self._configs[index] or not self._eligible(
                            index, config
                        ):
                            raise ValueError(
                                "Selected configuration violates fixed constraints"
                            )
                        selections.append(config)
                        elapsed.append(duration)
                        active, started = None, None
                        return {}
                    except Exception as error:
                        violations.append(f"{type(error).__name__}: {error}")
                        raise

                run_started = time.monotonic()
                output = self.runner.run(
                    source,
                    {
                        "target": self.target,
                        "settings": self.settings,
                        "cases": cases,
                    },
                    directory if repeat == 0 else directory / f"repeat-{repeat}",
                    simulate,
                )
                worker_seconds = time.monotonic() - run_started
                self.assert_unchanged()
                if output.get("error") or violations:
                    raise ValueError(output.get("error") or "; ".join(violations))
                if active is not None or len(selections) != len(cases):
                    raise ValueError("Optimizer omitted fixed scenarios")
                if output.get("configs") != [asdict(c) for c in selections]:
                    raise ValueError(
                        "Optimizer result differs from parent-validated selections"
                    )
                row = {
                    "configs": output["configs"],
                    "calls": calls,
                    "seconds": elapsed,
                    "worker_seconds": worker_seconds,
                }
                if runs and (row["configs"], calls) != (
                    runs[0]["configs"],
                    runs[0]["calls"],
                ):
                    raise ValueError(
                        "Joint optimizer choices/query counts must reproduce"
                    )
                runs.append(row)
            costs, wall, ratios = [], [], []
            for index, config in enumerate(runs[0]["configs"]):
                prediction = self._prediction(index, joint_config_from_dict(config))
                cost = self._cost(prediction)
                oracle = self._oracles[index]
                costs.append(cost)
                wall.append(prediction["seconds"])
                # Zero-cost cases are legal; use additive regret with a 1 GPU-s floor.
                ratios.append(1 + max(0, cost - oracle) / max(oracle, 1))
            seconds = [
                statistics.median(r["seconds"][i] for r in runs)
                for i in range(len(costs))
            ]
            worker_seconds = statistics.median(r["worker_seconds"] for r in runs)
            metrics = {
                "mean_gpu_seconds": statistics.fmean(costs),
                "mean_job_seconds": statistics.fmean(wall),
                "mean_relative_reference_cost": statistics.fmean(ratios),
                "mean_queries": statistics.fmean(runs[0]["calls"]),
                "mean_optimizer_seconds": statistics.fmean(seconds),
                "worker_seconds": worker_seconds,
                "cases": len(costs),
                "repetitions": self._repetitions,
                "reference_evaluations": sum(map(len, self._configs)),
                "selection_sha256": digest(
                    {
                        "configs": runs[0]["configs"],
                        "costs": costs,
                        "queries": runs[0]["calls"],
                    }
                ),
            }
            score = metrics["mean_relative_reference_cost"]
            atomic_json(
                directory / "joint-evaluation.json",
                {
                    "runs": runs,
                    "costs": costs,
                    "job_seconds": wall,
                    "optimizer_seconds": seconds,
                    "reference_costs": self._oracles,
                    "model_version": self._context["model_version"],
                },
            )
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        report = EvaluationReport(
            self.target,
            source_digest(source),
            self.benchmark_sha256,
            not errors,
            score,
            metrics,
            tuple(errors),
        )
        self._reports[digest(asdict(report))] = (
            None if errors else (costs, runs[0]["calls"], worker_seconds)
        )
        return report

    def _verified(self, report):
        if (
            report.benchmark_sha256 != self.benchmark_sha256
            or digest(asdict(report)) not in self._reports
        ):
            raise ValueError("Joint comparison requires unchanged issued reports")
        self.assert_unchanged()
        return self._reports[digest(asdict(report))]

    def compare(self, candidate, incumbent):
        new, old = self._verified(candidate), self._verified(incumbent)
        if not incumbent.valid or old is None:
            raise ValueError("Joint incumbent must evaluate successfully")
        if not candidate.valid or new is None:
            return False, "Joint optimizer evaluation failed"
        if any(
            n > o * (1 + self.criteria.max_cost_regression) + 1e-9
            for n, o in zip(new[0], old[0], strict=True)
        ):
            return False, "Per-scenario model cost regressed"
        threshold = self.criteria.min_relative_improvement
        if incumbent.score - candidate.score > max(
            1e-12, abs(incumbent.score) * threshold
        ):
            return True, "Improved fixed-model configuration quality"
        if candidate.score > incumbent.score + 1e-12:
            return False, "Configuration quality regressed"
        if any(n > o + 1e-9 for n, o in zip(new[0], old[0], strict=True)):
            return (
                False,
                "Search-efficiency acceptance requires no scenario cost increase",
            )
        before, after = statistics.fmean(old[1]), statistics.fmean(new[1])
        before_time, after_time = old[2], new[2]
        runtime_slack = max(
            self.criteria.min_runtime_seconds,
            before_time * self.criteria.min_runtime_improvement,
        )
        if (
            before - after > max(1e-12, before * threshold)
            and after_time <= before_time + runtime_slack
        ):
            return True, "Preserved model quality with fewer queries"
        # Timing cannot justify more queries. Use medians plus absolute/relative gates.
        if after <= before and before_time - after_time > max(
            self.criteria.min_runtime_seconds,
            before_time * self.criteria.min_runtime_improvement,
        ):
            return True, "Preserved model quality with faster solving"
        return False, "Insufficient quality or search-efficiency improvement"

    def feedback(self, report):
        self._verified(report)
        metrics = json.loads(json.dumps(report.metrics))
        metrics.pop("selection_sha256", None)
        return {
            "score": report.score,
            "metrics": metrics,
            "errors": list(report.errors),
        }
