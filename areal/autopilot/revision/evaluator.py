# SPDX-License-Identifier: Apache-2.0

"""Independent measurement and fixed-reference optimizer evaluations.

Candidate processes receive public inputs only. Simulation RPC, query accounting,
reference costs, progress checks and acceptance metrics are computed by the parent.
"""

import itertools
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from areal.autopilot.contracts import ResourceDecision, TrainingConfig
from areal.autopilot.elastic.policy.basic.optimizer import BasicOptimizer
from areal.autopilot.elastic.policy.basic.simulator import (
    PlanStep,
    forecast_window,
)
from areal.autopilot.revision.contracts import (
    BASELINE_SOURCE,
    EvaluationReport,
    Target,
    deployment_context,
    digest,
    forecast_from_dict,
    history_from_dict,
    lifecycle_profile_digest,
    observation_from_dict,
    plan_from_dict,
    source_digest,
    workflow_from_profile,
)
from areal.autopilot.revision.execution import CandidateRunner, load_source
from areal.autopilot.revision.phase import (
    SCOPE,
    build_phase_simulator,
    verify_phase_reference,
)


def finite_nonnegative(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def decision_for_plan(observation, first: PlanStep) -> ResourceDecision:
    ready = observation.batch_ready and observation.gate_available
    if (
        observation.training_active
        and first.release_before
        and not observation.batch_ready
    ):
        return ResourceDecision("release")
    if observation.training_active and first.config != observation.training_config:
        return (
            ResourceDecision("reconfigure", first.config, True)
            if ready
            else ResourceDecision("keep", observation.training_config)
        )
    if not observation.training_active:
        return (
            ResourceDecision("launch", first.config, True)
            if ready
            else ResourceDecision("keep")
        )
    return ResourceDecision("keep", first.config, ready)


@dataclass(frozen=True)
class AcceptanceCriteria:
    min_relative_improvement: float = 0.005
    max_cost_regression: float = 0.0

    def __post_init__(self):
        if not 0 <= self.min_relative_improvement < 1 or not math.isfinite(
            self.min_relative_improvement
        ):
            raise ValueError("Improvement threshold must be in [0, 1)")
        if not finite_nonnegative(self.max_cost_regression):
            raise ValueError("Cost regression tolerance must be nonnegative")

    def compare(
        self, candidate: EvaluationReport, incumbent: EvaluationReport
    ) -> tuple[bool, str]:
        if not candidate.valid:
            return False, "Candidate evaluation failed"
        if (
            candidate.target != incumbent.target
            or candidate.benchmark_sha256 != incumbent.benchmark_sha256
        ):
            raise ValueError("Cannot compare different evaluation references")
        if not incumbent.valid:
            raise ValueError("Incumbent must pass the fixed evaluation")
        if candidate.target == "simulator" and candidate.metrics["feasibility_errors"]:
            return False, "Predicted feasibility differs from held-out measurements"
        if (
            candidate.target == "optimizer"
            and candidate.metrics["mean_gpu_seconds"]
            > incumbent.metrics["mean_gpu_seconds"] * (1 + self.max_cost_regression)
            + 1e-9
        ):
            return False, "Reference GPU cost regressed"
        improvement = incumbent.score - candidate.score
        threshold = max(1e-12, abs(incumbent.score) * self.min_relative_improvement)
        return (
            (True, "Improved fixed-reference score")
            if improvement > threshold
            else (False, "Insufficient score improvement")
        )


class RevisionEvaluator:
    """Frozen calibration, held-out cases, reference code and search limits.

    Simulator cases supply independent observed seconds/GPU seconds/feasibility.
    Optimizer cases supply observations and historical windows; forecast and oracle
    are computed here. No candidate-reported score or query count is trusted.
    """

    def __init__(
        self,
        target: Target,
        profile: dict,
        cases: list[dict],
        *,
        settings: dict | None = None,
        overhead_weight: float = 0.05,
        max_optimizer_seconds: float = 5,
        runner: CandidateRunner | None = None,
        reference_simulator_source: str | None = None,
        phase_reference: dict | None = None,
    ):
        if target not in {"simulator", "optimizer"} or not cases:
            raise ValueError("Evaluation requires a valid target and nonempty cases")
        if (
            not finite_nonnegative(overhead_weight)
            or not finite_nonnegative(max_optimizer_seconds)
            or max_optimizer_seconds == 0
        ):
            raise ValueError("Invalid evaluator scoring or time limit")
        self.target = target
        if reference_simulator_source is not None and (
            target != "optimizer"
            or not isinstance(reference_simulator_source, str)
            or not reference_simulator_source.strip()
            or len(reference_simulator_source.encode()) > 128_000
        ):
            raise ValueError(
                "A frozen simulator source is supported for optimizer evaluation only"
            )
        self._profile_json = json.dumps(profile, sort_keys=True, allow_nan=False)
        self._settings_json = json.dumps(
            asdict(BasicOptimizer(**(settings or {}))), sort_keys=True
        )
        self.context = deployment_context(profile, self.settings)
        self.context["profile_sha256"] = lifecycle_profile_digest(profile)
        self.reference_simulator_source = (
            reference_simulator_source or BASELINE_SOURCE["simulator"]
        )
        if reference_simulator_source is not None:
            self.context["simulator_source"] = reference_simulator_source
        self.phase_only = target == "simulator" and all("phase" in c for c in cases)
        if (
            target == "simulator"
            and any("phase" in c for c in cases) != self.phase_only
        ):
            raise ValueError("Phase and whole-window cases require separate evaluators")
        self.phase_reference = (
            None
            if phase_reference is None
            else json.loads(json.dumps(phase_reference, allow_nan=False))
        )
        if phase_reference is not None and target != "optimizer":
            raise ValueError("Only optimizer evaluation takes a phase reference")
        self.runner = runner or CandidateRunner()
        self.overhead_weight, self.max_optimizer_seconds = (
            overhead_weight,
            max_optimizer_seconds,
        )
        self._reference = (
            build_phase_simulator(self.reference_simulator_source, profile)
            if phase_reference is not None
            else load_source(self.reference_simulator_source).build_simulator(
                workflow_from_profile(profile)
            )
        )
        if phase_reference is not None:
            verify_phase_reference(self._reference, self.phase_reference)
            self.context.update(
                model_scope=SCOPE,
                phase_cases_sha256=digest(self.phase_reference["cases"]),
                phase_prediction_sha256=self.phase_reference["prediction_sha256"],
            )
        if not callable(getattr(self._reference, "evaluate", None)):
            raise ValueError("Frozen simulator does not implement evaluate")
        public, labels, oracles = [], [], []
        self._predictions = []
        for case in json.loads(json.dumps(cases, allow_nan=False)):
            observation = observation_from_dict(case["observation"])
            history = history_from_dict(case.get("history", {}))
            if target == "simulator":
                if "phase" in case and (
                    case["phase"]
                    not in {
                        "start",
                        "restore",
                        "train",
                        "checkpoint",
                        "publish",
                        "release",
                    }
                    or observation.training_config is None
                    or observation.next_batch is None
                ):
                    raise ValueError(
                        "Phase evaluation requires a real config and workload"
                    )
                forecast = forecast_from_dict(case["forecast"])
                plan_from_dict(case["plan"])
                observed = case["observed"]
                if type(observed.get("feasible")) is not bool:
                    raise ValueError("A measured feasibility label is required")
                if observed["feasible"] and not all(
                    finite_nonnegative(observed.get(key))
                    for key in ("seconds", "total_gpu_seconds")
                ):
                    raise ValueError(
                        "Finite independent measurement labels are required"
                    )
                labels.append(observed)
                oracles.append(None)
            else:
                forecast = forecast_window(
                    observation, history, self.settings["horizon"]
                )
                while (
                    len(forecast) > 1
                    and (2 * observation.max_train_gpus) ** len(forecast)
                    > self.settings["max_candidates"]
                ):
                    forecast = forecast[:-1]
                if not forecast or observation.max_train_gpus < 1:
                    raise ValueError(
                        "Optimizer cases require a known workload and GPU pool"
                    )
                oracles.append(self._oracle(observation, forecast, history))
            public.append(
                {
                    "observation": asdict(observation),
                    "history": asdict(history),
                    "forecast": [asdict(x) for x in forecast],
                    **({"plan": case["plan"]} if target == "simulator" else {}),
                    **({"phase": case["phase"]} if "phase" in case else {}),
                }
            )
        self._public_json = json.dumps(public, sort_keys=True)
        if phase_reference is not None:
            verify_phase_reference(self._reference, self.phase_reference)
        if self.phase_only:
            self.context.update(model_scope=SCOPE, phase_cases_sha256=digest(public))
        self._labels_json = json.dumps(labels, sort_keys=True)
        self._oracles = tuple(oracles)
        self._reference_files = tuple(
            Path(__file__).parent / name
            for name in (
                "evaluator.py",
                "contracts.py",
                "execution.py",
                "worker.py",
                "phase.py",
                "online.py",
                "cycle.py",
            )
        ) + (
            Path(__file__).parents[1] / "elastic/policy/basic/simulator.py",
            Path(__file__).parents[1] / "elastic/policy/basic/optimizer.py",
            Path(__file__).parents[1] / "contracts.py",
            Path(__file__).parents[1] / "elastic/runtime/policy.py",
            Path(__file__).parents[1] / "elastic/policy/basic/policy.py",
            Path(__file__).parents[1] / "elastic/loop.py",
        )
        self._code = self._code_fingerprint()
        self.benchmark_sha256 = digest(
            {
                "target": target,
                "profile": profile,
                "cases": cases,
                "settings": self.settings,
                "overhead_weight": overhead_weight,
                "max_optimizer_seconds": max_optimizer_seconds,
                "reference_simulator_source": self.reference_simulator_source,
                "phase_reference": self.phase_reference,
                "code": self._code,
            }
        )

    def _code_fingerprint(self):
        return {
            str(path.relative_to(Path(__file__).parents[1])): source_digest(
                path.read_text()
            )
            for path in self._reference_files
        }

    @property
    def settings(self) -> dict:
        return json.loads(self._settings_json)

    @property
    def calibration(self) -> dict:
        return json.loads(self._profile_json)

    def _cost(self, result):
        return (
            result.total_gpu_seconds
            if self.settings["cost_scope"] == "total"
            else result.training_gpu_seconds
        )

    def _oracle(self, observation, forecast, history):
        predictions = {}
        self._predictions.append(predictions)
        if observation.pending_publication or not observation.committed:
            return (0.0, 0)
        options = [
            PlanStep(TrainingConfig(n), release)
            for n in range(1, observation.max_train_gpus + 1)
            for release in (False, True)
        ]
        count = len(options) ** len(forecast)
        if count > self.settings["max_candidates"]:
            raise ValueError("Reference search does not fit the fixed budget")
        costs = []
        for plan in itertools.product(options, repeat=len(forecast)):
            result = self._reference.evaluate(observation, forecast, plan, history)
            if result != self._reference.evaluate(observation, forecast, plan, history):
                raise ValueError(
                    "Frozen reference is not repeatable on the same instance"
                )
            predictions[digest([asdict(step) for step in plan])] = result
            if result.feasible and (
                self.settings["max_seconds"] is None
                or result.seconds <= self.settings["max_seconds"]
            ):
                costs.append(self._cost(result))
        if not costs:
            raise ValueError("Optimizer evaluation case has no feasible reference plan")
        return min(costs), count

    def _prediction(self, index, plan):
        try:
            return self._predictions[index][digest([asdict(step) for step in plan])]
        except KeyError as error:
            raise ValueError(
                "Plan is outside the frozen reference search space"
            ) from error

    @staticmethod
    def _waiting_seconds(observation, forecast, plan, events):
        phases = iter(events)
        current = observation.training_config if observation.training_active else None
        checkpoint = observation.has_checkpoint
        delays = []

        def duration(expected):
            phase, _, seconds = next(phases)
            if phase != expected:
                raise ValueError(
                    "Reference simulator returned inconsistent phase events"
                )
            return seconds

        for future, step in zip(forecast, plan, strict=True):
            delay = 0.0
            if current is not None and step.release_before:
                delay += max(0.0, duration("release") - future.wait_seconds)
                current = None
            if current is not None and current != step.config:
                delay += duration("release")
                current = None
            if current is None:
                delay += duration("restore" if checkpoint else "start")
                current = step.config
            for phase in ("train", "checkpoint", "publish"):
                duration(phase)
            delays.append(delay)
            checkpoint = True
        return statistics.fmean(delays)

    def evaluate(self, source: str, directory: Path) -> EvaluationReport:
        public = json.loads(self._public_json)
        calls = [0] * len(public)
        rpc_errors = []

        def simulate(message):
            if self.target != "optimizer":
                raise ValueError(
                    "Simulator candidates cannot query evaluation references"
                )
            index = message["case"]
            if type(index) is not int or not 0 <= index < len(public):
                rpc_errors.append("Invalid reference case")
                raise ValueError("Invalid reference case")
            if calls[index] >= self.settings["max_candidates"]:
                rpc_errors.append(f"case {index}: search budget exceeded")
                raise ValueError("Search budget exceeded")
            calls[index] += 1
            result = self._prediction(index, plan_from_dict(message["plan"]))
            return {"result": asdict(result)}

        errors = []
        metrics, score = {}, None
        try:
            if self._code != self._code_fingerprint():
                raise ValueError("Evaluation reference code changed")
            output = self.runner.run(
                source,
                {
                    "target": self.target,
                    "profile": self.calibration,
                    "settings": json.loads(self._settings_json),
                    "cases": public,
                    "phase_only": self.phase_only,
                },
                directory,
                simulate,
            )
            if self._code != self._code_fingerprint():
                raise ValueError("Candidate changed protected evaluation code")
            if "error" in output:
                raise ValueError(output["error"])
            rows = output["rows"]
            if len(rows) != len(public):
                raise ValueError("Candidate omitted evaluation cases")
            errors.extend(rpc_errors)
            if self.target == "simulator":
                relative_errors, feasibility_errors = [], 0
                for index, (case, row, label) in enumerate(
                    zip(public, rows, json.loads(self._labels_json), strict=True)
                ):
                    if "error" in row:
                        raise ValueError(f"case {index}: {row['error']}")
                    prediction = row["prediction"]
                    if "phase" in case:
                        if type(
                            prediction.get("feasible")
                        ) is not bool or not finite_nonnegative(
                            prediction.get("seconds")
                        ):
                            raise ValueError("Invalid phase estimate")
                        feasibility_errors += (
                            prediction["feasible"] != label["feasible"]
                        )
                        if label["feasible"]:
                            relative_errors.append(
                                abs(prediction["seconds"] - label["seconds"])
                                / max(label["seconds"], 1e-6)
                                if prediction["feasible"]
                                else 1.0
                            )
                        continue
                    if type(prediction["feasible"]) is not bool:
                        raise ValueError("Invalid predicted feasibility")
                    feasibility_errors += prediction["feasible"] != label["feasible"]
                    numeric = (
                        "seconds",
                        "training_gpu_seconds",
                        "inference_gpu_seconds",
                        "extra_gpu_seconds",
                    )
                    if not all(
                        finite_nonnegative(prediction.get(key)) for key in numeric
                    ):
                        raise ValueError("Nonfinite or negative prediction")
                    if label["feasible"] and not prediction["feasible"]:
                        relative_errors.extend((1.0, 1.0))
                    if label["feasible"] and prediction["feasible"]:
                        forecast = forecast_from_dict(case["forecast"])
                        if (prediction["samples"], prediction["effective_tokens"]) != (
                            sum(x.workload.samples for x in forecast),
                            sum(x.workload.effective_tokens for x in forecast),
                        ):
                            raise ValueError("Prediction changed logical data progress")
                        if (
                            type(prediction["samples"]) is not int
                            or type(prediction["effective_tokens"]) is not int
                        ):
                            raise ValueError(
                                "Prediction progress must use integer counts"
                            )
                        if not math.isclose(
                            prediction["inference_gpu_seconds"],
                            self.calibration.get("inference_gpus", 1)
                            * prediction["seconds"],
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        ):
                            raise ValueError(
                                "Prediction changed fixed inference residency"
                            )
                        for actual, expected in (
                            (prediction["seconds"], label["seconds"]),
                            (
                                sum(prediction[key] for key in numeric[1:]),
                                label["total_gpu_seconds"],
                            ),
                        ):
                            relative_errors.append(
                                abs(actual - expected) / max(expected, 1e-6)
                            )
                metrics = {
                    "mean_relative_error": statistics.fmean(relative_errors)
                    if relative_errors
                    else 0.0,
                    "feasibility_errors": feasibility_errors,
                    "cases": len(rows),
                    "prediction_sha256": digest(rows),
                }
                score = metrics["mean_relative_error"] + feasibility_errors / len(rows)
            else:
                costs, ratios, overheads, times = [], [], [], []
                wall, waiting, transitions, samples, tokens = [], [], [], [], []
                for index, (case, row, oracle) in enumerate(
                    zip(public, rows, self._oracles, strict=True)
                ):
                    if "error" in row:
                        raise ValueError(f"case {index}: {row['error']}")
                    observation = observation_from_dict(case["observation"])
                    decision_data = dict(row["decision"])
                    if decision_data.get("config") is not None:
                        decision_data["config"] = TrainingConfig(
                            **decision_data["config"]
                        )
                    decision = ResourceDecision(**decision_data)
                    if (
                        type(decision.train) is not bool
                        or not finite_nonnegative(row["seconds"])
                        or row["seconds"] > self.max_optimizer_seconds
                    ):
                        raise ValueError(
                            "Invalid decision or optimizer time limit exceeded"
                        )
                    if observation.pending_publication or not observation.committed:
                        if row["plan"] or decision.action != "keep" or decision.train:
                            raise ValueError(
                                "Optimizer bypassed the committed/publication guard"
                            )
                        cost = 0.0
                        telemetry = (0.0, 0.0, 0.0, 0, 0)
                    else:
                        plan = plan_from_dict(row["plan"])
                        forecast = forecast_from_dict(case["forecast"])
                        if len(plan) != len(forecast):
                            raise ValueError("Optimizer changed the fixed data horizon")
                        expected = decision_for_plan(observation, plan[0])
                        if (decision.action, decision.config, decision.train) != (
                            expected.action,
                            expected.config,
                            expected.train,
                        ):
                            raise ValueError(
                                "Optimizer action does not match its evaluated plan"
                            )
                        result = self._prediction(index, plan)
                        if not result.feasible or (
                            self.settings["max_seconds"] is not None
                            and result.seconds > self.settings["max_seconds"]
                        ):
                            raise ValueError("Optimizer selected an infeasible plan")
                        if (result.samples, result.effective_tokens) != (
                            sum(x.workload.samples for x in forecast),
                            sum(x.workload.effective_tokens for x in forecast),
                        ):
                            raise ValueError("Optimizer changed logical data progress")
                        cost = self._cost(result)
                        telemetry = (
                            result.seconds,
                            self._waiting_seconds(
                                observation, forecast, plan, result.events
                            ),
                            sum(
                                seconds
                                for phase, _, seconds in result.events
                                if phase in {"start", "restore", "release"}
                            ),
                            result.samples,
                            result.effective_tokens,
                        )
                    for bucket, value in zip(
                        (wall, waiting, transitions, samples, tokens),
                        telemetry,
                        strict=True,
                    ):
                        bucket.append(value)
                    costs.append(cost)
                    ratios.append(cost / oracle[0] if oracle[0] > 0 else 1.0)
                    overheads.append(calls[index] / max(oracle[1], 1))
                    times.append(row["seconds"])
                metrics = {
                    "mean_gpu_seconds": statistics.fmean(costs),
                    "mean_wall_seconds": statistics.fmean(wall),
                    "mean_update_start_delay_seconds": statistics.fmean(waiting),
                    "mean_transition_seconds": statistics.fmean(transitions),
                    "mean_completed_samples": statistics.fmean(samples),
                    "mean_effective_tokens": statistics.fmean(tokens),
                    "mean_relative_oracle_cost": statistics.fmean(ratios),
                    "mean_reference_queries": statistics.fmean(calls),
                    "mean_query_fraction": statistics.fmean(overheads),
                    "max_optimizer_seconds": max(times),
                    "cases": len(rows),
                    "selection_sha256": digest(
                        [{k: r[k] for k in ("plan", "decision")} for r in rows]
                    ),
                }
                score = (
                    metrics["mean_relative_oracle_cost"]
                    + self.overhead_weight * metrics["mean_query_fraction"]
                )
            if not finite_nonnegative(score):
                raise ValueError("Evaluation score must be finite and nonnegative")
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        if score is not None and not finite_nonnegative(score):
            score = None
        metrics = {
            key: (
                None if isinstance(value, float) and not math.isfinite(value) else value
            )
            for key, value in metrics.items()
        }
        return EvaluationReport(
            self.target,
            source_digest(source),
            self.benchmark_sha256,
            not errors,
            score,
            metrics,
            tuple(errors),
        )
