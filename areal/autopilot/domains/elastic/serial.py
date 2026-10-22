# SPDX-License-Identifier: Apache-2.0

"""Conditional timing revision and frozen joint search for the serial fixture.

Validation labels stay in this evaluator. Output token counts are conditioned on,
not forecast by this gate. Transition timing/allocation profiles remain fixed.
"""

from __future__ import annotations

import copy
import fcntl
import json
from dataclasses import asdict, dataclass

from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE
from areal.autopilot.domains.elastic.cycle import _fixed_json, _stage
from areal.autopilot.domains.elastic.joint_evaluator import JointOptimizerEvaluator
from areal.autopilot.revision.contracts import EvaluationReport, digest, source_digest
from areal.autopilot.revision.execution import load_source
from areal.autopilot.revision.loop import RevisionLoop
from areal.elastic.contracts import BatchWorkload
from areal.elastic.policy.basic.optimizer import JointOptimizer
from areal.elastic.policy.basic.serial_planning import (
    OnlineJointPolicy,
    SerialTimingInput,
    logical_workload,
    serial_config,
    serial_model_from_profile,
    serial_window_from_dict,
    timing_input_from_dict,
)

from .adapter import ElasticEvaluator, candidate_runner
from .identity import AREAL_ROOT, protected_files

SCOPE = "serial-joint-timing-v1"


def static_profile(profile):
    return copy.deepcopy(
        {k: v for k, v in profile.items() if k != "runtime_calibration"}
    )


def joint_settings(settings):
    return JointOptimizer(
        **{
            **settings,
            "candidates": tuple(serial_config(c) for c in settings["candidates"]),
        }
    )


def build_serial_model(source, profile, *, idle_seconds=30):
    """A fixed simulator owns coverage, ordering, allocations and all accounting."""
    model = serial_model_from_profile(
        static_profile(profile), idle_seconds=idle_seconds
    )
    predictor = load_source(source).build_serial_model(static_profile(profile))
    if not callable(getattr(predictor, "predict_timing", None)):
        raise ValueError("Serial model factory must implement predict_timing")
    model.timing_predictor = predictor
    return model


def timing_predictions(model, cases):
    def predict():
        return [asdict(model.estimate_timing(timing_input_from_dict(c))) for c in cases]

    rows = predict()
    if rows != predict():
        raise ValueError("Serial model predictions must reproduce on the same instance")
    return rows


def verify_timing_reference(model, reference):
    if (
        digest(timing_predictions(model, reference["cases"]))
        != reference["prediction_sha256"]
    ):
        raise ValueError("Serial model differs from accepted worker predictions")


def runtime_joint_problems(profile, window, *, settings, runner=None):
    updates = window["updates"]
    if len(updates) < 4:
        raise ValueError("Joint revision needs at least four committed updates")
    split = max(2, min(len(updates) - 1, int(len(updates) * 0.6)))
    calibration, validation = updates[:split], updates[split:]
    history = OnlineJointPolicy(
        serial_model_from_profile(profile), joint_settings(settings)
    )
    # Independent release boundaries lack episode attribution. They do not enter
    # this chronological split or become validation labels.
    for update in calibration:
        history.record_committed(update)
    phases, episodes = tuple(history.phases), tuple(history.episodes)
    cases = []
    for update in validation:
        config = serial_config(update.get("joint_config"))
        if config is None:
            raise ValueError("Joint history is missing actual full configurations")
        for phase in update["measurements"]:
            if phase["config"] != asdict(config.actor) or phase["workload"] is None:
                continue
            if update.get("config_receipt") is not None and phase["phase"] not in {
                "train",
                "checkpoint",
                "publish",
            }:
                continue
            workload = logical_workload(BatchWorkload(**phase["workload"]))
            cases.append(
                {
                    "input": asdict(
                        SerialTimingInput(
                            config, phase["phase"], workload, phases, episodes
                        )
                    ),
                    "seconds": phase["seconds"],
                }
            )
        episode = update.get("episode")
        if episode is not None:
            # Generation precedes ready-batch reconfiguration. Never synthesize
            # an Actor/Rollout combination that was not actually executed.
            source_row = episode.get("joint_config") or update.get(
                "joint_window", {}
            ).get("current")
            if source_row is None and update.get("config_receipt") is None:
                source_row = update["joint_config"]
            if source_row is None:
                continue  # Old switched rows without a complete source are unknown.
            source = serial_config(source_row)
            if asdict(source.rollout) != episode["rollout"]:
                raise ValueError(
                    "Episode source configuration and Rollout identity disagree"
                )
            cases.append(
                {
                    "input": asdict(
                        SerialTimingInput(
                            source,
                            "episode",
                            logical_workload(BatchWorkload(**episode["workload"])),
                            phases,
                            episodes,
                        )
                    ),
                    "seconds": episode["seconds"],
                }
            )
    # Search quality has no real timing labels. Use calibration planning windows
    # only, so deployment probes never expose held-out timing labels either.
    scenarios = tuple(
        serial_window_from_dict(u["joint_window"])
        for u in calibration
        if u.get("joint_window") is not None
    )
    if not cases or not scenarios:
        raise ValueError(
            "Joint revision needs measured phases and recorded planning windows"
        )
    return SerialModelEvaluator(
        profile, calibration, cases, settings=settings, runner=runner
    ), scenarios


@dataclass(frozen=True)
class SerialModelCriteria:
    min_relative_improvement: float = 0.005
    max_group_regression: float = 0.0


class SerialModelEvaluator(ElasticEvaluator):
    """Fixed labels, per-phase/full-config error and identical-sample coverage."""

    target = "serial_model"

    def __init__(self, profile, calibration, cases, *, settings, runner=None):
        self._profile_json = json.dumps(
            static_profile(profile), sort_keys=True, allow_nan=False
        )
        self._calibration_json = json.dumps(
            calibration, sort_keys=True, allow_nan=False
        )
        self._cases_json = json.dumps(cases, sort_keys=True, allow_nan=False)
        self._public_json = json.dumps(
            [c["input"] for c in cases], sort_keys=True, allow_nan=False
        )
        self._settings_json = json.dumps(
            asdict(joint_settings(settings)), sort_keys=True, allow_nan=False
        )
        self._criteria, self._runner = (
            SerialModelCriteria(),
            candidate_runner(runner),
        )
        self._files = protected_files()
        self._code = self._code_fingerprint()
        self._context = {
            "model_scope": SCOPE,
            "profile_sha256": digest(self.profile),
            "calibration_sha256": digest(json.loads(self._calibration_json)),
            "cases_sha256": digest(json.loads(self._public_json)),
            "labels_sha256": digest(json.loads(self._cases_json)),
            "settings": self.settings,
            "criteria": asdict(self.criteria),
            "worker": {
                "command": list(self.runner.worker_command),
                "timeout_seconds": self.runner.timeout_seconds,
                "max_output_bytes": self.runner.max_output_bytes,
            },
            "code": self._code,
        }
        self.benchmark_sha256 = digest(self._context)
        self._reports = {}

    @property
    def criteria(self):
        return self._criteria

    @property
    def runner(self):
        return self._runner

    @property
    def profile(self):
        return json.loads(self._profile_json)

    @property
    def calibration(self):
        return {
            "profile": self.profile,
            "runtime_calibration": json.loads(self._calibration_json),
        }

    @property
    def settings(self):
        return json.loads(self._settings_json)

    @property
    def context(self):
        return copy.deepcopy(self._context)

    def _code_fingerprint(self):
        base = AREAL_ROOT
        return {
            str(p.relative_to(base)): source_digest(p.read_text()) for p in self._files
        }

    def assert_unchanged(self):
        if self._code_fingerprint() != self._code:
            raise ValueError("Serial model evaluation/runtime code changed")

    def evaluate(self, source, directory):
        errors, metrics, score, groups, coverage = [], {}, None, {}, []
        try:
            self.assert_unchanged()
            result = self.runner.run(
                source,
                {
                    "target": self.target,
                    "profile": self.profile,
                    "cases": json.loads(self._public_json),
                },
                directory,
                lambda _: (_ for _ in ()).throw(ValueError("No model RPC")),
            )
            if result.get("error"):
                raise ValueError(result["error"])
            rows = result["rows"]
            model = build_serial_model(source, self.profile)
            if rows != timing_predictions(model, json.loads(self._public_json)):
                raise ValueError("Worker and deployed timing predictions differ")
            cases = json.loads(self._cases_json)
            if len(rows) != len(cases):
                raise ValueError("Model omitted timing cases")
            for case, row in zip(cases, rows, strict=True):
                available = row["seconds"] is not None
                coverage.append(available)
                key = case["input"]["phase"] + ":" + digest(case["input"]["config"])
                group = groups.setdefault(key, [0.0, 0.0, 0, 0])
                group[3] += 1
                if available:
                    group[0] += abs(row["seconds"] - case["seconds"])
                    group[1] += case["seconds"]
                    group[2] += 1
            if not any(coverage):
                raise ValueError("No timing validation cases covered")
            wape = {k: g[0] / max(g[1], 1e-9) for k, g in groups.items() if g[2]}
            score = sum(wape.values()) / len(wape)
            metrics = {
                "mean_group_wape": score,
                "groups": {
                    k: {"wape": wape.get(k), "covered": g[2], "total": g[3]}
                    for k, g in groups.items()
                },
                "covered": sum(coverage),
                "total": len(cases),
                "prediction_sha256": digest(rows),
            }
            self.assert_unchanged()
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
        self._reports[digest(asdict(report))] = (coverage, metrics.get("groups", {}))
        return report

    def _verified(self, report):
        self.assert_unchanged()
        if (
            report.benchmark_sha256 != self.benchmark_sha256
            or digest(asdict(report)) not in self._reports
        ):
            raise ValueError("Comparison requires issued serial model reports")
        return self._reports[digest(asdict(report))]

    def compare(self, candidate, incumbent):
        new, old = self._verified(candidate), self._verified(incumbent)
        if not incumbent.valid:
            raise ValueError("Incumbent serial timing model is invalid")
        if not candidate.valid:
            return False, "Timing model evaluation failed"
        if new[0] != old[0]:
            return False, "Timing coverage changed; compare identical covered samples"
        for key, group in old[1].items():
            if (
                group["wape"] is not None
                and new[1][key]["wape"]
                > group["wape"] + self.criteria.max_group_regression + 1e-9
            ):
                return False, "Per-phase/full-config timing error regressed"
        if incumbent.score - candidate.score > max(
            1e-12, incumbent.score * self.criteria.min_relative_improvement
        ):
            return True, "Improved historical conditional timing accuracy"
        return False, "Insufficient timing accuracy improvement"

    def feedback(self, report):
        self._verified(report)
        metrics = copy.deepcopy(report.metrics)
        metrics.pop("prediction_sha256", None)
        groups = metrics.get("groups", {})
        suppressed = not groups or any(g["covered"] < 5 for g in groups.values())
        for group in groups.values():
            group["error_suppressed"] = group["covered"] < 5
            if group["error_suppressed"]:
                group["wape"] = None
        if suppressed:
            metrics["mean_group_wape"] = None
        return {
            "score": None if suppressed else report.score,
            "metrics": metrics,
            "errors": list(report.errors),
        }


class SerialRevisionCycle:
    """Separate model and optimizer gates, one atomic executable policy bundle."""

    def __init__(
        self,
        model,
        scenarios,
        root,
        *,
        initial_model_source=None,
        initial_optimizer_source=None,
        parent_bundle=None,
        idle_seconds=30,
        horizon=2,
    ):
        self.model, self.scenarios, self.root = model, scenarios, root
        self.model_source = initial_model_source or BASELINE_SOURCE["serial_model"]
        self.optimizer_source = (
            initial_optimizer_source or BASELINE_SOURCE["joint_optimizer"]
        )
        self.parent, self.idle_seconds, self.horizon = (
            parent_bundle,
            idle_seconds,
            horizon,
        )
        root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        model_agent,
        optimizer_agent,
        *,
        max_model_rounds=2,
        max_optimizer_rounds=2,
    ):
        from areal.autopilot.domains.elastic.serial_policy import export_serial_policy

        with (self.root / "cycle.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _fixed_json(
                self.root / "context.json",
                {
                    "model": self.model.benchmark_sha256,
                    "scenarios": digest([asdict(s) for s in self.scenarios]),
                    "sources": [
                        source_digest(self.model_source),
                        source_digest(self.optimizer_source),
                    ],
                    "rounds": [max_model_rounds, max_optimizer_rounds],
                    "idle_seconds": self.idle_seconds,
                    "horizon": self.horizon,
                },
            )
            model = _stage(
                RevisionLoop(
                    model_agent,
                    self.model,
                    self.root / "model",
                    initial_source=self.model_source,
                ),
                max_model_rounds,
            )
            report = json.loads((model.root / model.record["report"]).read_text())
            reference = {
                "cases": json.loads(self.model._public_json),
                "prediction_sha256": report["metrics"]["prediction_sha256"],
            }
            simulator = build_serial_model(
                model.source.read_text(),
                self.model.profile,
                idle_seconds=self.idle_seconds,
            )
            verify_timing_reference(simulator, reference)
            settings = joint_settings(self.model.settings)
            model_version = "serial-model:" + digest(
                {
                    "source": model.record["source_sha256"],
                    "context": self.model.benchmark_sha256,
                    "reference": reference,
                    "scope": SCOPE,
                    "horizon": self.horizon,
                    "idle_seconds": self.idle_seconds,
                }
            )
            search = JointOptimizerEvaluator(
                simulator,
                self.scenarios,
                settings.candidates,
                model_version=model_version,
                max_queries=settings.max_evaluations,
                max_job_seconds=settings.max_seconds,
                switch_margin=settings.switch_margin,
                runner=self.model.runner,
            )
            _fixed_json(
                self.root / "handoff.json",
                {
                    "model_version": model_version,
                    "optimizer_benchmark": search.benchmark_sha256,
                },
            )
            optimizer = _stage(
                RevisionLoop(
                    optimizer_agent,
                    search,
                    self.root / "optimizer",
                    initial_source=self.optimizer_source,
                ),
                max_optimizer_rounds,
            )
            bundle = None
            if model.record["accepted"] or optimizer.record["accepted"]:
                bundle = export_serial_policy(
                    self.model,
                    search,
                    model,
                    optimizer,
                    self.root / "bundles",
                    reference=reference,
                    parent=self.parent,
                    idle_seconds=self.idle_seconds,
                    horizon=self.horizon,
                )
            result = {
                "status": "complete",
                "model": model.record,
                "optimizer": optimizer.record,
                "bundle": None if bundle is None else str(bundle.resolve()),
            }
            _fixed_json(self.root / "result.json", result)
            return result
