# SPDX-License-Identifier: Apache-2.0

"""Historical predictor factories in bounded workers, with parent-owned scoring."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType

from areal.autopilot.contracts import BatchWorkload, TrainingConfig, WorkloadStats
from areal.autopilot.elastic.policy.basic.cost_model.config import (
    ActorConfig,
    ParallelLayout,
)
from areal.autopilot.elastic.policy.basic.cost_model.rollout import (
    RolloutConfig,
    RolloutStage,
    RolloutWorkload,
)
from areal.autopilot.elastic.policy.basic.joint import (
    JointBatch,
    JointConfig,
    JointEvent,
    JointPrediction,
    JointScenario,
)
from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
from areal.autopilot.revision.contracts import EvaluationReport, digest, source_digest
from areal.autopilot.revision.execution import CandidateRunner
from areal.autopilot.revision.model_evaluator import (
    HistoricalCase,
    HistoricalInput,
    HistoricalModelEvaluator,
    HistoricalPrediction,
    HistoryReport,
    HistorySpan,
)


def _rollout_workload(data: dict | None) -> RolloutWorkload | None:
    return (
        None
        if data is None
        else RolloutWorkload(
            **{**data, "stages": tuple(RolloutStage(**s) for s in data["stages"])}
        )
    )


def joint_config_from_dict(data: dict | None) -> JointConfig | None:
    return (
        None
        if data is None
        else JointConfig(
            TrainingConfig(**data["actor"]), RolloutConfig(**data["rollout"])
        )
    )


def joint_scenario_from_dict(work: dict) -> JointScenario:
    if "observation" in work and "episodes" in work:
        from areal.autopilot.elastic.policy.basic.online import serial_window_from_dict

        return serial_window_from_dict(work)
    batches = []
    for value in work["batches"]:
        batch = dict(value)
        batch.pop("inferred_lengths", None)
        batch.pop("_prediction_stats", None)
        batch.update(
            actor=BatchWorkload(**batch["actor"]),
            rollout=_rollout_workload(batch["rollout"]),
            source_actor=TrainingConfig(**batch["source_actor"])
            if batch.get("source_actor") is not None
            else None,
            logical_stats=WorkloadStats(**batch["logical_stats"])
            if batch.get("logical_stats") is not None
            else None,
        )
        batches.append(JointBatch(**batch))
    return JointScenario(
        **{
            **work,
            "current": joint_config_from_dict(work["current"]),
            "batches": tuple(batches),
        }
    )


def historical_input_from_dict(data: dict) -> HistoricalInput:
    """Round-trip the public typed prediction inputs, without measurement fields."""
    data = dict(data)
    config, work = data["config"], data["workload"]
    if data["kind"] == "actor":
        config = ActorConfig(
            **{**config, "parallel": ParallelLayout(**config["parallel"])}
        )
        work = WorkloadStats(**work)
    elif data["kind"] == "rollout":
        config, work = RolloutConfig(**config), _rollout_workload(work)
    elif data["kind"] == "joint":
        config, work = joint_config_from_dict(config), joint_scenario_from_dict(work)
    return HistoricalInput(**{**data, "config": config, "workload": work})


def historical_case_from_dict(data: dict) -> HistoricalCase:
    return HistoricalCase(
        **{
            **data,
            "span": HistorySpan(**data["span"]),
            "inputs": historical_input_from_dict(data["inputs"]),
        }
    )


def joint_prediction_from_dict(data: dict) -> JointPrediction:
    return JointPrediction(
        **{
            **data,
            "config": joint_config_from_dict(data["config"]),
            "events": tuple(JointEvent(**e) for e in data["events"]),
        }
    )


class FrozenJointModel:
    """Immutable finite export; neither a fallback model nor an extrapolator."""

    def __init__(self, requests: list[dict], predictions: list[dict]):
        if not requests or len(requests) != len(predictions):
            raise ValueError("Joint export must cover every fixed model request")
        table, rows = {}, []
        for request, data in zip(requests, predictions, strict=True):
            value = joint_prediction_from_dict(data)
            if asdict(value.config) != request["config"]:
                raise ValueError("Joint export changed the requested configuration")
            scenario = joint_scenario_from_dict(request["scenario"])
            if value.status == "complete" and (
                value.completed_batches,
                value.samples,
                value.effective_tokens,
                value.sequence_tokens,
            ) != (
                len(scenario.batches),
                sum(b.actor.samples for b in scenario.batches),
                sum(b.actor.effective_tokens for b in scenario.batches),
                sum(b.actor.sequence_tokens for b in scenario.batches),
            ):
                raise ValueError("Joint export changed the complete workload window")
            key = digest(request["scenario"]), digest(request["config"])
            encoded = json.dumps(asdict(value), sort_keys=True, allow_nan=False)
            if key in table and table[key] != encoded:
                raise ValueError(
                    "Duplicate joint requests produced different predictions"
                )
            table[key] = encoded
            rows.append({"request": request, "prediction": asdict(value)})
        self._table = MappingProxyType(table)
        self._sha256 = digest(rows)

    @property
    def sha256(self) -> str:
        return self._sha256

    def evaluate(self, scenario: JointScenario, config: JointConfig) -> JointPrediction:
        key = digest(asdict(scenario)), digest(asdict(config))
        if key not in self._table:
            raise ValueError("Requested work/config is outside the frozen joint export")
        return joint_prediction_from_dict(json.loads(self._table[key]))


def prediction_fingerprint(rows: list[dict]) -> str:
    """Bind numerical predictions, independent of metric ordering/reason wording."""
    return digest(
        [
            sorted(dict(HistoricalPrediction(**row["prediction"]).values).items())
            for row in rows
        ]
    )


class HistoricalSourceEvaluator:
    """RevisionLoop adapter. Factory: build_model(calibration_cases, settings).

    Settings and declared pre-existing fit spans are caller-supplied and frozen.
    Factory calibration may use only the calibration partition. Workers receive
    no validation/final labels or identifiers, and no final prediction inputs.
    Candidate Python remains trusted code; this is not a filesystem sandbox.
    """

    target = "model"

    def __init__(
        self,
        history: HistoricalModelEvaluator,
        *,
        settings: dict,
        fitted_on: tuple[HistorySpan, ...],
        runner: CandidateRunner | None = None,
        joint_requests: list[dict] | None = None,
    ):
        self._history = history
        self._runner = runner or CandidateRunner()
        if not isinstance(settings, dict):
            raise ValueError("Historical model settings must be a JSON object")
        fitted_on = tuple(fitted_on)
        history.validate_fit(fitted_on)
        self._settings_json = json.dumps(settings, sort_keys=True, allow_nan=False)
        self._joint_requests_json = (
            None
            if joint_requests is None
            else json.dumps(joint_requests, sort_keys=True, allow_nan=False)
        )
        if joint_requests is not None and not joint_requests:
            raise ValueError("Required joint export needs nonempty fixed requests")
        self._fitted_on = tuple(
            sorted(
                set(fitted_on) | {case.span for case in history.calibration},
                key=lambda s: (s.run_id, s.first_step, s.last_step),
            )
        )
        history.validate_fit(self._fitted_on)
        self._reports: dict[str, HistoryReport | None] = {}
        root = Path(__file__).parent
        factory_files = tuple(
            root / name
            for name in (
                "model_execution.py",
                "contracts.py",
                "worker.py",
                "execution.py",
                "loop.py",
            )
        )
        basic = root.parent / "elastic/policy/basic"
        self._reference_files = factory_files + tuple(
            sorted(p for p in basic.rglob("*") if p.suffix in {".py", ".json"})
        )
        self._code = self._code_fingerprint()
        self._context = {
            "api_version": "historical-model-v1",
            "history_sha256": history.benchmark_sha256,
            "settings_sha256": digest(self.settings),
            "fitted_on": [asdict(s) for s in self._fitted_on],
            "worker_limits": {
                "timeout_seconds": self.runner.timeout_seconds,
                "max_output_bytes": self.runner.max_output_bytes,
                "command": list(
                    self.runner.command
                    or (sys.executable, "-m", "areal.autopilot.revision.worker")
                ),
            },
            "reproducibility_runs": 2,
            "joint_requests_sha256": None
            if joint_requests is None
            else digest(joint_requests),
            "reference_code": list(self._code),
        }
        self._benchmark_sha256 = digest(self._context)

    @property
    def history(self) -> HistoricalModelEvaluator:
        return self._history

    @property
    def runner(self) -> CandidateRunner:
        return self._runner

    @property
    def criteria(self):
        return self.history.criteria

    @property
    def benchmark_sha256(self) -> str:
        return self._benchmark_sha256

    def assert_unchanged(self) -> None:
        self.history.assert_unchanged()
        if self._code_fingerprint() != self._code:
            raise ValueError("Historical factory/worker code changed")

    def _code_fingerprint(self) -> tuple[str, ...]:
        root = Path(__file__).parent.parent
        return tuple(
            digest((str(path.relative_to(root)), source_digest(path.read_text())))
            for path in self._reference_files
        )

    @property
    def context(self) -> dict:
        return json.loads(json.dumps(self._context))

    @property
    def settings(self) -> dict:
        return json.loads(self._settings_json)

    @property
    def calibration(self) -> dict:
        return {
            "cases": [asdict(c) for c in self.history.calibration],
            **(
                {
                    "required_joint_export": "Factory predictor must expose .joint.evaluate(JointScenario, JointConfig) -> JointPrediction; preserve this model export while revising historical accuracy."
                }
                if self._joint_requests_json is not None
                else {}
            ),
        }

    def with_joint_export(self, requests: list[dict]) -> HistoricalSourceEvaluator:
        self.assert_unchanged()
        if (
            self._joint_requests_json is not None
            and json.loads(self._joint_requests_json) != requests
        ):
            raise ValueError("Cannot replace an already fixed joint export problem")
        return HistoricalSourceEvaluator(
            self.history,
            settings=self.settings,
            fitted_on=self._fitted_on,
            runner=self.runner,
            joint_requests=requests,
        )

    def evaluate(self, source: str, directory: Path) -> EvaluationReport:
        raw, score, metrics, errors = None, None, {}, []
        violations = []

        def no_reference(message):
            violations.append("Historical models cannot query evaluation references")
            raise ValueError(violations[-1])

        try:
            self.assert_unchanged()
            inputs = self.history.prediction_inputs
            request = {
                "target": self.target,
                "settings": self.settings,
                "calibration": self.calibration,
                "cases": [asdict(i) for i in inputs],
                **(
                    {"joint_cases": json.loads(self._joint_requests_json)}
                    if self._joint_requests_json is not None
                    else {}
                ),
            }
            output = self.runner.run(source, request, directory, no_reference)
            self.assert_unchanged()
            if output.get("error") or violations:
                raise ValueError(output.get("error") or "; ".join(violations))
            rows = output.get("rows")
            if not isinstance(rows, list) or len(rows) != len(inputs):
                raise ValueError(
                    "Candidate must return one prediction per historical input"
                )
            prediction_sha = None
            joint_sha = None
            if all(isinstance(row, dict) and not row.get("error") for row in rows):
                prediction_sha = prediction_fingerprint(rows)
                repeated = self.runner.run(
                    source, request, directory / "reproduce", no_reference
                )
                self.assert_unchanged()
                if (
                    repeated.get("error")
                    or violations
                    or not isinstance(repeated.get("rows"), list)
                    or len(repeated["rows"]) != len(inputs)
                ):
                    raise ValueError(
                        "Historical model factory failed independent reproduction"
                    )
                if prediction_fingerprint(repeated["rows"]) != prediction_sha:
                    raise ValueError(
                        "Historical model factory predictions are not reproducible"
                    )
                if self._joint_requests_json is not None:
                    first_joint = FrozenJointModel(
                        request["joint_cases"], output["joint_predictions"]
                    )
                    repeated_joint = FrozenJointModel(
                        request["joint_cases"], repeated["joint_predictions"]
                    )
                    if first_joint.sha256 != repeated_joint.sha256:
                        raise ValueError(
                            "Independent joint model exports are not reproducible"
                        )
                    joint_sha = first_joint.sha256

            class Predictions:
                def __init__(self):
                    self.rows = iter(rows)

                def predict(self, inputs):
                    row = next(self.rows)
                    if row.get("error"):
                        raise ValueError(row["error"])
                    return HistoricalPrediction(**row["prediction"])

            raw = self.history.evaluate(
                Predictions(), source_digest(source), fitted_on=self._fitted_on
            )
            atomic_json(directory / "history-evaluation.json", asdict(raw))
            score, metrics, errors = raw.score, raw.feedback(), list(raw.errors)
            metrics["prediction_sha256"] = prediction_sha
            if self._joint_requests_json is not None:
                metrics["joint_snapshot_sha256"] = joint_sha
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
        self._reports[digest(asdict(report))] = raw
        return report

    def _verified(self, report: EvaluationReport) -> HistoryReport | None:
        key = digest(asdict(report))
        if report.benchmark_sha256 != self.benchmark_sha256 or key not in self._reports:
            raise ValueError(
                "Historical source comparison requires unchanged issued reports"
            )
        self.assert_unchanged()
        return self._reports[key]

    def compare(
        self, candidate: EvaluationReport, incumbent: EvaluationReport
    ) -> tuple[bool, str]:
        old, new = self._verified(incumbent), self._verified(candidate)
        if not incumbent.valid or old is None:
            raise ValueError("Incumbent historical factory must evaluate successfully")
        if not candidate.valid or new is None:
            return False, "Candidate execution or historical prediction failed"
        result = self.history.compare(new, old)
        return result.accepted, result.reason

    def feedback(self, report: EvaluationReport) -> dict:
        self._verified(report)
        metrics = json.loads(json.dumps(report.metrics))
        metrics.pop("prediction_sha256", None)
        metrics.pop("joint_snapshot_sha256", None)
        return {
            "score": metrics.get("score"),
            "metrics": metrics,
            "errors": list(report.errors),
        }

    def verify_loaded(self, predictor, expected_sha: str) -> None:
        """Check a rebuilt frozen predictor twice against accepted public-input work."""
        self.assert_unchanged()
        if not isinstance(expected_sha, str) or not expected_sha:
            raise ValueError("Accepted model has no prediction fingerprint")
        for _ in range(2):
            rows = []
            for inputs in self.history.prediction_inputs:
                value = predictor.predict(copy.deepcopy(inputs))
                if not isinstance(value, HistoricalPrediction):
                    raise ValueError("Loaded model must return HistoricalPrediction")
                rows.append(
                    {
                        "prediction": asdict(
                            HistoricalPrediction(value.values, value.reason)
                        )
                    }
                )
            self.assert_unchanged()
            if prediction_fingerprint(rows) != expected_sha:
                raise ValueError(
                    "Loaded model predictions differ from accepted fitted model"
                )

    def export_joint(
        self,
        source: str,
        requests: list[dict],
        directory: Path,
        *,
        expected_prediction_sha: str,
        expected_joint_sha: str | None = None,
    ) -> FrozenJointModel:
        """Bind an explicit predictor.joint export to its historical prediction receipt."""
        if not isinstance(expected_prediction_sha, str) or not expected_prediction_sha:
            raise ValueError("Joint handoff requires a historical prediction receipt")
        self.assert_unchanged()
        if (
            self._joint_requests_json is not None
            and json.loads(self._joint_requests_json) != requests
        ):
            raise ValueError(
                "Joint export requests differ from the fixed model context"
            )
        request = {
            "target": self.target,
            "settings": self.settings,
            "calibration": self.calibration,
            "cases": [asdict(i) for i in self.history.prediction_inputs],
            "joint_cases": json.loads(json.dumps(requests, allow_nan=False)),
        }
        violations, snapshot = [], None

        def no_reference(message):
            violations.append("Model export cannot query evaluation references")
            raise ValueError(violations[-1])

        for repeat in range(2):
            output = self.runner.run(
                source, request, directory / f"run-{repeat}", no_reference
            )
            self.assert_unchanged()
            if output.get("error") or violations:
                raise ValueError(output.get("error") or "; ".join(violations))
            rows = output.get("rows")
            if (
                not isinstance(rows, list)
                or len(rows) != len(request["cases"])
                or prediction_fingerprint(rows) != expected_prediction_sha
            ):
                raise ValueError(
                    "Model export differs from historical prediction receipt"
                )
            values = output.get("joint_predictions")
            if not isinstance(values, list):
                raise ValueError("Model factory must export predictor.joint.evaluate")
            current = FrozenJointModel(request["joint_cases"], values)
            if expected_joint_sha is not None and current.sha256 != expected_joint_sha:
                raise ValueError(
                    "Joint export differs from historical model acceptance receipt"
                )
            if snapshot is not None and current.sha256 != snapshot.sha256:
                raise ValueError("Independent joint model exports are not reproducible")
            snapshot = current
        atomic_json(
            directory / "joint-export.json",
            {
                "source_sha256": source_digest(source),
                "historical_prediction_sha256": expected_prediction_sha,
                "joint_snapshot_sha256": snapshot.sha256,
            },
        )
        return snapshot
