# SPDX-License-Identifier: Apache-2.0

"""One resumable offline cycle: historical model, frozen joint model, optimizer."""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from areal.autopilot.elastic.runtime.state.checkpoint import atomic_json
from areal.autopilot.revision.agent import RevisionAgent
from areal.autopilot.revision.contracts import BASELINE_SOURCE, digest, source_digest
from areal.autopilot.revision.joint_evaluator import JointOptimizerEvaluator
from areal.autopilot.revision.loop import RevisionLoop
from areal.autopilot.revision.model_execution import HistoricalSourceEvaluator


def _fixed_json(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Saved {path.name} differs from the fixed revision cycle")
    else:
        atomic_json(path, value)


def _initial_source(path: Path, source: str) -> None:
    if path.exists():
        if path.read_text() != source:
            raise ValueError("Stage initial source changed")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class _Stage:
    root: Path
    source: Path
    record: dict

    def check(self):
        if (
            source_digest(self.source.read_text()) != self.record["source_sha256"]
            or digest(json.loads((self.root / self.record["report"]).read_text()))
            != self.record["report_sha256"]
        ):
            raise ValueError(
                "Completed stage source/evaluation changed after verification"
            )


def _stage(loop: RevisionLoop, budget: int) -> _Stage:
    """Recover durable child results; count started attempts against the total cap."""
    root, evaluator = loop.store.root, loop.evaluator
    initial = loop.initial_source or BASELINE_SOURCE[evaluator.target]
    initial_path = root / "initial.py"
    receipt_path = root / "stage.json"
    with (root / "revision.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        context_path = root / "context.json"
        if context_path.exists():
            if json.loads(context_path.read_text()) != loop.context:
                raise ValueError(
                    "Child stage belongs to another fixed evaluation context"
                )
        elif any(
            (root / name).exists()
            for name in (
                "active.json",
                "stage.json",
                "run.json",
                "baseline.json",
                "attempts",
            )
        ):
            raise ValueError("Child stage is missing its evaluation context")
        _initial_source(initial_path, initial)
        attempts = sorted((root / "attempts").glob("round-*"))
        if len(attempts) > budget:
            raise ValueError("Started revision attempts exceed the cycle budget")
        active = loop.store.active()
        # A promotion may have crashed after its acceptance receipt was durable.
        pending = [p for p in attempts if (p / "acceptance.json").exists()]
        if active is None and pending:
            if len(pending) != 1 or receipt_path.exists():
                raise ValueError("Missing active pointer for an already accepted stage")
            candidate = pending[0] / "candidate.py"
            loop.store.promote(
                candidate, json.loads((pending[0] / "evaluation.json").read_text())
            )
            active = loop.store.active()
        complete = (
            receipt_path.exists() or active is not None or (root / "run.json").exists()
        )
        remaining = budget - len(attempts)
        if not complete and remaining:
            # Already hold the same lock used by RevisionLoop.run; avoid relocking
            # and keep the budget check atomic with any new proposals.
            loop._run(remaining, True)
            active = loop.store.active()
        if not context_path.exists() or not (root / "baseline.json").exists():
            raise ValueError("Stage has no completed baseline evaluation")
        if (root / "run.json").exists():
            run = json.loads((root / "run.json").read_text())
            if (
                run["target"] != evaluator.target
                or run["benchmark_sha256"] != evaluator.benchmark_sha256
            ):
                raise ValueError(
                    "Child run result belongs to another evaluation context"
                )
            if active is None and run.get("active_source") is not None:
                raise ValueError("Accepted child run is missing its active source")
        source = active[0] if active is not None else initial_path
        report_path = (
            source.parent / "evaluation.json"
            if active is not None
            else root / "baseline.json"
        )
        report = json.loads(report_path.read_text())
        baseline_path = root / "baseline.json"
        record = {
            "target": evaluator.target,
            "context_sha256": digest(loop.context),
            "source": str(source.relative_to(root)),
            "source_sha256": source_digest(source.read_text()),
            "report": str(report_path.relative_to(root)),
            "report_sha256": digest(report),
            "baseline_sha256": digest(json.loads(baseline_path.read_text())),
            "accepted": active is not None,
        }
        if receipt_path.exists() and json.loads(receipt_path.read_text()) != record:
            raise ValueError("Completed stage source/receipt changed after handoff")
        if (
            not report["valid"]
            or report["target"] != evaluator.target
            or report["benchmark_sha256"] != evaluator.benchmark_sha256
            or report["candidate_sha256"] != record["source_sha256"]
        ):
            raise ValueError("Effective stage source has no matching valid evaluation")
        checked = evaluator.evaluate(
            source.read_text(), root / f"verify-{uuid.uuid4().hex}"
        )
        keys = (
            ("prediction_sha256", "joint_snapshot_sha256")
            if evaluator.target == "model"
            else ("prediction_sha256",)
            if evaluator.target in {"simulator", "serial_model"}
            else ("selection_sha256",)
        )
        if not checked.valid or any(
            checked.metrics.get(k) != report["metrics"].get(k) for k in keys
        ):
            raise ValueError(
                "Effective stage predictions/selections changed since evaluation"
            )
        _fixed_json(receipt_path, record)
        return _Stage(root, source, record)


class PolicyRevisionCycle:
    """Bounded offline model→optimizer handoff, idempotent for one fixed root.

    The model factory must export predictor.joint.evaluate. Baseline and accepted
    models follow identical export checks. A new training window or experiment
    uses a new root with explicit initial sources and fit provenance.
    """

    def __init__(
        self,
        model_evaluator: HistoricalSourceEvaluator,
        optimizer_problem: JointOptimizerEvaluator,
        root: Path,
        *,
        initial_model_source: str,
        initial_optimizer_source: str | None = None,
    ):
        if (
            not isinstance(initial_model_source, str)
            or not initial_model_source.strip()
        ):
            raise ValueError(
                "A revision cycle requires an explicit initial model factory"
            )
        if initial_optimizer_source is not None and (
            not isinstance(initial_optimizer_source, str)
            or not initial_optimizer_source.strip()
        ):
            raise ValueError("Initial optimizer source must be a nonempty factory")
        self._requests = optimizer_problem.model_requests
        self._model = model_evaluator.with_joint_export(self._requests)
        self._problem = optimizer_problem
        self._model_source = initial_model_source
        self._optimizer_source = (
            initial_optimizer_source or BASELINE_SOURCE["joint_optimizer"]
        )
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._code = source_digest(Path(__file__).read_text())

    def _check(self):
        self._model.assert_unchanged()
        self._problem.assert_unchanged()
        if source_digest(Path(__file__).read_text()) != self._code:
            raise ValueError("Revision cycle implementation changed")

    def run(
        self,
        model_agent: RevisionAgent,
        optimizer_agent: RevisionAgent,
        *,
        max_model_rounds: int = 3,
        max_optimizer_rounds: int = 3,
    ) -> dict:
        if any(
            type(n) is not int or n < 1
            for n in (max_model_rounds, max_optimizer_rounds)
        ):
            raise ValueError("Cycle round budgets must be positive integers")
        with (self.root / "cycle.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._check()
            context = {
                "format_version": 1,
                "model_context_sha256": self._model.benchmark_sha256,
                "optimizer_problem_sha256": self._problem.problem_sha256,
                "initial_model_sha256": source_digest(self._model_source),
                "initial_optimizer_sha256": source_digest(self._optimizer_source),
                "max_model_rounds": max_model_rounds,
                "max_optimizer_rounds": max_optimizer_rounds,
                "cycle_code_sha256": self._code,
            }
            _fixed_json(self.root / "context.json", context)
            model_stage = _stage(
                RevisionLoop(
                    model_agent,
                    self._model,
                    self.root / "model",
                    initial_source=self._model_source,
                ),
                max_model_rounds,
            )
            self._check()
            model_stage.check()
            model_report = json.loads(
                (self.root / "model" / model_stage.record["report"]).read_text()
            )
            joint = self._model.export_joint(
                model_stage.source.read_text(),
                self._requests,
                self.root / f"export-{uuid.uuid4().hex}",
                expected_prediction_sha=model_report["metrics"]["prediction_sha256"],
                expected_joint_sha=model_report["metrics"]["joint_snapshot_sha256"],
            )
            model_version = "history-model:" + digest(
                {
                    "source_sha256": model_stage.record["source_sha256"],
                    "history_context_sha256": self._model.benchmark_sha256,
                    "prediction_sha256": model_report["metrics"]["prediction_sha256"],
                    "joint_snapshot_sha256": joint.sha256,
                }
            )
            search = self._problem.with_model(joint, model_version=model_version)
            handoff = {
                "model_stage_sha256": digest(model_stage.record),
                "model_version": model_version,
                "joint_snapshot_sha256": joint.sha256,
                "optimizer_context_sha256": search.benchmark_sha256,
            }
            _fixed_json(self.root / "handoff.json", handoff)
            optimizer_stage = _stage(
                RevisionLoop(
                    optimizer_agent,
                    search,
                    self.root / "optimizer",
                    initial_source=self._optimizer_source,
                ),
                max_optimizer_rounds,
            )
            self._check()
            model_stage.check()
            optimizer_stage.check()
            baseline = json.loads((self.root / "optimizer/baseline.json").read_text())
            selected = json.loads(
                (self.root / "optimizer" / optimizer_stage.record["report"]).read_text()
            )
            result = {
                "status": "complete",
                "cycle_context_sha256": digest(context),
                "model": model_stage.record,
                "optimizer": optimizer_stage.record,
                "handoff": handoff,
                "optimizer_scores": {
                    "benchmark_sha256": search.benchmark_sha256,
                    "baseline": baseline["score"],
                    "effective": selected["score"],
                },
            }
            _fixed_json(self.root / "result.json", result)
            return result
