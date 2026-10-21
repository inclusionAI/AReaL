# SPDX-License-Identifier: Apache-2.0

"""Evaluator port implemented by each revision domain."""

from pathlib import Path
from typing import Any, Protocol

from .contracts import EvaluationReport


class RevisionEvaluator(Protocol):
    target: str
    benchmark_sha256: str
    initial_source: str | None
    owns_acceptance: bool
    replay_fingerprints: tuple[str, ...]
    criteria: Any
    context: dict
    settings: dict
    calibration: dict
    proposal_context: dict

    def evaluate(self, source: str, directory: Path) -> EvaluationReport: ...
    def compare(
        self, candidate: EvaluationReport, incumbent: EvaluationReport
    ) -> tuple[bool, str]: ...
    def feedback(self, report: EvaluationReport) -> dict: ...
