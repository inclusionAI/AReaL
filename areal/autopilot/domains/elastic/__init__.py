# SPDX-License-Identifier: Apache-2.0

"""Elastic evaluators and revision adapters for AutoPilot."""

from .cycle import PolicyRevisionCycle
from .evaluator import AcceptanceCriteria, RevisionEvaluator
from .joint_evaluator import JointOptimizerCriteria, JointOptimizerEvaluator
from .loaders import apply_accepted_revision, load_accepted_model
from .model_adapters import BasicHistoryPredictor
from .model_evaluator import (
    HistoricalCase,
    HistoricalInput,
    HistoricalModelEvaluator,
    HistoricalPrediction,
    HistoryReport,
    HistorySpan,
    ModelAcceptanceCriteria,
    ModelComparison,
)
from .model_execution import HistoricalSourceEvaluator

__all__ = [
    "PolicyRevisionCycle",
    "AcceptanceCriteria",
    "RevisionEvaluator",
    "JointOptimizerCriteria",
    "JointOptimizerEvaluator",
    "apply_accepted_revision",
    "load_accepted_model",
    "BasicHistoryPredictor",
    "HistoricalCase",
    "HistoricalInput",
    "HistoricalModelEvaluator",
    "HistoricalPrediction",
    "HistoryReport",
    "HistorySpan",
    "ModelAcceptanceCriteria",
    "ModelComparison",
    "HistoricalSourceEvaluator",
]
