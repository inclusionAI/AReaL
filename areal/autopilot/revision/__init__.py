# SPDX-License-Identifier: Apache-2.0

"""Independent, bounded revisions of simulator or joint optimizer code."""

from areal.autopilot.revision.agent import CommandRevisionAgent, RevisionAgent
from areal.autopilot.revision.contracts import Proposal, ProposalRequest
from areal.autopilot.revision.cycle import PolicyRevisionCycle
from areal.autopilot.revision.evaluator import AcceptanceCriteria, RevisionEvaluator
from areal.autopilot.revision.joint_evaluator import (
    JointOptimizerCriteria,
    JointOptimizerEvaluator,
)
from areal.autopilot.revision.llm import LLMRevisionAgent
from areal.autopilot.revision.loop import (
    RevisionLoop,
    RevisionStore,
    apply_accepted_revision,
    load_accepted_model,
)
from areal.autopilot.revision.model_adapters import BasicHistoryPredictor
from areal.autopilot.revision.model_evaluator import (
    HistoricalCase,
    HistoricalInput,
    HistoricalModelEvaluator,
    HistoricalPrediction,
    HistoryReport,
    HistorySpan,
    ModelAcceptanceCriteria,
    ModelComparison,
)
from areal.autopilot.revision.model_execution import HistoricalSourceEvaluator

__all__ = [
    "AcceptanceCriteria",
    "BasicHistoryPredictor",
    "CommandRevisionAgent",
    "HistoricalCase",
    "HistoricalInput",
    "HistoricalModelEvaluator",
    "HistoricalPrediction",
    "HistoricalSourceEvaluator",
    "HistoryReport",
    "HistorySpan",
    "LLMRevisionAgent",
    "JointOptimizerCriteria",
    "JointOptimizerEvaluator",
    "ModelAcceptanceCriteria",
    "ModelComparison",
    "Proposal",
    "ProposalRequest",
    "PolicyRevisionCycle",
    "RevisionAgent",
    "RevisionEvaluator",
    "RevisionLoop",
    "RevisionStore",
    "apply_accepted_revision",
    "load_accepted_model",
]
