"""
Scaffolding Framework Integration for AReaL.

This module provides integration between TensorRT-LLM's Scaffolding framework
and AReaL's RL training pipeline.

Key Components:
- ScaffoldingWorkflow: RolloutWorkflow implementation that wraps ScaffoldingLlm
- RLVRRewardTask: Task for computing verifiable rewards
- RLVRRewardController: Controller for computing verifiable rewards
- PipelineTrajectoryMaker: Controller for composing generation and reward pipelines
- ChatTracer: TaskCollection for tracing multi-turn chat conversations
- TraceTrajectoryMaker: Controller that traces ChatTask objects during rollout
- TraceGenerationTask: Task for tracing multi-turn generation
- ChatRewardTask: Task for computing rewards on traced interactions
- CreateWorkerFromEngine: Creates a scaffolding Worker from AReaL's InferenceEngine
- SGLangWorker: Worker implementation for SGLang engines

Note: Requires tensorrt_llm to be installed for full functionality.
"""

from areal.experimental.scaffolding._compat import HAS_TENSORRT_LLM
from areal.experimental.scaffolding.controllers import (
    ChatTracer,
    PipelineTrajectoryMaker,
    RLVRRewardController,
    TraceTrajectoryMaker,
)
from areal.experimental.scaffolding.task import (
    ChatRewardTask,
    RLVRRewardTask,
    TraceGenerationTask,
)
from areal.experimental.scaffolding.worker import CreateWorkerFromEngine, SGLangWorker
from areal.experimental.scaffolding.workflow import ScaffoldingWorkflow

__all__ = [
    "HAS_TENSORRT_LLM",
    "ScaffoldingWorkflow",
    "RLVRRewardTask",
    "RLVRRewardController",
    "PipelineTrajectoryMaker",
    "ChatTracer",
    "TraceTrajectoryMaker",
    "TraceGenerationTask",
    "ChatRewardTask",
    "CreateWorkerFromEngine",
    "SGLangWorker",
]
