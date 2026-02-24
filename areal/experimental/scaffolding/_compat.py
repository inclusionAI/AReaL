"""Scaffolding framework primitives vendored from TensorRT-LLM.

This module re-exports the core scaffolding classes from the vendored copy
at ``areal.experimental.scaffolding.core``, so the rest of AReaL can import
them from a single location.
"""

from areal.experimental.scaffolding.core.controller import (
    BestOfNController,
    Controller,
    MajorityVoteController,
    NativeChatController,
    NativeGenerationController,
    NativeRewardController,
    ParallelProcess,
)
from areal.experimental.scaffolding.core.result import ScaffoldingOutput
from areal.experimental.scaffolding.core.scaffolding_llm import ScaffoldingLlm
from areal.experimental.scaffolding.core.task import (
    AssistantMessage,
    ChatTask,
    GenerationTask,
    OpenAIToolDescription,
    RoleMessage,
    StreamGenerationTask,
    SystemMessage,
    Task,
    TaskStatus,
    UserMessage,
)
from areal.experimental.scaffolding.core.task_collection import (
    TaskCollection,
    with_task_collection,
)
from areal.experimental.scaffolding.core.worker import OpenaiWorker, Worker

__all__ = [
    "AssistantMessage",
    "BestOfNController",
    "ChatTask",
    "Controller",
    "GenerationTask",
    "MajorityVoteController",
    "NativeChatController",
    "NativeGenerationController",
    "NativeRewardController",
    "OpenAIToolDescription",
    "OpenaiWorker",
    "ParallelProcess",
    "RoleMessage",
    "ScaffoldingLlm",
    "ScaffoldingOutput",
    "StreamGenerationTask",
    "SystemMessage",
    "Task",
    "TaskCollection",
    "TaskStatus",
    "UserMessage",
    "Worker",
    "with_task_collection",
]
