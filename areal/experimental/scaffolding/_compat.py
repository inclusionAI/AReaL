"""Compatibility layer for optional tensorrt_llm.scaffolding dependency.

Provides imports from tensorrt_llm.scaffolding when available, or lightweight
standalone implementations when not installed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

try:
    from tensorrt_llm.scaffolding import (
        NativeGenerationController,
        ScaffoldingLlm,
    )
    from tensorrt_llm.scaffolding.controller import Controller
    from tensorrt_llm.scaffolding.result import ScaffoldingOutput
    from tensorrt_llm.scaffolding.task import (
        AssistantMessage,
        ChatTask,
        GenerationTask,
        Task,
        TaskStatus,
    )
    from tensorrt_llm.scaffolding.task_collection import (
        TaskCollection,
        with_task_collection,
    )
    from tensorrt_llm.scaffolding.worker import OpenaiWorker, Worker

    HAS_TENSORRT_LLM = True

except ImportError:
    HAS_TENSORRT_LLM = False

    # ---- Standalone lightweight implementations ----
    # These provide the scaffolding interfaces so the framework works
    # without tensorrt_llm installed.

    class Controller:
        """Lightweight Controller base class."""

        def process(self, tasks: list, **kwargs) -> Any:
            yield tasks

    @dataclass
    class Task:
        """Lightweight Task base class."""

        worker_tag: Any = None

    @dataclass
    class GenerationTask(Task):
        """Lightweight GenerationTask."""

        input_str: str | None = None
        output_str: str | None = None
        input_tokens: list | None = None
        output_tokens: list | None = None
        logprobs: Any = None
        finish_reason: str | None = None
        perf_metrics: Any = None
        customized_result_fields: dict = field(default_factory=dict)

    @dataclass
    class ChatTask(Task):
        """Lightweight ChatTask."""

        messages: list = field(default_factory=list)
        completion: Any = None
        tools: list | None = None
        finish_reason: str | None = None
        input_tokens: list | None = None
        output_tokens: list | None = None
        enable_token_counting: bool = False
        prompt_tokens_num: int = 0
        completion_tokens_num: int = 0
        reasoning_tokens_num: int = 0
        perf_metrics: Any = None

        @staticmethod
        def create_from_prompt(prompt: str) -> ChatTask:
            return ChatTask(messages=[{"role": "user", "content": prompt}])

        def messages_to_dict_content(self) -> list:
            return self.messages

    class TaskStatus(enum.Enum):
        """Lightweight TaskStatus."""

        SUCCESS = "success"
        WORKER_EXECEPTION = "worker_exception"  # noqa: S105 (matches upstream typo)

    class AssistantMessage:
        """Lightweight AssistantMessage."""

        def __init__(
            self,
            content: str | None = None,
            reasoning: str | None = None,
            reasoning_content: str | None = None,
            tool_calls: list | None = None,
        ):
            self.content = content
            self.reasoning = reasoning
            self.reasoning_content = reasoning_content
            self.tool_calls = tool_calls

    class TaskCollection:
        """Lightweight TaskCollection base class."""

        def before_yield(self, tasks: list) -> None:
            pass

        def after_yield(self, tasks: list) -> None:
            pass

    def with_task_collection(name: str, collection_cls: type):
        """Decorator that attaches a TaskCollection to a Controller class."""

        def decorator(cls):
            if not hasattr(cls, "task_collections"):
                cls.task_collections = {}
            cls.task_collections[name] = collection_cls()
            return cls

        return decorator

    class Worker:
        """Lightweight Worker base class."""

    class OpenaiWorker(Worker):
        """Lightweight OpenaiWorker base class."""

        def __init__(self, async_client: Any = None, model: str = "", **kwargs):
            self.async_client = async_client
            self.model = model

        def convert_task_params(self, task: Any) -> dict:
            return {}

    @dataclass
    class ScaffoldingOutput:
        """Lightweight ScaffoldingOutput."""

        text: str = ""
        token_ids: list = field(default_factory=list)

    class NativeGenerationController(Controller):
        """Lightweight NativeGenerationController."""

        class WorkerTag(enum.Enum):
            GENERATION = "generation"

        def process(self, tasks: list, **kwargs) -> Any:
            yield tasks

    class ScaffoldingLlm:
        """Lightweight ScaffoldingLlm."""

        def __init__(self, controller: Controller, workers: dict | None = None):
            self.controller = controller
            self.workers = workers or {}

        def generate(self, prompt: str, **kwargs) -> Any:
            return None

        async def generate_async(self, prompt: str, **kwargs) -> Any:
            return None

        def shutdown(self) -> None:
            pass


__all__ = [
    "HAS_TENSORRT_LLM",
    "AssistantMessage",
    "ChatTask",
    "Controller",
    "GenerationTask",
    "NativeGenerationController",
    "OpenaiWorker",
    "ScaffoldingLlm",
    "ScaffoldingOutput",
    "Task",
    "TaskCollection",
    "TaskStatus",
    "Worker",
    "with_task_collection",
]
