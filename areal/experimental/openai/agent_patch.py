import asyncio
import inspect
from contextvars import ContextVar
from dataclasses import fields, is_dataclass, replace

from agents import RunConfig
from agents import Runner as OpenAIRunner

_CONTEXT_RUN_CONFIG = ContextVar("context_run_config", default=None)


class AReaLOpenAIClientContext:
    _lock = asyncio.Lock()
    _original_run = None
    _active_contexts = 0

    def __init__(self, run_config: RunConfig | None = None):
        self.run_config_to_set = run_config
        self._context_token = None

    async def __aenter__(self):
        cls = self.__class__
        async with cls._lock:
            if cls._active_contexts == 0:
                # print("🚀 First context entered. Applying patch to OpenAIRunner.run.")
                cls._original_run = OpenAIRunner.run
                OpenAIRunner.run = cls._patched_run

            cls._active_contexts += 1

        self._context_token = _CONTEXT_RUN_CONFIG.set(self.run_config_to_set)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        cls = self.__class__
        if self._context_token:
            _CONTEXT_RUN_CONFIG.reset(self._context_token)
            self._context_token = None
        async with cls._lock:
            cls._active_contexts -= 1

            if cls._active_contexts == 0:
                # print("🔴 Last context exited. Restoring original OpenAIRunner.run.")
                if cls._original_run:
                    OpenAIRunner.run = cls._original_run
                    cls._original_run = None

    @staticmethod
    async def _patched_run(*args, **kwargs):
        original_run = AReaLOpenAIClientContext._original_run

        context_run_config = _CONTEXT_RUN_CONFIG.get()
        if context_run_config is None or original_run is None:
            if original_run:
                return await original_run(*args, **kwargs)
            else:
                raise RuntimeError(
                    "Patch is active, but original OpenAIRunner.run was not saved."
                )
        original_sig = inspect.signature(original_run)
        bound_args = original_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        original_run_config = bound_args.arguments.get("run_config", None)

        merged_run_config = AReaLOpenAIClientContext._merge_run_config(
            original_run_config, context_run_config
        )

        bound_args.arguments["run_config"] = merged_run_config
        return await original_run(*bound_args.args, **bound_args.kwargs)

    @staticmethod
    def _merge_run_config(
        base_run_config: RunConfig | None, override_run_config: RunConfig | None
    ):
        if not override_run_config:
            return base_run_config
        if not base_run_config:
            return override_run_config

        merged_run_config = replace(base_run_config)
        for field in fields(override_run_config):
            update_value = getattr(override_run_config, field.name)
            if update_value is None:
                continue
            base_value = getattr(merged_run_config, field.name)
            if is_dataclass(base_value) and is_dataclass(update_value):
                setattr(
                    merged_run_config,
                    field.name,
                    AReaLOpenAIClientContext._merge_run_config(
                        base_value, update_value
                    ),
                )
            else:
                setattr(merged_run_config, field.name, update_value)
        return merged_run_config
