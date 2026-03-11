"""AEnvironment adapter for AReaL workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from areal.infra.aenv.config import AenvConfig
from areal.utils import logging

logger = logging.getLogger("AenvAdapter")


def _import_aenv_environment():
    """Import aenvironment lazily to keep it optional."""
    try:
        from aenv import Environment

        return Environment
    except ImportError as exc:
        raise ImportError(
            "aenvironment is required for AEnvironment integration. "
            "Install it with: pip install aenvironment"
        ) from exc


@dataclass
class AenvToolCallResult:
    """Normalized tool call output from AEnvironment SDK."""

    content: Any
    is_error: bool


class AenvEnvironmentAdapter:
    """Adapter bridging AReaL workflows to the aenvironment SDK."""

    def __init__(self, config: AenvConfig):
        self.config = config
        self._env: Any | None = None
        self._tools_cache: list[dict[str, Any]] | None = None

    async def initialize(self) -> None:
        """Initialize the underlying AEnvironment instance."""
        if self._env is not None:
            return

        environment_cls = _import_aenv_environment()
        logger.info(
            "Initializing AEnvironment instance",
            extra={"env_name": self.config.env_name, "aenv_url": self.config.aenv_url},
        )

        self._env = environment_cls(
            env_name=self.config.env_name,
            datasource=self.config.datasource,
            ttl=self.config.ttl,
            environment_variables=self.config.environment_variables,
            arguments=self.config.arguments,
            aenv_url=self.config.aenv_url,
            timeout=self.config.timeout,
            startup_timeout=self.config.startup_timeout,
            max_retries=self.config.max_retries,
        )
        await self._env.initialize()

    async def release(self, force: bool = False) -> None:
        """Release environment resources.

        Args:
            force: Force releasing resources regardless of ``auto_release`` config.
        """
        if self._env is None:
            return

        if not self.config.auto_release and not force:
            logger.info("Skipping release because auto_release is disabled")
            return

        try:
            await self._env.release()
        except Exception as exc:  # pragma: no cover - defensive cleanup path
            logger.warning(f"Failed to release AEnvironment instance: {exc}")
        finally:
            self._env = None
            self._tools_cache = None

    async def cleanup(self) -> None:
        """Alias for release to align with common adapter naming."""
        await self.release(force=False)

    async def list_tools(self, use_cache: bool = True) -> list[dict[str, Any]]:
        """List tools exposed by the current environment."""
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        tools = await self._env.list_tools()
        if not isinstance(tools, list):
            raise TypeError(
                f"Expected list from list_tools(), got {type(tools).__name__}"
            )

        self._tools_cache = tools
        return tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> AenvToolCallResult:
        """Execute a tool with classified retries for transient failures."""
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        if not isinstance(arguments, dict):
            raise TypeError(
                f"Tool arguments must be a dict, got {type(arguments).__name__}"
            )

        max_attempts = self.config.max_retries + 1
        call_timeout = self.config.tool_call_timeout if timeout is None else timeout
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                result = await self._env.call_tool(
                    tool_name,
                    arguments,
                    timeout=call_timeout,
                )
                content = result.content if hasattr(result, "content") else result
                is_error = bool(getattr(result, "is_error", False))
                return AenvToolCallResult(content=content, is_error=is_error)
            except Exception as exc:  # noqa: PERF203 - explicit retry loop
                last_error = exc
                if attempt >= self.config.max_retries or not self._is_retriable_error(
                    exc
                ):
                    break

                delay = self.config.retry_delay * (2**attempt)
                logger.warning(
                    "Retriable tool call failure",
                    extra={
                        "tool_name": tool_name,
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                        "delay": delay,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Tool call failed after {max_attempts} attempts: {tool_name}"
        ) from last_error

    async def __aenter__(self) -> AenvEnvironmentAdapter:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.release(force=True)

    @staticmethod
    def _is_retriable_error(exc: Exception) -> bool:
        """Best-effort classification for transient transport failures."""
        if isinstance(
            exc, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)
        ):
            return True

        exc_module = exc.__class__.__module__
        exc_name = exc.__class__.__name__
        if exc_module.startswith("httpx") and exc_name in {
            "ConnectError",
            "ConnectTimeout",
            "PoolTimeout",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "WriteError",
            "WriteTimeout",
        }:
            return True

        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "connection reset",
                "connection refused",
                "temporarily unavailable",
                "timed out",
                "timeout",
                "502",
                "503",
                "504",
            )
        )
