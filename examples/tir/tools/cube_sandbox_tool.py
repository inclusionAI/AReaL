"""CubeSandbox-backed Python execution tool for TIR workflows.

Drop-in replacement for :class:`PythonTool` that executes code in a
KVM-isolated CubeSandbox instance instead of in-process.  Provides
production-grade security while maintaining the same ``BaseTool``
interface.

Usage
-----
Pass a ``SandboxConfig`` to ``ToolRegistry`` and it will automatically
use this tool for ``ToolType.PYTHON``::

    from areal.api.sandbox_api import SandboxConfig

    registry = ToolRegistry(
        sandbox_config=SandboxConfig(
            enabled=True,
            api_url="http://your-cubesandbox:3000",
            template_id="your-template",
        ),
    )

Or construct directly::

    tool = CubeSandboxPythonTool(
        sandbox_config=SandboxConfig(
            enabled=True,
            api_url="http://your-cubesandbox:3000",
            template_id="your-template",
        ),
    )

Prerequisites
-------------
- ``pip install e2b-code-interpreter``
- Running CubeSandbox service or E2B-compatible API.
"""

from __future__ import annotations

import asyncio
from typing import Any

from areal.api.sandbox_api import SandboxConfig
from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType
from .python_tool import extract_python_code

logger = logging.getLogger("CubeSandboxPythonTool")


class CubeSandboxPythonTool(BaseTool):
    """Python code execution via CubeSandbox (KVM isolated).

    Conforms to the TIR ``BaseTool`` interface so it can be used as a
    drop-in replacement for ``PythonTool``.

    Parameters
    ----------
    timeout : int
        Per-execution timeout in seconds.
    debug_mode : bool
        If True, returns dummy output without executing.
    sandbox_config : SandboxConfig | None
        Sandbox configuration.  If ``None``, reads from env vars
        (``SANDBOX_API_URL``, ``SANDBOX_API_KEY``, ``CUBE_TEMPLATE_ID``,
        ``CUBE_SSL_CERT_FILE``).
    """

    def __init__(
        self,
        timeout: int = 30,
        debug_mode: bool = False,
        sandbox_config: SandboxConfig | None = None,
    ):
        super().__init__(timeout, debug_mode)
        self._config = sandbox_config or SandboxConfig(enabled=True)
        self._sandbox = None
        self._sandbox_lock = asyncio.Lock()

    async def _get_sandbox(self):
        """Lazily create a persistent sandbox for this tool instance."""
        if self._sandbox is not None:
            return self._sandbox

        async with self._sandbox_lock:
            if self._sandbox is not None:
                return self._sandbox

            from areal.infra.sandbox.cube_sandbox import CubeSandboxExecutor

            self._sandbox = await CubeSandboxExecutor.create(
                api_url=self._config.api_url,
                api_key=self._config.api_key,
                template_id=self._config.template_id or None,
                ssl_cert_file=self._config.ssl_cert_file,
            )
            logger.info("Created CubeSandbox instance for TIR tool.")
            return self._sandbox

    @property
    def tool_type(self) -> ToolType:
        return ToolType.PYTHON

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="sandbox_python_executor",
            description=(
                "Execute Python code in an isolated sandbox environment. "
                "Supports variable calculation, data processing, algorithm "
                "implementation, package imports, and file I/O."
            ),
            parameters={"code": "The Python code string to execute"},
            parameter_prompt=(
                "Please provide the Python code to execute. The code runs "
                "in an isolated sandbox with full Python standard library."
            ),
            example=(
                "```python\n"
                "import math\n"
                "result = math.factorial(10)\n"
                "print(f'10! = {result}')\n"
                "```"
            ),
        )

    @property
    def markers(self) -> ToolMarkers:
        return ToolMarkers(
            start_markers=["```python", "<python>"],
            end_markers=["```", "</python>"],
        )

    def parse_parameters(self, text: str) -> dict[str, Any]:
        """Extract Python code from text."""
        code = extract_python_code(text)
        return {"code": code}

    def execute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        """Execute Python code in CubeSandbox (sync wrapper).

        Bridges to the async sandbox API.  In async contexts, prefer
        :meth:`async_execute` to avoid unnecessary thread-pool overhead.
        """
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug(f"[FAKE] Sandbox executing: {code[:100]}...")
            return "dummy sandbox output", ToolCallStatus.SUCCESS

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                from areal.infra.utils.concurrent import run_async_task

                return run_async_task(self._async_execute, code)
            else:
                return loop.run_until_complete(self._async_execute(code))
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            return f"Error: {str(e)}", ToolCallStatus.ERROR

    async def _async_execute(self, code: str) -> tuple[str, ToolCallStatus]:
        """Async execution in sandbox."""
        sandbox = await self._get_sandbox()
        result = await sandbox.run_code(code, timeout=self.timeout)

        if result.success:
            output = result.text or result.stdout or "(no output)"
            # Truncate long output
            if len(output) > 2000:
                output = output[:1000] + "\n...(truncated)...\n" + output[-500:]
            return str(output), ToolCallStatus.SUCCESS
        else:
            error_msg = result.error or result.stderr or "Unknown error"
            return f"Error: {error_msg}", ToolCallStatus.ERROR

    async def async_execute(
        self, parameters: dict[str, Any]
    ) -> tuple[str, ToolCallStatus]:
        """Native async execution — preferred in async workflows."""
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR

        if self.debug_mode:
            return "dummy sandbox output", ToolCallStatus.SUCCESS

        try:
            return await self._async_execute(code)
        except Exception as e:
            logger.error(f"Sandbox async execution error: {e}")
            return f"Error: {str(e)}", ToolCallStatus.ERROR

    async def close(self) -> None:
        """Close the sandbox."""
        if self._sandbox is not None:
            await self._sandbox.close()
            self._sandbox = None

    def __del__(self):
        if self._sandbox is not None:
            logger.warning(
                "CubeSandboxPythonTool was not explicitly closed. "
                "Call close() to release sandbox resources."
            )
