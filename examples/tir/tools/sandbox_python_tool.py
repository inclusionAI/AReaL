# SPDX-License-Identifier: Apache-2.0

"""Sandbox-backed Python code execution tool.

Drop-in alternative to :class:`PythonTool` that delegates execution to
a :class:`~areal.api.sandbox_api.SandboxExecutor` (E2B-compatible or
local) instead of running code in-process.

Usage
-----
Swap the tool in your config to enable sandbox execution::

    # In your TIR yaml config:
    tir:
      enable_tools: "sandbox_python"
      sandbox:
        enabled: true
        backend: e2b
        api_url: "https://..."
        api_key: "..."

The tool shares the same markers and parameter format as ``PythonTool``,
so the model prompt and parsing logic remain identical.
"""

from __future__ import annotations

import asyncio
from typing import Any

from areal.api.sandbox_api import ExecutionResult, SandboxExecutor
from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType
from .python_tool import extract_python_code

logger = logging.getLogger("SandboxPythonTool")

_DEFAULT_MAX_OUTPUT_CHARS = 2000


class SandboxPythonTool(BaseTool):
    """Python code execution tool backed by a SandboxExecutor.

    Implements the same :class:`BaseTool` interface as :class:`PythonTool`,
    but delegates execution to a :class:`SandboxExecutor` for isolation.

    Parameters
    ----------
    timeout : int
        Per-execution timeout in seconds.
    debug_mode : bool
        If True, return dummy output without executing.
    sandbox_executor : SandboxExecutor
        The sandbox backend to delegate execution to.
    max_output_chars : int
        Truncate output beyond this length to avoid exploding RL context.
    """

    def __init__(
        self,
        timeout: int = 30,
        debug_mode: bool = False,
        sandbox_executor: SandboxExecutor | None = None,
        max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    ):
        super().__init__(timeout, debug_mode)
        if sandbox_executor is None:
            raise ValueError(
                "SandboxPythonTool requires a SandboxExecutor instance. "
                "Use PythonTool for local (in-process) execution."
            )
        self._sandbox = sandbox_executor
        self._max_output = max_output_chars

    @property
    def tool_type(self) -> ToolType:
        return ToolType.SANDBOX_PYTHON

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="sandbox_python_executor",
            description=(
                "Execute Python code in an isolated sandbox environment. "
                "Supports variable calculation, data processing, "
                "algorithm implementation, etc."
            ),
            parameters={"code": "The Python code string to execute"},
            parameter_prompt=(
                "Please provide the Python code to execute. "
                "Supports variable calculation, data processing, "
                "algorithm implementation, etc."
            ),
            example=(
                "```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```\n"
                " or \n<python>\na=1\nb=1\n"
                "print(f'The a+b result is {a+b}')\n</python>"
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
        """Execute Python code in the sandbox."""
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug(f"[FAKE] Sandbox executing Python code: {code[:100]}...")
            return "dummy sandbox python output", ToolCallStatus.SUCCESS

        return self._execute_in_sandbox(code)

    def _execute_in_sandbox(self, code: str) -> tuple[str, ToolCallStatus]:
        """Sync wrapper over async SandboxExecutor.run_code()."""

        async def _run() -> ExecutionResult:
            return await self._sandbox.run_code(
                code, language="python", timeout=self.timeout
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an event loop (e.g., called from TIRWorkflow).
            # Schedule in a new thread to avoid blocking.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(asyncio.run, _run()).result()
        else:
            result = asyncio.run(_run())

        return self._format_result(result)

    def _format_result(
        self, result: ExecutionResult
    ) -> tuple[str, ToolCallStatus]:
        """Format ExecutionResult into (text, status) tuple."""
        if result.success:
            output = result.text or "(no output)"
            # Truncate to avoid exploding RL context
            if len(output) > self._max_output:
                half = self._max_output // 2
                output = (
                    output[:half]
                    + "\n...(truncated)...\n"
                    + output[-half // 2 :]
                )
            return str((output, "Done")), ToolCallStatus.SUCCESS

        error_msg = result.error or result.stderr or "Unknown error"
        return str(("", error_msg)), ToolCallStatus.ERROR
