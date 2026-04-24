# SPDX-License-Identifier: Apache-2.0

"""Local (unsafe) sandbox executor for development and debugging.

.. warning::
    This executor runs code in the current process with NO isolation.
    Do NOT use in production. Use :class:`E2BSandboxExecutor` instead.
"""

from __future__ import annotations

import asyncio
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout

from areal.api.sandbox_api import ExecutionResult
from areal.utils import logging

logger = logging.getLogger("LocalSandbox")


class LocalSandboxExecutor:
    """Unsafe local code execution for development/testing only.

    Implements the :class:`~areal.api.sandbox_api.SandboxExecutor`
    protocol by running code in a subprocess via ``asyncio``.

    .. warning::
        No isolation whatsoever. Only use for local debugging.
    """

    def __init__(self) -> None:
        self._closed = False
        logger.warning(
            "LocalSandboxExecutor is NOT sandboxed. "
            "Do not use for production workloads."
        )

    async def run_code(
        self,
        code: str,
        language: str = "python",
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute Python code locally (unsafe).

        Parameters
        ----------
        code : str
            Python source code.
        language : str
            Only ``"python"`` is supported.
        timeout : float
            Execution timeout in seconds.

        Returns
        -------
        ExecutionResult
            Execution result.
        """
        if self._closed:
            raise RuntimeError("Cannot run code on a closed sandbox.")
        if language != "python":
            return ExecutionResult(
                error=f"LocalSandboxExecutor only supports Python, got {language}",
                exit_code=1,
            )

        loop = asyncio.get_running_loop()

        def _execute():
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, {"__builtins__": __builtins__})  # noqa: S102
                return ExecutionResult(
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue(),
                    exit_code=0,
                    output_text=stdout_capture.getvalue(),
                )
            except Exception:
                return ExecutionResult(
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue() + traceback.format_exc(),
                    exit_code=1,
                    error=traceback.format_exc().strip().split("\n")[-1],
                )

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _execute),
                timeout=timeout,
            )
        except TimeoutError:
            result = ExecutionResult(
                error=f"Execution timed out after {timeout}s",
                exit_code=1,
            )
        return result

    async def run_command(
        self,
        command: str,
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute a shell command locally (unsafe).

        Parameters
        ----------
        command : str
            Shell command to execute.
        timeout : float
            Execution timeout in seconds.

        Returns
        -------
        ExecutionResult
            Execution result.
        """
        if self._closed:
            raise RuntimeError("Cannot run command on a closed sandbox.")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            return ExecutionResult(
                error=f"Command timed out after {timeout}s",
                exit_code=1,
            )
        except Exception as exc:
            return ExecutionResult(
                error=f"Command execution error: {exc}",
                exit_code=1,
            )

        exit_code = proc.returncode or 0
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            error=stderr if exit_code != 0 else None,
        )

    async def close(self) -> None:
        """No-op for local executor."""
        self._closed = True

    @property
    def is_closed(self) -> bool:
        """Whether this sandbox has been closed."""
        return self._closed
