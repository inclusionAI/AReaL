# SPDX-License-Identifier: Apache-2.0

"""E2B-compatible sandbox backend implementation.

Works with any deployment that implements the E2B Code Interpreter
protocol, including:

- `E2B Cloud <https://e2b.dev>`_ (managed SaaS)
- `CubeSandbox <https://github.com/TencentCloud/CubeSandbox>`_
  (TencentCloud self-hosted, KVM-isolated, <60ms cold start, <5MB
  memory per instance — current recommended backend for RL training)
- Other self-hosted E2B-compatible services

Performance optimizations:

- **SSL context manager**: temporarily sets ``SSL_CERT_FILE`` only
  during E2B SDK calls for self-hosted deployments with custom TLS
  certificates, avoiding interference with other HTTPS connections.
- **Shared API client**: ``AsyncSandbox`` is used directly for now;
  a shared async client with connection pooling may be added later.

Prerequisites
-------------
- Install the SDK: ``pip install e2b-code-interpreter``
- Set ``SANDBOX_API_URL`` and ``SANDBOX_API_KEY`` environment variables,
  or pass them via :class:`~areal.api.sandbox_api.SandboxConfig`.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any

from areal.api.sandbox_api import ExecutionResult
from areal.utils import logging

logger = logging.getLogger("E2BSandbox")

# ---------------------------------------------------------------------------
# SSL context manager
# ---------------------------------------------------------------------------


@contextmanager
def _ssl_context(ssl_cert_file: str = ""):
    """Temporarily set ``SSL_CERT_FILE`` for E2B SDK calls, then restore.

    Self-hosted E2B-compatible deployments (e.g. CubeSandbox) typically
    use custom TLS certificates.  The E2B SDK reads ``SSL_CERT_FILE``
    internally, so we patch it right before the call and restore
    immediately after.

    Parameters
    ----------
    ssl_cert_file : str
        Path to the CA cert file.  If empty, the context manager is a
        no-op.
    """
    if not ssl_cert_file:
        yield
        return

    old = os.environ.get("SSL_CERT_FILE")
    os.environ["SSL_CERT_FILE"] = ssl_cert_file
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SSL_CERT_FILE", None)
        else:
            os.environ["SSL_CERT_FILE"] = old


# ---------------------------------------------------------------------------
# E2BSandboxExecutor — async executor conforming to SandboxExecutor protocol
# ---------------------------------------------------------------------------


class E2BSandboxExecutor:
    """E2B-protocol implementation of the SandboxExecutor protocol.

    Wraps an E2B-compatible ``AsyncSandbox`` instance.  Created via
    :func:`~areal.infra.sandbox.factory.create_sandbox` or
    :meth:`E2BSandboxExecutor.create`.

    Works with any E2B-compatible deployment including E2B Cloud,
    CubeSandbox (self-hosted), and other compatible services.

    Parameters
    ----------
    sandbox : AsyncSandbox
        An already-created E2B AsyncSandbox instance.
    ssl_cert_file : str
        CA cert path for SDK calls that need custom TLS verification.
    """

    def __init__(self, sandbox: Any, ssl_cert_file: str = "") -> None:
        self._sandbox = sandbox
        self._closed = False
        self._ssl_cert_file = ssl_cert_file

    @classmethod
    async def create(
        cls,
        api_url: str = "",
        api_key: str = "",
        template_id: str | None = None,
        timeout: float = 300.0,
        ssl_cert_file: str = "",
    ) -> E2BSandboxExecutor:
        """Create a new E2B sandbox instance.

        Parameters
        ----------
        api_url : str
            E2B-compatible API endpoint.
        api_key : str
            API key for authentication.
        template_id : str | None
            Optional template ID for pre-configured environments.
        timeout : float
            Sandbox keep-alive timeout in seconds.
        ssl_cert_file : str
            CA cert path for self-hosted deployments.

        Returns
        -------
        E2BSandboxExecutor
            A new executor wrapping the created sandbox.

        Raises
        ------
        ImportError
            If ``e2b_code_interpreter`` is not installed.
        """
        try:
            from e2b_code_interpreter import AsyncSandbox
        except ImportError as exc:
            raise ImportError(
                "E2B sandbox backend requires `e2b-code-interpreter` package. "
                "Install with: pip install e2b-code-interpreter"
            ) from exc

        kwargs: dict[str, Any] = {}
        if api_url:
            kwargs["api_url"] = api_url
        if api_key:
            kwargs["api_key"] = api_key
        if template_id:
            kwargs["template"] = template_id
        kwargs["timeout"] = int(timeout)

        logger.info(
            "Creating E2B sandbox instance (api_url=%s, template=%s)",
            api_url or "<default>",
            template_id or "<default>",
        )

        t1 = time.monotonic()
        with _ssl_context(ssl_cert_file):
            sandbox = await AsyncSandbox.create(**kwargs)
        api_ms = round((time.monotonic() - t1) * 1000)
        logger.info(
            "E2B sandbox created: %s (api_call=%dms)",
            getattr(sandbox, "sandbox_id", "?"),
            api_ms,
        )
        return cls(sandbox, ssl_cert_file=ssl_cert_file)

    async def run_code(
        self,
        code: str,
        language: str = "python",
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute code in the E2B sandbox.

        Parameters
        ----------
        code : str
            Source code to execute.
        language : str
            Programming language (currently ``"python"`` is supported).
        timeout : float
            Execution timeout in seconds.

        Returns
        -------
        ExecutionResult
            Structured execution result.
        """
        if self._closed:
            raise RuntimeError("Cannot run code on a closed sandbox.")

        try:
            with _ssl_context(self._ssl_cert_file):
                execution = await self._sandbox.run_code(
                    code, timeout=int(timeout)
                )
        except Exception as exc:
            return ExecutionResult(
                error=f"Sandbox execution error: {exc}",
                exit_code=1,
            )

        # E2B execution result structure:
        # execution.logs.stdout: list[str], execution.logs.stderr: list[str]
        # execution.error: SandboxError | None (.name, .value, .traceback)
        # execution.text: str | None (convenience text output)
        stdout = ""
        stderr = ""
        if hasattr(execution, "logs"):
            if hasattr(execution.logs, "stdout"):
                stdout = (
                    "\n".join(execution.logs.stdout)
                    if isinstance(execution.logs.stdout, list)
                    else str(execution.logs.stdout)
                )
            if hasattr(execution.logs, "stderr"):
                stderr = (
                    "\n".join(execution.logs.stderr)
                    if isinstance(execution.logs.stderr, list)
                    else str(execution.logs.stderr)
                )

        error = None
        if execution.error:
            error_val = getattr(execution.error, "value", str(execution.error))
            error = str(error_val)

        output_text = ""
        if hasattr(execution, "text") and execution.text:
            output_text = execution.text

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=1 if error else 0,
            error=error,
            output_text=output_text,
        )

    async def run_command(
        self,
        command: str,
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute a shell command in the E2B sandbox.

        Parameters
        ----------
        command : str
            Shell command to execute.
        timeout : float
            Execution timeout in seconds.

        Returns
        -------
        ExecutionResult
            Structured execution result.
        """
        if self._closed:
            raise RuntimeError("Cannot run command on a closed sandbox.")

        try:
            with _ssl_context(self._ssl_cert_file):
                result = await self._sandbox.commands.run(
                    command, timeout=int(timeout)
                )
        except Exception as exc:
            return ExecutionResult(
                error=f"Sandbox command error: {exc}",
                exit_code=1,
            )

        exit_code = getattr(result, "exit_code", 0) or 0
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            error=stderr if exit_code != 0 else None,
        )

    async def close(self) -> None:
        """Destroy the sandbox and release resources."""
        if self._closed:
            return
        self._closed = True
        try:
            with _ssl_context(self._ssl_cert_file):
                await self._sandbox.close()
        except Exception as exc:
            logger.warning("Error closing E2B sandbox: %s", exc)

    @property
    def is_closed(self) -> bool:
        """Whether this sandbox has been closed."""
        return self._closed
