# SPDX-License-Identifier: Apache-2.0

"""Sandbox execution abstraction for agent RL training.

Provides the core protocol and data structures for isolated code/command
execution backends. Any sandbox implementation (E2B-compatible services,
Docker, local process) should conform to the :class:`SandboxExecutor` protocol.

Architecture
------------
- ``areal/api/sandbox_api.py``   — Pure abstractions (this file)
- ``areal/infra/sandbox/``       — Runtime implementations
- ``examples/sandbox/``          — AgentWorkflow with sandbox code execution

See Also
--------
areal.infra.sandbox : Concrete sandbox implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ExecutionResult:
    """Result of a sandbox code or command execution.

    Attributes
    ----------
    stdout : str
        Standard output captured during execution.
    stderr : str
        Standard error captured during execution.
    exit_code : int
        Process exit code. 0 indicates success.
    error : str | None
        Error message if execution failed, None on success.
    output_text : str
        Concatenated text output (convenience field for code execution
        backends that return structured results).
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str | None = None
    output_text: str = ""

    @property
    def success(self) -> bool:
        """Whether the execution completed without errors."""
        return self.error is None and self.exit_code == 0

    @property
    def text(self) -> str:
        """Primary text output: output_text if available, else stdout."""
        return self.output_text or self.stdout


@runtime_checkable
class SandboxExecutor(Protocol):
    """Protocol for sandbox code/command execution backends.

    Any sandbox implementation should conform to this protocol.
    The protocol uses ``runtime_checkable`` to allow ``isinstance()``
    checks, consistent with ``RemoteInfBackendProtocol``.

    Methods are async to support non-blocking I/O with remote sandbox
    services (e.g., E2B Cloud, CubeSandbox, other E2B-compatible APIs).
    """

    async def run_code(
        self,
        code: str,
        language: str = "python",
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute code in the sandbox.

        Parameters
        ----------
        code : str
            Source code to execute.
        language : str
            Programming language. Default is ``"python"``.
        timeout : float
            Maximum execution time in seconds.

        Returns
        -------
        ExecutionResult
            Execution result with stdout/stderr/error.
        """
        ...

    async def run_command(
        self,
        command: str,
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute a shell command in the sandbox.

        Parameters
        ----------
        command : str
            Shell command to execute.
        timeout : float
            Maximum execution time in seconds.

        Returns
        -------
        ExecutionResult
            Execution result with stdout/stderr/exit_code.
        """
        ...

    async def close(self) -> None:
        """Release sandbox resources.

        Implementations should be idempotent — calling close() on an
        already-closed sandbox must not raise.
        """
        ...


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution in RL training workflows.

    This config is used by the AgentWorkflow sandbox example
    (``examples/sandbox/``) and
    :func:`~areal.infra.sandbox.factory.create_sandbox`.

    Attributes
    ----------
    enabled : bool
        Whether sandbox execution is enabled.
    backend : str
        Sandbox backend to use. Currently supported: ``"e2b"``, ``"local"``.
    api_url : str
        E2B-compatible API endpoint URL.
    api_key : str
        API key for authentication with the sandbox service.
        Can also be set via ``SANDBOX_API_KEY`` environment variable.
    template_id : str
        Sandbox template ID for pre-configured environments.
    ssl_cert_file : str
        Path to a CA certificate file for self-hosted E2B-compatible
        deployments (e.g. CubeSandbox) with custom TLS certificates.
        When set, it is applied as ``SSL_CERT_FILE`` only during E2B
        SDK calls so that other HTTPS connections are not affected.
        Can also be set via ``SANDBOX_SSL_CERT_FILE`` environment variable.
    timeout : float
        Default per-execution timeout in seconds.
    """

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable sandbox execution for tool-use workflows."},
    )
    backend: str = field(
        default="e2b",
        metadata={
            "help": (
                "Sandbox backend. 'e2b' covers any E2B-compatible deployment "
                "(E2B Cloud, CubeSandbox, etc.). 'local' is unsafe, debug only."
            ),
            "choices": ["e2b", "local"],
        },
    )
    api_url: str = field(
        default="",
        metadata={
            "help": "E2B-compatible API URL. Also reads from SANDBOX_API_URL env var."
        },
    )
    api_key: str = field(
        default="",
        metadata={
            "help": "API key for sandbox service. Also reads from SANDBOX_API_KEY env var."
        },
    )
    template_id: str = field(
        default="",
        metadata={"help": "Sandbox template ID for pre-configured environments."},
    )
    ssl_cert_file: str = field(
        default="",
        metadata={
            "help": (
                "Path to CA cert for self-hosted E2B-compatible deployments "
                "with custom TLS (e.g. CubeSandbox). "
                "Also reads from SANDBOX_SSL_CERT_FILE env var."
            ),
        },
    )
    timeout: float = field(
        default=30.0,
        metadata={"help": "Default per-execution timeout in seconds."},
    )

    def __post_init__(self):
        import os

        if self.backend == "cube":
            raise ValueError(
                "backend='cube' has been removed; use backend='e2b' instead. "
                "The underlying implementation is unchanged — CubeSandbox is an "
                "E2B-compatible deployment and is fully supported under 'e2b'."
            )
        if not self.api_url:
            self.api_url = os.environ.get("SANDBOX_API_URL", "")
        if not self.api_key:
            self.api_key = os.environ.get("SANDBOX_API_KEY", "")
        if not self.ssl_cert_file:
            self.ssl_cert_file = os.environ.get("SANDBOX_SSL_CERT_FILE", "")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
