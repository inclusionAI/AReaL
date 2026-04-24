# SPDX-License-Identifier: Apache-2.0

"""Factory function for creating sandbox executors from config."""

from __future__ import annotations

from areal.api.sandbox_api import SandboxConfig, SandboxExecutor
from areal.utils import logging

logger = logging.getLogger("SandboxFactory")


async def create_sandbox(config: SandboxConfig) -> SandboxExecutor:
    """Create a sandbox executor based on the given configuration.

    Parameters
    ----------
    config : SandboxConfig
        Sandbox configuration specifying backend and connection details.

    Returns
    -------
    SandboxExecutor
        A sandbox executor conforming to the protocol.

    Raises
    ------
    ValueError
        If the specified backend is not supported.
    """
    if config.backend == "e2b":
        from areal.infra.sandbox.e2b_sandbox import E2BSandboxExecutor

        return await E2BSandboxExecutor.create(
            api_url=config.api_url,
            api_key=config.api_key,
            template_id=config.template_id or None,
            timeout=max(config.timeout * config.max_tool_turns * 2, 300.0),
            ssl_cert_file=config.ssl_cert_file,
        )
    elif config.backend == "local":
        from areal.infra.sandbox.local_sandbox import LocalSandboxExecutor

        return LocalSandboxExecutor()
    else:
        raise ValueError(
            f"Unsupported sandbox backend: {config.backend!r}. "
            f"Supported backends: 'e2b', 'local'."
        )
