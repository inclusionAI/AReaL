# SPDX-License-Identifier: Apache-2.0

"""Sandbox runtime infrastructure for agent RL training.

Provides concrete sandbox backend implementations, connection pooling,
and lifecycle management for use by workflow layers.

Components
----------
- :class:`CubeSandboxExecutor` — CubeSandbox / E2B SDK backend.
- :class:`LocalSandboxExecutor` — Unsafe local execution (debug only).
- :class:`SandboxManager` — Per-thread sandbox pool and lifecycle manager.

See Also
--------
areal.api.sandbox_api : Abstract protocol and config definitions.
areal.workflow.sandbox_tool : Workflow that consumes sandbox executors.
"""

__all__ = [
    "CubeSandboxExecutor",
    "LocalSandboxExecutor",
    "SandboxManager",
    "batch_create_sandboxes",
    "create_sandbox",
    "create_sandbox_info_sync",
]

_LAZY_IMPORTS = {
    "CubeSandboxExecutor": "areal.infra.sandbox.cube_sandbox",
    "LocalSandboxExecutor": "areal.infra.sandbox.local_sandbox",
    "SandboxManager": "areal.infra.sandbox.manager",
    "batch_create_sandboxes": "areal.infra.sandbox.cube_sandbox",
    "create_sandbox": "areal.infra.sandbox.factory",
    "create_sandbox_info_sync": "areal.infra.sandbox.cube_sandbox",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
