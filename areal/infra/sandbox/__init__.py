# SPDX-License-Identifier: Apache-2.0

"""Sandbox runtime infrastructure for agent RL training.

Provides concrete sandbox backend implementations for use by workflow layers.

Components
----------
- :class:`E2BSandboxExecutor` — E2B-compatible SDK backend.
- :class:`LocalSandboxExecutor` — Unsafe local execution (debug only).

See Also
--------
areal.api.sandbox_api : Abstract protocol and config definitions.
"""

__all__ = [
    "E2BSandboxExecutor",
    "LocalSandboxExecutor",
    "create_sandbox",
]

_LAZY_IMPORTS = {
    "E2BSandboxExecutor": "areal.infra.sandbox.e2b_sandbox",
    "LocalSandboxExecutor": "areal.infra.sandbox.local_sandbox",
    "create_sandbox": "areal.infra.sandbox.factory",
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
