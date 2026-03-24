"""Pytest configuration for archon tests.

NOTE: Archon engine requires PyTorch >= 2.9.1.
Qwen3.5-specific tests additionally require transformers >= 5.2.
"""

import pytest
import torch

# --- PyTorch version gate (all archon tests) ---
_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
_MIN_TORCH_VERSION = (2, 9, 1)

# --- Transformers version gate (Qwen3.5 tests) ---
try:
    from importlib.metadata import version as _pkg_version

    _TF_VERSION = tuple(int(x) for x in _pkg_version("transformers").split(".")[:2])
except Exception:
    _TF_VERSION = (0, 0)

_MIN_TF_QWEN3_5 = (5, 2)

# Prevent pytest from importing/collecting test files that would fail
collect_ignore_glob: list[str] = []
if _TORCH_VERSION < _MIN_TORCH_VERSION:
    collect_ignore_glob.append("test_*.py")
if _TF_VERSION < _MIN_TF_QWEN3_5:
    collect_ignore_glob.extend(["test_qwen3_5*.py", "test_hf_parity_qwen3_5*.py"])


def pytest_collection_modifyitems(config, items):
    """Skip archon tests based on version requirements."""
    if _TORCH_VERSION >= _MIN_TORCH_VERSION:
        return

    skip_marker = pytest.mark.skip(
        reason=f"Archon tests require PyTorch >= 2.9.1, but found {torch.__version__}"
    )
    for item in items:
        if "experimental/archon" in str(item.fspath):
            item.add_marker(skip_marker)
