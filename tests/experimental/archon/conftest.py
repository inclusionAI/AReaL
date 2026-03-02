"""Pytest configuration for archon tests.

NOTE: Archon engine requires PyTorch >= 2.9.1.
"""

import pytest
import torch

# Require PyTorch >= 2.9.1 for archon tests
_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
_MIN_TORCH_VERSION = (2, 9, 1)

if _TORCH_VERSION < _MIN_TORCH_VERSION:
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(config, items):
    """Skip all archon tests if PyTorch version is too old."""
    if _TORCH_VERSION >= _MIN_TORCH_VERSION:
        return

    skip_marker = pytest.mark.skip(
        reason=f"Archon tests require PyTorch >= 2.9.1, but found {torch.__version__}"
    )
    for item in items:
        if "experimental/archon" in str(item.fspath):
            item.add_marker(skip_marker)
