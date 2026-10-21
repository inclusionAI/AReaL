# SPDX-License-Identifier: Apache-2.0

"""Run CPU AutoPilot tests without importing AReaL's optional training stack.

Only the top-level package initializer is bypassed. Submodules are real source
files; no Ray, engine, tensor, or distributed behavior is mocked.
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
spec = importlib.machinery.ModuleSpec("areal", loader=None, is_package=True)
module = importlib.util.module_from_spec(spec)
module.__path__ = [str(root / "areal")]
sys.modules["areal"] = module

if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
