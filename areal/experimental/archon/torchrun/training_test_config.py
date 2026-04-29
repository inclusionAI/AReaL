# SPDX-License-Identifier: Apache-2.0

"""Stable `areal.*` import path for Archon training test config helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType


# region agent log
def _debug_log(hypothesis_id: str, message: str, data: dict[str, object]) -> None:
    payload = {
        "sessionId": "b1b34b",
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "timestamp": int(time.time() * 1000),
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": "areal/experimental/archon/torchrun/training_test_config.py",
        "message": message,
        "data": data,
    }
    with open(
        "/data/jiarui/dta/AReaL/.cursor/debug-b1b34b.log", "a", encoding="utf-8"
    ) as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


# endregion
def _load_impl_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[4]
    impl_path = (
        repo_root
        / "tests"
        / "experimental"
        / "archon"
        / "torchrun"
        / "training_test_config.py"
    )
    if not impl_path.is_file():
        raise FileNotFoundError(
            f"Archon training test config implementation not found: {impl_path}"
        )
    # region agent log
    _debug_log(
        "H2",
        "impl_path_resolved",
        {
            "repo_root": str(repo_root),
            "impl_path": str(impl_path),
            "impl_exists": impl_path.is_file(),
        },
    )
    # endregion

    import areal.api.alloc_mode as alloc_mode

    # region agent log
    _debug_log(
        "H1",
        "alloc_mode_symbol_check",
        {
            "has_AllocationMode": hasattr(alloc_mode, "AllocationMode"),
            "has__AllocationMode": hasattr(alloc_mode, "_AllocationMode"),
            "alloc_mode_file": getattr(alloc_mode, "__file__", None),
        },
    )
    # endregion

    spec = importlib.util.spec_from_file_location(
        "areal.experimental.archon.torchrun._training_test_config_impl",
        str(impl_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from: {impl_path}")

    module = importlib.util.module_from_spec(spec)
    # region agent log
    _debug_log(
        "H4",
        "module_exec_start",
        {
            "spec_name": spec.name,
            "loader_type": type(spec.loader).__name__,
        },
    )
    # endregion
    # region agent log
    _debug_log(
        "H5",
        "pre_exec_sys_modules_state",
        {
            "spec_name": spec.name,
            "module_name": getattr(module, "__name__", None),
            "spec_name_in_sys_modules_before_exec": spec.name in sys.modules,
            "module_name_in_sys_modules_before_exec": getattr(module, "__name__", "")
            in sys.modules,
        },
    )
    # endregion
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        # region agent log
        _debug_log(
            "H6",
            "exec_module_exception",
            {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "spec_name_in_sys_modules_on_error": spec.name in sys.modules,
                "module_name_in_sys_modules_on_error": getattr(module, "__name__", "")
                in sys.modules,
            },
        )
        # endregion
        sys.modules.pop(spec.name, None)
        raise
    # region agent log
    _debug_log(
        "H7",
        "exec_module_success",
        {
            "spec_name": spec.name,
            "module_name": getattr(module, "__name__", None),
            "spec_name_in_sys_modules_after_exec": spec.name in sys.modules,
            "module_name_in_sys_modules_after_exec": getattr(module, "__name__", "")
            in sys.modules,
        },
    )
    # endregion
    return module


_IMPL = _load_impl_module()

ArchonTrainingTestConfig = _IMPL.ArchonTrainingTestConfig
ensure_dump_dir = _IMPL.ensure_dump_dir
load_training_test_config = _IMPL.load_training_test_config

__all__ = [
    "ArchonTrainingTestConfig",
    "ensure_dump_dir",
    "load_training_test_config",
]
