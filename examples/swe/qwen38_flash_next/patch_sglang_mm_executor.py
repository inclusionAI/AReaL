# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the common SGLang multimodal executor patch."""

from areal.engine.sglang_mm_executor import main, patched_source

__all__ = ["main", "patched_source"]


if __name__ == "__main__":
    main()
