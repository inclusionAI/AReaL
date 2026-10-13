# SPDX-License-Identifier: Apache-2.0
"""Prevent exhausted SGLang multimodal iterators from stranding asyncio requests.

Apply explicitly before starting any supported SGLang multimodal recipe with
``python -m areal.engine.sglang_mm_executor``. The patch is independent of the
model and workflow, but accepts only the validated executor source revision.
"""

from pathlib import Path

from areal.engine._sglang_source_patch import patch_installed_module, patch_source

EXPECTED_SHA256 = "b69d19801ea61a36f6359e3e4f5464d9eaa1de825368a2c1aaa36005098f63b2"
PATCHED_SHA256 = "77b2e364f2e259a488e54f16de441361c4b33498460596990a3fdc78c73c424e"
TARGET_RELATIVE = Path("srt/multimodal/processors/executor.py")
OLD = "        return function(*args, processor=processor, **kwargs)\n"
NEW = """        try:
            return function(*args, processor=processor, **kwargs)
        except StopIteration as exc:
            # asyncio cannot copy StopIteration into a Future. Convert it here
            # so preprocessing unwinds and releases the model-update reader lock.
            raise ValueError(
                "Multimodal processor exhausted an input iterator; "
                "check image/video placeholders against supplied data."
            ) from exc
"""


def patched_source(source: bytes) -> bytes:
    """Convert StopIteration before it crosses the thread-to-asyncio boundary."""
    return patch_source(
        source,
        target_relative=TARGET_RELATIVE,
        replacements_by_sha256={EXPECTED_SHA256: [(OLD, NEW)]},
        patched_sha256=PATCHED_SHA256,
    )


def main() -> None:
    patch_installed_module(__doc__, TARGET_RELATIVE, patched_source)


if __name__ == "__main__":
    main()
