# SPDX-License-Identifier: Apache-2.0
"""Prevent exhausted SGLang multimodal iterators from stranding asyncio requests.

Apply explicitly before starting any supported SGLang multimodal recipe with
``python -m areal.engine.sglang_mm_executor``. The patch is independent of the
model and workflow, but accepts only the validated executor source revision.
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path

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
    before = hashlib.sha256(source).hexdigest()
    if before == PATCHED_SHA256:
        return source
    if before != EXPECTED_SHA256:
        raise RuntimeError("Unknown SGLang multimodal executor; refusing to modify")
    text = source.decode()
    if text.count(OLD) != 1:
        raise RuntimeError("Expected exactly one multimodal processor invocation")
    patched = text.replace(OLD, NEW).encode()
    if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
        raise RuntimeError(
            "Patched multimodal executor does not match validated source"
        )
    compile(patched, str(TARGET_RELATIVE), "exec")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, help="Disposable executor.py copy")
    args = parser.parse_args()
    target = args.target
    if target is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or spec.origin is None:
            raise RuntimeError("SGLang is not installed; pass --target")
        target = Path(spec.origin).parent / TARGET_RELATIVE
    source = target.read_bytes()
    patched = patched_source(source)
    if patched != source:
        target.write_bytes(patched)


if __name__ == "__main__":
    main()
