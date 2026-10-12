# SPDX-License-Identifier: Apache-2.0
"""Shared source validation and installation for pinned SGLang patches."""

import argparse
import hashlib
import importlib.util
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


def patch_source(
    source: bytes,
    *,
    target_relative: Path,
    replacements_by_sha256: Mapping[str, Sequence[tuple[str, str]]],
    patched_sha256: str,
    idempotent: bool = True,
) -> bytes:
    before = hashlib.sha256(source).hexdigest()
    if idempotent and before == patched_sha256:
        return source
    if before not in replacements_by_sha256:
        raise RuntimeError(
            f"Unknown SGLang {target_relative.name} source; refusing to modify"
        )
    text = source.decode()
    for old, new in replacements_by_sha256[before]:
        if text.count(old) != 1:
            raise RuntimeError("Expected exactly one source replacement site")
        text = text.replace(old, new)
    patched = text.encode()
    if hashlib.sha256(patched).hexdigest() != patched_sha256:
        raise RuntimeError("Patched SGLang source does not match validated source")
    compile(patched, str(target_relative), "exec")
    return patched


def patch_installed_module(
    description: str | None,
    target_relative: Path,
    transform: Callable[[bytes], bytes],
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--target", type=Path, help=f"Disposable {target_relative.name} copy"
    )
    target = parser.parse_args().target
    if target is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or spec.origin is None:
            raise RuntimeError("SGLang is not installed; pass --target")
        target = Path(spec.origin).parent / target_relative
    source = target.read_bytes()
    patched = transform(source)
    if patched != source:
        target.write_bytes(patched)
