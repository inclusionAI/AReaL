# SPDX-License-Identifier: Apache-2.0
"""Shared source validation and installation for pinned SGLang patches."""

import argparse
import fcntl
import hashlib
import importlib.util
import os
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


def patch_source(
    source: bytes,
    *,
    target_relative: Path,
    replacements_by_sha256: Mapping[str, Sequence[tuple[str, str]]],
    patched_sha256: str,
) -> bytes:
    before = hashlib.sha256(source).hexdigest()
    if before == patched_sha256:
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


def installed_sglang_root() -> Path:
    """Locate the current interpreter's installation without importing SGLang."""
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("SGLang is not installed; pass --target for a source copy")
    return Path(spec.origin).parent


def _atomic_write(target: Path, source: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(source)
            os.fchmod(stream.fileno(), stat.S_IMODE(target.stat().st_mode))
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install_source_patches(
    targets: Sequence[tuple[Path, Callable[[bytes], bytes]]], *, lock_path: Path
) -> None:
    """Validate the whole set before atomically installing any changed file.

    Concurrent starts share a lock. Fully patched installations require no writes,
    including no lock-file creation, so they also work on read-only filesystems.
    """

    def prepare() -> list[tuple[Path, bytes]]:
        changes = []
        for target, transform in targets:
            source = target.read_bytes()
            patched = transform(source)
            if patched != source:
                changes.append((target, patched))
        return changes

    if not prepare():
        return
    try:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # A concurrent start may have installed the patches while we waited.
            for target, patched in prepare():
                _atomic_write(target, patched)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot install SGLang patches at {lock_path.parent}; "
            "use a writable installation or prepatched runtime"
        ) from exc


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
        root = installed_sglang_root()
        target = root / target_relative
    else:
        root = target.parent
    install_source_patches(
        [(target, transform)], lock_path=root / ".areal-sglang-patches.lock"
    )
