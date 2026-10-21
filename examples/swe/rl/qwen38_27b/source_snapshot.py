"""Stage and verify a Git-independent, per-trial source snapshot using rsync."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROFILE = Path("examples/swe/rl/qwen38_27b/site.env")
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
SOURCE_DIRS = {"areal", "examples", "tests", "scripts"}


def excluded(path: Path) -> bool:
    return (
        bool(SKIP_DIRS.intersection(path.parts))
        or path.suffix in {".pyc", ".pyo", ".env"}
        or path.name in {".env", ".netrc"}
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def describe(root: Path, relative: Path) -> dict[str, str | int]:
    path = root / relative
    if path.is_symlink():
        target = path.resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Source symlink escapes the checkout: {relative}")
        return {"target": os.readlink(path)}
    if not path.is_file():
        raise ValueError(f"Source is not a regular file: {relative}")
    return {"sha256": digest(path), "executable": path.stat().st_mode & 0o111}


def inventory(root: Path) -> dict[str, dict[str, str | int]]:
    """Ignore only generated caches; the explicitly staged profile is included."""
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS)
        for name in [
            *files,
            *(name for name in dirs if (Path(directory) / name).is_symlink()),
        ]:
            path = (Path(directory) / name).relative_to(root)
            if path.suffix in {".pyc", ".pyo"}:
                continue
            result[path.as_posix()] = describe(root, path)
    return result


def revision(files: dict) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def verify(snapshot: Path, expected_revision: str) -> None:
    manifest = json.loads((snapshot / "manifest.json").read_text())
    actual = inventory(snapshot / "code")
    if (
        manifest.get("revision") != expected_revision
        or revision(manifest["files"]) != expected_revision
        or actual != manifest["files"]
    ):
        raise ValueError(
            "Source snapshot changed; use a new trial for changed code/profile"
        )


def stage(source: Path, profile: Path, destination: Path) -> str:
    source, profile, destination = (
        source.resolve(),
        profile.resolve(),
        destination.resolve(),
    )
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and snapshot directories must not overlap")
    if not profile.is_file():
        raise ValueError("Site profile does not exist")

    def git(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={source}", "-C", str(source), *args]
        )

    tracked = git("ls-files", "--cached", "-z").split(b"\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    # Include edits/deletions and new runtime source, without copying unrelated
    # root-level datasets, archives, experiment outputs, or ignored credentials.
    candidates = {Path(os.fsdecode(name)) for name in tracked if name}
    candidates.update(
        Path(os.fsdecode(name))
        for name in untracked
        if name and Path(os.fsdecode(name)).parts[0] in SOURCE_DIRS
    )
    selected = sorted(
        path
        for path in candidates
        if not excluded(path)
        and ((source / path).exists() or (source / path).is_symlink())
    )
    files = {path.as_posix(): describe(source, path) for path in selected}
    files[PROFILE.as_posix()] = {
        "sha256": digest(profile),
        "executable": profile.stat().st_mode & 0o111,
    }
    source_revision = revision(files)
    if destination.exists():
        verify(destination, source_revision)
        return source_revision

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        code = temporary / "code"
        code.mkdir()
        names = temporary / "files.list"
        names.write_bytes(b"\0".join(os.fsencode(path) for path in selected) + b"\0")
        subprocess.run(
            [
                "rsync",
                "-a",
                "--no-owner",
                "--no-group",
                "--from0",
                f"--files-from={names}",
                f"{source}/",
                f"{code}/",
            ],
            check=True,
        )
        target_profile = code / PROFILE
        target_profile.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(profile, target_profile)
        if inventory(code) != files:
            raise ValueError("Source/profile changed during rsync; retry submission")
        names.unlink()
        manifest = {
            "revision": source_revision,
            "git_head": git("rev-parse", "HEAD").decode().strip(),
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        # A completed snapshot is never overwritten, including concurrent starts.
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            verify(destination, source_revision)
        return source_revision
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("stage")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--profile", type=Path, required=True)
    create.add_argument("--destination", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--snapshot", type=Path, required=True)
    check.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.command == "stage":
        sys.stdout.write(stage(args.source, args.profile, args.destination) + "\n")
    else:
        verify(args.snapshot, args.revision)
        sys.stdout.write("Source snapshot verified: " + args.revision + "\n")


if __name__ == "__main__":
    main()
