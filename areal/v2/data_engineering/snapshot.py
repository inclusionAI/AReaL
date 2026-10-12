# SPDX-License-Identifier: Apache-2.0

"""Content-verified task snapshots and explicit training-data materialization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT = "areal.data.snapshot.v1"


def json_bytes(value: Any) -> bytes:
    """Fixed snapshot encoding; these bytes determine content identities."""
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return decode_jsonl(path.read_bytes())


def decode_jsonl(data: bytes) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class DatasetSnapshot:
    root: Path
    dataset_id: str

    @property
    def public_path(self) -> Path:
        return self.root / "public.jsonl"

    @property
    def training_path(self) -> Path:
        return self.root / "train.jsonl"

    def training_rows(self) -> list[dict[str, Any]]:
        manifest, files = _verified_files(self.root)
        if manifest["dataset_id"] != self.dataset_id:
            raise ValueError("Snapshot differs from the requested dataset identity")
        return decode_jsonl(files["train.jsonl"])


def _verified_files(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    manifest = json.loads((root / "manifest.json").read_bytes())
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("Unsupported data snapshot format")
    payload = {key: value for key, value in manifest.items() if key != "dataset_id"}
    if manifest.get("dataset_id") != digest(json_bytes(payload)):
        raise ValueError("Snapshot identity mismatch")
    required = {
        "public.jsonl",
        "references.jsonl",
        "train.jsonl",
        "probes.jsonl",
        "report.json",
    }
    if (
        not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != required
    ):
        raise ValueError("Snapshot has unexpected or missing files")
    files = {}
    for name, expected in manifest["files"].items():
        path = root / name
        if path.is_symlink():
            raise ValueError("Snapshot files must be regular files")
        value = path.read_bytes()
        if digest(value) != expected:
            raise ValueError(f"Snapshot digest mismatch: {name}")
        files[name] = value
    public = decode_jsonl(files["public.jsonl"])
    references = decode_jsonl(files["references.jsonl"])
    ids = [row["task_id"] for row in public]
    if (
        any(not isinstance(task_id, str) or not task_id for task_id in ids)
        or len(ids) != len(set(ids))
        or ids != [row["task_id"] for row in references]
    ):
        raise ValueError("Snapshot public/reference task identities differ")
    if len(ids) != manifest["count"] or len(decode_jsonl(files["train.jsonl"])) != len(
        ids
    ):
        raise ValueError("Snapshot row count mismatch")
    return manifest, files


def load_snapshot(root: str | Path) -> DatasetSnapshot:
    path = Path(root).resolve()
    manifest, _ = _verified_files(path)
    return DatasetSnapshot(path, manifest["dataset_id"])


def publish_snapshot(
    destination: Path,
    *,
    public: list[dict[str, Any]],
    references: list[dict[str, Any]],
    training: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    report: dict[str, Any],
    provenance: dict[str, Any],
) -> DatasetSnapshot:
    if destination.exists():
        raise FileExistsError(f"Snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-", dir=destination.parent))
    try:
        values = {
            "public.jsonl": b"".join(json_bytes(row) for row in public),
            "references.jsonl": b"".join(json_bytes(row) for row in references),
            "train.jsonl": b"".join(json_bytes(row) for row in training),
            "probes.jsonl": b"".join(json_bytes(row) for row in probes),
            "report.json": json_bytes(report),
        }
        manifest = {
            "format": FORMAT,
            "count": len(public),
            "files": {name: digest(value) for name, value in values.items()},
            "provenance": provenance,
        }
        manifest["dataset_id"] = digest(json_bytes(manifest))
        for name, value in values.items():
            write_file(staging / name, value)
        write_file(staging / "manifest.json", json_bytes(manifest))
        _verified_files(staging)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DatasetSnapshot(destination.resolve(), manifest["dataset_id"])


def export_hf_dataset(snapshot: DatasetSnapshot, destination: str | Path) -> Path:
    """Materialize a task recipe's training view for existing AReaL dataset loaders.

    Tokenization and RL/SFT/OPD-specific tensor conversion remain downstream.
    """
    from datasets import Dataset

    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.export.lock"
    with lock.open("x"):
        staging: Path | None = None
        try:
            if destination.exists():
                raise FileExistsError(f"Training dataset already exists: {destination}")
            rows = snapshot.training_rows()
            staging = Path(tempfile.mkdtemp(prefix=".hf-data-", dir=destination.parent))
            Dataset.from_list(rows).save_to_disk(str(staging))
            write_file(
                staging / "areal-source.json",
                json_bytes({"dataset_id": snapshot.dataset_id}),
            )
            staging.rename(destination)
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            lock.unlink(missing_ok=True)
    return destination
