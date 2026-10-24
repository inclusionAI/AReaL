# SPDX-License-Identifier: Apache-2.0
"""Pinned Arena combination and public task index; contains no credentials."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class ArenaConfig:
    path: Path
    value: dict[str, Any]
    tasks: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, path: Path, *, task_index: Path | None = None) -> ArenaConfig:
        path = path.resolve()
        value = json.loads(path.read_text())
        if value["format_version"] != 1 or value["benchmark"] != "dapo_math":
            raise ValueError("This adapter supports the DAPO config schema v1")
        for field in ("harness_ref", "reward_ref"):
            if set(value[field]) != {"key", "version"} or not all(
                value[field].values()
            ):
                raise ValueError("Use explicit key/version references")
        sampling = value["sampling"]
        if set(sampling) != {"max_tokens", "temperature", "top_p"}:
            raise ValueError("Freeze max_tokens, temperature and top_p")
        if type(sampling["max_tokens"]) is not int or sampling["max_tokens"] < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not 0 <= sampling["temperature"] <= 2 or not 0 < sampling["top_p"] <= 1:
            raise ValueError("Invalid sampling parameters")
        if value["model_mode"] not in ("external", "mock"):
            raise ValueError("Declare external or mock model mode explicitly")
        arena = value["arena"]
        if not all(
            arena.get(k) for k in ("binary", "base_url", "profile", "stream_id")
        ):
            raise ValueError("Supply the target CLI, API base, profile and Stream")
        dataset = value["dataset"]
        payload = (task_index or path.parent / dataset["task_index"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != dataset["task_index_sha256"]:
            raise ValueError("Task index differs from its pinned digest")
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        tasks = {row["task_id"]: row for row in rows}
        if len(tasks) != len(rows) or len(rows) != dataset["count"]:
            raise ValueError("Task index count or identity is ambiguous")
        if len({row["data_id"] for row in rows}) != len(rows):
            raise ValueError("Each task must map to a distinct data_id")
        for row in rows:
            if not row["data_id"].startswith("env:") or "@" not in row["data_id"]:
                raise ValueError("Expected env:key@version data IDs")
        return cls(path, value, tasks)

    @property
    def fingerprint(self) -> str:
        return digest(self.value)

    def match(self, task_id: str, prompt: str) -> dict[str, Any]:
        row = self.tasks[task_id]
        if hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Task prompt differs from its published index")
        return row

    def spec_hash(self, reference: dict[str, str]) -> str:
        return self.value["component_spec_sha256"][
            reference["key"] + "@" + reference["version"]
        ]
