from __future__ import annotations

import json
import os
from typing import Iterable


def iter_meta_info_files(result_path: str, filename: str = "meta_info.jsonl") -> Iterable[str]:
    for name in os.listdir(result_path):
        subdir = os.path.join(result_path, name)
        if not os.path.isdir(subdir):
            continue
        meta_path = os.path.join(subdir, filename)
        if os.path.isfile(meta_path):
            yield meta_path


def load_records(meta_path: str) -> Iterable[dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line in {meta_path}: {line}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSON line is not an object in {meta_path}: {line}")
            yield payload
