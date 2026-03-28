#!/usr/bin/env python3
"""Grade average accuracy for JSONL outputs by comparing boxed answers.

This script extracts the last boxed-style answer from each completion and compares
it against the boxed answer in ground truth.

Supported answer tags:
- \\boxed{...}
- \\boexd{...}  (common typo supported intentionally)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


BOXED_CMD_RE = re.compile(r"\\(boxed|boexd)\s*\{")


def normalize_answer(text: str) -> str:
    """Normalize extracted math answer text for robust string comparison."""
    s = text.strip()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", "", s)
    return s


def extract_last_boxed_payload(text: str) -> Optional[str]:
    """Extract the payload of the last \\boxed{...} / \\boexd{...} expression.

    Handles nested braces in the payload.
    """
    if not text:
        return None

    matches = list(BOXED_CMD_RE.finditer(text))
    for match in reversed(matches):
        start = match.end()  # position right after the opening '{'
        depth = 1
        i = start

        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1

        if depth == 0:
            payload = text[start : i - 1]
            return payload.strip()

    return None


def get_ground_truth_payload(obj: dict) -> Optional[str]:
    """Read and extract boxed payload from source.reward_model.ground_truth."""
    ground_truth = (
        obj.get("source", {})
        .get("reward_model", {})
        .get("ground_truth")
    )
    if not isinstance(ground_truth, str):
        return None
    return extract_last_boxed_payload(ground_truth)


def grade_file(jsonl_path: Path) -> None:
    total = 0
    correct = 0
    skipped = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                print(f"[skip] line {line_no}: invalid JSON")
                continue

            completion = obj.get("completion", "")
            if not isinstance(completion, str):
                skipped += 1
                print(f"[skip] line {line_no}: missing/invalid completion")
                continue

            pred = extract_last_boxed_payload(completion)
            gt = get_ground_truth_payload(obj)

            if pred is None or gt is None:
                skipped += 1
                missing = []
                if pred is None:
                    missing.append("prediction boxed answer")
                if gt is None:
                    missing.append("ground truth boxed answer")
                print(f"[skip] line {line_no}: missing {' and '.join(missing)}")
                continue

            total += 1
            if normalize_answer(pred) == normalize_answer(gt):
                correct += 1

    accuracy = (correct / total) if total else 0.0

    print("\n=== Boxed Accuracy Report ===")
    print(f"file: {jsonl_path}")
    print(f"graded: {total}")
    print(f"correct: {correct}")
    print(f"skipped: {skipped}")
    print(f"accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade JSONL by comparing boxed answers in completion vs ground truth."
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to input JSONL file",
    )
    args = parser.parse_args()

    if not args.jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {args.jsonl_path}")

    grade_file(args.jsonl_path)


if __name__ == "__main__":
    main()
