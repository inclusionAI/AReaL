from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from pathlib import Path
from typing import Any

from api import call_model_claude


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def has_two_or_more_alternatively(text: str) -> bool:
    # Support both "Alternatively ... Alternatively" and "Wait ... Wait" patterns.
    pattern = re.compile(r"(?i)\b(?:alternatively|wait)\b")
    return sum(1 for _ in pattern.finditer(text)) >= 2


def extract_alternatively_spans(text: str) -> list[tuple[int, int, str]]:
    """Return spans [start, end) between consecutive marker words.

    Supported markers: "Alternatively", "Wait" (case-insensitive).
    """
    pattern = re.compile(r"(?i)\b(?:alternatively|wait)\b")
    matches = list(pattern.finditer(text))
    spans: list[tuple[int, int, str]] = []
    for i in range(len(matches) - 1):
        start = matches[i].start()
        end = matches[i + 1].start()
        spans.append((start, end, text[start:end]))
    return spans


def build_judge_prompt(segment: str) -> str:
    return (
        "You are judging a math reasoning segment.\n"
        "Task: Decide whether this segment is a failed trial that should be deleted.\n"
        "Rules:\n"
        "1) Return failed=true only if this segment is clearly an incorrect segment. Or this segment has tried a method, but this method doesn't work or it is too complicated to continue work with this method.\n"
        "2) Return failed=false if the segment looks valid, useful, or uncertain.\n"
        "3) Return ONLY valid JSON object: {\"failed\": true_or_false}.\n"
        "\n"
        "Segment:\n"
        "-----\n"
        f"{segment}\n"
        "-----"
    )


def llm_is_failed_segment(segment: str, model: str, max_retries: int = 2) -> bool | None:
    prompt = build_judge_prompt(segment)

    for _ in range(max_retries + 1):
        response = call_model_claude(
            prompt,
            model=model,
            temperature=0.0,
        )
        if response is None:
            continue

        try:
            content = response["choices"][0]["message"]["content"]
        except Exception:
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        # First try direct JSON parse.
        try:
            payload = json.loads(content)
            failed = payload.get("failed")
            if isinstance(failed, bool):
                return failed
        except Exception:
            pass

        # Fallback: extract fenced JSON block if model wrapped output.
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                try:
                    payload = json.loads(candidate)
                    failed = payload.get("failed")
                    if isinstance(failed, bool):
                        return failed
                except Exception:
                    continue

    # If LLM parsing fails, return unknown.
    return None


def judge_spans_parallel(
    spans: list[tuple[int, int, str]],
    model: str,
    max_workers: int,
) -> list[bool]:
    """Judge all spans concurrently and return failed flags in original order."""
    if not spans:
        return []

    workers = max(1, min(max_workers, len(spans)))
    if workers == 1:
        return [llm_is_failed_segment(segment, model=model) is True for _, _, segment in spans]

    failed_flags: list[bool] = [False] * len(spans)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(llm_is_failed_segment, segment, model): idx
            for idx, (_start, _end, segment) in enumerate(spans)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                decision = future.result()
            except Exception:
                decision = None
            # Conservative fallback: keep segment if undecidable.
            failed_flags[idx] = decision is True

    return failed_flags


def remove_failed_spans_by_judgment(text: str, failed_flags: list[bool]) -> tuple[str, int]:
    spans = extract_alternatively_spans(text)
    if not spans:
        return text, 0

    if len(spans) != len(failed_flags):
        return text, 0

    # If step i is deleted, also delete i+2, i+3, ... while keeping i+1.
    adjusted_flags = list(failed_flags)
    for i, is_failed in enumerate(failed_flags):
        if is_failed:
            if i + 1 < len(adjusted_flags):
                adjusted_flags[i + 1] = False
            for j in range(i + 2, len(adjusted_flags)):
                adjusted_flags[j] = True
            break

    out_parts: list[str] = []
    cursor = 0
    removed = 0

    for (start, end, _segment), is_failed in zip(spans, adjusted_flags):
        out_parts.append(text[cursor:start])
        if is_failed:
            removed += 1
        else:
            out_parts.append(text[start:end])
        cursor = end

    out_parts.append(text[cursor:])
    return "".join(out_parts), removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean failed trial spans enclosed by two 'Alternatively' markers in completion text via LLM."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSONL path.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--field",
        type=str,
        default="completion",
        help="Text field to clean in each JSON row. Default: completion",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model name passed to call_model_claude.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only process first N rows.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent threads used for LLM calls per row.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    scanned = 0
    llm_calls = 0
    judged_segments = 0
    removed_segments = 0
    changed = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for row in iter_jsonl(input_path):
            if args.limit > 0 and scanned >= args.limit:
                break

            scanned += 1
            new_row = dict(row)

            text = row.get(args.field)
            if isinstance(text, str) and has_two_or_more_alternatively(text):
                spans = extract_alternatively_spans(text)
                llm_calls += len(spans)
                judged_segments += len(spans)
                failed_flags = judge_spans_parallel(
                    spans,
                    model=args.model,
                    max_workers=args.max_workers,
                )

                cleaned, removed_now = remove_failed_spans_by_judgment(text, failed_flags)
                removed_segments += removed_now
                if cleaned != text:
                    changed += 1
                new_row[args.field] = cleaned

            out_f.write(json.dumps(new_row, ensure_ascii=False) + "\n")

            if scanned % 10 == 0:
                print(
                    (
                        f"Progress: {scanned} | llm_calls={llm_calls} "
                        f"| judged_segments={judged_segments} "
                        f"| removed_segments={removed_segments} | changed={changed}"
                    ),
                    flush=True,
                )

    if scanned % 10 != 0:
        print(
            (
                f"Progress: {scanned} | llm_calls={llm_calls} "
                f"| judged_segments={judged_segments} "
                f"| removed_segments={removed_segments} | changed={changed}"
            ),
            flush=True,
        )

    print(
        (
            f"Done. rows={scanned} llm_calls={llm_calls} "
            f"judged_segments={judged_segments} removed_segments={removed_segments} "
            f"changed={changed} output={output_path}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
