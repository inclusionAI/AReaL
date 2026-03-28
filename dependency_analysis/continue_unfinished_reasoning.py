from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from sglang import RuntimeEndpoint, function, gen
from sglang.lang.api import set_default_backend

QWEN3_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>", "</s>"]


@function
def _continue_once(
    s,
    prompt_with_prefix: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    s += prompt_with_prefix
    s += gen(
        "continuation",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=STOP_TOKENS,
    )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_problem_text(row: dict[str, Any]) -> str:
    if "problem" in row:
        return str(row["problem"])

    source = row.get("source")
    if isinstance(source, dict):
        if "problem" in source:
            return str(source["problem"])

        prompt = source.get("prompt")
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                    return str(msg["content"])

    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                return str(msg["content"])

    raise ValueError("Cannot find problem text in row.")


def extract_system_prompt(row: dict[str, Any]) -> str:
    system_prompt = row.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt.strip():
        return system_prompt

    source = row.get("source")
    if isinstance(source, dict):
        prompt = source.get("prompt")
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "system" and "content" in msg:
                    value = str(msg["content"]).strip()
                    if value:
                        return value

    return QWEN3_DEFAULT_SYSTEM_PROMPT


def strip_terminal_stop_tokens(text: str) -> str:
    out = text
    changed = True
    while changed:
        changed = False
        stripped = out.rstrip()
        for token in STOP_TOKENS:
            if stripped.endswith(token):
                stripped = stripped[: -len(token)].rstrip()
                out = stripped
                changed = True
                break
    return out


def build_prompt_for_continuation(system_prompt: str, problem: str, completion_prefix: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n{completion_prefix}"
    )


def continue_completion(
    system_prompt: str,
    problem: str,
    completion_prefix: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = build_prompt_for_continuation(system_prompt, problem, completion_prefix)
    state = _continue_once(prompt, max_tokens, temperature, top_p)
    continuation = str(state["continuation"])
    return completion_prefix + continuation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a cleaned JSONL file, continue generation from each existing completion, "
            "and write rows in the same JSONL schema."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--backend",
        type=str,
        default="127.0.0.1:30000",
        help="SGLang backend host:port.",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="completion",
        help="Text field containing model response to continue. Default: completion",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of worker threads for concurrent continuation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max new tokens to append for each row.",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only process first N rows.")
    args = parser.parse_args()

    if args.threads <= 0:
        raise ValueError("--threads must be >= 1")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_default_backend(RuntimeEndpoint(f"http://{args.backend}"))

    rows = list(iter_jsonl(input_path))
    if args.limit > 0:
        rows = rows[: args.limit]

    tasks: list[tuple[int, dict[str, Any]]] = [(idx, row) for idx, row in enumerate(rows)]
    total = len(tasks)

    results: list[dict[str, Any] | None] = [None] * total
    done = 0

    def _worker(task: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        idx, row = task
        new_row = dict(row)
        original = row.get(args.field)
        if not isinstance(original, str) or not original.strip():
            return idx, new_row

        problem = extract_problem_text(row)
        system_prompt = extract_system_prompt(row)
        prefix = strip_terminal_stop_tokens(original)
        continued = continue_completion(
            system_prompt,
            problem,
            prefix,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        new_row[args.field] = continued
        return idx, new_row

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for fut in concurrent.futures.as_completed(futures):
            idx, row = fut.result()
            results[idx] = row
            done += 1
            if done % 10 == 0 or done == total:
                print(f"Progress: {done}/{total}", flush=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for row in results:
            if row is None:
                continue
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {total} rows to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
