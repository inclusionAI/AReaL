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


@function
def _generate_once(
    s,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    s += prompt
    s += gen(
        "answer",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|im_end|>", "<|endoftext|>", "</s>"],
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_problem_text(row: dict[str, Any]) -> str:
    # Supports both the attached schema (`prompt` as chat list) and eval/aime.jsonl schema (`problem`).
    if isinstance(row.get("prompt"), list):
        for msg in row["prompt"]:
            if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                return str(msg["content"])

    if "problem" in row:
        return str(row["problem"])

    raise ValueError("Cannot find problem text in row (expected `prompt` or `problem`).")


def build_qwen3_prompt(problem: str) -> str:
    return (
        f"<|im_start|>system\n{QWEN3_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_one_completion(
    problem: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = build_qwen3_prompt(problem)
    state = _generate_once(prompt, max_tokens, temperature, top_p)
    return str(state["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Connect to SGLang backend at 127.0.0.1:30000 (default), "
            "load AIME-style problems, prepend Qwen3 default system prompt, "
            "and solve each problem multiple times (default: 8)."
        )
    )
    parser.add_argument("--input", type=str, default="AIME_test.jsonl", help="Input JSONL file.")
    parser.add_argument("--output", type=str, default="eval/aime_qwen3_8x_outputs.jsonl", help="Output JSONL file.")
    parser.add_argument(
        "--backend",
        type=str,
        default="127.0.0.1:30000",
        help="SGLang backend host:port.",
    )
    parser.add_argument("--repeat", type=int, default=8, help="Number of completions per problem.")
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of worker threads for concurrent generation.",
    )
    parser.add_argument("--max-tokens", type=int, default=40960, help="Max new tokens per completion.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only process first N problems.")
    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be >= 1")
    if args.threads <= 0:
        raise ValueError("--threads must be >= 1")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_default_backend(RuntimeEndpoint(f"http://{args.backend}"))

    rows = read_jsonl(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    tasks: list[tuple[int, int, str, dict[str, Any]]] = []
    for problem_idx, row in enumerate(rows):
        problem = extract_problem_text(row)
        for repeat_idx in range(args.repeat):
            tasks.append((problem_idx, repeat_idx, problem, row))

    total_requests = len(tasks)
    done = 0
    records: list[dict[str, Any]] = []

    def _worker(task: tuple[int, int, str, dict[str, Any]]) -> dict[str, Any]:
        problem_idx, repeat_idx, problem, row = task
        completion = generate_one_completion(
            problem,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        return {
            "problem_index": problem_idx,
            "repeat_index": repeat_idx,
            "system_prompt": QWEN3_DEFAULT_SYSTEM_PROMPT,
            "problem": problem,
            "completion": completion,
            "source": row,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for fut in concurrent.futures.as_completed(futures):
            records.append(fut.result())
            done += 1
            if done % 10 == 0 or done == total_requests:
                print(f"Progress: {done}/{total_requests}")

    records.sort(key=lambda r: (int(r["problem_index"]), int(r["repeat_index"])))
    with output_path.open("w", encoding="utf-8") as out_f:
        for record in records:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {total_requests} generations to: {output_path}")


if __name__ == "__main__":
    main()
