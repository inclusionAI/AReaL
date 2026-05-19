#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert collected.jsonl into a Hugging Face dataset, with an option
to override the 'thinking' trajectory using a glob of .txt files.

Usage examples
--------------
# Basic: just the JSONL
python save_collected_jsonl_to_hf_dataset.py \
  --input /path/to/collected.jsonl \
  --output ./out_dataset

# Replace 'thinking' with trajectories from *.txt (index-matched), then build chat
python save_collected_jsonl_to_hf_dataset.py \
  --input /path/to/collected.jsonl \
  --output ./out_dataset \
  --trajectory-glob "/path/to/step6/**/*.txt" \
  --build-chat \
  --qwen-model Qwen/Qwen3-8B
"""
import argparse
import glob
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

from datasets import Dataset
from tqdm import tqdm
# transformers is only required if --build-chat is used
try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


def load_collected(jsonl_path: str) -> List[Dict]:
    """Load collected.jsonl and normalize fields."""
    records: List[Dict] = []
    
    # First pass to count lines for progress bar
    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        with tqdm(total=total_lines, desc="Loading JSONL") as pbar:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                question = d.get("problem", "") or ""
                thinking = d.get("thinking", "") or ""
                output = d.get("output", "") or ""

                rec = {
                    "uuid": d.get("uuid", ""),
                    "question": question,
                    "response_reasoning": thinking,
                    "response_content": output,
                    "correctness": bool(d.get("correctness", True)),
                }
                # Try to pull trailing index from uuid "...-<int>"
                m = re.search(r"-(\d+)$", rec["uuid"])
                rec["_uuid_index"] = int(m.group(1)) if m else None
                records.append(rec)
                pbar.update(1)
    
    if not records:
        raise ValueError(f"No records found in {jsonl_path}")
    return records


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


def clean_thinking(text: str) -> str:
    """Normalize any <Think> tags and strip them; return inner content."""
    if not text:
        return ""
    norm = text.replace("<Think>", "<think>").replace("</Think>", "</think>")
    # Remove any explicit think tags if present
    norm = re.sub(r"</?\s*think\s*>", "", norm, flags=re.IGNORECASE)
    return norm.strip()


_PATH_SANITIZE_RE = re.compile(r"(<Thread>)\s*(\d+):\s*")
_PATH_CLOSE_SANITIZE_RE = re.compile(r"\s*</Thread>")
_PATH_ADJACENT_SANITIZE_RE = re.compile(r"</Thread>\s*<Thread>")


def sanitize_text_block(text: str) -> str:
    """Normalize <Thread> blocks to use '<Thread>\\n<idx>: ' and ensure newline before </Thread>."""
    if not text:
        return text
    text = _PATH_SANITIZE_RE.sub(r"\1\n\2: ", text)
    text = _PATH_CLOSE_SANITIZE_RE.sub(r"\n</Thread>", text)
    return _PATH_ADJACENT_SANITIZE_RE.sub(r"</Thread><Thread>", text)


def sanitize_records(records: List[Dict]) -> None:
    """Apply path block sanitization to key text fields."""
    for rec in records:
        for field in ("question", "response_reasoning", "response_content"):
            value = rec.get(field)
            if isinstance(value, str) and value:
                rec[field] = sanitize_text_block(value)


def maybe_override_with_txt(
    records: List[Dict],
    glob_pattern: Optional[str],
    strict: bool,
) -> Tuple[int, int, str]:
    """
    If glob_pattern is provided, read .txt trajectories and replace response_reasoning.
    Returns (matched, total_txt, mode_used).
    """
    if not glob_pattern:
        # Mark all records as unmatched when no trajectory glob is provided
        for rec in records:
            rec["_trajectory_matched"] = False
        return 0, 0, "none"

    txt_paths = [p for p in glob.glob(glob_pattern, recursive=True) if os.path.isfile(p)]
    if not txt_paths:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    # Try index-based alignment first (e.g., '17.txt' -> index 17)
    def idx_from_path(p: str) -> Optional[int]:
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.fullmatch(r"(\d+)", base)
        return int(m.group(1)) if m else None

    idx_to_txt: Dict[int, str] = {}
    for p in tqdm(txt_paths, desc="Processing trajectory files"):
        idx = idx_from_path(p)
        if idx is not None:
            idx_to_txt[idx] = p

    matched = 0
    mode_used = "auto-lex"
    if idx_to_txt and any(rec.get("_uuid_index") is not None for rec in records):
        # Index mode: match on shared integer indices between uuid and filename
        mode_used = "index"
        for rec in tqdm(records, desc="Matching trajectories by index"):
            idx = rec.get("_uuid_index")
            if idx is not None and idx in idx_to_txt:
                rec["response_reasoning"] = clean_thinking(read_text(idx_to_txt[idx]))
                rec["trajectory_source"] = idx_to_txt[idx]
                rec["_trajectory_matched"] = True
                matched += 1
            else:
                rec["_trajectory_matched"] = False

        # If strict, we also require that every record had a matching txt
        if strict:
            exp = sum(1 for r in records if r.get("_uuid_index") is not None)
            if matched != exp:
                raise ValueError(
                    f"--trajectory-strict: matched {matched} / {exp} records by index."
                )
        return matched, len(txt_paths), mode_used

    # Lexicographic fallback: sort both and pair in order
    txt_paths_sorted = sorted(txt_paths)
    n = min(len(records), len(txt_paths_sorted))
    
    # Mark all records as unmatched initially for lexicographic mode
    for rec in records:
        rec["_trajectory_matched"] = False
    
    for i in tqdm(range(n), desc="Matching trajectories lexicographically"):
        rec = records[i]
        p = txt_paths_sorted[i]
        rec["response_reasoning"] = clean_thinking(read_text(p))
        rec["trajectory_source"] = p
        rec["_trajectory_matched"] = True
        matched += 1

    if strict and (len(records) != len(txt_paths_sorted)):
        raise ValueError(
            f"--trajectory-strict: record count ({len(records)}) != txt count ({len(txt_paths_sorted)})."
        )

    return matched, len(txt_paths), mode_used


def maybe_build_chat(
    records: List[Dict],
    build_chat: bool,
    qwen_model: str,
    instruction: str,
) -> None:
    """Optionally add qwen_text + token counts to each record."""
    if not build_chat:
        return
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers is required for --build-chat, but is not available.")

    qwen_tok = None

    if qwen_model:
        qwen_tok = AutoTokenizer.from_pretrained(qwen_model)
        # Qwen3 templates typically already retain <think>

    for rec in tqdm(records, desc="Building chat templates"):
        q = rec["question"]
        r_think = rec["response_reasoning"] or ""
        r_out = rec["response_content"] or ""

        # Reconstruct assistant content as <think>…</think>\nfinal
        assistant_text = f"<think>\n{r_think.strip()}\n</think>\n{r_out.strip()}"

        messages = [
            {"role": "user", "content": f"{q} {instruction}".strip()},
            {"role": "assistant", "content": assistant_text},
        ]
        rec["raw_messages"] = messages  # for reference/debug

        if qwen_tok:
            qwen_text = qwen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            rec["qwen_text"] = qwen_text
            try:
                rec["num_qwen_tokens"] = len(qwen_tok(qwen_text)["input_ids"])
            except Exception:
                rec["num_qwen_tokens"] = -1


def save_variants(
    records: List[Dict],
    out_dir: str,
    sample_size: int = 0,
    repeat: int = 0,
    seed: int = 0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("Saving base dataset...")
    base_ds = Dataset.from_list(records)

    # Optional sample
    if sample_size and sample_size > 0:
        n = len(base_ds)
        if n <= sample_size:
            sampled = base_ds
        else:
            print(f"Sampling {sample_size} records from {n} total...")
            rng = random.Random(seed)
            idx = rng.sample(range(n), sample_size)
            sampled = base_ds.select(idx)

        sample_dir = os.path.join(out_dir, f"sample_{len(sampled)}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Saving sampled dataset ({len(sampled)} records)...")
        # sampled.save_to_disk(os.path.join(sample_dir, "dataset"))
        sampled.to_parquet(os.path.join(sample_dir, "train.parquet"))

        # Optional repeated/shuffled expansion
        if repeat and repeat > 1:
            print(f"Creating {repeat}× shuffled expansion...")
            base_list = [sampled[i] for i in range(len(sampled))]
            rng = random.Random(seed)
            expanded: List[Dict] = []
            for _ in tqdm(range(repeat), desc="Creating shuffled copies"):
                tmp = list(base_list)
                rng.shuffle(tmp)
                expanded.extend(tmp)

            print(f"Saving expanded dataset ({len(expanded)} records)...")
            ds_rep = Dataset.from_list(expanded)
            rep_dir = os.path.join(out_dir, f"sample_{len(sampled)}_{repeat}x")
            os.makedirs(rep_dir, exist_ok=True)
            # ds_rep.save_to_disk(os.path.join(rep_dir, "dataset"))
            ds_rep.to_parquet(os.path.join(rep_dir, "train.parquet"))

    # Manifest
    # manifest = {
    #     "num_rows": len(base_ds),
    #     "paths": {
    #         "save_to_disk": os.path.join(out_dir, "dataset"),
    #         "parquet": os.path.join(out_dir, "train.parquet"),
    #     },
    #     "sample": sample_size,
    #     "repeat": repeat,
    #     "seed": seed,
    # }
    # with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    #     json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Save collected.jsonl to a HF dataset, optionally overriding trajectories from .txt files.")
    ap.add_argument("--input", required=True, help="Path to collected.jsonl")
    ap.add_argument("--output", required=True, help="Output directory for the dataset")
    ap.add_argument("--trajectory-glob", default=None,
                    help="Glob for raw trajectory .txt files to override 'thinking' (e.g., '/path/to/step6/**/*.txt').")
    ap.add_argument("--trajectory-strict", action="store_true",
                    help="Require 1:1 match when using --trajectory-glob (index or count).")
    ap.add_argument("--trajectory-matched-only", action="store_true",
                    help="Only save records that have matched trajectory files (requires --trajectory-glob).")
    ap.add_argument("--build-chat", action="store_true",
                    help="Build qwen_text with <think> retained and add token counts.")
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-8B",
                    help="Qwen model for chat template (only used with --build-chat).")
    ap.add_argument("--instruction", default="Let's think step by step and output the final answer within \\boxed{}.",
                    help="Instruction appended to the user question when building chat texts.")
    ap.add_argument("--sample-size", type=int, default=0,
                    help="If >0, create a sampled subset of this size.")
    ap.add_argument("--repeat", type=int, default=0,
                    help="If >1 and sample-size>0, create an N× shuffled expansion of the sample.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling/shuffling.")
    ap.add_argument("--no-sanitize", action="store_true",
                    help="Skip normalizing <Thread> blocks in collected data.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow writing into an existing output directory.")
    args = ap.parse_args()

    sanitize = not args.no_sanitize
    base_out_dir = os.path.abspath(args.output)
    out_dir = base_out_dir

    sanitize_status = "enabled" if sanitize else "disabled"

    if os.path.exists(out_dir):
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}")
        print(f"[overwrite] existing contents may be replaced in {out_dir}")

    records = load_collected(args.input)

    matched, total_txt, mode = maybe_override_with_txt(
        records=records,
        glob_pattern=args.trajectory_glob,
        strict=args.trajectory_strict,
    )
    if args.trajectory_glob:
        print(f"[trajectory] mode={mode} matched={matched} txt_files={total_txt} strict={args.trajectory_strict}")

    if sanitize:
        sanitize_records(records)
        print("[sanitize] normalized <Thread> blocks")

    # Filter to only matched records if requested
    if args.trajectory_matched_only:
        if not args.trajectory_glob:
            raise ValueError("--trajectory-matched-only requires --trajectory-glob to be specified")
        original_count = len(records)
        records = [rec for rec in records if rec.get("_trajectory_matched", False)]
        print(f"[filtering] kept {len(records)} records with matched trajectories (was {original_count})")

    maybe_build_chat(
        records=records,
        build_chat=args.build_chat,
        qwen_model=args.qwen_model,
        instruction=args.instruction,
    )

    save_variants(
        records=records,
        out_dir=out_dir,
        sample_size=args.sample_size,
        repeat=args.repeat,
        seed=args.seed,
    )
    print(f"[sanitize] status={sanitize_status} saved_output={out_dir}")
    print(f"Saved dataset with {len(records)} rows to {out_dir}")


if __name__ == "__main__":
    main()
