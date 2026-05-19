#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import re
import glob
import statistics
import concurrent.futures
from typing import Dict, Tuple, List, Optional

import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Hugging Face tokenizer (same as your eval script) ---
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# --- HF datasets loader ---
from datasets import load_dataset

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"
PARALLELIZE_DELIMITER_START = "<Parallel>"
PARALLELIZE_DELIMITER_END = "</Parallel>"

tags_to_check = ["Think", "Parallel", "Outlines", "Outline", "Thread", "Conclusion"]


def _whitespace_token_count(text: str) -> int:
    """Count tokens using whitespace as delimiter."""
    return len(text.split())


def _hf_token_count(text: str, tokenizer: Optional[PreTrainedTokenizerBase]) -> int:
    """Count tokens using HuggingFace tokenizer."""
    if tokenizer is None:
        print("WARNING: tokenizer is None, falling back to using whitespace as delimiter")
        return _whitespace_token_count(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def compute_parallel_metrics_and_thread_tokens(
    model_response: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None
) -> tuple[
    Optional[float],  # parallel_ratio
    Optional[int],    # num_tokens_in_the_longest_thread
    int,              # total_num_tokens
    Optional[float],  # avg_thread_length
    Optional[float],  # avg_tokens_per_parallel_block
    Optional[float],  # avg_threads_per_parallel_block
    Optional[float],  # avg_outlines_block_tokens
    Optional[float],  # avg_conclusion_block_tokens
]:
    """Compute metrics about <Parallel> structure."""
    start, end = PARALLELIZE_DELIMITER_START, PARALLELIZE_DELIMITER_END

    total_num_tokens = _hf_token_count(model_response, tokenizer)

    starts = model_response.count(start)
    ends = model_response.count(end)

    # Case 1: No parallel tags. The "longest thread" is the entire response.
    if starts == 0 and ends == 0:
        return 0.0, total_num_tokens, total_num_tokens, None, None, None, 0, 0

    # Case 2: Mismatched tags are an error.
    if starts != ends:
        return None, None, total_num_tokens, None, None, None, None, None

    # Case 3: no tokens at all (will not happen in practice)
    if total_num_tokens == 0:
        return None, None, 0, None, None, None, None, None

    inside_ratio_tokens = 0
    num_tokens_in_the_longest_thread = 0
    last_end_idx = 0
    all_thread_token_counts = []
    parallel_block_token_counts = []
    parallel_block_thread_counts = []

    parallel_block_matches = list(re.finditer(rf'{re.escape(start)}(.*?){re.escape(end)}', model_response, re.DOTALL))
    outlines_block_token_counts: list[int] = []
    conclusion_block_token_counts: list[int] = []

    for match in parallel_block_matches:
        # Accumulate tokens from the non-parallel text segment before this block
        non_parallel_segment = model_response[last_end_idx:match.start()]
        num_tokens_in_the_longest_thread += _hf_token_count(non_parallel_segment, tokenizer)

        # Process the content within the current parallel block
        block_content = match.group(1)

        # --- Ratio Calculation ---
        block_content_without_placeholders = re.sub(r'<Thread>\s*\d+:\s*\[placeholder\]\s*</Thread>', '', block_content, flags=re.IGNORECASE)
        block_token_count = _hf_token_count(block_content_without_placeholders, tokenizer)
        inside_ratio_tokens += block_token_count

        # Track tokens per parallel block
        parallel_block_token_counts.append(_hf_token_count(start + block_content_without_placeholders + end, tokenizer))

        # --- Longest Thread Calculation ---
        thread_tokens_for_this_block = _hf_token_count(start + end, tokenizer)

        # Find and add tokens for the full <Outlines>...</Outlines> block
        outlines_block_match = re.search(r'\s*<Outlines>.*?</Outlines>\s*', block_content, re.DOTALL)
        if outlines_block_match:
            outlines_tokens = _hf_token_count(outlines_block_match.group(0), tokenizer)
            thread_tokens_for_this_block += outlines_tokens
            outlines_block_token_counts.append(outlines_tokens)

        # Find and add tokens for the full <Conclusion>...</Conclusion> block
        conclusion_block_match = re.search(r'\s*<Conclusion>.*?</Conclusion>\s*', block_content, re.DOTALL)
        if conclusion_block_match:
            conclusion_tokens = _hf_token_count(conclusion_block_match.group(0), tokenizer)
            thread_tokens_for_this_block += conclusion_tokens
            conclusion_block_token_counts.append(conclusion_tokens)

        # Find all full <Thread>...</Thread> blocks
        thread_block_matches = re.finditer(r'<Thread>.*?</Thread>', block_content, re.DOTALL)
        thread_token_counts = []
        for t_match in thread_block_matches:
            thread_content = t_match.group(0)
            # Skip placeholder threads
            if not re.search(r'<Thread>\s*\d+:\s*\[placeholder\]\s*</Thread>', thread_content, re.IGNORECASE):
                thread_token_counts.append(_hf_token_count(thread_content, tokenizer))
        if thread_token_counts:
            thread_tokens_for_this_block += max(thread_token_counts)
            all_thread_token_counts.extend(thread_token_counts)

        # Track threads per parallel block
        parallel_block_thread_counts.append(len(thread_token_counts))

        num_tokens_in_the_longest_thread += thread_tokens_for_this_block
        last_end_idx = match.end()

    # Add tokens from the final text segment after the last parallel block
    final_segment = model_response[last_end_idx:]
    num_tokens_in_the_longest_thread += _hf_token_count(final_segment, tokenizer)

    # Empty parallel blocks are an error for the ratio
    if inside_ratio_tokens == 0:
        outlines_avg = (sum(outlines_block_token_counts) / len(outlines_block_token_counts)) if outlines_block_token_counts else 0
        conclusion_avg = (sum(conclusion_block_token_counts) / len(conclusion_block_token_counts)) if conclusion_block_token_counts else 0
        return None, num_tokens_in_the_longest_thread, total_num_tokens, None, None, None, outlines_avg, conclusion_avg

    ratio = inside_ratio_tokens / total_num_tokens

    # Calculate averages
    avg_thread_length = sum(all_thread_token_counts) / len(all_thread_token_counts) if all_thread_token_counts else None
    avg_tokens_per_parallel_block = sum(parallel_block_token_counts) / len(parallel_block_token_counts) if parallel_block_token_counts else None
    avg_threads_per_parallel_block = sum(parallel_block_thread_counts) / len(parallel_block_thread_counts) if parallel_block_thread_counts else None
    avg_outlines_length = (sum(outlines_block_token_counts) / len(outlines_block_token_counts)) if outlines_block_token_counts else None
    avg_conclusion_length = (sum(conclusion_block_token_counts) / len(conclusion_block_token_counts)) if conclusion_block_token_counts else None

    return (
        ratio,
        num_tokens_in_the_longest_thread,
        total_num_tokens,
        avg_thread_length,
        avg_tokens_per_parallel_block,
        avg_threads_per_parallel_block,
        avg_outlines_length,
        avg_conclusion_length,
    )


def has_tags(txt):
    """Check if text contains any XML-like tags."""
    tag_pattern = r'<(/?)(\w+)>'
    return bool(re.search(tag_pattern, txt, re.IGNORECASE))


def is_parallel_format_correct_v2(
    model_response: str,
    treat_no_parallel_as_format_error,
    allow_nonempty_whitespace: bool = False,
    skip_conclusion_check: bool = False,
    verbose: bool = True
) -> bool:
    """
    Checks if the response's <Parallel> blocks follow the strict rules:
      - Any number of non-nested <Parallel>…</Parallel> blocks
      - Inside each:
          <Outlines> containing only properly numbered <Outline> tags
          one-to-one <Thread> tags matching those outlines, adjacent with no whitespace
          a single <Conclusion> after all threads
      - No other tags or nested Parallels
    """
    if not (PARALLELIZE_DELIMITER_START in model_response and PARALLELIZE_DELIMITER_END in model_response):
        if treat_no_parallel_as_format_error:
            return False

    # 1) Equal count of opening and closing Parallel tags
    if model_response.count(PARALLELIZE_DELIMITER_START) != model_response.count(PARALLELIZE_DELIMITER_END):
        if verbose:
            print("Unequal count of opening and closing Parallel tags")
        return False

    # 2) Extract and validate each top-level <Parallel>…</Parallel> block
    for pm in re.finditer(r'<Parallel>(.*?)</Parallel>', model_response, re.DOTALL):
        block = pm.group(1)

        # 2a) No nested Parallel
        if '<Parallel>' in block or '</Parallel>' in block:
            if verbose:
                print("Nested Parallel tags")
            return False

        # 2b) No disallowed tags inside this Parallel
        for tag in re.findall(r'<(/?)(\w+)>', block):
            if tag[1].lower() not in ('outlines', 'outline', 'thread', 'conclusion'):
                if verbose:
                    print("Disallowed tag", tag)
                return False

        # 2c) Match the exact sequence: Outlines → Outline + Thread → Conclusion
        num_outline_start, num_outline_end = block.count('<Outline>'), block.count('</Outline>')
        num_thread_start, num_thread_end = block.count('<Thread>'), block.count('</Thread>')
        num_conclusion_start, num_conclusion_end = block.count('<Conclusion>'), block.count('</Conclusion>')

        if num_outline_start != num_thread_start:
            if verbose:
                print(f"Mismatched number of Outline and Thread tags: {num_outline_start}, {num_thread_start}")
            return False

        if num_outline_start != num_outline_end:
            if verbose:
                print(f"Mismatched number of Outline opening and closing tags: {num_outline_start}, {num_outline_end}")
            return False

        if num_thread_start != num_thread_end:
            if verbose:
                print(f"Mismatched number of Thread opening and closing tags: {num_thread_start}, {num_thread_end}")
            return False

        if num_conclusion_start != num_conclusion_end:
            if verbose:
                print(f"Mismatched number of Conclusion opening and closing tags: {num_conclusion_start}, {num_conclusion_end}")
            return False

        if not skip_conclusion_check and num_conclusion_start != 1:
            if verbose:
                print(f"Incorrect number of Conclusion tags: {num_conclusion_start}")
            return False

        if num_outline_start > 50:
            if verbose:
                print(f"Too many Outline tags: {num_outline_start}")
            return False

        if num_conclusion_start > 0:
            seq_pattern = re.compile(
                r'^\s*'
                r'<Outlines>(?P<outlines>.*?)</Outlines>'
                r'\s*'
                r'(?P<threads>(?:<Thread>.*?</Thread>)+)'
                r'\s*'
                r'<Conclusion>(?P<conclusion>.*?)</Conclusion>'
                r'\s*$',
                re.DOTALL
            )
        else:
            if skip_conclusion_check:
                seq_pattern = re.compile(
                    r'^\s*'
                    r'<Outlines>(?P<outlines>.*?)</Outlines>'
                    r'\s*'
                    r'(?P<threads>(?:<Thread>.*?</Thread>)+)'
                    r'\s*$',
                    re.DOTALL
                )
            else:
                if verbose:
                    print("No Conclusion tag found, and not skipping conclusion check")
                return False

        m = seq_pattern.match(block)

        if not m:
            if skip_conclusion_check:
                if verbose:
                    print(f"Incorrect sequence of tags. Expected Outlines, Outline, Thread, and optionally Conclusion. Got: {block=}")
                return False
            else:
                if verbose:
                    print(f"Incorrect sequence of tags. Expected Outlines, Outline, Thread, Conclusion. Got: {block=}")
                return False

        outlines_content = m.group('outlines')
        threads_content = m.group('threads')
        if not skip_conclusion_check:
            conclusion_content = m.group('conclusion').strip()

        # 3) Validate <Outline> inside <Outlines>
        outlines = re.findall(r'<Outline>(.*?)</Outline>', outlines_content, re.DOTALL)
        if not outlines:
            if verbose:
                print("No Outline tags inside Outlines")
            return False

        outline_numbers = []
        for text in outlines:
            # no tags inside Outline
            if has_tags(text):
                if verbose:
                    print("Tags inside Outline")
                return False
            # must match "n: description"
            num_match = re.match(r'^\s*(\d+):\s*(.+)$', text.strip(), re.DOTALL)
            if not num_match:
                if verbose:
                    print(f"Invalid Outline format: {text.strip()}")
                return False
            outline_numbers.append(int(num_match.group(1)))

        # numbering must be 1,2,3,... with no gaps
        if outline_numbers != list(range(1, len(outlines) + 1)):
            if verbose:
                print(f"Invalid Outline numbering: {outline_numbers}")
            return False

        # 4) Validate <Thread> siblings
        thread_matches = list(re.finditer(r'<Thread>(.*?)</Thread>', threads_content, re.DOTALL))
        if len(thread_matches) != len(outlines):
            if verbose:
                print(f"Mismatched number of Thread tags. Expected {len(outlines)}, got {len(thread_matches)}")
            return False

        thread_numbers = []
        for tmatch in thread_matches:
            txt = tmatch.group(1)
            # no tags inside Thread
            if has_tags(txt):
                if verbose:
                    print(f"Tags inside Thread: {txt}\n", end='')
                return False
            num_match = re.match(r'^\s*(\d+):\s*(.+)$', txt.strip(), re.DOTALL)
            if not num_match:
                if verbose:
                    print(f"Invalid Thread format: {txt.strip()=}\n", end='')
                return False
            thread_numbers.append(int(num_match.group(1)))

        # must exactly match outline_numbers
        if thread_numbers != outline_numbers:
            if verbose:
                print("Mismatched Thread numbering", thread_numbers, outline_numbers)
            return False

        # must be adjacent with no whitespace between </Thread> and <Thread>
        if not allow_nonempty_whitespace:
            for first, second in zip(thread_matches, thread_matches[1:]):
                between = threads_content[first.end(): second.start()]
                if between != '':
                    if verbose:
                        print(f"Nonempty whitespace between Thread tags: {between}\n", end='')
                    return False

        # 5) Validate <Conclusion>
        if (not skip_conclusion_check) and (not conclusion_content or has_tags(conclusion_content)):
            if verbose:
                print(f"Invalid Conclusion content: {conclusion_content}\n", end='')
            return False

    # All blocks passed
    return True


def deepscaler_reward_fn(
    solution_str: str,
    ground_truth,
    config: dict,
    correctness_as_reward: bool,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    verbose: bool = True
):
    """
    Simplified version of deepscaler reward function that only checks format correctness.
    This is used for filtering trajectories based on parallel format.
    """
    # Normalize Think tags
    norm = solution_str.replace("<Think>", "<think>").replace("</Think>", "</think>")

    # Get config values with defaults
    parallel_format_error_v2_reward_enabled = config.get("parallel_format_error_v2_reward_enabled", True)
    allow_nonempty_whitespace = config.get("parallel_format_error_v2_allow_nonempty_whitespace", True)
    skip_conclusion_check = config.get("parallel_format_error_v2_skip_conclusion_check", False)
    treat_no_parallel_as_format_error = config.get("treat_no_parallel_as_format_error", False)

    # Check parallel format correctness
    parallel_format_correct_v2 = is_parallel_format_correct_v2(
        norm,
        treat_no_parallel_as_format_error=treat_no_parallel_as_format_error,
        allow_nonempty_whitespace=allow_nonempty_whitespace,
        skip_conclusion_check=skip_conclusion_check,
        verbose=verbose,
    )

    # Compute parallel metrics
    (
        parallel_ratio,
        num_tokens_in_the_longest_thread,
        total_num_tokens,
        avg_thread_length,
        avg_tokens_per_parallel_block,
        avg_threads_per_parallel_block,
        avg_outlines_block_length,
        avg_conclusion_block_length,
    ) = compute_parallel_metrics_and_thread_tokens(norm, tokenizer=tokenizer)

    # Calculate acceleration ratio
    acceleration_ratio = (
        1 - num_tokens_in_the_longest_thread / total_num_tokens
        if num_tokens_in_the_longest_thread is not None and total_num_tokens
        else None
    )

    # Build extra info
    parallel_count_match = norm.count(PARALLELIZE_DELIMITER_START) == norm.count(PARALLELIZE_DELIMITER_END)
    parallel_count = norm.count(PARALLELIZE_DELIMITER_START)
    with_parallel = parallel_count > 0

    extra_info = {
        "parallel_format_correct_v2": parallel_format_correct_v2,
        "parallel_ratio": parallel_ratio,
        "parallel_count_match": parallel_count_match,
        "parallel_count": parallel_count,
        "with_parallel": with_parallel,
        "num_tokens_in_the_longest_thread": num_tokens_in_the_longest_thread,
        "total_num_tokens": total_num_tokens,
        "acceleration_ratio": acceleration_ratio,
        "avg_thread_length": avg_thread_length,
        "avg_tokens_per_parallel_block": avg_tokens_per_parallel_block,
        "avg_threads_per_parallel_block": avg_threads_per_parallel_block,
        "avg_outlines_block_length": avg_outlines_block_length,
        "avg_conclusion_block_length": avg_conclusion_block_length,
    }

    return 0.0, extra_info


# =============================================================================
# Global variables for worker processes
# =============================================================================
_worker_tokenizer = None


def init_worker_tokenizer():
    """Initialize tokenizer in worker processes to avoid pickling issues."""
    global _worker_tokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained("Multiverse4FM/Multiverse-32B")


def process_chunk(chunk_args: tuple) -> List[Dict]:
    """
    Helper function to process a chunk of items.
    This function is intended to be run in a separate process.
    """
    indices_chunk, ds, text_column, strip_reasoning, require_think_end, skip_conclusion_check, out_dir, sidecar_root = chunk_args
    
    # Use global tokenizer initialized in the worker
    global _worker_tokenizer
    tokenizer = _worker_tokenizer
    
    def _safe_identifier(val) -> str:
        s = str(val)
        return re.sub(r"[^A-Za-z0-9_.-]", "_", s)
    
    results = []
    for idx in indices_chunk:
        row = ds[int(idx)]
        row_id = row.get("id") if "id" in ds.column_names else None

        # Prepare identifiers
        ident = _safe_identifier(row_id) if row_id is not None else str(idx)
        rel = f"row_{ident}.txt"
        dst_path = os.path.join(out_dir, rel) if out_dir else None
        dst_sidecar = os.path.join(sidecar_root, f"row_{ident}.json") if sidecar_root else None
        
        # Prepare question file path for saving question alongside row
        question_rel = f"row_{ident}_question.txt"
        question_dst_path = os.path.join(out_dir, question_rel) if out_dir else None

        # Get text
        txt = row.get(text_column, None)
        if txt is None:
            result = {
                "file": f"dataset/split#{idx}",
                "rel": rel,
                "format_correct": False,
                "copied": False,
                "error": f"missing_text_column:{text_column}",
                "row_index": idx,
                "row_id": row_id,
            }
            results.append(result)
            continue
            
        if not isinstance(txt, str):
            txt = str(txt)
        
        # Get question if available
        question = row.get("question", None)
        if question is not None and not isinstance(question, str):
            question = str(question)
        
        # Strip reasoning if requested
        if strip_reasoning:
            try:
                txt = extract_reasoning_content(txt)
            except Exception as e:
                result = {
                    "file": f"dataset/split#{idx}",
                    "rel": rel,
                    "format_correct": False,
                    "copied": False,
                    "error": f"reasoning_extraction_error: {e}",
                    "row_index": idx,
                    "row_id": row_id,
                }
                results.append(result)
                continue

        fmt_ok, extra = evaluate_trajectory(
            txt,
            tokenizer=tokenizer,
            require_think_end=require_think_end,
            skip_conclusion_check=skip_conclusion_check,
        )

        # Save sidecar JSON with extra info
        sidecar_payload = {
            "file": f"dataset/split#{idx}",
            "relative_path": rel,
            "format_correct": bool(fmt_ok),
            "extra_info": extra,
            "row_index": idx,
            "row_id": row_id,
        }
        if dst_sidecar:
            try:
                with open(dst_sidecar, "w", encoding="utf-8") as jf:
                    json.dump(sidecar_payload, jf, ensure_ascii=False, indent=2)
            except Exception as e:
                sidecar_payload["sidecar_write_error"] = str(e)

        copied = False
        question_copied = False
        if fmt_ok and dst_path:
            # Write the same content into the step-6 mirror (one .txt per passing row)
            try:
                with open(dst_path, "w", encoding="utf-8") as out_f:
                    out_f.write(txt)
                copied = True
            except Exception as e:
                sidecar_payload["copy_error"] = str(e)
            
            # Save question alongside if available
            if question is not None and question_dst_path:
                try:
                    with open(question_dst_path, "w", encoding="utf-8") as question_f:
                        question_f.write(question)
                    question_copied = True
                except Exception as e:
                    sidecar_payload["question_copy_error"] = str(e)

        sidecar_payload["copied"] = copied
        sidecar_payload["question_copied"] = question_copied
        results.append(sidecar_payload)
    
    return results


def extract_reasoning_content(text: str) -> str:
    """
    Extract content between <Think> and </Think> tags.
    If multiple Think blocks exist, concatenate them.
    If no Think blocks found, return the original text.
    """
    # Case-insensitive search for think tags
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    assert len(matches) == 1, f"Expected exactly one <Think> block, found {len(matches)} in text: {text[:100]}..."
    if matches:
        # Concatenate all thinking blocks
        return matches[0].strip()
    else:
        # No think blocks found, return original text
        return text


def evaluate_trajectory(
    text: str,
    tokenizer=None,
    require_think_end: bool = True,
    skip_conclusion_check: bool = False,
) -> Tuple[bool, Dict]:
    """
    Run deepscaler_reward_fn on a single trajectory and return:
      (format_correct, extra_info_dict)

    We only care about 'parallel_format_correct_v2' to decide format pass/fail.
    Ground truth isn't needed for format checks, so we pass a stub list.
    """
    # Use global tokenizer if none provided (for worker processes)
    if tokenizer is None:
        global _worker_tokenizer
        tokenizer = _worker_tokenizer
    
    # Normalize Think tags as in the provided script
    norm = text.replace("<Think>", "<think>").replace("</Think>", "</think>")

    # Match the config used in your evaluation script
    cfg = {
        "version": "v1",
        "parallel_format_error_v2_reward_enabled": True,
        "parallel_format_error_v2_allow_nonempty_whitespace": True,
        "parallel_format_error_v2_skip_conclusion_check": skip_conclusion_check,
        "treat_no_parallel_as_format_error": False,
        "parallel_ratio_reward": 0.1,
        "parallel_ratio_reward_factor": 1.0,
        "strip_comma_from_answer": True,
        "require_think_end": require_think_end,
    }

    # We only need format correctness signals; pass a stub GT list
    try:
        _score, extra = deepscaler_reward_fn(
            norm,
            [" "],  # stub ground truth; not used for format validity
            config=cfg,
            correctness_as_reward=False,
            tokenizer=tokenizer,
        )
    except Exception as e:
        # On any scoring error, mark as not format-correct, but return the reason
        return False, {"error": f"reward_fn_exception: {e.__class__.__name__}: {e}"}

    # Some versions return float(0/1); others bool. Treat any truthy value as pass.
    fmt_val = extra["parallel_format_correct_v2"]
    return fmt_val, extra


def process_file_chunk(chunk_args: tuple) -> List[Dict]:
    """
    Helper function to process a chunk of files.
    This function is intended to be run in a separate process.
    """
    files_chunk, in_dir, out_dir, sidecar_root, require_think_end, skip_conclusion_check = chunk_args
    
    # Use global tokenizer initialized in the worker
    global _worker_tokenizer
    tokenizer = _worker_tokenizer
    
    results = []
    for src_path in files_chunk:
        rel = os.path.relpath(src_path, in_dir)
        dst_path = os.path.join(out_dir, rel) if out_dir else None
        dst_sidecar = (
            os.path.join(sidecar_root, re.sub(r"[\\/]", "__", rel) + ".json")
            if sidecar_root
            else None
        )

        # Ensure destination subdirs exist
        if dst_path:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            with open(src_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            result = {
                "file": src_path,
                "rel": rel,
                "format_correct": False,
                "copied": False,
                "error": f"read_error: {e}",
            }
            results.append(result)
            continue

        fmt_ok, extra = evaluate_trajectory(
            text, tokenizer=tokenizer, require_think_end=require_think_end, skip_conclusion_check=skip_conclusion_check
        )

        # Save sidecar JSON with extra info (and a few convenience fields)
        sidecar_payload = {
            "file": src_path,
            "relative_path": rel,
            "format_correct": bool(fmt_ok),
            "extra_info": extra,
        }
        if dst_sidecar:
            try:
                with open(dst_sidecar, "w", encoding="utf-8") as jf:
                    json.dump(sidecar_payload, jf, ensure_ascii=False, indent=2)
            except Exception as e:
                # Keep going even if sidecar write fails
                sidecar_payload["sidecar_write_error"] = str(e)

        copied = False
        if fmt_ok and dst_path:
            # Copy (really: write) the same content into the step-6 mirror
            try:
                # Use write rather than shutil.copyfile to avoid re-reading
                with open(dst_path, "w", encoding="utf-8") as out_f:
                    out_f.write(text)
                copied = True
            except Exception as e:
                sidecar_payload["copy_error"] = str(e)
        elif fmt_ok:
            # Format correct but output disabled; no copy performed
            pass
        else:
            print(f"Skipping copy for non-format-correct file: {src_path}")

        sidecar_payload["copied"] = copied
        results.append(sidecar_payload)
    
    return results


def main():
    p = argparse.ArgumentParser(
        description="Filter format-correct trajectories from input files or a Hugging Face dataset using POLARIS deepscaler_reward_fn."
    )
    p.add_argument(
        "--input",
        default=None,
        help="Step-5 directory containing trajectories (e.g., ${DATA_PATH}_step5_${VERSION}).",
    )
    p.add_argument(
        "--glob",
        default="**/*.txt",
        help='Glob for input files, relative to --input (default: "**/*.txt").',
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="HF dataset repo_id or local dataset path for datasets.load_dataset (e.g., 'my-org/my-ds').",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    p.add_argument(
        "--text_column",
        default="deepseek_thinking_trajectory_parallel",
        help="Column name containing the trajectories "
             "(default: deepseek_thinking_trajectory_parallel).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Step-6 directory to write format-correct trajectories and summaries (e.g., ${DATA_PATH}_step6_${VERSION}).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing --output directory.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: min(32, os.cpu_count()+4)). Set to 0 to use sequential processing.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for multiprocessing batches.",
    )
    p.add_argument(
        "--executor-type",
        type=str,
        choices=["thread", "process"],
        default="process",
        help="Type of executor: 'thread' for ThreadPoolExecutor or 'process' for ProcessPoolExecutor.",
    )
    p.add_argument(
        "--no_require_think_end",
        action="store_true",
        help="If set, does NOT require </Think> for correctness.",
    )
    p.add_argument(
        "--skip_conclusion_check",
        action="store_true",
        help="If set, skips the conclusion check.",
    )
    p.add_argument(
        "--summary_name",
        default="format_check_summary.json",
        help="Filename for the aggregate JSON summary written in --output (default: format_check_summary.json).",
    )
    p.add_argument(
        "--sidecar_dirname",
        default="_reward_extra",
        help="Subdir in --output to store per-row extra_info JSONs (default: _reward_extra).",
    )
    p.add_argument(
        "--strip_reasoning",
        action="store_true",
        help="If set, extract and process only the content between <Think> and </Think> tags.",
    )
    p.add_argument(
        "--keep_first",
        type=int,
        default=None,
        help="Extract only the first NUM entries from the dataset.",
    )
    p.add_argument(
        "--keep_last",
        type=int,
        default=None,
        help="Extract only the last NUM entries from the dataset.",
    )
    args = p.parse_args()

    # Validate that exactly one source is specified
    if args.input is None and args.dataset is None:
        raise ValueError("Must specify exactly one source: either --input or --dataset")
    
    if args.input is not None and args.dataset is not None:
        raise ValueError("Cannot specify both --input and --dataset. Choose exactly one source.")

    out_dir = os.path.abspath(args.output) if args.output else None
    if out_dir:
        if os.path.exists(out_dir) and not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Use --overwrite to replace it.")
        os.makedirs(out_dir, exist_ok=True)
        sidecar_root = os.path.join(out_dir, args.sidecar_dirname)
        os.makedirs(sidecar_root, exist_ok=True)
    else:
        sidecar_root = None

    write_outputs = out_dir is not None

    # Workers
    if args.workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    else:
        max_workers = max(1, args.workers)

    chunk_size = getattr(args, 'chunk_size', 1000)
    executor_type = getattr(args, 'executor_type', 'process')

    # Load tokenizer once for sequential processing or main thread
    tokenizer = AutoTokenizer.from_pretrained("Multiverse4FM/Multiverse-32B")

    require_think_end = not args.no_require_think_end

    results = []
    fails = 0
    passes = 0

    if args.input is not None:
        # Process input directory with glob pattern
        in_dir = os.path.abspath(args.input)
        pattern = os.path.join(in_dir, args.glob)

        # Resolve files
        files = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
        if not files:
            raise FileNotFoundError(f"No files matched: {pattern}")

        print(f"Processing {len(files)} files from {in_dir} with pattern {args.glob}")

        def _process_one_file(src_path: str) -> Dict:
            rel = os.path.relpath(src_path, in_dir)
            dst_path = os.path.join(out_dir, rel) if out_dir else None
            dst_sidecar = (
                os.path.join(sidecar_root, re.sub(r"[\\/]", "__", rel) + ".json")
                if sidecar_root
                else None
            )

            # Ensure destination subdirs exist
            if dst_path:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            try:
                with open(src_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                return {
                    "file": src_path,
                    "rel": rel,
                    "format_correct": False,
                    "copied": False,
                    "error": f"read_error: {e}",
                }

            fmt_ok, extra = evaluate_trajectory(
                text, tokenizer=tokenizer, require_think_end=require_think_end, skip_conclusion_check=args.skip_conclusion_check
            )

            # Save sidecar JSON with extra info (and a few convenience fields)
            sidecar_payload = {
                "file": src_path,
                "relative_path": rel,
                "format_correct": bool(fmt_ok),
                "extra_info": extra,
            }
            if dst_sidecar:
                try:
                    with open(dst_sidecar, "w", encoding="utf-8") as jf:
                        json.dump(sidecar_payload, jf, ensure_ascii=False, indent=2)
                except Exception as e:
                    # Keep going even if sidecar write fails
                    sidecar_payload["sidecar_write_error"] = str(e)

            copied = False
            if fmt_ok and dst_path:
                # Copy (really: write) the same content into the step-6 mirror
                try:
                    # Use write rather than shutil.copyfile to avoid re-reading
                    with open(dst_path, "w", encoding="utf-8") as out_f:
                        out_f.write(text)
                    copied = True
                except Exception as e:
                    sidecar_payload["copy_error"] = str(e)
            elif fmt_ok:
                # Format correct but output disabled; nothing to copy
                pass

            sidecar_payload["copied"] = copied
            return sidecar_payload

        # Parallel processing logic for files
        if max_workers == 0:
            print("Starting sequential file processing...")
            # Initialize tokenizer for sequential processing
            init_worker_tokenizer()
            
            for file_path in tqdm(files, desc="Processing files sequentially"):
                try:
                    result = _process_one_file(file_path)
                    results.append(result)
                    if result.get("format_correct"):
                        passes += 1
                    else:
                        fails += 1
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    fails += 1
        else:
            executor_class = concurrent.futures.ProcessPoolExecutor if executor_type == "process" else concurrent.futures.ThreadPoolExecutor
            executor_name = "ProcessPoolExecutor" if executor_type == "process" else "ThreadPoolExecutor"
            
            print(f"Starting parallel file processing with {max_workers} workers, chunk size {chunk_size}, using {executor_name}...")
            
            # Process in chunks to avoid memory issues
            with executor_class(max_workers=max_workers, initializer=init_worker_tokenizer) as executor:
                
                # Create chunks of files
                file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
                
                # Prepare chunk arguments
                chunk_args_list = [
                    (chunk, in_dir, out_dir, sidecar_root, require_think_end, args.skip_conclusion_check)
                    for chunk in file_chunks
                ]
                
                future_to_chunk_start_index = {
                    executor.submit(process_file_chunk, chunk_args): i * chunk_size
                    for i, chunk_args in enumerate(chunk_args_list)
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_chunk_start_index), 
                                 total=len(future_to_chunk_start_index), desc="Processing file chunks"):
                    try:
                        chunk_results = future.result()
                        for result in chunk_results:
                            results.append(result)
                            if result.get("format_correct"):
                                passes += 1
                            else:
                                fails += 1
                    except Exception as e:
                        start_index = future_to_chunk_start_index[future]
                        print(f"Error processing file chunk starting at index {start_index}: {e}")
                        # Mark all results in this chunk as failed
                        chunk_size_actual = min(chunk_size, len(files) - start_index)
                        fails += chunk_size_actual

    else:
        # Process dataset (original dataset processing logic)
        print(f"Processing dataset: {args.dataset}")
        
        # Load dataset
        ds = load_dataset(args.dataset, split=args.split)
        if args.text_column not in ds.column_names:
            raise KeyError(
                f"Column '{args.text_column}' not found in dataset. "
                f"Available columns: {ds.column_names}"
            )

        # Create indices for processing
        indices = list(range(len(ds)))
        
        # Apply subsetting based on --keep_first and --keep_last arguments
        if args.keep_first is not None and args.keep_last is not None:
            raise ValueError("Cannot specify both --keep_first and --keep_last at the same time")
        elif args.keep_first is not None:
            if args.keep_first <= 0:
                raise ValueError("--keep_first must be a positive integer")
            indices = indices[:args.keep_first]
            print(f"Processing first {len(indices)} entries out of {len(ds)} total")
        elif args.keep_last is not None:
            if args.keep_last <= 0:
                raise ValueError("--keep_last must be a positive integer")
            indices = indices[-args.keep_last:]
            print(f"Processing last {len(indices)} entries out of {len(ds)} total")

        def _safe_identifier(val) -> str:
            s = str(val)
            return re.sub(r"[^A-Za-z0-9_.-]", "_", s)

        def _process_one(idx: int) -> Dict:
            row = ds[int(idx)]
            row_id = row.get("id") if "id" in ds.column_names else None

            # Prepare identifiers
            ident = _safe_identifier(row_id) if row_id is not None else str(idx)
            rel = f"row_{ident}.txt"
            dst_path = os.path.join(out_dir, rel) if out_dir else None
            dst_sidecar = os.path.join(sidecar_root, f"row_{ident}.json") if sidecar_root else None
            
            # Prepare question file path for saving question alongside row
            question_rel = f"question_{ident}.txt"
            question_dst_path = os.path.join(out_dir, question_rel) if out_dir else None

            # Get text
            txt = row.get(args.text_column, None)
            if txt is None:
                return {
                    "file": f"{args.dataset}/{args.split}#{idx}",
                    "rel": rel,
                    "format_correct": False,
                    "copied": False,
                    "error": f"missing_text_column:{args.text_column}",
                    "row_index": idx,
                    "row_id": row_id,
                }
            if not isinstance(txt, str):
                txt = str(txt)
            
            # Get question if available
            question = row.get("question", None)
            if question is not None and not isinstance(question, str):
                question = str(question)
            
            # Strip reasoning if requested
            if args.strip_reasoning:
                try:
                    txt = extract_reasoning_content(txt)
                except Exception as e:
                    return {
                        "file": f"{args.dataset}/{args.split}#{idx}",
                        "rel": rel,
                        "format_correct": False,
                        "copied": False,
                        "error": f"reasoning_extraction_error: {e}",
                        "row_index": idx,
                        "row_id": row_id,
                    }

            fmt_ok, extra = evaluate_trajectory(
                txt,
                tokenizer=tokenizer,
                require_think_end=require_think_end,
                skip_conclusion_check=args.skip_conclusion_check,
            )

            # Save sidecar JSON with extra info
            sidecar_payload = {
                "file": f"{args.dataset}/{args.split}#{idx}",
                "relative_path": rel,
                "format_correct": bool(fmt_ok),
                "extra_info": extra,
                "row_index": idx,
                "row_id": row_id,
            }
            if dst_sidecar:
                try:
                    with open(dst_sidecar, "w", encoding="utf-8") as jf:
                        json.dump(sidecar_payload, jf, ensure_ascii=False, indent=2)
                except Exception as e:
                    sidecar_payload["sidecar_write_error"] = str(e)

            copied = False
            question_copied = False
            if fmt_ok and dst_path:
                # Write the same content into the step-6 mirror (one .txt per passing row)
                try:
                    with open(dst_path, "w", encoding="utf-8") as out_f:
                        out_f.write(txt)
                    copied = True
                except Exception as e:
                    sidecar_payload["copy_error"] = str(e)
                
                # Save question alongside if available
                if question is not None and question_dst_path:
                    try:
                        with open(question_dst_path, "w", encoding="utf-8") as question_f:
                            question_f.write(question)
                        question_copied = True
                    except Exception as e:
                        sidecar_payload["question_copy_error"] = str(e)

            sidecar_payload["copied"] = copied
            sidecar_payload["question_copied"] = question_copied
            return sidecar_payload

        # Parallel processing logic for dataset
        if max_workers == 0:
            print("Starting sequential data processing...")
            # Initialize tokenizer for sequential processing
            init_worker_tokenizer()
            
            for idx in tqdm(indices, desc="Processing items sequentially"):
                try:
                    result = _process_one(idx)
                    results.append(result)
                    if result.get("format_correct"):
                        passes += 1
                    else:
                        fails += 1
                except Exception as e:
                    print(f"Error processing item {idx}: {e}")
                    fails += 1
        else:
            executor_class = concurrent.futures.ProcessPoolExecutor if executor_type == "process" else concurrent.futures.ThreadPoolExecutor
            executor_name = "ProcessPoolExecutor" if executor_type == "process" else "ThreadPoolExecutor"
            
            print(f"Starting parallel data processing with {max_workers} workers, chunk size {chunk_size}, using {executor_name}...")
            
            # Process in chunks to avoid memory issues
            with executor_class(max_workers=max_workers, initializer=init_worker_tokenizer) as executor:
                
                # Create chunks of indices
                index_chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
                
                # Prepare chunk arguments
                chunk_args_list = [
                    (chunk, ds, args.text_column, args.strip_reasoning, require_think_end, 
                     args.skip_conclusion_check, out_dir, sidecar_root)
                    for chunk in index_chunks
                ]
                
                future_to_chunk_start_index = {
                    executor.submit(process_chunk, chunk_args): i * chunk_size
                    for i, chunk_args in enumerate(chunk_args_list)
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_chunk_start_index), 
                                 total=len(future_to_chunk_start_index), desc="Processing chunks"):
                    try:
                        chunk_results = future.result()
                        for result in chunk_results:
                            results.append(result)
                            if result.get("format_correct"):
                                passes += 1
                            else:
                                fails += 1
                    except Exception as e:
                        start_index = future_to_chunk_start_index[future]
                        print(f"Error processing chunk starting at index {start_index}: {e}")
                        # Mark all results in this chunk as failed
                        chunk_size_actual = min(chunk_size, len(indices) - start_index)
                        fails += chunk_size_actual

    # Calculate means of parallel_ratio, acceleration_ratio, num_tokens_in_the_longest_thread, total_num_tokens, avg_thread_length, parallel_count, avg_tokens_per_parallel_block, and avg_threads_per_parallel_block
    parallel_ratios = []
    acceleration_ratios = []
    longest_thread_tokens = []
    total_num_tokens = []
    avg_thread_lengths = []
    parallel_counts = []
    avg_tokens_per_parallel_blocks = []
    avg_threads_per_parallel_blocks = []
    # Newly added metrics
    avg_outlines_block_lengths = []
    avg_conclusion_block_lengths = []

    for res in results:
        if res.get("format_correct") and "extra_info" in res:
            extra = res["extra_info"]
            if extra["parallel_ratio"] is not None:
                parallel_ratios.append(extra["parallel_ratio"])
            if extra["acceleration_ratio"] is not None:
                acceleration_ratios.append(extra["acceleration_ratio"])
            if extra["num_tokens_in_the_longest_thread"] is not None:
                longest_thread_tokens.append(extra["num_tokens_in_the_longest_thread"])
            if extra["total_num_tokens"] is not None:
                total_num_tokens.append(extra["total_num_tokens"])
            if extra["avg_thread_length"] is not None:
                avg_thread_lengths.append(extra["avg_thread_length"])
            if extra.get("parallel_count") is not None:
                parallel_counts.append(extra["parallel_count"])
            if extra.get("avg_tokens_per_parallel_block") is not None:
                avg_tokens_per_parallel_blocks.append(extra["avg_tokens_per_parallel_block"])
            if extra.get("avg_threads_per_parallel_block") is not None:
                avg_threads_per_parallel_blocks.append(extra["avg_threads_per_parallel_block"])
            if extra.get("avg_outlines_block_length") is not None:
                avg_outlines_block_lengths.append(extra["avg_outlines_block_length"])
            if extra.get("avg_conclusion_block_length") is not None:
                avg_conclusion_block_lengths.append(extra["avg_conclusion_block_length"])

    mean_parallel_ratio = sum(parallel_ratios) / len(parallel_ratios) if parallel_ratios else 0
    mean_acceleration_ratio = sum(acceleration_ratios) / len(acceleration_ratios) if acceleration_ratios else 0
    mean_longest_thread_tokens = sum(longest_thread_tokens) / len(longest_thread_tokens) if longest_thread_tokens else 0
    mean_total_num_tokens = sum(total_num_tokens) / len(total_num_tokens) if total_num_tokens else 0
    mean_avg_thread_length = sum(avg_thread_lengths) / len(avg_thread_lengths) if avg_thread_lengths else 0
    mean_parallel_count = sum(parallel_counts) / len(parallel_counts) if parallel_counts else 0
    mean_avg_tokens_per_parallel_block = sum(avg_tokens_per_parallel_blocks) / len(avg_tokens_per_parallel_blocks) if avg_tokens_per_parallel_blocks else 0
    mean_avg_threads_per_parallel_block = sum(avg_threads_per_parallel_blocks) / len(avg_threads_per_parallel_blocks) if avg_threads_per_parallel_blocks else 0
    mean_avg_outlines_block_length = sum(avg_outlines_block_lengths) / len(avg_outlines_block_lengths) if avg_outlines_block_lengths else 0
    mean_avg_conclusion_block_length = sum(avg_conclusion_block_lengths) / len(avg_conclusion_block_lengths) if avg_conclusion_block_lengths else 0

    parallel_hist_path = None
    acceleration_hist_path = None
    longest_thread_hist_path = None
    total_tokens_hist_path = None
    avg_thread_length_hist_path = None
    parallel_count_hist_path = None
    avg_tokens_per_parallel_block_hist_path = None
    avg_threads_per_parallel_block_hist_path = None
    avg_outlines_block_length_hist_path = None
    avg_conclusion_block_length_hist_path = None

    if write_outputs:
        # Generate histograms and save to sidecar directory
        if parallel_ratios:
            plt.figure(figsize=(10, 6))
            plt.hist(parallel_ratios, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Parallel Ratios')
            plt.xlabel('Parallel Ratio')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            parallel_hist_path = os.path.join(sidecar_root, "parallel_ratio_histogram.png")
            plt.savefig(parallel_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

        if acceleration_ratios:
            plt.figure(figsize=(10, 6))
            plt.hist(acceleration_ratios, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Acceleration Ratios')
            plt.xlabel('Acceleration Ratio')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            acceleration_hist_path = os.path.join(sidecar_root, "acceleration_ratio_histogram.png")
            plt.savefig(acceleration_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

        if longest_thread_tokens:
            plt.figure(figsize=(10, 6))
            plt.hist(longest_thread_tokens, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Number of Tokens in the Longest Thread')
            plt.xlabel('Number of Tokens')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            longest_thread_hist_path = os.path.join(sidecar_root, "longest_thread_tokens_histogram.png")
            plt.savefig(longest_thread_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

        if total_num_tokens:
            plt.figure(figsize=(10, 6))
            plt.hist(total_num_tokens, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Total Number of Tokens')
            plt.xlabel('Total Number of Tokens')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            total_tokens_hist_path = os.path.join(sidecar_root, "total_num_tokens_histogram.png")
            plt.savefig(total_tokens_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

        if avg_thread_lengths:
            plt.figure(figsize=(10, 6))
            plt.hist(avg_thread_lengths, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Average Thread Length')
            plt.xlabel('Average Thread Length')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            avg_thread_length_hist_path = os.path.join(sidecar_root, "avg_thread_length_histogram.png")
            plt.savefig(avg_thread_length_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

        if parallel_counts:
            plt.figure(figsize=(10, 6))
            plt.hist(parallel_counts, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Parallel Count')
            plt.xlabel('Parallel Count')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            parallel_count_hist_path = os.path.join(sidecar_root, "parallel_count_histogram.png")
            plt.savefig(parallel_count_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

        if avg_tokens_per_parallel_blocks:
            plt.figure(figsize=(10, 6))
            plt.hist(avg_tokens_per_parallel_blocks, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Average Tokens per Parallel Block')
            plt.xlabel('Average Tokens per Parallel Block')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            avg_tokens_per_parallel_block_hist_path = os.path.join(sidecar_root, "avg_tokens_per_parallel_block_histogram.png")
            plt.savefig(avg_tokens_per_parallel_block_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

        if avg_threads_per_parallel_blocks:
            plt.figure(figsize=(10, 6))
            plt.hist(avg_threads_per_parallel_blocks, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Average Threads per Parallel Block')
            plt.xlabel('Average Threads per Parallel Block')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            avg_threads_per_parallel_block_hist_path = os.path.join(sidecar_root, "avg_threads_per_parallel_block_histogram.png")
            plt.savefig(avg_threads_per_parallel_block_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

        if avg_outlines_block_lengths:
            plt.figure(figsize=(10, 6))
            plt.hist(avg_outlines_block_lengths, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Average Outlines Block Length')
            plt.xlabel('Average Outlines Block Length')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            avg_outlines_block_length_hist_path = os.path.join(sidecar_root, "avg_outlines_block_length_histogram.png")
            plt.savefig(avg_outlines_block_length_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

        if avg_conclusion_block_lengths:
            plt.figure(figsize=(10, 6))
            plt.hist(avg_conclusion_block_lengths, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Average Conclusion Block Length')
            plt.xlabel('Average Conclusion Block Length')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            avg_conclusion_block_length_hist_path = os.path.join(sidecar_root, "avg_conclusion_block_length_histogram.png")
            plt.savefig(avg_conclusion_block_length_hist_path, dpi=300, bbox_inches='tight')
            plt.close()

    # Find top 10 highest/lowest trajectories for each metric (keeping the original logic)
    def get_top_trajectories(metric_name, metric_values, results_with_metric, top_k=5):
        if not metric_values:
            return [], []
        sorted_results = sorted(results_with_metric, key=lambda x: x[1])
        lowest_k = sorted_results[:min(top_k, len(sorted_results))]
        highest_k = sorted_results[-min(top_k, len(sorted_results)):][::-1]
        return lowest_k, highest_k

    results_with_parallel_ratio = []
    results_with_acceleration_ratio = []
    results_with_longest_thread = []
    results_with_total_tokens = []
    results_with_avg_thread_length = []
    results_with_parallel_count = []
    results_with_avg_tokens_per_parallel_block = []
    results_with_avg_threads_per_parallel_block = []
    results_with_avg_outlines_block_length = []
    results_with_avg_conclusion_block_length = []

    for res in results:
        if res.get("format_correct") and "extra_info" in res:
            extra = res["extra_info"]
            if "parallel_ratio" in extra and extra["parallel_ratio"] is not None:
                results_with_parallel_ratio.append((res, extra["parallel_ratio"]))
            if "acceleration_ratio" in extra and extra["acceleration_ratio"] is not None:
                results_with_acceleration_ratio.append((res, extra["acceleration_ratio"]))
            if "num_tokens_in_the_longest_thread" in extra and extra["num_tokens_in_the_longest_thread"] is not None:
                results_with_longest_thread.append((res, extra["num_tokens_in_the_longest_thread"]))
            if "total_num_tokens" in extra and extra["total_num_tokens"] is not None:
                results_with_total_tokens.append((res, extra["total_num_tokens"]))
            if "avg_thread_length" in extra and extra["avg_thread_length"] is not None:
                results_with_avg_thread_length.append((res, extra["avg_thread_length"]))
            if "parallel_count" in extra and extra["parallel_count"] is not None:
                results_with_parallel_count.append((res, extra["parallel_count"]))
            if "avg_tokens_per_parallel_block" in extra and extra["avg_tokens_per_parallel_block"] is not None:
                results_with_avg_tokens_per_parallel_block.append((res, extra["avg_tokens_per_parallel_block"]))
            if "avg_threads_per_parallel_block" in extra and extra["avg_threads_per_parallel_block"] is not None:
                results_with_avg_threads_per_parallel_block.append((res, extra["avg_threads_per_parallel_block"]))
            if "avg_outlines_block_length" in extra and extra["avg_outlines_block_length"] is not None:
                results_with_avg_outlines_block_length.append((res, extra["avg_outlines_block_length"]))
            if "avg_conclusion_block_length" in extra and extra["avg_conclusion_block_length"] is not None:
                results_with_avg_conclusion_block_length.append((res, extra["avg_conclusion_block_length"]))

    lowest_parallel, highest_parallel = get_top_trajectories(
        "parallel_ratio", parallel_ratios, results_with_parallel_ratio
    )
    lowest_acceleration, highest_acceleration = get_top_trajectories(
        "acceleration_ratio", acceleration_ratios, results_with_acceleration_ratio
    )
    lowest_longest_thread, highest_longest_thread = get_top_trajectories(
        "longest_thread", longest_thread_tokens, results_with_longest_thread
    )
    lowest_total_tokens, highest_total_tokens = get_top_trajectories(
        "total_tokens", total_num_tokens, results_with_total_tokens
    )
    lowest_avg_thread_length, highest_avg_thread_length = get_top_trajectories("avg_thread_length", avg_thread_lengths, results_with_avg_thread_length)
    lowest_parallel_count, highest_parallel_count = get_top_trajectories("parallel_count", parallel_counts, results_with_parallel_count)
    lowest_avg_tokens_per_parallel_block, highest_avg_tokens_per_parallel_block = get_top_trajectories("avg_tokens_per_parallel_block", avg_tokens_per_parallel_blocks, results_with_avg_tokens_per_parallel_block)
    lowest_avg_threads_per_parallel_block, highest_avg_threads_per_parallel_block = get_top_trajectories("avg_threads_per_parallel_block", avg_threads_per_parallel_blocks, results_with_avg_threads_per_parallel_block)
    lowest_avg_outlines_block_length, highest_avg_outlines_block_length = get_top_trajectories("avg_outlines_block_length", avg_outlines_block_lengths, results_with_avg_outlines_block_length)
    lowest_avg_conclusion_block_length, highest_avg_conclusion_block_length = get_top_trajectories("avg_conclusion_block_length", avg_conclusion_block_lengths, results_with_avg_conclusion_block_length)

    top_trajectories = {
        "parallel_ratio": {
            "highest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in highest_parallel],
            "lowest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in lowest_parallel]
        },
        "acceleration_ratio": {
            "highest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in highest_acceleration],
            "lowest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in lowest_acceleration]
        },
        "longest_thread_tokens": {
            "highest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in highest_longest_thread],
            "lowest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in lowest_longest_thread]
        },
        "total_num_tokens": {
            "highest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in highest_total_tokens],
            "lowest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in lowest_total_tokens]
        },
        "avg_thread_length": {
            "highest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in highest_avg_thread_length],
            "lowest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in lowest_avg_thread_length]
        },
        "parallel_count": {
            "highest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in highest_parallel_count],
            "lowest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in lowest_parallel_count]
        },
        "avg_tokens_per_parallel_block": {
            "highest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in highest_avg_tokens_per_parallel_block],
            "lowest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in lowest_avg_tokens_per_parallel_block]
        },
        "avg_threads_per_parallel_block": {
            "highest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in highest_avg_threads_per_parallel_block],
            "lowest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in lowest_avg_threads_per_parallel_block]
        },
        "avg_outlines_block_length": {
            "highest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in highest_avg_outlines_block_length],
            "lowest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in lowest_avg_outlines_block_length]
        },
        "avg_conclusion_block_length": {
            "highest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in highest_avg_conclusion_block_length],
            "lowest": [{"file": res[0]["file"], "value": res[1], "extra_info": res[0]["extra_info"]} for res in lowest_avg_conclusion_block_length]
        }
    }

    top_trajectories_path = None
    if write_outputs:
        top_trajectories_path = os.path.join(sidecar_root, "top_trajectories.json")
        with open(top_trajectories_path, "w", encoding="utf-8") as f:
            json.dump(top_trajectories, f, ensure_ascii=False, indent=2)

    # Calculate statistics for all extra_info fields
    stats_summary = {}
    all_extra_fields = {}
    for res in results:
        if res.get("format_correct") and "extra_info" in res:
            extra = res["extra_info"]
            for key, value in extra.items():
                if not isinstance(value, (list, str)):
                    if value is not None:
                        all_extra_fields.setdefault(key, []).append(value)

    for field_name, values in all_extra_fields.items():
        if values:
            try:
                stats_summary[field_name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0
                }
            except (TypeError, statistics.StatisticsError):
                continue

    stats_path = None
    if write_outputs:
        stats_path = os.path.join(sidecar_root, "summary.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_summary, f, ensure_ascii=False, indent=2)

    # Build summary with appropriate fields for input source
    if args.input is not None:
        # File-based input
        summary = {
            "input_dir": in_dir,
            "output_dir": out_dir,
            "glob": args.glob,
            "workers": max_workers,
            "require_think_end": require_think_end,
            "total_files": len(files),
            "format_pass": passes,
            "format_fail": fails,
            "mean_parallel_ratio": mean_parallel_ratio,
            "mean_acceleration_ratio": mean_acceleration_ratio,
            "mean_longest_thread_tokens": mean_longest_thread_tokens,
            "mean_total_num_tokens": mean_total_num_tokens,
            "mean_avg_thread_length": mean_avg_thread_length,
            "mean_parallel_count": mean_parallel_count,
            "mean_avg_tokens_per_parallel_block": mean_avg_tokens_per_parallel_block,
            "mean_avg_threads_per_parallel_block": mean_avg_threads_per_parallel_block,
            "mean_avg_outlines_block_length": mean_avg_outlines_block_length,
            "mean_avg_conclusion_block_length": mean_avg_conclusion_block_length,
            "files": results,  # one record per input file
        }
        total_processed = len(files)
    else:
        # Dataset-based input
        summary = {
            "input_dataset": args.dataset,
            "split": args.split,
            "text_column": args.text_column,
            "output_dir": out_dir,
            "workers": max_workers,
            "require_think_end": require_think_end,
            "strip_reasoning": args.strip_reasoning,
            "total_rows": len(indices),
            "format_pass": passes,
            "format_fail": fails,
            "mean_parallel_ratio": mean_parallel_ratio,
            "mean_acceleration_ratio": mean_acceleration_ratio,
            "mean_longest_thread_tokens": mean_longest_thread_tokens,
            "mean_total_num_tokens": mean_total_num_tokens,
            "mean_avg_thread_length": mean_avg_thread_length,
            "mean_parallel_count": mean_parallel_count,
            "mean_avg_tokens_per_parallel_block": mean_avg_tokens_per_parallel_block,
            "mean_avg_threads_per_parallel_block": mean_avg_threads_per_parallel_block,
            "mean_avg_outlines_block_length": mean_avg_outlines_block_length,
            "mean_avg_conclusion_block_length": mean_avg_conclusion_block_length,
            "files": results,  # one record per dataset row
        }
        total_processed = len(indices)

    summary_path = None
    if write_outputs:
        summary_path = os.path.join(out_dir, args.summary_name)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"Done. Format-correct: {passes}/{total_processed}\n"
        f"Mean parallel_ratio: {mean_parallel_ratio:.4f}\n"
        f"Mean acceleration_ratio: {mean_acceleration_ratio:.4f}\n"
        f"Mean longest_thread_tokens: {mean_longest_thread_tokens:.2f}\n"
        f"Mean total_num_tokens: {mean_total_num_tokens:.2f}\n"
        f"Mean avg_thread_length: {mean_avg_thread_length:.2f}\n"
        f"Mean parallel_count: {mean_parallel_count:.2f}\n"
        f"Mean avg_tokens_per_parallel_block: {mean_avg_tokens_per_parallel_block:.2f}\n"
        f"Mean avg_threads_per_parallel_block: {mean_avg_threads_per_parallel_block:.2f}\n"
        f"Mean avg_outlines_block_length: {mean_avg_outlines_block_length:.2f}\n"
        f"Mean avg_conclusion_block_length: {mean_avg_conclusion_block_length:.2f}\n"
        f"Summary: {summary_path or 'N/A'}\n"
        f"Sidecars: {sidecar_root or 'N/A'}\n"
        f"Statistics summary: {stats_path or 'N/A'}\n"
        f"Top trajectories analysis: {top_trajectories_path or 'N/A'}\n"
        f"Parallel ratio histogram: {parallel_hist_path or 'N/A'}\n"
        f"Acceleration ratio histogram: {acceleration_hist_path or 'N/A'}\n"
        f"Longest thread tokens histogram: {longest_thread_hist_path or 'N/A'}\n"
        f"Total num tokens histogram: {total_tokens_hist_path or 'N/A'}\n"
        f"Avg thread length histogram: {avg_thread_length_hist_path or 'N/A'}\n"
        f"Parallel count histogram: {parallel_count_hist_path or 'N/A'}\n"
        f"Avg tokens per parallel block histogram: {avg_tokens_per_parallel_block_hist_path or 'N/A'}\n"
        f"Avg threads per parallel block histogram: {avg_threads_per_parallel_block_hist_path or 'N/A'}\n"
        f"Avg outlines block length histogram: {avg_outlines_block_length_hist_path or 'N/A'}\n"
        f"Avg conclusion block length histogram: {avg_conclusion_block_length_hist_path or 'N/A'}\n"
        f"Output writing enabled: {'yes' if write_outputs else 'no'}\n"
    )

if __name__ == "__main__":
    main()
