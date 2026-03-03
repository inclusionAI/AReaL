#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Rewrite a reasoning trajectory by replacing ranges indicated by <parallel> blocks
in an annotations file, then SAVE THE RESULT to a separate output directory
(does not modify the original file).

New filtering behavior:
    --filter-diff-threshold (default 100) skips emitting a <Parallel> replacement
    if the combined character count of all extracted thread snippets minus the
    largest single snippet's character count is below the threshold. This avoids
    producing low-information parallel sections where one thread dominates.

Trace logging:
    All warnings / informational messages printed during processing are also
    captured and written to a parallel trace file at:
        <output_dir>/trace/<trajectory_filename>
    Dry runs also generate this trace file for inspection.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

# -------- Logging accumulator

LOG_MESSAGES: List[str] = []  # Collected warning / debug / error messages

def _log(msg: str, level: str = "INFO", stderr: bool = False) -> None:
    """Record a message and print it immediately.

    Parameters
    ----------
    msg : str
        Message body (without trailing newline).
    level : str
        One of INFO, WARNING, ERROR (free-form accepted); prepended for trace file.
    stderr : bool
        If True, print to stderr, else stdout.
    """
    line = f"[{level}] {msg}" if not msg.startswith("[") else msg
    LOG_MESSAGES.append(line)
    if stderr:
        print(line, file=sys.stderr)
    else:
        print(line)


# -------- Regexes

RANGE_RE = re.compile(r"range:\s*L(\d+)\s*-\s*L(\d+)", re.IGNORECASE)
OUTER_PARALLEL_TAG_RE = re.compile(r"</?parallel>", re.IGNORECASE)


# -------- Data structures

@dataclass
class ThreadSeg:
    """Represents a single <thread>...</thread> segment within a block."""
    text: str
    range_start: Optional[int]
    range_end: Optional[int]

    def cleaned_outline(self) -> str:
        """The text content of the thread serves as the outline."""
        return self.text.strip()


@dataclass
class ParallelBlock:
    """Represents a top-level <parallel> ... </parallel> block."""
    reason: str
    threads: List[ThreadSeg]
    block_range: Optional[Tuple[int, int]]


# -------- Helpers

def extract_first_range(text: str) -> Optional[Tuple[int, int]]:
    """Extract the first 'range: Lx-Ly' from text; returns (min, max) or None."""
    m = RANGE_RE.search(text)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if a <= b else (b, a)


def iter_top_level_parallel_spans(s: str) -> List[Tuple[int, int]]:
    """
    Return list of (start_index, end_index) byte offsets for TOP-LEVEL
    <parallel> ... </parallel> blocks in `s`, using depth-based pairing.
    """
    spans: List[Tuple[int, int]] = []
    depth = 0
    open_start: Optional[int] = None
    for m in OUTER_PARALLEL_TAG_RE.finditer(s):
        tag = m.group(0).lower()
        if tag == "<parallel>":
            if depth == 0:
                open_start = m.start()
            depth += 1
        else:  # </parallel>
            depth -= 1
            if depth == 0 and open_start is not None:
                # --- CORRECTED LOGIC ---
                # The span must end after the full line containing the closing tag.
                line_end_pos = s.find('\n', m.end())
                if line_end_pos == -1:
                    # The tag is on the very last line of the file.
                    span_end = len(s)
                else:
                    # Include the newline character in the span.
                    span_end = line_end_pos + 1
                
                spans.append((open_start, span_end))
                # --- END CORRECTED LOGIC ---
                open_start = None
    return spans

def parse_threads_from_block(block_text: str) -> List[ThreadSeg]:
    """Parses all <thread>...</thread> segments from the text of a single parallel block."""
    threads: List[ThreadSeg] = []
    thread_open_re = re.compile(r"<thread>", re.IGNORECASE)
    thread_close_re = re.compile(r"</thread>", re.IGNORECASE)
    
    open_matches = list(thread_open_re.finditer(block_text))
    close_matches = list(thread_close_re.finditer(block_text))

    if len(open_matches) != len(close_matches):
        return []

    for open_match, close_match in zip(open_matches, close_matches):
        content_start = open_match.end()
        content_end = close_match.start()
        thread_text = block_text[content_start:content_end].strip()

        line_end_pos = block_text.find('\n', close_match.end())
        if line_end_pos == -1:
            line_end_pos = len(block_text)
        
        line_start_pos = block_text.rfind('\n', 0, close_match.start()) + 1
        closing_tag_line = block_text[line_start_pos:line_end_pos]
        
        thread_range = extract_first_range(closing_tag_line)
        start, end = (thread_range[0], thread_range[1]) if thread_range else (None, None)
        threads.append(ThreadSeg(text=thread_text, range_start=start, range_end=end))
        
    return threads


def parse_all_parallel_blocks(annotation_text: str) -> List[ParallelBlock]:
    """Parse all top-level <parallel> blocks from an annotations file."""
    blocks: List[ParallelBlock] = []
    for start, end in iter_top_level_parallel_spans(annotation_text):
        block_text = annotation_text[start:end]
        
        # Find the last line to extract the block's range
        last_line_start_idx = block_text.rfind('\n', 0, -1)
        last_line = block_text[last_line_start_idx+1:]
        block_range = extract_first_range(last_line)
        
        first_newline_idx = block_text.find('\n')
        if first_newline_idx == -1: first_newline_idx = len(block_text)
        first_line = block_text[:first_newline_idx]
        reason_match = re.search(r"\[Parallel reason:\s*(.*?)\]", first_line, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else ""

        threads = parse_threads_from_block(block_text)

        # If the block range was not found, infer it from the threads.
        if block_range is None:
            valid_thread_ranges = [
                (t.range_start, t.range_end) for t in threads if t.range_start is not None and t.range_end is not None
            ]
            if valid_thread_ranges:
                min_start = min(r[0] for r in valid_thread_ranges)
                max_end = max(r[1] for r in valid_thread_ranges)
                block_range = (min_start, max_end)

        blocks.append(ParallelBlock(reason=reason, threads=threads, block_range=block_range))
    return blocks


def build_parallel_replacement(block: ParallelBlock, traj_lines: List[str], filter_diff_threshold: int) -> Optional[str]:
    """Build the <Parallel> ... </Parallel> text for a given block.

    Skips building if diversity threshold not met: computes total characters across
    all extracted thread snippets and the maximum single snippet length; if the
    difference (total - max_single) is below filter_diff_threshold, we skip.
    """
    if not block.block_range:
        return None

    good_threads: List[Tuple[ThreadSeg, Tuple[int, int]]] = []
    for t in block.threads:
        if t.range_start is None or t.range_end is None:
            continue
        a, b = (t.range_start, t.range_end) if t.range_start <= t.range_end else (t.range_end, t.range_start)
        good_threads.append((t, (a, b)))

    # Ignore parallel block if only one thread
    if len(good_threads) <= 1:
        _log(f"Skipping parallel block with insufficient threads: {len(good_threads)}", level="WARNING", stderr=True)
        return None

    if not good_threads:
        _log("Skipping parallel block with no valid threads", level="WARNING", stderr=True)
        return None

    ranges_sorted = sorted([rng for _, rng in good_threads])
    for i in range(len(ranges_sorted) - 1):
        if ranges_sorted[i][1] >= ranges_sorted[i+1][0]:
            _log(f"Skipping parallel block due to overlapping thread ranges: {ranges_sorted[i]} and {ranges_sorted[i+1]}", level="WARNING", stderr=True)
            return None

    
    # out_lines.append("<Outlines>")
    # Do not add outlines at the current stage
    # for idx, (t, _) in enumerate(good_threads, start=1):
    #     outline = t.cleaned_outline()
    #     out_lines.append(f"<Outline>{idx}: {outline}</Outline>")
    # out_lines.append("</Outlines>")
    idx_snippets = []
    idx = 1
    for _, (a, b) in good_threads:
        start_i = max(1, a)
        end_i = min(len(traj_lines), b)
        snippet = "".join(traj_lines[start_i - 1:end_i]).rstrip() if start_i <= end_i else ""
        if snippet.strip():
            idx_snippets.append((idx, snippet))
            idx += 1

    if len(idx_snippets) <= 1:
        _log("Skipping parallel block with only one or zero valid thread after extraction", level="WARNING", stderr=True)
        return None

    # Diversity / difference filter: ensure combined content isn't dominated by a single thread.
    lengths = [len(snippet) for _, snippet in idx_snippets]
    total_len = sum(lengths)
    max_len = max(lengths)
    diff = total_len - max_len
    if diff < filter_diff_threshold:
        _log(
            f"Skipping parallel block due to low diversity diff (total={total_len}, max={max_len}, diff={diff} < threshold={filter_diff_threshold}). snippets: {[snippet for _, snippet in idx_snippets]}",
            level="WARNING",
            stderr=True,
        )
        return None

    out_lines: List[str] = ["<Parallel>"]
    for idx, snippet in idx_snippets:
        out_lines.append("<Thread>")
        out_lines.append(f"{idx}: {snippet}")
        out_lines.append("</Thread>")
    out_lines.append("</Parallel>")
    return "\n".join(out_lines) + "\n"


def apply_replacements(traj_text: str, replacements: List[Tuple[int, int, str]]) -> str:
    """Apply multiple line-numbered replacements to traj_text."""
    lines = traj_text.splitlines(keepends=True)
    norm: List[Tuple[int, int, str]] = []
    for a, b, s in replacements:
        if a > b: a, b = b, a
        a = max(1, a)
        b = min(len(lines), b)
        if a > b: continue
        norm.append((a, b, s))

    norm.sort(key=lambda x: (x[0], x[1]))
    filtered: List[Tuple[int, int, str]] = []
    last_end = 0
    for a, b, s in norm:
        if a < last_end:
            _log(f"Skipping overlapping replacement for L{a}-L{b}", level="WARNING", stderr=True)
            continue
        filtered.append((a, b, s))
        last_end = b

    for a, b, s in sorted(filtered, key=lambda x: x[0], reverse=True):
        start_idx_0 = a - 1
        end_idx_0_excl = b
        rep = s if s.endswith("\n") else s + "\n"
        lines[start_idx_0:end_idx_0_excl] = [rep]
    return "".join(lines)


def main() -> None:
    """Main execution function."""
    ap = argparse.ArgumentParser(description="Replace trajectory ranges based on <parallel> annotations.")
    ap.add_argument("--trajectory", "-t", required=True, help="Path to the reasoning trajectory file to READ.")
    ap.add_argument("--annotations", "-a", required=True, help="Path to the parallel annotations file.")
    ap.add_argument("--output-dir", "-o", required=True, help="Directory to WRITE the modified trajectory.")
    ap.add_argument("--encoding", default="utf-8", help="Text encoding for both files (default: utf-8).")
    ap.add_argument("--dry-run", action="store_true", help="Preview planned replacements and output path; do not write.")
    ap.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    ap.add_argument("--filter-diff-threshold", type=int, default=100,
                    help="Minimum difference (sum(chars in all thread snippets) - max(single snippet chars)) required to keep a parallel replacement. "
                         "If the difference is below this threshold, the replacement is skipped (default: 100).")
    args = ap.parse_args()

    try:
        with io.open(args.trajectory, "r", encoding=args.encoding, newline="") as f:
            traj_text = f.read()
    except FileNotFoundError:
        _log(f"Trajectory file not found at {args.trajectory}", level="ERROR", stderr=True)
        sys.exit(1)
        
    try:
        with io.open(args.annotations, "r", encoding=args.encoding, newline="") as f:
            annotation_text = f.read()
    except FileNotFoundError:
        _log(f"Annotations file not found at {args.annotations}", level="ERROR", stderr=True)
        sys.exit(1)

    blocks = parse_all_parallel_blocks(annotation_text)
    
    traj_lines = traj_text.splitlines(keepends=True)
    replacements: List[Tuple[int, int, str]] = []
    for block in blocks:
        rep = build_parallel_replacement(block, traj_lines, args.filter_diff_threshold)
        if rep is None or not block.block_range:
            continue
        a, b = block.block_range
        if a > b: a, b = b, a
        replacements.append((a, b, rep))

    if not replacements:
        _log("No eligible replacements found (nothing to write).", level="WARNING", stderr=True)

    new_text = apply_replacements(traj_text, replacements)
    out_filename = os.path.basename(args.trajectory)
    out_path = os.path.join(args.output_dir, out_filename)

    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {out_path}. Use --overwrite to replace it.")

    os.makedirs(args.output_dir, exist_ok=True)

    trace_dir = os.path.join(args.output_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, out_filename)

    if args.dry_run:
        _log("Planned replacements (1-based inclusive line ranges in trajectory):", level="INFO")
        if replacements:
            for a, b, _ in sorted(replacements):
                _log(f"  - L{a}-L{b}", level="INFO")
        else:
            _log("  - None", level="INFO")
        _log(f"(DRY RUN) Would write to: {out_path}", level="INFO")
        # Even in dry-run, write trace file for transparency
        try:
            with io.open(trace_path, "w", encoding=args.encoding, newline="") as tf:
                tf.write("\n".join(LOG_MESSAGES) + ("\n" if LOG_MESSAGES else ""))
        except Exception as e:
            _log(f"Failed to write trace file at {trace_path}: {e}", level="ERROR", stderr=True)
        return

    try:
        with io.open(out_path, "w", encoding=args.encoding, newline="") as f:
            f.write(new_text)
        _log(f"Wrote updated trajectory to: {out_path}", level="INFO")
    except Exception as e:
        _log(f"Failed writing updated trajectory to {out_path}: {e}", level="ERROR", stderr=True)

    # Write trace log
    try:
        with io.open(trace_path, "w", encoding=args.encoding, newline="") as tf:
            tf.write("\n".join(LOG_MESSAGES) + ("\n" if LOG_MESSAGES else ""))
        _log(f"Wrote trace log to: {trace_path}", level="INFO")
    except Exception as e:
        _log(f"Failed to write trace file at {trace_path}: {e}", level="ERROR", stderr=True)

if __name__ == "__main__":
    main()
