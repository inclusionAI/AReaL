#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import json
import argparse
import logging
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

# -------------------------------------------------------------
# Logging
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outline_generator")

# -------------------------------------------------------------
# API Key
# -------------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key is missing. Set OPENAI_API_KEY in your environment.")

# -------------------------------------------------------------
# Global token accounting (thread-safe) — PRESERVED
# -------------------------------------------------------------
_token_lock = Lock()
_total_input_tokens = 0
_total_output_tokens = 0

def _accumulate_usage(input_tokens: int, output_tokens: int) -> None:
    global _total_input_tokens, _total_output_tokens
    with _token_lock:
        _total_input_tokens += int(input_tokens or 0)
        _total_output_tokens += int(output_tokens or 0)

# -------------------------------------------------------------
# OpenAI Responses helper (sync) — PRESERVED STYLE
# -------------------------------------------------------------
def _create_response(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    effort: str = "medium",
    verbosity: str = "high",
) -> str:
    """Call the OpenAI Responses API and return best-effort plain text output."""
    resp = client.responses.create(
        model=model,
        input=messages,
        reasoning={"effort": effort},
        text={"verbosity": verbosity},
    )
    try:
        in_tok = int(getattr(getattr(resp, "usage", None), "input_tokens", 0) or 0)
        out_tok = int(getattr(getattr(resp, "usage", None), "output_tokens", 0) or 0)
        _accumulate_usage(in_tok, out_tok)
        logger.debug(f"Usage — input: {in_tok}, output: {out_tok}")
    except Exception:
        pass

    text = getattr(resp, "output_text", None)
    if text is not None:
        return text

    try:
        parts = []
        for item in getattr(resp, "output", []):
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "output_text":
                    parts.append(getattr(content, "text", ""))
        return "".join(parts) if parts else str(resp)
    except Exception:
        return str(resp)


def _create_response_chat_completions(client: OpenAI, model: str, messages: List[Dict[str, Any]]) -> str:
    """
    Call the OpenAI Chat Completions API with a running list of role-based messages.
    Returns best-effort plain text output (SYNC).
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    try:
        usage = getattr(resp, "usage", None)
        in_tok = int(
            (getattr(usage, "input_tokens", None) or
             getattr(usage, "prompt_tokens", None) or 0)
        )
        out_tok = int(
            (getattr(usage, "output_tokens", None) or
             getattr(usage, "completion_tokens", None) or 0)
        )
        _accumulate_usage(in_tok, out_tok)
        logger.debug(f"Usage — input: {in_tok}, output: {out_tok}")
    except Exception:
        pass

    try:
        choice = resp.choices[0]
        text = getattr(choice.message, "content", None)
        return text if text is not None else ""
    except Exception:
        return str(resp)


# -------------------------------------------------------------
# Prompt builder
# -------------------------------------------------------------
def build_outline_prompt(
    instruction_preamble: str,
    context_text: str,
    parallel_content_with_threads: str,
    num_threads: int,
) -> str:
    """
    Construct the prompt for generating all outlines in a Parallel block.

    - Use the caller-provided instruction preamble.
    - Provide 'Context:' from BEFORE the current <Parallel> block.
    - Create a <Outlines> block with placeholders for each thread.
    - Append the content of the <Parallel> block (threads, etc.) for the model to use.
    """
    placeholders = [
        f"<Outline>{i}: [Outline to be generated for thread {i} below]</Outline>"
        for i in range(1, num_threads + 1)
    ]
    outlines_placeholder_block = "<Outlines>\n" + "\n".join(placeholders) + "\n</Outlines>"

    # The model is tasked with filling in the placeholders in the <Outlines> block
    # using the context and the subsequent <Thread> contents.
    parts = [
        instruction_preamble.strip(),
        f"Context:\n{context_text.strip()}",
        f"Segment:\n<Parallel>\n{outlines_placeholder_block.strip()}\n{parallel_content_with_threads.strip()}\n</Parallel>",
    ]
    return "\n\n".join(parts)

# -------------------------------------------------------------
# Parsing helpers
# -------------------------------------------------------------
PARALLEL_RE = re.compile(r"<Parallel>(.*?)</Parallel>", re.DOTALL)
THREAD_RE = re.compile(r"<Thread>(.*?)</Thread>", re.DOTALL)
OUTLINES_RE = re.compile(r"<Outlines>(.*?)</Outlines>", re.DOTALL)

def _last_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    # Filter lines with tags
    lines = [re.sub(r"</?[A-Za-z]+>", "", line) for line in lines]
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines[-n:]) if lines else ""


def _first_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    # Filter lines with tags
    lines = [re.sub(r"</?[A-Za-z]+>", "", line) for line in lines]
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines[:n]) if lines else ""

def _extract_outlines_block(raw_text: str) -> Optional[str]:
    """Find and return the first <Outlines>...</Outlines> block from the model's output."""
    # Use a non-greedy search to find the first complete <Outlines> block
    match = re.search(r"<Outlines>.*?</Outlines>", raw_text, re.DOTALL)
    if match:
        return match.group(0)
    logger.warning("Could not parse a <Outlines> block from the model response.")
    return None

# -------------------------------------------------------------
# Core processing
# -------------------------------------------------------------
def process_file(
    filepath: str,
    outdir: str,
    client: OpenAI,
    model: str,
    instruction_preamble: str,
    effort: str = "medium",
    verbosity: str = "high",
    trace_dir: Optional[str] = None,
    ctx_lines: int = 50,
    args: argparse.Namespace = None,
) -> Tuple[str, bool, str]:
    """
    For each <Parallel> block in a file, generate all outlines for its <Thread>s
    with a single API call, then insert the resulting <Outlines> block.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        new_text = text
        offset = 0

        for p_match in re.finditer(PARALLEL_RE, text):
            p_start, p_end = p_match.span()
            p_content = p_match.group(1)
            original_parallel_block = p_match.group(0)

            # Context: last N lines of the text BEFORE this <Parallel>
            allowed_ctx = _last_n_lines(text[:p_start], ctx_lines)

            # Find all <Thread> blocks to determine how many outlines to generate
            thread_matches = list(re.finditer(THREAD_RE, p_content))
            if not thread_matches:
                continue
            num_threads = len(thread_matches)

            # Remove any pre-existing <Outlines> block from the parallel content
            p_content_no_outlines = re.sub(OUTLINES_RE, '', p_content, count=1).strip()

            # Build the single prompt for the entire <Parallel> block
            prompt = build_outline_prompt(
                instruction_preamble,
                allowed_ctx,
                p_content_no_outlines,
                num_threads,
            )

            # Make a single API call for all threads in this block
            messages = [{"role": "user", "content": prompt}]
            # Try up to 3 times to get a valid response
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if args and args.use_chat_completions:
                        response_text = _create_response_chat_completions(client, model, messages)
                    else:
                        response_text = _create_response(client, model, messages, effort, verbosity)

                    # Extract the generated <Outlines> block from the response
                    new_outlines_block = _extract_outlines_block(response_text)
                    if not new_outlines_block:
                        raise ValueError(f"Could not find <Outlines> block in response for parallel block at pos {p_start} in {filepath} in response {response_text}")

                    # Ensure the number of outlines matches the number of threads
                    outlines_found = re.findall(r"<Outline>\s*(\d+):", new_outlines_block)
                    if len(outlines_found) != num_threads:
                        raise ValueError(
                            f"Number of outlines ({len(outlines_found)}) does not match number of threads "
                            f"({num_threads}) in parallel block at pos {p_start} in {filepath}."
                        )
                    
                    # If we get here, everything is valid
                    break
                
                except ValueError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt+1}/{max_retries}: {str(e)}")
                        continue
                    else:
                        raise e
                        # logger.warning(
                        #     f"All {max_retries} attempts failed for parallel block "
                        #     f"at pos {p_start} in {filepath}. Inserting an error block."
                        # )
                        # error_outlines = "\n".join([f\"<Outline>{i+1}: (Error: Failed to generate outline)</Outline>\" for i in range(num_threads)])
                        # new_outlines_block = f\"<Outlines>\\n{error_outlines}\\n</Outlines>\"

            # Optional trace dump for the entire parallel block
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
                base = os.path.basename(filepath)
                trace_path = os.path.join(
                    trace_dir,
                    f"{os.path.splitext(base)[0]}_parallel_{p_start}.json",
                )
                with open(trace_path, "w", encoding="utf-8") as tf:
                    json.dump(
                        {
                            "prompt": prompt,
                            "response": response_text,
                            "extracted_outlines": new_outlines_block,
                        },
                        tf,
                        ensure_ascii=False,
                        indent=2,
                    )
            
            # Reconstruct the new inner content of the <Parallel> block
            # New outlines block at the start, followed by the original content (sans old outlines).
            inner_new = new_outlines_block + "\n" + p_content_no_outlines

            # Replace original inner content with new content to preserve surrounding whitespace
            new_parallel_block = original_parallel_block.replace(p_content, "\n" + inner_new.strip() + "\n", 1)

            # Splice into the full document text using the current global offset
            p_start_adj = p_start + offset
            p_end_adj = p_end + offset
            new_text = new_text[:p_start_adj] + new_parallel_block + new_text[p_end_adj:]

            # Update global offset to account for the change in length
            offset += len(new_parallel_block) - (p_end - p_start)

        # Post-processing to clean up excessive newlines
        new_text = re.sub(r'\n{3,}', '\n\n', new_text)

        # Write output
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, os.path.basename(filepath))
        print(f"Outlining {filepath} → {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new_text)

        return filepath, True, ""

    except Exception as e:
        logger.error(f"Failed to process file {filepath}", exc_info=e)
        return filepath, False, str(e)

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate <Outline> entries for each <Thread> in <Parallel> blocks using a GPT outline prompt."
    )
    parser.add_argument("--input", required=True, help="Input file or directory containing trajectory text files.")
    parser.add_argument("--output", required=True, help="Directory to write the outlined files.")
    parser.add_argument("--glob", default="**/*.txt", help="Glob pattern when --input is a directory (default: **/*.txt).")
    parser.add_argument("--openai_model", default="gpt-4-turbo", help="Model to use (default: gpt-4-turbo).")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads (default: min(32, os.cpu_count()+4)).")
    parser.add_argument("--trace", default=None, help="Optional directory to save prompts/responses per processed <Parallel> block.")
    parser.add_argument("--prompt", required=True, help="Path to the file containing the instruction preamble template.")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"], help="Reasoning effort level (default: medium).")
    parser.add_argument("--verbosity", default="high", choices=["low", "medium", "high"], help="Text verbosity level (default: high).")
    parser.add_argument("--ctx_lines", type=int, default=50, help="Number of context lines preceding each <Parallel> to include (default: 50).")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of files to process from glob (default: no limit).")
    parser.add_argument("--in_cost_per_million", type=float, default=1.25, help="USD per 1M input tokens (default: 1.25).")
    parser.add_argument("--out_cost_per_million", type=float, default=10.0, help="USD per 1M output tokens (default: 10.0).")
    parser.add_argument('--use_chat_completions', action='store_true', help='Use chat.completions API instead of responses API.')
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")

    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {args.output}. Use --overwrite to replace it.")

    try:
        with open(args.prompt, "r", encoding="utf-8") as f:
            instruction_preamble = f.read().strip()
    except Exception as e:
        raise FileNotFoundError(f"Could not read prompt file '{args.prompt}': {e}")

    files: List[str] = []
    if os.path.isdir(args.input):
        import glob as _glob
        pattern = os.path.join(args.input, args.glob)
        files = [p for p in _glob.glob(pattern, recursive=True) if os.path.isfile(p)]
        if args.max_samples is not None and args.max_samples > 0:
            files = files[:args.max_samples]
    else:
        files = [args.input]

    if not files:
        raise FileNotFoundError("No input files found.")

    if args.workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    else:
        max_workers = max(1, args.workers)

    logger.info(f"Discovered {len(files)} file(s). Using {max_workers} worker thread(s). Model: {args.openai_model}")

    def _worker(path: str) -> Tuple[str, bool, str]:
        client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_API_BASE", None))
        return process_file(
            path,
            args.output,
            client,
            args.openai_model,
            instruction_preamble,
            args.effort,
            args.verbosity,
            args.trace,
            args.ctx_lines,
            args,
        )

    results: List[Tuple[str, bool, str]] = []
    errors: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_worker, p) for p in files]
        for fut in as_completed(futs):
            fp, ok, msg = fut.result()
            if not ok:
                logger.error(f"ERROR in {fp}: {msg}")
                errors.append((fp, msg))
            else:
                logger.info(f"Processed {fp}")
            results.append((fp, ok, msg))

    with _token_lock:
        total_in = _total_input_tokens
        total_out = _total_output_tokens

    cost_in = (total_in / 1_000_000) * float(args.in_cost_per_million)
    cost_out = (total_out / 1_000_000) * float(args.out_cost_per_million)
    total_cost = cost_in + cost_out

    logger.info(f"Token usage — input: {total_in:,} | output: {total_out:,}")
    logger.info(
        f"Estimated cost — input ${cost_in:,.4f} + output ${cost_out:,.4f} = ${total_cost:,.4f}"
    )

    os.makedirs(args.output, exist_ok=True)
    summary_path = os.path.join(args.output, "usage_summary.json")
    summary_data = {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "input_cost_usd": cost_in,
        "output_cost_usd": cost_out,
        "total_cost_usd": total_cost,
        "files": [
            {"file": fp, "ok": ok, "error": msg if not ok else ""} for fp, ok, msg in results
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved usage summary to {summary_path}")

    if errors:
        logger.info(f"{len(errors)} files had errors.")
    else:
        logger.info("All files processed successfully.")

if __name__ == "__main__":
    main()
