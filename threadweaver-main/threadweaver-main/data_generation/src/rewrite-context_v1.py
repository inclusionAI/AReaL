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
logger = logging.getLogger("parallel_thread_rewriter")

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

def _create_response(client: OpenAI, model: str, messages: List[Dict[str, str]], effort: str = "medium", verbosity: str = "high") -> str:
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
    # Make the chat.completions call (remove Responses-specific args)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        # Optional knobs: uncomment if you want them
        # temperature=0.2,
        # max_tokens=2048,
        # extra_body={"reasoning": {"effort": "medium"}},  # for reasoning-capable models
    )

    # Usage accounting (supports both classic and newer field names)
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
        pass  # don't let accounting failures break the flow

    # Extract assistant text
    try:
        choice = resp.choices[0]
        text = getattr(choice.message, "content", None)
        return text if text is not None else ""
    except Exception:
        return str(resp)


# -------------------------------------------------------------
# Prompt template
# -------------------------------------------------------------

def build_prompt(instruction_preamble: str, allowed_ctx: str, forbidden_head: str, forbidden_tail: str, thread_text: str) -> str:
    parts = [instruction_preamble]
    parts.append("Allowed Context:\n" + allowed_ctx.strip())
    parts.append(
        "Forbidden Context:\n" + forbidden_head.strip() + "\n...\n" + forbidden_tail.strip()
    )
    parts.append("Text Under Review:\n" + thread_text.strip())
    return "\n\n".join(parts) + "\n"

# -------------------------------------------------------------
# Parsing helpers
# -------------------------------------------------------------

PARALLEL_RE = re.compile(r"<Parallel>(.*?)</Parallel>", re.DOTALL)
THREAD_RE = re.compile(r"<Thread>(.*?)</Thread>", re.DOTALL)


def _is_flagged_prompt_error(err: Exception) -> bool:
    """
    Detect 400-level invalid prompt errors so we can skip rewriting that block
    without failing the entire file.
    """
    try:
        status = getattr(err, "status_code", None)
        if status is None and hasattr(err, "response"):
            status = getattr(err.response, "status_code", None)
        message = str(err).lower()
        if status == 400 and ("invalid prompt" in message or "flagged as potentially violating" in message):
            return True
    except Exception:
        pass
    return False


def _last_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    # Filter lines with tags
    lines = [line for line in lines if line.strip() and not re.match(r"<[A-Za-z]+>", line.strip())]
    return "\n".join(lines[-n:]) if lines else ""


def _first_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    # Filter lines with tags
    lines = [line for line in lines if line.strip() and not re.match(r"<[A-Za-z]+>", line.strip())]
    return "\n".join(lines[:n]) if lines else ""


def _apply_replacements(original: str, repls: List[Dict[str, str]]) -> str:
    original = original.strip()
    out = original
    for r in repls:
        src = r.get("src", "")
        dest = r.get("dest", "")
        if not isinstance(src, str) or not isinstance(dest, str):
            continue
        if src == "":
            continue
        out = out.replace(src, dest)
    if out.strip() == "":
        logger.warning("All content replaced with empty string; returning original.")
        return original
    # if original matches regex with "^\d+: " but out does not, we need to add it back
    # If the original has empty content (e.g., "2: \n", then it will be stripped to just "2:", which will not pass and is right).
    assert re.match(r"^\d+: ", original), f"Expected original to start with a digit and colon: {original}"

    if not re.match(r"^\d+: ", out):
        # The ":" should have been removed too (not just the digit).
        assert any([":" in item['src'] for item in repls]), f"Expected at least one replacement to contain a colon to remove that from the source: {repls}"
        # Add the prefix back to the output
        match = re.match(r"^(\d+): ", original)
        if match:
            prefix = match.group(0)
            out = prefix + out.strip()
    out = out.strip()
    out = "\n" + out + "\n"
    return out


def _parse_response_to_replacements(text: str) -> Optional[List[Dict[str, str]]]:
    """Return list of {src, dest} dicts or None if parsing fails."""
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            # validate
            good: List[Dict[str, str]] = []
            for item in obj:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("src"), str)
                    and isinstance(item.get("dest"), str)
                ):
                    good.append({"src": item["src"], "dest": item["dest"]})
            return good
    except Exception:
        pass

    # Try to find the first JSON array in the text
    try:
        m = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", text)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                good: List[Dict[str, str]] = []
                for item in obj:
                    if (
                        isinstance(item, dict)
                        and isinstance(item.get("src"), str)
                        and isinstance(item.get("dest"), str)
                    ):
                        good.append({"src": item["src"], "dest": item["dest"]})
                return good
    except Exception:
        pass
    return None

# -------------------------------------------------------------
# Path rewriting helper (extracts repeated logic incl. retry)
# -------------------------------------------------------------

def _call_and_parse_replacements(
    *,
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    args: argparse.Namespace,
    effort: str,
    verbosity: str,
) -> Tuple[Optional[List[Dict[str, str]]], str]:
    """Single model attempt returning (replacements or None, raw_response_text)."""
    if args.use_chat_completions:
        response_text = _create_response_chat_completions(client, model, messages)
    else:
        response_text = _create_response(client, model, messages, effort, verbosity)
    repls = _parse_response_to_replacements(response_text)
    return repls, response_text

def _rewrite_thread_with_retry(
    *,
    content: str,
    prev_content: str,
    allowed_ctx: str,
    filepath: str,
    thread_index: int,
    parallel_start: int,
    client: OpenAI,
    model: str,
    instruction_preamble: str,
    effort: str,
    verbosity: str,
    trace_dir: Optional[str],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    """Rewrite a single <Thread> block (excluding the first in a <Parallel>). Returns (changed, new_content).

    Handles:
      - Prompt construction (allowed + forbidden context)
      - Single model call + JSON replacement parsing
      - Retry once if shrinks below 50%
      - Trace dumping (initial + retry) if enabled
    """
    # Forbidden context derived from previous thread
    f_head = _first_n_lines(prev_content, 5)
    f_tail = _last_n_lines(prev_content, 5)

    prompt = build_prompt(instruction_preamble, allowed_ctx, f_head, f_tail, content)
    messages = [{"role": "user", "content": prompt}]

    # Initial attempt
    try:
        repls, response_text = _call_and_parse_replacements(
            client=client,
            model=model,
            messages=messages,
            args=args,
            effort=effort,
            verbosity=verbosity,
        )
    except Exception as e:
        if _is_flagged_prompt_error(e):
            logger.error(
                f"{os.path.basename(filepath)}: <Parallel> thread #{thread_index+1}: prompt flagged by API; skipping rewrite for this thread."
            )
            return False, content
        raise
    if repls is None:
        logger.warning(
            f"{os.path.basename(filepath)}: <Parallel> thread #{thread_index+1}: could not parse JSON; leaving unchanged."
        )
        return False, content  # unchanged

    # Trace (initial)
    if trace_dir:
        try:
            os.makedirs(trace_dir, exist_ok=True)
            base = os.path.basename(filepath)
            trace_path = os.path.join(
                trace_dir, f"{os.path.splitext(base)[0]}_parallel_{parallel_start}_thread_{thread_index+1}.json"
            )
            with open(trace_path, "w", encoding="utf-8") as tf:
                json.dump({
                    "prompt": prompt,
                    "response": response_text,
                    "replacements": repls,
                }, tf, ensure_ascii=False, indent=2)
        except Exception as te:
            logger.debug(f"Trace write failed: {te}")

    new_content = _apply_replacements(content, repls)

    # Retry policy for shrinkage (<50%)
    shrink_threshold = int(len(content) * 0.5)
    if len(new_content) < shrink_threshold:
        logger.warning(
            f"{os.path.basename(filepath)}: <Parallel> thread #{thread_index+1}: replacement <50% size; retrying once."
        )
        try:
            repls_retry, response_text_retry = _call_and_parse_replacements(
                client=client,
                model=model,
                messages=messages,
                args=args,
                effort=effort,
                verbosity=verbosity,
            )
            if repls_retry is not None:
                new_content_retry = _apply_replacements(content, repls_retry)
                if trace_dir:
                    try:
                        os.makedirs(trace_dir, exist_ok=True)
                        base = os.path.basename(filepath)
                        trace_path_retry = os.path.join(
                            trace_dir,
                            f"{os.path.splitext(base)[0]}_parallel_{parallel_start}_thread_{thread_index+1}_retry.json",
                        )
                        with open(trace_path_retry, "w", encoding="utf-8") as tf:
                            json.dump({
                                "prompt": prompt,
                                "response": response_text_retry,
                                "replacements": repls_retry,
                            }, tf, ensure_ascii=False, indent=2)
                    except Exception as te:
                        logger.debug(f"Retry trace write failed: {te}")
                # Prefer retry if it's larger OR meets threshold
                if (
                    len(new_content_retry) >= shrink_threshold
                    or len(new_content_retry) > len(new_content)
                ):
                    new_content = new_content_retry
                if len(new_content) < shrink_threshold:
                    logger.warning(
                        f"{os.path.basename(filepath)}: <Parallel> thread #{thread_index+1}: retry still <50% of original; accepting anyway."
                    )
            else:
                logger.warning(
                    f"{os.path.basename(filepath)}: <Parallel> thread #{thread_index+1}: retry JSON parse failed; keeping first replacement."
                )
        except Exception as retry_e:
            if _is_flagged_prompt_error(retry_e):
                logger.error(
                    f"{os.path.basename(filepath)}: <Parallel> thread #{thread_index+1}: retry prompt flagged by API; keeping first replacement."
                )
            else:
                logger.warning(
                    f"{os.path.basename(filepath)}: <Parallel> thread #{thread_index+1}: retry attempt raised {retry_e}; keeping first replacement."
                )

    return True, new_content

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
    args: argparse.Namespace = None,
) -> Tuple[str, bool, str]:
    """Process a single trajectory file, rewriting each <Thread> (except the first) in every <Parallel>."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        new_text = text
        offset = 0  # track shifts due to replacements

        for p_match in re.finditer(PARALLEL_RE, text):
            p_start, p_end = p_match.span()
            p_content = p_match.group(1)

            # Allowed context: 5 lines before the <Parallel> block
            allowed_ctx = _last_n_lines(text[:p_start], 5)

            # Work within this parallel's content
            inner = p_content
            inner_new = inner
            inner_offset = 0

            # enumerate <Thread> blocks in order
            thread_matches = list(re.finditer(THREAD_RE, inner))
            for idx, m in enumerate(thread_matches):
                full_start, full_end = m.span()
                content = m.group(1)

                if idx == 0:
                    continue  # skip the first <Thread>
                prev_content = thread_matches[idx - 1].group(1)

                changed, new_content = _rewrite_thread_with_retry(
                    content=content,
                    prev_content=prev_content,
                    allowed_ctx=allowed_ctx,
                    filepath=filepath,
                    thread_index=idx,
                    parallel_start=p_start,
                    client=client,
                    model=model,
                    instruction_preamble=instruction_preamble,
                    effort=effort,
                    verbosity=verbosity,
                    trace_dir=trace_dir,
                    args=args,
                )
                if not changed:
                    continue

                # Rebuild this <Thread> block preserving tags/whitespace
                original_block = m.group(0)  # <Thread>...content...</Thread>
                new_block = original_block.replace(content, new_content, 1)

                # Splice into inner_new using current inner_offset
                full_start_adj = full_start + inner_offset
                full_end_adj = full_end + inner_offset
                inner_new = inner_new[:full_start_adj] + new_block + inner_new[full_end_adj:]

                # Update inner_offset due to size change within this parallel
                delta = len(new_block) - (full_end - full_start)
                inner_offset += delta

            # After finishing this <Parallel>, splice back into document
            p_start_adj = p_start + offset
            p_end_adj = p_end + offset
            # Reconstruct full <Parallel> with original tags preserved
            original_parallel_block = p_match.group(0)
            # Replace only the inner content once
            new_parallel_block = original_parallel_block.replace(p_content, inner_new, 1)
            new_text = new_text[:p_start_adj] + new_parallel_block + new_text[p_end_adj:]

            # Update global offset
            offset += len(new_parallel_block) - (p_end - p_start)

        # Write output
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, os.path.basename(filepath))
        print(f"Rewriting {filepath} → {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new_text)

        return filepath, True, ""

    except Exception as e:
        return filepath, False, str(e)

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rewrite <Thread> blocks in <Parallel> using GPT-generated replacements.")
    parser.add_argument("--input", required=True, help="Input file or directory containing trajectory text files.")
    parser.add_argument("--output", required=True, help="Directory to write the rewritten files.")
    parser.add_argument("--glob", default="**/*.txt", help="Glob pattern when --input is a directory (default: **/*.txt).")
    parser.add_argument("--openai_model", default="gpt-5", help="Model to use (default: gpt-5).")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads (default: min(32, os.cpu_count()+4)).")
    parser.add_argument("--trace", default=None, help="Optional directory to save prompts/responses per processed <Thread>.")
    parser.add_argument("--prompt", required=True, help="Path to the file containing the instruction preamble template.")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"], help="Reasoning effort level (default: medium).")
    parser.add_argument("--verbosity", default="high", choices=["low", "medium", "high"], help="Text verbosity level (default: high).")
    parser.add_argument("--in_cost_per_million", type=float, default=1.25, help="USD per 1M input tokens (default: 1.25).")
    parser.add_argument("--out_cost_per_million", type=float, default=10.0, help="USD per 1M output tokens (default: 10.0).")
    parser.add_argument('--use_chat_completions', action='store_true', help='Use chat.completions API instead of responses API.')
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")

    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {args.output}. Use --overwrite to replace it.")
    # Read instruction preamble from file
    try:
        with open(args.prompt, "r", encoding="utf-8") as f:
            instruction_preamble = f.read().strip()
    except Exception as e:
        raise FileNotFoundError(f"Could not read prompt file '{args.prompt}': {e}")

    # Resolve files
    files: List[str] = []
    if os.path.isdir(args.input):
        import glob as _glob
        pattern = os.path.join(args.input, args.glob)
        files = [p for p in _glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    else:
        files = [args.input]

    if not files:
        raise FileNotFoundError("No input files found.")

    # Threads
    if args.workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    else:
        max_workers = max(1, args.workers)

    logger.info(f"Discovered {len(files)} file(s). Using {max_workers} worker thread(s). Model: {args.openai_model}")

    # Client per worker task, created inside worker
    def _worker(path: str) -> Tuple[str, bool, str]:
        client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_API_BASE", None))
        return process_file(path, args.output, client, args.openai_model, instruction_preamble, args.effort, args.verbosity, args.trace, args=args)

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

    # Cost summary — PRESERVED
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
