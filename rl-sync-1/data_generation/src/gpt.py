# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This script references code from Multiverse's data processing script https://github.com/Multiverse4FM/Multiverse/blob/main/data/src/model/gemini.py

The original script as well as the part from the original script used in this script are under Apache License 2.0 https://github.com/Multiverse4FM/Multiverse/blob/main/LICENSE
"""

import os
import argparse
import json
import logging
import re
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock  # NEW: for thread-safe token accounting

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key is missing. Set OPENAI_API_KEY in your environment.")

# ----------------------------
# Global token accounting (thread-safe)
# ----------------------------
_token_lock = Lock()
_total_input_tokens = 0
_total_output_tokens = 0

def _accumulate_usage(input_tokens: int, output_tokens: int) -> None:
    global _total_input_tokens, _total_output_tokens
    with _token_lock:
        _total_input_tokens += int(input_tokens or 0)
        _total_output_tokens += int(output_tokens or 0)

# ----------------------------
# Helpers for OpenAI Responses (sync)
# ----------------------------

def _create_response(client: OpenAI, model: str, messages: List[Dict[str, Any]]) -> str:
    """
    Call the OpenAI Responses API with a running list of role-based messages.
    Returns best-effort plain text output (SYNC).
    """
    resp = client.responses.create(
        model=model,
        input=messages,
        reasoning={"effort": "medium"},
        text={"verbosity": "high"},
        # reasoning={"effort": "low"},
        # text={"verbosity": "low"},
    )

    # EXACT usage extraction based on the known response format
    try:
        in_tok = int(getattr(getattr(resp, "usage", None), "input_tokens", 0) or 0)
        out_tok = int(getattr(getattr(resp, "usage", None), "output_tokens", 0) or 0)
        _accumulate_usage(in_tok, out_tok)
        logger.debug(f"Usage — input: {in_tok}, output: {out_tok}")
    except Exception:
        # Don't let accounting issues affect core flow
        pass

    # Prefer the convenience property if available:
    text = getattr(resp, "output_text", None)
    if text is not None:
        return text

    # Fallback: stitch together text segments from the structured output, if needed.
    try:
        parts = []
        for item in getattr(resp, "output", []):
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "output_text":
                    parts.append(getattr(content, "text", ""))
        return "".join(parts) if parts else str(resp)
    except Exception:
        # Absolute last resort — just cast to string
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


def _append_user(messages: List[Dict[str, Any]], text: str) -> None:
    messages.append({"role": "user", "content": text})


def _append_assistant(messages: List[Dict[str, Any]], text: str) -> None:
    messages.append({"role": "assistant", "content": text})


# ----------------------------
# Core chat flows
# ----------------------------

def run_chat(client: OpenAI, prompt: List[str], thinking: str, model: str, use_chat_completions: bool) -> Tuple[List[Any], List[Any]]:
    """
    Mirrors original run_chat():
      - Creates a 'session' by accumulating messages locally and sending them on each turn
      - At i == 3, finds P{n} cases in the most recent response and fans out sub-prompts
    """
    messages: List[Dict[str, Any]] = []
    prompt_list: List[Any] = []
    response_list: List[Any] = []

    for i, p in enumerate(prompt):
        if i == 3:
            # Fan out on detected cases (P{n}) from the most recent assistant response
            all_case_response = []
            all_case_prompt = []
            recent_response = response_list[-1]  # last response text
            all_case = re.findall(r'P(\d+)', recent_response)
            all_case = sorted(set(all_case), key=lambda x: int(x))

            for case in all_case:
                p_case = p.format(i=case)
                _append_user(messages, p_case)
                if use_chat_completions:
                    r_text = _create_response_chat_completions(client, model, messages)
                else:
                    r_text = _create_response(client, model, messages)
                _append_assistant(messages, r_text)
                all_case_response.append(r_text)
                all_case_prompt.append(p_case)

            response_list.append(all_case_response)
            prompt_list.append(all_case_prompt)
            continue

        elif i == 0:
            # Prepend the "thinking" to the first prompt
            p = thinking + '\n\n' + p

        _append_user(messages, p)
        if use_chat_completions:
            r_text = _create_response_chat_completions(client, model, messages)
        else:
            r_text = _create_response(client, model, messages)
        _append_assistant(messages, r_text)
        response_list.append(r_text)
        prompt_list.append(p)

    return response_list, prompt_list


def run_chat_step1_v1(client: OpenAI, prompt: List[str], thinking: str, model: str, use_chat_completions: bool) -> Tuple[List[Any], List[Any]]:
    """
    Mirrors original run_chat_step1():
      - Adds Original Reasoning Chain at i == 0 and i == 5
    """
    messages: List[Dict[str, Any]] = []
    prompt_list: List[Any] = []
    response_list: List[Any] = []

    for i, p in enumerate(prompt):
        if i == 0 or i == 5:
            original_text = "Original Reasoning Chain: \n```markdown\n" + thinking + "\n```"
            p = original_text + "\n\n" + p

        _append_user(messages, p)
        if use_chat_completions:
            r_text = _create_response_chat_completions(client, model, messages)
        else:
            r_text = _create_response(client, model, messages)
        _append_assistant(messages, r_text)
        response_list.append(r_text)
        prompt_list.append(p)

    return response_list, prompt_list


# ----------------------------
# Worker for multithreaded execution
# ----------------------------

def process_record(
    record: Dict[str, Any],
    args: argparse.Namespace,
    prompt_segments: List[str],
    skip_existing: bool = False,
) -> Tuple[str, bool, str]:
    """
    Process a single dataset record synchronously.
    Returns (uuid, success, error_message_if_any)
    """
    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_API_BASE", None))  # new client per record (mirrors original pattern)

    uuid = record["uuid"]
    thinking = record["thinking"]
    prompt_list = prompt_segments.copy()

    reasoning_path = os.path.join(args.output, f"{uuid}_reasoning.txt")
    chat_path = os.path.join(args.chat, f"{uuid}_chat.txt")

    if os.path.exists(reasoning_path) and os.path.exists(chat_path) and skip_existing:
        logger.info(f"Skipping {uuid} as output files already exist.")
        return uuid, True, ""

    logger.info(f"Working on {uuid}.")

    try:
        # Choose flow based on prompt file name
        if 'step1' in args.prompt:
            response_list, used_prompts = run_chat_step1_v1(client, prompt_list, thinking, args.openai_model, use_chat_completions=args.use_chat_completions)
        else:
            response_list, used_prompts = run_chat(client, prompt_list, thinking, args.openai_model, use_chat_completions=args.use_chat_completions)

        final_response = response_list[-1]

        # Write primary outputs
        with open(reasoning_path, 'w', encoding='utf-8') as f:
            f.write(final_response if isinstance(final_response, str) else json.dumps(final_response, ensure_ascii=False))

        # Write turn-by-turn trace
        with open(chat_path, 'w', encoding='utf-8') as f:
            for p, r_out in zip(used_prompts, response_list):
                f.write(f"Prompt: {p}\n")
                if isinstance(r_out, list):
                    f.write(f"Response: {json.dumps(r_out, ensure_ascii=False)}\n")
                else:
                    f.write(f"Response: {r_out}\n")

        return uuid, True, ""
    except Exception as e:
        return uuid, False, str(e)


# ----------------------------
# Main: CLI + multithreaded IO
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Run OpenAI reasoning.')
    parser.add_argument('--prompt', type=str, help='The prompt file to use.')
    parser.add_argument('--input', type=str, help='The input file to use.')
    parser.add_argument('--output', type=str, help='The output directory to store results.')
    parser.add_argument('--chat', type=str, help='The chat directory to store turn-by-turn traces.')
    parser.add_argument('--start_idx', type=int, help='The start index of the input file.')
    parser.add_argument('--end_idx', type=int, help='The end index of the input file.')
    parser.add_argument('--openai_model', type=str, help='Model to use.', default='gpt-5')
    parser.add_argument('--workers', type=int, help='Number of worker threads (default: min(32, os.cpu_count() + 4)).', default=None)
    parser.add_argument('--in_cost_per_million', type=float, default=1.25, help='USD per 1M input tokens (default: 1.25).')
    parser.add_argument('--out_cost_per_million', type=float, default=10.0, help='USD per 1M output tokens (default: 10.0).')
    parser.add_argument('--use_chat_completions', action='store_true', help='Use chat.completions API instead of responses API.')
    parser.add_argument('--skip_existing', action='store_true', help='Skip processing records if output files already exist.')
    parser.add_argument('--overwrite', action='store_true', help='Allow writing into existing output/chat directories.')
    args = parser.parse_args()

    if args.output is None:
        raise ValueError("--output is required.")
    if args.chat is None:
        raise ValueError("--chat is required.")

    # Ensure output folders exist
    for label, path in (("output", args.output), ("chat", args.chat)):
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f"{label.capitalize()} directory already exists: {path}. Use --overwrite to replace it.")
        os.makedirs(path, exist_ok=True)

    logger.info(f"args: {args}")

    # Load input dataset (JSONL with fields including 'uuid' and 'thinking')
    r1_dataset = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            r1_dataset.append(json.loads(line))
    logger.info(f"Loaded {len(r1_dataset)} examples from {args.input}")

    # Slicing
    start_idx = args.start_idx if args.start_idx is not None and args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx is not None and args.end_idx < len(r1_dataset) else len(r1_dataset)
    r1_dataset = r1_dataset[start_idx:end_idx]
    logger.info(f"Loaded {len(r1_dataset)} examples from {args.input} from {start_idx} to {end_idx}")

    # Load and split prompt template by '---\n'
    with open(args.prompt, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    prompt_segments = prompt_text.split('---\n')

    logger.info(f"Prompt Number: {len(prompt_segments)}")
    for p in prompt_segments:
        logger.info(p)
        logger.info('-' * 100)

    # Thread pool
    if args.workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    else:
        max_workers = max(1, args.workers)

    logger.info(f"Using {max_workers} worker threads.")
    logger.info(f"Using model: {args.openai_model}, processing {len(r1_dataset)} records.")

    # Submit all records
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for r in r1_dataset:
            futures.append(
                executor.submit(
                    process_record,
                    r,
                    args,
                    prompt_segments,
                    args.skip_existing,
                )
            )

        # Progress tracking
        errors = []
        for fut in tqdm(as_completed(futures), total=len(futures)):
            uuid, ok, msg = fut.result()
            if not ok:
                errors.append((uuid, msg))
                logger.error(f"Error in {uuid}: {msg}")

    if errors:
        logger.info(f"{len(errors)} records failed.")
    else:
        logger.info("All records processed successfully.")

    # ----------------------------
    # Report token usage and estimated costs
    # ----------------------------
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

    summary_path = os.path.join(args.output, "usage_summary.json")
    summary_data = {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "input_cost_usd": cost_in,
        "output_cost_usd": cost_out,
        "total_cost_usd": total_cost
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved usage summary to {summary_path}")

if __name__ == "__main__":
    main()
