"""
Process train.jsonl to add parallel branches at "Alternatively" keywords.

For each assistant response with n "Alternatively" occurrences, the first n-1
are wrapped in <Parallel><Trial>...</Trial></Parallel> with generated branches
from a sglang server.

Usage:
    python process_branches.py \
        --input ../train.jsonl \
        --output ../train_branched.jsonl \
        --num_branches 2 \
        --sglang_url http://localhost:30000 \
        --max_tokens 2048 \
        --temperature 0.7
"""

import argparse
import json
import re
import os
import sys
import time
import logging
import threading
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def split_by_alternatively(text: str) -> List[str]:
    """
    Split text by the keyword 'Alternatively'.
    Returns a list of segments.  The keyword itself is NOT included in any
    segment – it lives at the boundary.

    Example
    -------
    >>> split_by_alternatively("AAA Alternatively BBB Alternatively CCC")
    ['AAA ', ' BBB ', ' CCC']
    """
    # Use a regex that keeps the split points but we manually reconstruct
    parts = re.split(r'(Alternatively)', text)
    # parts looks like ['AAA ', 'Alternatively', ' BBB ', 'Alternatively', ' CCC']
    # We want segments: ['AAA ', ' BBB ', ' CCC']
    segments = []
    for i, p in enumerate(parts):
        if p == 'Alternatively':
            continue
        segments.append(p)
    return segments


def count_alternatively(text: str) -> int:
    """Count occurrences of 'Alternatively' in text."""
    return len(re.findall(r'Alternatively', text))


# ---------------------------------------------------------------------------
# Chat template formatting
# ---------------------------------------------------------------------------

def format_chat_template(problem: str, previous_reasoning: str) -> str:
    """
    Manually formulate a chat template prompt from the problem and the
    reasoning generated so far.

    Customize this function to match your model's chat template (e.g.
    Llama-3, Qwen-2.5, DeepSeek, Mistral, etc.).

    The returned string is the **raw text** that will be sent to the sglang
    ``/generate`` endpoint.  The server will continue generating from the
    end of this text (i.e. ``s += gen()``).

    Parameters
    ----------
    problem : str
        The user/problem turn content.
    previous_reasoning : str
        The assistant's reasoning produced so far (the prefix we want the
        model to continue from).

    Returns
    -------
    str
        A fully formatted prompt string ready for the ``/generate`` endpoint.

    Example  (Llama-3 style)
    -------
    >>> prompt = format_chat_template(
    ...     "Solve x^2 = 4",
    ...     "We can take the square root of both sides."
    ... )
    >>> print(prompt)  # doctest: +SKIP
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Solve x^2 = 4<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    We can take the square root of both sides.
    """
    # ---------------------------------------------------------------
    # TODO: Adapt the template below to your model's chat format.
    #       The default uses a generic ChatML-style template.
    # ---------------------------------------------------------------

    # --- Generic ChatML template ---
    s = ""
    s += "<|im_start|>user\n"
    s += problem
    s += "<|im_end|>\n"
    s += "<|im_start|>assistant\n"
    s += previous_reasoning  + "Alternatively"# model will continue from here via gen()
    return s


def format_chat_template_from_messages(
    messages: list,
    assistant_prefix: str,
) -> str:
    """
    Build a raw prompt from a list of message dicts and an assistant prefix.

    Iterates over prior messages and appends the partial assistant response
    so the sglang server can continue generation (``s += gen()``).

    Parameters
    ----------
    messages : list
        Conversation turns *before* the assistant message, each a dict
        with ``role`` and ``content``.
    assistant_prefix : str
        The assistant text generated so far.

    Returns
    -------
    str
        Raw prompt string.
    """
    # Extract the user/problem content from messages
    problem_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            problem_parts.append(f"[System] {content}")
        elif role == "user":
            problem_parts.append(content)
        # skip other roles for the problem text

    problem = "\n".join(problem_parts)
    return format_chat_template(problem, assistant_prefix)


# ---------------------------------------------------------------------------
# Generation via sglang native /generate endpoint  (s += gen() style)
# ---------------------------------------------------------------------------

def generate_branch(
    sglang_url: str,
    messages_prefix: list,
    assistant_prefix: str,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    stop_words: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> str:
    """
    Call the sglang server's native ``/generate`` endpoint to continue
    generation from a hand-crafted prompt.

    Conceptually this mirrors the sglang frontend pattern::

        @sgl.function
        def branch(s, problem, previous_reasoning):
            s += format_chat_template(problem, previous_reasoning)
            s += sgl.gen("continuation",
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop="Alternatively")

    Parameters
    ----------
    sglang_url : str
        Base URL of the sglang server, e.g. ``http://localhost:30000``.
    messages_prefix : list
        The conversation messages *before* the assistant message.
    assistant_prefix : str
        The assistant's content generated so far (prefix to continue from).
    max_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.
    stop_words : list[str] | None
        Stop strings (defaults to ``["Alternatively"]``).
    model : str | None
        Not used for the native endpoint but kept for API compatibility.

    Returns
    -------
    str
        The generated continuation text.
    """
    if stop_words is None:
        stop_words = ["Alternatively"]

    # -- Build the raw prompt using our hand-crafted chat template ----------
    #    (mirrors:  s += format_chat_template(...)  then  s += gen(...)  )
    prompt = format_chat_template_from_messages(messages_prefix, assistant_prefix)
    print(f"[DEBUG] Generated prompt for sglang /generate:\n{prompt}\n")
    # -- Call sglang native /generate endpoint ------------------------------
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop_words,
        },
    }

    url = f"{sglang_url}/generate"
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        result = resp.json()
        # sglang /generate returns {"text": "<generated>"}
        generated = result["text"]
        print(f"[DEBUG] Received generated text from sglang:\n{generated}\n")
        return "Alternatively" + generated
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to sglang server at %s", sglang_url)
        raise
    except Exception as e:
        logger.error("Generation failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

def process_assistant_content(
    user_messages: list,
    assistant_content: str,
    sglang_url: str,
    num_branches: int = 2,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> str:
    """
    Process a single assistant response:
    1. Find all 'Alternatively' occurrences.
    2. For the first n-1 of them, generate alternative branches.
    3. Wrap them in <Parallel><Trial>...</Trial></Parallel>.

    Parameters
    ----------
    user_messages : list
        All messages before the assistant message (user, system, etc.).
    assistant_content : str
        The original assistant response text.
    sglang_url : str
        sglang server URL.
    num_branches : int
        Number of alternative branches to generate per Alternatively block.
    max_tokens : int
        Max tokens per generation.
    temperature : float
        Sampling temperature.
    model : str | None
        Model name.

    Returns
    -------
    str
        Modified assistant content with <Parallel>/<Trial> tags.
    """
    n = count_alternatively(assistant_content)

    if n <= 1:
        # 0 or 1 "Alternatively" → nothing to branch
        return assistant_content

    # Split into segments
    segments = split_by_alternatively(assistant_content)
    # segments has n+1 elements:
    #   segments[0]  = content before 1st Alternatively
    #   segments[i]  = content after i-th Alternatively, before (i+1)-th
    #   segments[n]  = content after last Alternatively

    # We process the first n-1 Alternatively blocks (indices 1 to n-1)
    # The n-th Alternatively (last one) is left as-is.

    # Build the result incrementally
    result_parts = [segments[0]]  # start with content before first Alternatively

    for i in range(1, n):  # i = 1 .. n-1 → first n-1 Alternatively blocks
        # Original block: "Alternatively" + segments[i]
        original_block = "Alternatively" + segments[i]

        # Build the assistant prefix for generation:
        # Everything from the original content up to (but not including) this
        # Alternatively keyword.
        # prefix = segments[0] + "Alternatively" + segments[1] + ... + segments[i-1]
        prefix_parts = [segments[0]]
        for j in range(1, i):
            prefix_parts.append("Alternatively")
            prefix_parts.append(segments[j])
        assistant_prefix = "".join(prefix_parts)

        # Generate alternative branches
        branches = []
        for b in range(num_branches):
            try:
                generated = generate_branch(
                    sglang_url=sglang_url,
                    messages_prefix=user_messages,
                    assistant_prefix=assistant_prefix,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_words=["Alternatively"],
                    model=model,
                )
                branches.append(generated.strip())
            except Exception as e:
                logger.warning(
                    "Failed to generate branch %d for Alternatively #%d: %s",
                    b + 1, i, e,
                )
                # Skip this branch on failure
                continue

        # Build the <Parallel>...</Parallel> block
        trial_parts = [f"<Trial>{original_block}</Trial>"]
        for branch_text in branches:
            trial_parts.append(f"<Trial>{branch_text}</Trial>")

        parallel_block = "<Parallel>" + "".join(trial_parts) + "</Parallel>"
        result_parts.append(parallel_block)

    # Append the last Alternatively block (the n-th one) as-is
    result_parts.append("Alternatively" + segments[n])

    return "".join(result_parts)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def detect_model(sglang_url: str) -> Optional[str]:
    """Try to detect the model name from the sglang server."""
    try:
        resp = requests.get(f"{sglang_url}/v1/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        model = data["data"][0]["id"]
        logger.info("Detected model: %s", model)
        return model
    except Exception as e:
        logger.warning("Could not detect model from server: %s", e)
        return None


def _process_single_line(
    idx: int,
    line: str,
    sglang_url: str,
    num_branches: int,
    max_tokens: int,
    temperature: float,
    model: Optional[str],
) -> Tuple[int, str, bool]:
    """
    Process a single JSONL line.  Returns (idx, output_json_line, was_processed).

    This function is designed to be called from a thread pool.
    """
    line = line.strip()
    if not line:
        return (idx, "", False)

    data = json.loads(line)
    messages = data.get("messages", [])

    # Find the assistant message
    assistant_idx = None
    for mi, msg in enumerate(messages):
        if msg["role"] == "assistant":
            assistant_idx = mi
            break

    if assistant_idx is None:
        return (idx, json.dumps(data, ensure_ascii=False) + "\n", False)

    assistant_content = messages[assistant_idx]["content"]
    n_alt = count_alternatively(assistant_content)

    if n_alt <= 1:
        return (idx, json.dumps(data, ensure_ascii=False) + "\n", False)

    # Messages before the assistant message (for generation context)
    user_messages = messages[:assistant_idx]

    was_processed = False
    try:
        new_content = process_assistant_content(
            user_messages=user_messages,
            assistant_content=assistant_content,
            sglang_url=sglang_url,
            num_branches=num_branches,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )
        messages[assistant_idx]["content"] = new_content
        data["messages"] = messages
        was_processed = True
    except Exception as e:
        logger.error("[%d] Processing failed: %s", idx + 1, e)
        # Keep original on failure

    return (idx, json.dumps(data, ensure_ascii=False) + "\n", was_processed)


def process_file(
    input_path: str,
    output_path: str,
    sglang_url: str,
    num_branches: int = 2,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    max_workers: int = 32,
):
    """Process the entire JSONL file using multithreading."""
    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("sglang URL: %s", sglang_url)
    logger.info("Branches per Alternatively: %d", num_branches)
    logger.info("Max tokens: %d", max_tokens)
    logger.info("Temperature: %.2f", temperature)
    logger.info("Max workers (concurrent requests): %d", max_workers)

    # Detect model
    model = detect_model(sglang_url)

    # Read all lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    logger.info("Total samples: %d", total)

    # Pre-allocate results array to preserve ordering
    results = [None] * total
    processed = 0
    skipped = 0
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, line in enumerate(lines):
            future = executor.submit(
                _process_single_line,
                idx,
                line,
                sglang_url,
                num_branches,
                max_tokens,
                temperature,
                model,
            )
            futures[future] = idx

        for future in as_completed(futures):
            try:
                line_idx, output_line, was_processed = future.result()
                results[line_idx] = output_line
                with lock:
                    completed += 1
                    if was_processed:
                        processed += 1
                    else:
                        skipped += 1
                    if completed % 10 == 0:
                        logger.info(
                            "[%d/%d] Processed=%d  Skipped=%d",
                            completed, total, processed, skipped,
                        )
            except Exception as e:
                line_idx = futures[future]
                logger.error("[%d] Future failed: %s", line_idx + 1, e)
                # Write original line on failure
                results[line_idx] = lines[line_idx].strip() + "\n" if lines[line_idx].strip() else ""
                with lock:
                    completed += 1
                    skipped += 1

    # Write results in original order
    with open(output_path, "w", encoding="utf-8") as fout:
        for result in results:
            if result:  # skip empty lines
                fout.write(result)

    logger.info("Done! Processed=%d  Skipped=%d  Total=%d", processed, skipped, total)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Add parallel branches at 'Alternatively' keywords in assistant responses."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../train.jsonl",
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../train_branched.jsonl",
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--sglang_url",
        type=str,
        default="http://localhost:30000",
        help="URL of the sglang server.",
    )
    parser.add_argument(
        "--num_branches",
        type=int,
        default=2,
        help="Number of alternative branches to generate per Alternatively block (hyperparameter).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per branch.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Maximum number of concurrent requests to the sglang server.",
    )

    args = parser.parse_args()
    process_file(
        input_path=args.input,
        output_path=args.output,
        sglang_url=args.sglang_url,
        num_branches=args.num_branches,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
