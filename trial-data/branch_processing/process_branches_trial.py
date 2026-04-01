"""
Process a JSONL file (input/output format) to add parallel branches at <Trial>
tags inside <Goal> blocks of <Parallel> stages.

Dataset format:
  {"input": "...", "output": "..."}

The output contains one or more <Parallel> blocks structured as:

  [Content Before]
  <Parallel>
  <Goal>
  <Trial>
  1. [Goal description 1]
  </Trial>
  <Trial>
  2. [Goal description 2]
  </Trial>
  ...
  </Goal>
  <Path>
  1. ...
  </Path>
  <Path>
  2. ...
  </Path>
  ...
  <Conclusion>
  ...
  </Conclusion>
  </Parallel>

For each <Parallel> block whose <Goal> uses <Trial> tags (not <Subtask>):
  1. Strip ALL special tokens and their enclosed content from the output text
     that precedes this stage to form a clean "previous_reasoning" prefix.
  2. For each Trial in the <Goal>, call the model with:
       - The original input (user + system turns)
       - previous_reasoning prefilled, then "Alternatively" appended
       - Stop generation at the next "Alternatively"
  3. Append the generated text as new <Path> entries to the <Parallel> block.
  4. Append a blank <Trial> entry for each generated branch.

Usage:
    python process_branches_trial.py \\
        --input  ../parallel-only-1-conv-prompt-with-original-modified.jsonl \\
        --output ../parallel-branched-output.jsonl \\
        --num_branches 2 \\
        --sglang_url http://localhost:30000 \\
        --max_tokens 2048 \\
        --temperature 0.7
"""

import argparse
import json
import re
import sys
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
# Regex helpers
# ---------------------------------------------------------------------------

# Matches an entire <Parallel>...</Parallel> block (non-greedy, DOTALL)
_RE_PARALLEL = re.compile(r"<Parallel>(.*?)</Parallel>", re.DOTALL)

# Matches a <Goal>...</Goal> block inside a Parallel
_RE_GOAL = re.compile(r"<Goal>(.*?)</Goal>", re.DOTALL)

# Matches individual <Trial>...</Trial> entries inside a Goal
_RE_TRIAL = re.compile(r"<Trial>(.*?)</Trial>", re.DOTALL)

# Matches individual <Subtask>...</Subtask> entries inside a Goal
_RE_SUBTASK = re.compile(r"<Subtask>(.*?)</Subtask>", re.DOTALL)

# Tags whose content should be stripped when building the clean prefix
# (all special parallel-reasoning tokens)
_SPECIAL_TAG_PATTERN = re.compile(
    r"<(?:Parallel|Goal|Trial|Subtask|Path|Conclusion)>.*?</(?:Parallel|Goal|Trial|Subtask|Path|Conclusion)>",
    re.DOTALL,
)

# The trigger phrase inserted between parallel blocks (kept in clean prefix)
_PARALLEL_TRIGGER = re.compile(r"Let's think in parallel\s*\n?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Text-cleaning helpers
# ---------------------------------------------------------------------------

def strip_special_tokens(text: str) -> str:
    """
    Remove all content inside special parallel-reasoning tags
    (Parallel, Goal, Trial, Subtask, Path, Conclusion) **and** their tags,
    as well as the "Let's think in parallel" trigger phrase.

    The result is the plain reasoning text without any scaffolding.
    """
    cleaned = _SPECIAL_TAG_PATTERN.sub("", text)
    cleaned = _PARALLEL_TRIGGER.sub("", cleaned)
    # Collapse runs of blank lines left by removals
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def text_before_parallel_block(full_output: str, parallel_start: int) -> str:
    """
    Return the output text that appears *before* the given character position
    (i.e. before the <Parallel> opening tag), then strip all special tokens
    from that prefix.

    Parameters
    ----------
    full_output : str
        The complete assistant output string.
    parallel_start : int
        Character index of the '<' in the <Parallel> tag we are processing.

    Returns
    -------
    str
        Clean reasoning prefix (no special tokens, no enclosed content).
    """
    raw_prefix = full_output[:parallel_start]
    return strip_special_tokens(raw_prefix)


# ---------------------------------------------------------------------------
# Goal-type detection
# ---------------------------------------------------------------------------

def goal_uses_trials(goal_content: str) -> bool:
    """Return True if the <Goal> block uses <Trial> tags (not <Subtask>)."""
    has_trial = bool(_RE_TRIAL.search(goal_content))
    has_subtask = bool(_RE_SUBTASK.search(goal_content))
    # Process only when Trial tags are present (and Subtask are not,
    # or when Trial are explicitly present regardless)
    return has_trial and not has_subtask


# ---------------------------------------------------------------------------
# Chat-template formatting
# ---------------------------------------------------------------------------

def format_prompt(input_text: str, previous_reasoning: str) -> str:
    """
    Build a raw prompt from the input string (which already contains the
    full chat template header, e.g. <|im_start|>user ... <|im_end|>
    <|im_start|>assistant\\n) and the assistant reasoning produced so far.

    We append "Alternatively" at the end so the model continues from there.

    Conceptually mirrors the sglang frontend pattern::

        @sgl.function
        def branch(s, input_text, previous_reasoning):
            s += input_text
            s += previous_reasoning
            s += sgl.gen("continuation",
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop="Alternatively")

    Parameters
    ----------
    input_text : str
        The raw ``input`` field from the JSONL entry, typically ending with
        ``<|im_start|>assistant\\n``.
    previous_reasoning : str
        Assistant text generated before the current branch point.

    Returns
    -------
    str
        The fully concatenated prompt ready for the sglang /generate endpoint.
    """
    s = ""
    s += input_text
    s += previous_reasoning
    s += "Alternatively"  # model will continue from here via gen()
    return s


# ---------------------------------------------------------------------------
# Generation via sglang /generate
# ---------------------------------------------------------------------------

def generate_branch(
    sglang_url: str,
    input_text: str,
    previous_reasoning: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    stop_words: Optional[List[str]] = None,
) -> str:
    """
    Call the sglang server's native ``/generate`` endpoint to continue
    generation from a hand-crafted prompt.

    Conceptually this mirrors the sglang frontend pattern::

        @sgl.function
        def branch(s, input_text, previous_reasoning):
            s += format_prompt(input_text, previous_reasoning)
            s += sgl.gen("continuation",
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop="Alternatively")

    Parameters
    ----------
    sglang_url : str
        Base URL of the sglang server.
    input_text : str
        The full ``input`` field from the JSONL entry.
    previous_reasoning : str
        Clean assistant reasoning up to the branch point.
    max_tokens : int
        Maximum new tokens to generate.
    temperature : float
        Sampling temperature.
    stop_words : list[str] | None
        Stop strings; defaults to ["Alternatively"].

    Returns
    -------
    str
        The generated continuation (prefixed with "Alternatively").
    """
    if stop_words is None:
        stop_words = ["Alternatively"]

    prompt = format_prompt(input_text, previous_reasoning)
    print(f"[DEBUG] Generated prompt for sglang /generate:\n{prompt}\n")

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
        generated = resp.json()["text"]
        print(f"[DEBUG] Received generated text from sglang:\n{generated}\n")
        return "Alternatively" + generated
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to sglang server at %s", sglang_url)
        raise
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Parallel-block processing
# ---------------------------------------------------------------------------

def process_parallel_block(
    parallel_content: str,
    input_text: str,
    previous_reasoning: str,
    sglang_url: str,
    num_branches: int,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Given the *inner* content of a <Parallel> block (everything between the
    outer tags), generate additional branches and return the modified inner
    content.

    Strategy
    --------
    * Parse the <Goal> section to find all <Trial> entries.
    * For each Trial, call the model ``num_branches`` times to get alternative
      continuations.
    * Append ``num_branches`` new blank <Trial> tags to the <Goal> section.
    * Append ``num_branches`` new <Path> blocks filled with the generated text.

    Parameters
    ----------
    parallel_content : str
        Everything between ``<Parallel>`` and ``</Parallel>``.
    input_text : str
        The original ``input`` field (full chat prompt up to assistant turn).
    previous_reasoning : str
        Clean assistant text before this <Parallel> stage.
    sglang_url : str
    num_branches : int
    max_tokens : int
    temperature : float

    Returns
    -------
    str
        The modified inner content.
    """
    goal_match = _RE_GOAL.search(parallel_content)
    if goal_match is None:
        return parallel_content

    goal_content = goal_match.group(1)

    # Only process Goal blocks that use <Trial> tags
    if not goal_uses_trials(goal_content):
        return parallel_content

    trial_matches = _RE_TRIAL.findall(goal_content)
    n_trials = len(trial_matches)

    if n_trials == 0:
        return parallel_content

    # -----------------------------------------------------------------------
    # Generate new branches
    # Each Trial in the Goal represents one reasoning direction; we use the
    # previous_reasoning as the common prefix and generate num_branches new
    # alternative continuations.
    # -----------------------------------------------------------------------
    new_paths: List[str] = []

    for branch_idx in range(num_branches):
        try:
            generated = generate_branch(
                sglang_url=sglang_url,
                input_text=input_text,
                previous_reasoning=previous_reasoning,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            new_paths.append(generated.strip())
            logger.info("  Generated branch %d/%d", branch_idx + 1, num_branches)
        except Exception as exc:
            logger.warning("  Branch %d generation failed: %s", branch_idx + 1, exc)
            new_paths.append("")  # empty fallback

    # -----------------------------------------------------------------------
    # Rebuild the Parallel block content
    # -----------------------------------------------------------------------

    # 1. Add blank <Trial> tags to the Goal section
    extra_trials = "".join(f"\n<Trial>\n</Trial>" for _ in range(num_branches))
    new_goal_content = goal_content + extra_trials
    new_goal_block = f"<Goal>{new_goal_content}</Goal>"

    # 2. Build new <Path> blocks from the generated text
    #    Count existing paths so we can number new ones correctly
    existing_paths = re.findall(r"<Path>", parallel_content)
    base_path_number = len(existing_paths) + 1

    new_path_blocks = []
    for i, path_text in enumerate(new_paths):
        path_num = base_path_number + i
        new_path_blocks.append(f"<Path>\n{path_num}. {path_text}\n</Path>")

    new_paths_str = "\n".join(new_path_blocks)

    # 3. Replace the Goal block in the inner content
    new_inner = parallel_content[: goal_match.start()] \
                + new_goal_block \
                + parallel_content[goal_match.end():]

    # 4. Find the end of the last </Path> block to insert new paths before
    #    </Conclusion> (if present) or before </Parallel> (implicit).
    conclusion_match = re.search(r"<Conclusion>", new_inner)
    if conclusion_match:
        insert_pos = conclusion_match.start()
    else:
        # Insert right at the end of the inner content
        insert_pos = len(new_inner)

    new_inner = new_inner[:insert_pos] + new_paths_str + "\n" + new_inner[insert_pos:]

    return new_inner


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------

def process_entry(
    entry: dict,
    sglang_url: str,
    num_branches: int,
    max_tokens: int,
    temperature: float,
) -> Tuple[dict, bool]:
    """
    Process a single JSONL entry.

    Parameters
    ----------
    entry : dict
        Parsed JSON object with at least ``"input"`` and ``"output"`` keys.
    sglang_url, num_branches, max_tokens, temperature : as above.

    Returns
    -------
    (modified_entry, was_processed)
    """
    input_text: str = entry.get("input", "")
    output_text: str = entry.get("output", "")

    if not output_text:
        return entry, False

    # Find all <Parallel> blocks in the output
    parallel_iter = list(_RE_PARALLEL.finditer(output_text))
    if not parallel_iter:
        return entry, False

    # Check whether any Parallel block has a Trial-based Goal
    any_trial_goal = False
    for m in parallel_iter:
        goal_m = _RE_GOAL.search(m.group(1))
        if goal_m and goal_uses_trials(goal_m.group(1)):
            any_trial_goal = True
            break

    if not any_trial_goal:
        return entry, False

    # -----------------------------------------------------------------------
    # Process each <Parallel> block in order, rebuilding the output string
    # as we go (offsets shift after each replacement).
    # -----------------------------------------------------------------------
    new_output = output_text
    offset = 0  # running offset from prior replacements

    for m in parallel_iter:
        # Recompute the match in the (possibly already modified) string
        # by searching from the last insertion point.
        search_start = m.start() + offset
        # Re-find the parallel block at this position in the updated string
        rematch = _RE_PARALLEL.search(new_output, search_start)
        if rematch is None:
            break

        parallel_inner = rematch.group(1)

        # Build clean prefix (output text before this <Parallel> open tag)
        previous_reasoning = text_before_parallel_block(new_output, rematch.start())

        # Check if this block should be processed
        goal_m = _RE_GOAL.search(parallel_inner)
        if goal_m is None or not goal_uses_trials(goal_m.group(1)):
            # Skip; advance offset past this block
            offset += rematch.end() - m.end()
            continue

        logger.info(
            "Processing Parallel block at char %d (previous reasoning: %d chars)",
            rematch.start(),
            len(previous_reasoning),
        )

        new_inner = process_parallel_block(
            parallel_content=parallel_inner,
            input_text=input_text,
            previous_reasoning=previous_reasoning,
            sglang_url=sglang_url,
            num_branches=num_branches,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        replacement = f"<Parallel>{new_inner}</Parallel>"
        new_output = (
            new_output[: rematch.start()]
            + replacement
            + new_output[rematch.end():]
        )

        # Update offset: difference between replacement and original lengths
        old_len = rematch.end() - rematch.start()
        new_len = len(replacement)
        offset += new_len - old_len

    entry = dict(entry)  # shallow copy to avoid mutating caller's dict
    entry["output"] = new_output
    return entry, True


# ---------------------------------------------------------------------------
# File-level pipeline
# ---------------------------------------------------------------------------

def _process_single_line(
    idx: int,
    line: str,
    sglang_url: str,
    num_branches: int,
    max_tokens: int,
    temperature: float,
) -> Tuple[int, str, bool]:
    """Process one JSONL line; returns (idx, output_line, was_processed)."""
    line = line.strip()
    if not line:
        return idx, "", False

    try:
        entry = json.loads(line)
    except json.JSONDecodeError as exc:
        logger.error("[%d] JSON parse error: %s", idx + 1, exc)
        return idx, line + "\n", False

    try:
        new_entry, processed = process_entry(
            entry=entry,
            sglang_url=sglang_url,
            num_branches=num_branches,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        logger.error("[%d] Processing failed: %s", idx + 1, exc)
        return idx, json.dumps(entry, ensure_ascii=False) + "\n", False

    return idx, json.dumps(new_entry, ensure_ascii=False) + "\n", processed


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
    logger.info("Input:        %s", input_path)
    logger.info("Output:       %s", output_path)
    logger.info("sglang URL:   %s", sglang_url)
    logger.info("Branches:     %d", num_branches)
    logger.info("Max tokens:   %d", max_tokens)
    logger.info("Temperature:  %.2f", temperature)
    logger.info("Max workers:  %d", max_workers)

    with open(input_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    total = len(lines)
    logger.info("Total samples: %d", total)

    results = [None] * total
    processed_count = 0
    skipped_count = 0
    completed_count = 0
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
            )
            futures[future] = idx

        for future in as_completed(futures):
            try:
                line_idx, output_line, was_processed = future.result()
                results[line_idx] = output_line
                with lock:
                    completed_count += 1
                    if was_processed:
                        processed_count += 1
                    else:
                        skipped_count += 1
                    if completed_count % 10 == 0 or completed_count == total:
                        logger.info(
                            "[%d/%d] processed=%d skipped=%d",
                            completed_count,
                            total,
                            processed_count,
                            skipped_count,
                        )
            except Exception as exc:
                line_idx = futures[future]
                logger.error("[%d] Future failed: %s", line_idx + 1, exc)
                results[line_idx] = lines[line_idx].strip() + "\n" if lines[line_idx].strip() else ""
                with lock:
                    completed_count += 1
                    skipped_count += 1

    with open(output_path, "w", encoding="utf-8") as fout:
        for result in results:
            if result:
                fout.write(result)

    logger.info(
        "Done! processed=%d skipped=%d total=%d",
        processed_count,
        skipped_count,
        total,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add parallel branches at <Trial> tags inside <Goal> blocks "
            "of <Parallel> stages in an input/output JSONL dataset."
        )
    )
    parser.add_argument(
        "--input",
        default="../parallel-only-1-conv-prompt-with-original-modified.jsonl",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="../parallel-branched-trial-output.jsonl",
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--sglang_url",
        default="http://localhost:30000",
        help="Base URL of the sglang server.",
    )
    parser.add_argument(
        "--num_branches",
        type=int,
        default=2,
        help="Number of alternative branches to generate per Parallel block.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per branch.",
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
