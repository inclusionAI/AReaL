"""
Filter similar branches from the branched JSONL dataset produced by process_branches.py.

For each <Parallel> block, the first <Trial> is the original branch.
All other <Trial> blocks are compared to the first one using an LLM.
Branches deemed too similar to the original are removed.

Outputs two versions:
1. Special tokens (<Parallel>, </Parallel>, <Trial>, </Trial>) preserved,
   similar branches deleted.
2. Special tokens stripped, similar branches deleted (plain text).

Usage:
    python filter_similar_branches.py \
        --input ../train_branched.jsonl \
        --output_with_tokens ../train_filtered_with_tokens.jsonl \
        --output_without_tokens ../train_filtered_no_tokens.jsonl
"""

import argparse
import copy
import json
import logging
import re
import sys
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM API (placeholder — fill in your implementation)
# ---------------------------------------------------------------------------

def call_llm_api(system_prompt: str, user_prompt: str) -> str:
    """
    Call an LLM API and return the model's text response.

    Parameters
    ----------
    system_prompt : str
        The system / instruction prompt.
    user_prompt : str
        The user-turn prompt containing the data to evaluate.

    Returns
    -------
    str
        The model's raw text response.

    TODO: Implement with your preferred API (OpenAI, Anthropic, Together, etc.)
    Example skeleton for OpenAI:

        import openai
        client = openai.OpenAI(api_key="...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    """
    raise NotImplementedError(
        "call_llm_api() is not implemented. "
        "Please fill in your LLM API call in filter_similar_branches.py."
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_trials(parallel_inner: str) -> List[str]:
    """
    Extract every <Trial>…</Trial> content from the *inner* text of a
    <Parallel> block (i.e. the string between <Parallel> and </Parallel>).

    Returns a list of raw string contents, *without* the Trial tags.
    """
    return re.findall(r"<Trial>(.*?)</Trial>", parallel_inner, re.DOTALL)


def iter_parallel_blocks(text: str):
    """
    Yield (start, end, inner_content) for every <Parallel>…</Parallel>
    block found in *text*.
    """
    for m in re.finditer(r"<Parallel>(.*?)</Parallel>", text, re.DOTALL):
        yield m.start(), m.end(), m.group(1)


# ---------------------------------------------------------------------------
# LLM-based similarity judgment
# ---------------------------------------------------------------------------

_SIMILARITY_SYSTEM_PROMPT = """\
You are an expert at evaluating the diversity of reasoning approaches in \
mathematical and logical problem-solving.

Your task: given an original reasoning branch and one or more alternative \
branches, identify which alternatives are *too similar* to the original.

Definition of "similar":
- Uses the same core method, technique, or algorithmic approach.
- Arrives at the answer through essentially identical steps.
- Is a minor paraphrase, reordering, or notational variant of the original.

Definition of "different" (keep these):
- Applies a genuinely distinct method or perspective.
- Introduces new reasoning steps not present in the original.
- Adds meaningful value by approaching the problem differently.

Respond with ONLY a valid JSON array of 0-based indices into the list of \
alternatives (NOT counting the original).
- Index 0 refers to Alternative 0, index 1 to Alternative 1, and so on.
- If no alternatives are similar: []
- Example where alternatives 0 and 2 are similar: [0, 2]
Do NOT include any explanation or extra text — only the JSON array."""


def find_similar_indices(trials: List[str]) -> List[int]:
    """
    Send all branches to the LLM and return the 0-based indices (into
    ``trials[1:]``) of alternatives that are too similar to the original
    (``trials[0]``).

    Returns an empty list if there is nothing to compare or if the LLM
    call fails.
    """
    if len(trials) <= 1:
        return []

    original = trials[0]
    alternatives = trials[1:]

    lines = ["Original branch:\n" + original.strip()]
    for i, alt in enumerate(alternatives):
        lines.append(f"\nAlternative {i}:\n" + alt.strip())

    user_prompt = (
        "Below is the original reasoning branch followed by the alternatives.\n"
        "Identify which alternatives (by 0-based index) are too similar to "
        "the original.\n\n"
        + "\n".join(lines)
        + "\n\nRespond with ONLY a JSON array of 0-based indices."
    )

    try:
        response = call_llm_api(_SIMILARITY_SYSTEM_PROMPT, user_prompt)

        # Extract the first JSON array from the response
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if not match:
            logger.warning(
                "LLM response contained no JSON array; keeping all branches. "
                "Response: %s",
                response[:300],
            )
            return []

        raw = json.loads(match.group())
        # Validate elements: must be integers within range
        valid = [
            idx
            for idx in raw
            if isinstance(idx, int) and 0 <= idx < len(alternatives)
        ]
        if len(valid) != len(raw):
            logger.warning(
                "Some LLM-returned indices were out of range and were ignored. "
                "Raw=%s  Valid=%s",
                raw,
                valid,
            )
        return valid

    except NotImplementedError:
        raise  # propagate so the caller can abort early
    except Exception as exc:
        logger.warning("LLM similarity check failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Per-block filtering
# ---------------------------------------------------------------------------

def filter_parallel_block(
    parallel_inner: str,
) -> Tuple[str, str]:
    """
    Filter similar branches out of a single <Parallel> block.

    Parameters
    ----------
    parallel_inner : str
        The raw text *between* ``<Parallel>`` and ``</Parallel>``.

    Returns
    -------
    with_tokens : str
        The full ``<Parallel>…</Parallel>`` block with similar ``<Trial>``
        entries removed but all tags kept.
    without_tokens : str
        The kept trial *contents* joined by newlines, with all tags stripped.
    """
    trials = parse_trials(parallel_inner)

    if not trials:
        # Malformed block — return as-is
        return f"<Parallel>{parallel_inner}</Parallel>", parallel_inner

    if len(trials) == 1:
        # Nothing to compare
        return f"<Parallel><Trial>{trials[0]}</Trial></Parallel>", trials[0]

    # Ask the LLM which alternatives are too similar to the original
    similar_set = set(find_similar_indices(trials))
    logger.debug("Similar indices for this block: %s", sorted(similar_set))

    # Always keep the original (trials[0]); drop similar alternatives
    kept: List[str] = [trials[0]]
    for i, trial_content in enumerate(trials[1:]):
        if i not in similar_set:
            kept.append(trial_content)

    n_removed = len(trials) - len(kept)
    if n_removed:
        logger.debug("Removed %d similar branch(es) from block.", n_removed)

    # Version 1 — preserve tags
    trial_tags = "".join(f"<Trial>{t}</Trial>" for t in kept)
    with_tokens = f"<Parallel>{trial_tags}</Parallel>"

    # Version 2 — strip all tags; join kept contents
    without_tokens = "\n".join(t.strip() for t in kept)

    return with_tokens, without_tokens


# ---------------------------------------------------------------------------
# Full assistant-content processing
# ---------------------------------------------------------------------------

def process_assistant_content(content: str) -> Tuple[str, str]:
    """
    Process a complete assistant response string.

    Finds every ``<Parallel>…</Parallel>`` block, filters similar branches
    via the LLM, and reconstructs two variants of the full text.

    Returns
    -------
    version_with_tokens : str
        Full content with similar branches removed but tags preserved.
    version_without_tokens : str
        Full content with similar branches removed *and* all
        ``<Parallel>``/``<Trial>`` tags stripped.
    """
    blocks = list(iter_parallel_blocks(content))

    if not blocks:
        return content, content

    parts_tok: List[str] = []
    parts_notok: List[str] = []

    prev_end = 0
    for start, end, inner in blocks:
        # Literal text before this block
        before = content[prev_end:start]
        parts_tok.append(before)
        parts_notok.append(before)

        with_tok, without_tok = filter_parallel_block(inner)
        parts_tok.append(with_tok)
        parts_notok.append(without_tok)

        prev_end = end

    # Trailing text after the last block
    tail = content[prev_end:]
    parts_tok.append(tail)
    parts_notok.append(tail)

    return "".join(parts_tok), "".join(parts_notok)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_file(
    input_path: str,
    output_with_tokens_path: str,
    output_without_tokens_path: str,
) -> None:
    """
    Read the branched JSONL file, filter similar branches, and write two
    output files.

    Parameters
    ----------
    input_path : str
        Path to the input JSONL (output of process_branches.py).
    output_with_tokens_path : str
        Output JSONL — similar branches removed, special tokens preserved.
    output_without_tokens_path : str
        Output JSONL — similar branches removed, all special tokens stripped.
    """
    logger.info("Input:                   %s", input_path)
    logger.info("Output (with tokens):    %s", output_with_tokens_path)
    logger.info("Output (without tokens): %s", output_without_tokens_path)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    logger.info("Total lines: %d", total)

    processed = skipped = 0

    with (
        open(output_with_tokens_path, "w", encoding="utf-8") as f_tok,
        open(output_without_tokens_path, "w", encoding="utf-8") as f_notok,
    ):
        for idx, raw_line in enumerate(lines):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            data = json.loads(raw_line)
            messages = data.get("messages", [])

            # Locate the assistant message
            assistant_idx: Optional[int] = None
            for mi, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    assistant_idx = mi
                    break

            # No assistant turn or no Parallel blocks → pass through unchanged
            if assistant_idx is None or "<Parallel>" not in messages[assistant_idx]["content"]:
                out = json.dumps(data, ensure_ascii=False) + "\n"
                f_tok.write(out)
                f_notok.write(out)
                skipped += 1
                continue

            assistant_content = messages[assistant_idx]["content"]

            try:
                content_tok, content_notok = process_assistant_content(assistant_content)

                # Deep-copy to avoid aliasing between the two output dicts
                data_tok = copy.deepcopy(data)
                data_tok["messages"][assistant_idx]["content"] = content_tok

                data_notok = copy.deepcopy(data)
                data_notok["messages"][assistant_idx]["content"] = content_notok

                f_tok.write(json.dumps(data_tok, ensure_ascii=False) + "\n")
                f_notok.write(json.dumps(data_notok, ensure_ascii=False) + "\n")
                processed += 1

            except NotImplementedError:
                logger.error(
                    "call_llm_api() is not implemented. "
                    "Please add your API code before running this script."
                )
                sys.exit(1)
            except Exception as exc:
                logger.error("[line %d] Processing failed: %s", idx + 1, exc)
                fallback = json.dumps(data, ensure_ascii=False) + "\n"
                f_tok.write(fallback)
                f_notok.write(fallback)
                skipped += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                logger.info(
                    "[%d/%d] processed=%d  skipped=%d",
                    idx + 1, total, processed, skipped,
                )

    logger.info(
        "Finished. processed=%d  skipped=%d  total=%d",
        processed, skipped, total,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter similar branches from a branched JSONL dataset. "
            "Produces two output files: one with special tokens preserved "
            "and one with all special tokens stripped."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../train_branched.jsonl",
        help="Input branched JSONL file (output of process_branches.py).",
    )
    parser.add_argument(
        "--output_with_tokens",
        type=str,
        default="../train_filtered_with_tokens.jsonl",
        help="Output JSONL: similar branches removed, special tokens kept.",
    )
    parser.add_argument(
        "--output_without_tokens",
        type=str,
        default="../train_filtered_no_tokens.jsonl",
        help="Output JSONL: similar branches removed, special tokens stripped.",
    )
    args = parser.parse_args()

    process_file(
        input_path=args.input,
        output_with_tokens_path=args.output_with_tokens,
        output_without_tokens_path=args.output_without_tokens,
    )


if __name__ == "__main__":
    main()
