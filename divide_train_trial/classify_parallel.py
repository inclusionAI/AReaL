"""
classify_parallel.py
--------------------
Reads train_new_special_tok.jsonl, finds every <Parallel>…</Parallel> block in the
`qwen_text` field (configurable via --field), asks the LLM whether each block is a
"Trial" or a "Subtask", and rewrites the <Outline>/<Outline> tags accordingly.

Output is written to  train_classified.jsonl  in the same directory.

Usage:
    python classify_parallel.py [--input FILE] [--output FILE]
                                [--workers N] [--model NAME]
                                [--field FIELD_NAME] [--dry-run]
"""

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from api_call import call_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
CLASSIFY_PROMPT = """\
You are an expert at analysing structured reasoning traces.

Below is the content of a single <Parallel> block extracted from a mathematical reasoning process.
It contains a <Goal> section with multiple <Outline> entries that describe different approaches, \
and <Path> sections that actually work through those approaches.

Your task is to classify whether this parallel block represents:

- **Trial**: The paths are independent *attempts* or *explorations* of the same goal, \
  where each path tries a different strategy but they are all working toward the same answer. \
  Typically, the paths explore different approaches and may converge to the same result.

- **Subtask**: The paths are *complementary sub-problems* that together constitute the solution. \
  Each path addresses a distinct part of the overall problem, and their results must be \
  combined to reach the final answer.

Respond with ONLY one word: either  Trial  or  Subtask.

<Parallel>
{parallel_content}
</Parallel>
"""

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_OUTLINE_OPEN_RE = re.compile(r"<Outline>")
_OUTLINE_CLOSE_RE = re.compile(r"</Outline>")

_TAG_RE = re.compile(r"<(/?)Parallel>")


def extract_parallel_blocks(text: str):
    """Return list of (match_start, match_end, full_match, inner_content).

    Uses a depth-tracking scan so that nested <Parallel> blocks are handled
    correctly: only the *outermost* <Parallel>…</Parallel> spans are returned,
    just like a proper bracket matcher.  The naive non-greedy regex
      r"(<Parallel>)(.*?)(</Parallel>)"
    stops at the *first* </Parallel> it finds, so it mis-parses nested blocks
    and leaves outer <Outline> tags unreplaced.
    """
    results = []
    depth = 0
    start = None
    for m in _TAG_RE.finditer(text):
        if not m.group(1):          # opening tag
            if depth == 0:
                start = m.start()
            depth += 1
        else:                       # closing tag
            depth -= 1
            if depth == 0 and start is not None:
                end = m.end()
                full_match = text[start:end]
                inner = full_match[len("<Parallel>"):-len("</Parallel>")]
                results.append((start, end, full_match, inner))
                start = None
    return results


def replace_outline_tags(parallel_inner: str, label: str) -> str:
    """Replace <Outline>/<Outline> with <Trial>/<Trial> or <Subtask>/<Subtask>."""
    open_tag  = f"<{label}>"
    close_tag = f"</{label}>"
    result = _OUTLINE_OPEN_RE.sub(open_tag,  parallel_inner)
    result = _OUTLINE_CLOSE_RE.sub(close_tag, result)
    return result


def parse_llm_label(response_text: str) -> str | None:
    """Extract 'Trial' or 'Subtask' from the LLM response text."""
    if response_text is None:
        return None
    cleaned = response_text.strip().split()[0].rstrip(".,;:").capitalize()
    if cleaned in ("Trial", "Subtask"):
        return cleaned
    # Fallback: search anywhere in the response
    lower = response_text.lower()
    if "subtask" in lower:
        return "Subtask"
    if "trial" in lower:
        return "Trial"
    return None


def build_prompt(parallel_inner: str) -> str:
    return CLASSIFY_PROMPT.format(parallel_content=parallel_inner)


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(record: dict, model: str, dry_run: bool, field: str = "qwen_text") -> dict:
    """Classify all <Parallel> blocks in one JSONL record and rewrite the tags."""
    text: str = record.get(field, "")
    blocks = extract_parallel_blocks(text)

    if not blocks:
        return record  # nothing to do

    # ---- Build all prompts for this record --------------------------------
    prompts = []   # list of (block_index, parallel_inner)
    for idx, (_, _, _, inner) in enumerate(blocks):
        prompts.append((idx, inner))

    # ---- Call LLM (sequentially within this record; parallelism is across records)
    labels: dict[int, str] = {}
    for idx, inner in prompts:
        if dry_run:
            label = "Trial"   # placeholder
        else:
            prompt = build_prompt(inner)
            result = call_model(prompt, model=model, max_tokens=500)
            if result is None:
                logger.warning("LLM call failed for block %d; defaulting to 'Trial'", idx)
                label = "Trial"
            else:
                raw = result["choices"][0]["message"]["content"]
                label = parse_llm_label(raw)
                if label is None:
                    logger.warning(
                        "Could not parse label from response %r; defaulting to 'Trial'", raw
                    )
                    label = "Trial"
        labels[idx] = label
        logger.debug("Block %d → %s", idx, label)

    # ---- Rewrite the text (process from right to left to preserve offsets) --
    modified = text
    # Collect (start, end, replacement) sorted by start descending
    replacements = []
    for idx, (start, end, full_match, inner) in enumerate(blocks):
        label = labels[idx]
        new_inner = replace_outline_tags(inner, label)
        new_block = f"<Parallel>{new_inner}</Parallel>"
        replacements.append((start, end, new_block))

    # Apply replacements right-to-left so earlier offsets stay valid
    for start, end, new_block in sorted(replacements, key=lambda x: x[0], reverse=True):
        modified = modified[:start] + new_block + modified[end:]

    new_record = dict(record)
    new_record[field] = modified
    return new_record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classify <Parallel> blocks as Trial or Subtask.")
    parser.add_argument("--input",   default="train_new_special_tok.jsonl",
                        help="Input JSONL file (default: train_new_special_tok.jsonl)")
    parser.add_argument("--output",  default="train_classified.jsonl",
                        help="Output JSONL file (default: train_classified.jsonl)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel worker threads (default: 2)")
    parser.add_argument("--model",   default="gemini-2.5-flash",
                        help="LLM model name (default: gemini-2.5-flash)")
    parser.add_argument("--field",   default="qwen_text",
                        help="JSONL field to process (default: qwen_text)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip real API calls; label everything as 'Trial'")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # ---- Load all records --------------------------------------------------
    logger.info("Loading records from %s …", input_path)
    records = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records.", len(records))

    # ---- Process in parallel threads (one thread per record) ---------------
    processed: dict[int, dict] = {}

    def _worker(idx: int, record: dict):
        result = process_record(record, model=args.model, dry_run=args.dry_run, field=args.field)
        return idx, result

    logger.info(
        "Processing with %d worker threads%s (field: %s) …",
        args.workers,
        " [DRY RUN]" if args.dry_run else "",
        args.field,
    )

    completed = 0
    total = len(records)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, i, r): i for i, r in enumerate(records)}
        for future in as_completed(futures):
            idx, result = future.result()
            processed[idx] = result
            completed += 1
            if completed % 1 == 0 or completed == total:
                logger.info("Progress: %d / %d", completed, total)

    # ---- Write output in original order ------------------------------------
    logger.info("Writing output to %s …", output_path)
    with output_path.open("w", encoding="utf-8") as fh:
        for i in range(total):
            fh.write(json.dumps(processed[i], ensure_ascii=False) + "\n")

    logger.info("Done. Output written to %s", output_path)


if __name__ == "__main__":
    main()
