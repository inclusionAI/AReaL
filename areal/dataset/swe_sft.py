"""SWE SFT dataset loader.

Loads SWE-bench trajectory data and converts it into progressive SFT
training pairs.  Each trajectory is split at assistant-turn boundaries
so that every pair ends with an assistant segment (assistant message +
its subsequent tool responses).

Example trajectory::

    [system, user, asst1, tool1a, tool1b, asst2, tool2, asst3]

Produces three pairs::

    Pair 1: [system, user, asst1, tool1a, tool1b]
    Pair 2: [system, user, asst1, tool1a, tool1b, asst2, tool2]
    Pair 3: [system, user, asst1, tool1a, tool1b, asst2, tool2, asst3]

In each pair, only the **last** assistant segment is trained (loss=1);
earlier assistant turns are treated as context (loss=0).

By default, pairs whose current segment contains a tool result with
``is_error=True`` are discarded.  Set ``filter_errors=False`` to keep them.
"""

import json
import os
import re

from datasets import Dataset

from areal.utils import logging

logger = logging.getLogger("SWESFTDataset")

DATASET_NUM_PROC = 16

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_thinking(content):
    """Strip ``<think>...</think>`` blocks from *content*.

    Returns:
        Cleaned content with thinking blocks removed, or the original
        content unchanged if no thinking tags are found.
    """
    if not content:
        return content
    cleaned = _THINK_RE.sub("", content).strip()
    return cleaned if cleaned != content.strip() else content


def _find_segments(messages):
    """Find assistant+tools segment boundaries.

    Returns:
        List of ``(assistant_start_idx, segment_end_idx)`` tuples.
    """
    segments = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "assistant":
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            segments.append((i, j))
            i = j
        else:
            i += 1
    return segments


def _segment_has_error(messages, start, end):
    """Check if any tool message in ``messages[start:end]`` has ``is_error=True``."""
    for m in messages[start:end]:
        if m.get("role") == "tool" and m.get("is_error") is True:
            return True
    return False


def _clean_message(msg, strip_thinking=True):
    """Remove non-standard fields before tokenization.

    Keeps only the fields expected by tokenizer chat templates:
    role, content, tool_calls (for assistant), tool_call_id (for tool).

    Args:
        msg: Raw message dict.
        strip_thinking: If True, remove ``<think>...</think>`` blocks from
            assistant content (used for context turns).  If False, preserve
            content as-is (used for the training-target assistant turn).
    """
    cleaned = {"role": msg["role"]}

    # Handle content — some assistant messages have content=None when
    # they only contain tool_calls.  Preserve None so chat templates
    # that distinguish None vs "" render correctly.
    content = msg.get("content")
    if content is not None:
        if msg["role"] == "assistant" and strip_thinking:
            cleaned["content"] = _extract_thinking(content)
        else:
            cleaned["content"] = content
    elif msg["role"] == "assistant" and msg.get("tool_calls"):
        # Assistant with tool_calls but content=None: preserve None explicitly
        # so chat templates that distinguish None vs "" render correctly.
        cleaned["content"] = None
    else:
        # Non-assistant messages without content: default to empty string.
        cleaned["content"] = ""

    # Copy tool_calls for assistant messages
    if msg["role"] == "assistant" and msg.get("tool_calls"):
        cleaned_tool_calls = []
        for tc in msg["tool_calls"]:
            cleaned_tc = {
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.dumps(tc["function"]["arguments"])
                    if isinstance(tc["function"]["arguments"], dict)
                    else tc["function"]["arguments"],
                },
            }
            if "id" in tc:
                cleaned_tc["id"] = tc["id"]
            cleaned_tool_calls.append(cleaned_tc)
        cleaned["tool_calls"] = cleaned_tool_calls

    # Copy tool_call_id for tool messages
    if msg["role"] == "tool" and msg.get("tool_call_id"):
        cleaned["tool_call_id"] = msg["tool_call_id"]

    return cleaned


def _split_and_filter(messages, filter_errors=True):
    """Split trajectory into progressive pairs and optionally filter by ``is_error``.

    For each pair, thinking (``<think>...</think>``) is stripped from all
    context assistant turns.  Only the last assistant turn (the training
    target) keeps its content unchanged.

    Args:
        messages: Raw trajectory messages.
        filter_errors: If True (default), discard pairs whose current segment
            contains a tool result with ``is_error=True``.  Set to False to
            keep all pairs regardless of tool errors.

    Returns:
        List of cleaned message sequences (one per valid pair).
    """
    segments = _find_segments(messages)
    pairs = []
    n_filtered = 0

    for asst_start, seg_end in segments:
        # Check if current segment has any tool errors
        if filter_errors and _segment_has_error(messages, asst_start, seg_end):
            n_filtered += 1
            continue

        pair = []
        for idx, m in enumerate(messages[:seg_end]):
            # Only the last assistant (at asst_start) keeps thinking;
            # all earlier assistant turns get thinking stripped.
            is_target = m.get("role") == "assistant" and idx == asst_start
            pair.append(_clean_message(m, strip_thinking=not is_target))
        pairs.append(pair)

    return pairs, n_filtered


_TEMPLATE_PATTERNS = [
    # ChatML (Qwen, etc.):  <|im_start|>assistant\n ... <|im_end|>
    (r"<\|im_start\|>assistant\n", r"<\|im_end\|>"),
    # Llama 3:  <|start_header_id|>assistant<|end_header_id|>\n\n ... <|eot_id|>
    (r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n", r"<\|eot_id\|>"),
]


class _TokenizeAndMask:
    """Picklable callable for ``Dataset.map(num_proc=N)``."""

    def __init__(self, tokenizer, assistant_pattern):
        self.tokenizer = tokenizer
        self.assistant_pattern = assistant_pattern

    def __call__(self, sample):
        messages = sample["messages"]

        # 1) Render the full template text.
        try:
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            logger.warning(
                "apply_chat_template failed (likely unsupported tool_calls "
                "or tool role): %s. Skipping sample.",
                e,
            )
            return {"input_ids": [], "loss_mask": []}

        # 2) Tokenize with offset mapping so we can map char→token.
        encoding = self.tokenizer(
            full_text, add_special_tokens=False, return_offsets_mapping=True
        )
        input_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]

        loss_mask = [0] * len(input_ids)

        # 3) Find the LAST assistant segment char range via regex.
        # In progressive SFT, only the newest (last) assistant turn is
        # the training target; earlier turns are context.
        last_match = None
        for m in self.assistant_pattern.finditer(full_text):
            last_match = m
        if last_match is not None:
            # m.start(1) = first char of assistant content
            # m.end(0)   = char after end-of-turn token
            rs, re_ = last_match.start(1), last_match.end(0)
            for tok_idx, (cs, ce) in enumerate(offset_mapping):
                if ce > rs and cs < re_:
                    loss_mask[tok_idx] = 1

        return {"input_ids": input_ids, "loss_mask": loss_mask}


def _detect_template_pattern(tokenizer):
    """Detect the assistant role delimiter used by this tokenizer's template.

    Strategy:
        1. Try known ``_TEMPLATE_PATTERNS`` (fast, battle-tested).
        2. Fall back to double-probe diff: render the template with a known
           marker and with empty content, then diff the two strings to extract
           the exact header and end-of-turn delimiters.

    Raises:
        ValueError: If both strategies fail to detect a usable pattern.
    """
    _PROBE_CONTENT = "PROBE_MARKER"

    probe_msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": _PROBE_CONTENT},
    ]
    probe_text = tokenizer.apply_chat_template(probe_msgs, tokenize=False)

    # --- Strategy 1: known patterns ---
    for hdr_re, eot_re in _TEMPLATE_PATTERNS:
        if re.search(hdr_re, probe_text):
            pattern = re.compile(hdr_re + r"(.*?)" + eot_re, re.DOTALL)
            logger.info(
                f"Detected template style (known pattern): "
                f"header_re={hdr_re!r}, eot_re={eot_re!r}"
            )
            return pattern

    # --- Strategy 2: double-probe diff ---
    try:
        probe_empty = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": ""},
        ]
        text_empty = tokenizer.apply_chat_template(probe_empty, tokenize=False)

        marker_idx = probe_text.index(_PROBE_CONTENT)
        header = probe_text[:marker_idx]
        tail = probe_text[marker_idx + len(_PROBE_CONTENT) :]

        if text_empty == header + tail:
            # Extract the assistant-specific header by removing the shared
            # user-only prefix.
            user_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}], tokenize=False
            )
            asst_header = header[len(user_only) :]
            # end-of-turn delimiter: strip leading newlines, then take
            # up to the first newline (or the full string if none).
            eot_stripped = tail.lstrip("\n")
            eot = eot_stripped.split("\n")[0] if "\n" in eot_stripped else eot_stripped

            if asst_header and eot:
                hdr_re = re.escape(asst_header)
                eot_re = re.escape(eot)
                pattern = re.compile(hdr_re + r"(.*?)" + eot_re, re.DOTALL)
                logger.info(
                    f"Detected template style (probe diff): "
                    f"header={asst_header!r}, eot={eot!r}"
                )
                return pattern
    except (ValueError, IndexError):
        pass  # PROBE_CONTENT not found in rendered text, skip

    raise ValueError(
        "Could not detect chat template assistant delimiters. "
        "Unable to build a reliable loss mask. "
        f"Probe text: {probe_text[:200]!r}"
    )


def _load_trajectory_pairs(path: str, filter_errors: bool = True):
    """Load trajectory JSONL and split into progressive pairs.

    Each line has ``{"conversations": [{"messages": [...]}]}``.

    Args:
        path: Path to the JSONL file.
        filter_errors: If True, discard pairs whose current segment contains
            a tool result with ``is_error=True``.
    """
    all_pairs = []
    records_in = 0
    total_filtered = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records_in += 1

            convs = record.get("conversations", [])
            if not convs:
                continue

            if len(convs) > 1:
                logger.warning(
                    "Record %d has %d conversations, using only the last one.",
                    records_in,
                    len(convs),
                )

            messages = convs[-1].get("messages", [])
            if not messages:
                continue

            pairs, n_filtered = _split_and_filter(messages, filter_errors=filter_errors)
            total_filtered += n_filtered
            all_pairs.extend(pairs)

    logger.info(
        f"Loaded {records_in} trajectories, "
        f"generated {len(all_pairs)} pairs "
        f"(filtered {total_filtered} with tool errors)"
    )
    return all_pairs


def _load_presplit_pairs(path: str):
    """Load pre-split pair JSONL where each line is ``{"messages": [...]}``.

    Messages are cleaned but no splitting or error-filtering is performed.
    """
    all_pairs = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if not messages:
                continue
            all_pairs.append([_clean_message(m) for m in messages])

    logger.info(f"Loaded {len(all_pairs)} pre-split pairs from {path}")
    return all_pairs


def get_swe_sft_dataset(
    path: str,
    split: str | None = None,
    tokenizer=None,
    max_length: int | None = None,
    num_proc: int | None = None,
    pre_split: bool = False,
    filter_errors: bool = True,
):
    """Load SWE trajectory data and convert to SFT training pairs.

    Args:
        path: Path to the JSONL file containing SWE trajectories.
        split: Unused, kept for API compatibility.
        tokenizer: Tokenizer with ``apply_chat_template`` support.
        max_length: Max token length.  Longer sequences are filtered out.
        num_proc: Number of parallel workers for tokenization.
            Defaults to ``min(os.cpu_count(), DATASET_NUM_PROC)``.
        pre_split: If True, treat input as pre-split pairs (each line is
            ``{"messages": [...]}``) instead of full trajectories.
        filter_errors: If True (default), discard pairs whose current segment
            contains a tool result with ``is_error=True``.  Set to False to
            keep all pairs regardless of tool errors.

    Returns:
        A HuggingFace ``Dataset`` with ``input_ids`` and ``loss_mask`` columns.
    """
    if num_proc is None:
        num_proc = max(1, min(os.cpu_count() or 1, DATASET_NUM_PROC))

    if pre_split:
        all_pairs = _load_presplit_pairs(path)
    else:
        all_pairs = _load_trajectory_pairs(path, filter_errors=filter_errors)

    if not all_pairs:
        raise ValueError(f"No valid SFT pairs generated from {path}")

    dataset = Dataset.from_dict({"messages": all_pairs})

    assistant_pattern = _detect_template_pattern(tokenizer)
    process_fn = _TokenizeAndMask(tokenizer, assistant_pattern)

    dataset = dataset.map(process_fn, num_proc=num_proc).remove_columns(["messages"])

    # Filter out empty samples (e.g. from apply_chat_template failures).
    before_empty = len(dataset)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
    if before_empty - len(dataset) > 0:
        logger.info(
            f"Filtered {before_empty - len(dataset)} empty samples "
            f"(likely from template rendering failures)"
        )

    if max_length is not None:
        before_filter = len(dataset)
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
        logger.info(
            f"Filtered {before_filter - len(dataset)} samples "
            f"exceeding max_length={max_length}"
        )

    logger.info(f"Final dataset: {len(dataset)} samples")
    return dataset


if __name__ == "__main__":
    import argparse
    import sys
    import tempfile

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Verify SWE SFT pair generation and loss masking.",
    )
    parser.add_argument("path", help="Path to SWE trajectory JSONL file")
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="HuggingFace tokenizer name or path (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Filter samples exceeding this token length",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=None,
        help="Number of pairs to process.  Controls loading, tokenization,"
        " display, and export.  Default: all pairs.",
    )
    parser.add_argument(
        "--show-transitions",
        action="store_true",
        help="Show loss mask transition boundaries with decoded tokens",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: min(cpu_count, {DATASET_NUM_PROC}))",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help='Write pairs to a JSONL file (each line: {"messages": [...]})',
    )
    parser.add_argument(
        "--pre-split",
        action="store_true",
        help='Input is already in pair format (each line: {"messages": [...]}).'
        " Skip trajectory splitting and error filtering.",
    )
    parser.add_argument(
        "--pairs-only",
        action="store_true",
        help="Only verify pair generation (no tokenization). "
        "Combine with --output to export pairs.",
    )
    parser.add_argument(
        "--no-filter-errors",
        action="store_true",
        help="Keep pairs whose current segment contains tool results with "
        "is_error=True (by default these are discarded).",
    )
    args = parser.parse_args()

    # --- Load pairs (always full load, then truncate) ---
    filter_errors = not args.no_filter_errors
    if args.pre_split:
        all_pairs = _load_presplit_pairs(args.path)
    else:
        all_pairs = _load_trajectory_pairs(args.path, filter_errors=filter_errors)

    total_pairs = len(all_pairs)
    if args.num_samples is not None:
        all_pairs = all_pairs[: args.num_samples]

    print(f"Total pairs:    {total_pairs}")
    if args.num_samples is not None:
        print(f"Using:          {len(all_pairs)}")

    if all_pairs:
        lengths = [len(p) for p in all_pairs]
        print(
            f"Messages/pair:  min={min(lengths)}, "
            f"max={max(lengths)}, avg={sum(lengths) / len(lengths):.1f}"
        )

    # --- Export pairs ---
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fout:
            for pair in all_pairs:
                fout.write(json.dumps({"messages": pair}, ensure_ascii=False))
                fout.write("\n")
        print(f"Wrote {len(all_pairs)} pairs to {args.output}")

    if args.pairs_only:
        # Show pair structure and exit.
        for i, pair in enumerate(all_pairs):
            roles = [m["role"] for m in pair]
            print(f"  Pair {i}: {len(pair)} msgs  roles={roles}")
        sys.exit(0)

    # --- Tokenize (only the selected pairs) ---
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Write selected pairs to a temp file so get_swe_sft_dataset can load
    # them without re-processing the entire original file.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as fout:
        for pair in all_pairs:
            fout.write(json.dumps({"messages": pair}, ensure_ascii=False))
            fout.write("\n")

    try:
        ds = get_swe_sft_dataset(
            path=tmp_path,
            tokenizer=tok,
            max_length=args.max_length,
            num_proc=args.num_proc,
            pre_split=True,
            filter_errors=filter_errors,
        )
    finally:
        os.unlink(tmp_path)

    print(f"\nTokenized: {len(ds)} samples")

    # --- Per-sample stats ---
    total_tokens = 0
    total_loss = 0
    for i in range(len(ds)):
        ids = ds[i]["input_ids"]
        mask = ds[i]["loss_mask"]
        n_loss = sum(mask)
        total_tokens += len(ids)
        total_loss += n_loss
        pct = 100.0 * n_loss / len(ids) if ids else 0
        print(f"  Sample {i}: {len(ids)} tokens, loss={n_loss} ({pct:.1f}%)")

    avg_pct = 100.0 * total_loss / total_tokens if total_tokens else 0
    print(
        f"\nAggregate: {total_tokens} tokens, {total_loss} loss tokens ({avg_pct:.1f}%)"
    )

    # --- Show transitions ---
    if args.show_transitions:
        for i in range(len(ds)):
            ids = ds[i]["input_ids"]
            mask = ds[i]["loss_mask"]
            print(f"\nSample {i} transitions (loss 0->1 or 1->0):")
            for j in range(1, len(mask)):
                if mask[j] != mask[j - 1]:
                    ctx_start = max(0, j - 2)
                    ctx_end = min(len(ids), j + 3)
                    ctx = tok.decode(ids[ctx_start:ctx_end])
                    print(f"  Token {j}: {mask[j - 1]}->{mask[j]}  context: {ctx!r}")
