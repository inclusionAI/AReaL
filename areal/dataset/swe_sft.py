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

The file is organized into the following sections:

1. **Constants & Infrastructure** — shared constants, distributed sync
2. **Cleaning** — message content transforms (thinking tags, field cleanup)
3. **Filters** — keep/discard predicates (error, empty, bare-text, truncation)
4. **Splitting** — trajectory → progressive pairs (segment detection + split)
5. **Tokenization** — template detection, render→tokenize→loss_mask, dump
6. **Pipeline** — loading, processing, distributed cache, public API
7. **CLI** — ``python -m areal.dataset.swe_sft`` entry point
"""

import json
import os
import re
import time

from datasets import Dataset

from areal.utils import logging

logger = logging.getLogger("SWESFTDataset")


# ============================================================
# 1. Constants & Infrastructure
# ============================================================

DATASET_NUM_PROC = 1

# Timeout (seconds) for non-rank-0 workers waiting for rank 0 to finish
# dataset processing.  Large datasets with tokenization can take minutes;
# 30 min is a generous upper bound.
_RANK0_CACHE_TIMEOUT = 3600
_RANK0_CACHE_POLL_INTERVAL = 5


def _wait_for_marker(marker_path: str):
    """Block until *marker_path* exists on disk, with timeout."""
    start = time.monotonic()
    while not os.path.exists(marker_path):
        elapsed = time.monotonic() - start
        if elapsed > _RANK0_CACHE_TIMEOUT:
            raise TimeoutError(
                f"Waited {_RANK0_CACHE_TIMEOUT}s for rank 0 to finish dataset "
                f"processing (marker: {marker_path}). Check rank 0 logs."
            )
        time.sleep(_RANK0_CACHE_POLL_INTERVAL)


# ============================================================
# 2. Cleaning — message content transforms
# ============================================================

# Match reasoning blocks with any common tag variant:
#   <think>...</think>      (Qwen standard)
#   <thinking>...</thinking> (Claude)
# The opening and closing tag names need not match exactly — mixed pairs
# like ``<think>...</thinking>`` (seen in distillation data) are handled.
_THINK_OPEN_RE = re.compile(r"<think(?:ing)?>")
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>")
_THINK_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)


def _normalize_thinking_tags(content):
    """Normalise all thinking tag variants to ``<think>``/``</think>``.

    Distillation data from different models may use ``<thinking>`` (Claude)
    vs ``<think>`` (Qwen).  Non-standard variants are multi-token for the
    Qwen tokenizer which breaks think/tool_call boundaries.
    """
    if not content:
        return content
    content = _THINK_OPEN_RE.sub("<think>", content)
    content = _THINK_CLOSE_RE.sub("</think>", content)
    return content


def _extract_thinking(content):
    """Strip thinking blocks from *content*.

    Callers must run ``_normalize_thinking_tags`` first so that all
    tag variants have been converted to ``<think>``/``</think>``.

    Returns:
        Cleaned content with thinking blocks removed, or the original
        content unchanged if no thinking tags are found.
    """
    if not content:
        return content
    cleaned = _THINK_RE.sub("", content).strip()
    return cleaned if cleaned != content.strip() else content


def _clean_message(msg, strip_thinking=True, ensure_thinking=False):
    """Remove non-standard fields before tokenization.

    Keeps only the fields expected by tokenizer chat templates:
    role, content, tool_calls (for assistant), tool_call_id (for tool).

    Args:
        msg: Raw message dict.
        strip_thinking: If True, remove ``<think>...</think>`` blocks from
            assistant content (used for context turns).  If False, preserve
            content as-is (used for the training-target assistant turn).
        ensure_thinking: If True, set ``reasoning_content`` on assistant
            tool_call turns that lack a thinking block, so that templates
            without ``loop.last`` logic (e.g. Ling2.5/Bailing) still
            render ``<think></think>``.  Mirrors Qwen3's ``loop.last``
            behavior at the data level.
    """
    cleaned = {"role": msg["role"]}

    # Handle content — some assistant messages have content=None when
    # they only contain tool_calls.  Preserve None so chat templates
    # that distinguish None vs "" render correctly.
    content = msg.get("content")
    has_thinking = False
    if content is not None:
        if msg["role"] == "assistant":
            content = _normalize_thinking_tags(content)
            has_thinking = bool(_THINK_RE.search(content))
            if strip_thinking:
                content = _extract_thinking(content)
        cleaned["content"] = content
    elif msg["role"] == "assistant" and msg.get("tool_calls"):
        # Assistant with tool_calls but content=None: preserve None explicitly
        # so chat templates that distinguish None vs "" render correctly.
        cleaned["content"] = None
    else:
        # Non-assistant messages without content: default to empty string.
        cleaned["content"] = ""

    # For the target assistant turn (last in pair) without a thinking
    # block, inject reasoning_content='\n' so that templates without
    # loop.last logic (e.g. Ling2.5/Bailing, which gate on
    # `reasoning_content != ''`) still render <think></think>.
    # NOTE: this relies on the template applying
    # `reasoning_content.strip('\n')` before rendering.  Both Bailing
    # and Qwen3 do this, so '\n' is equivalent to '' and the output is
    # identical to the native non-empty reasoning_content path.  Verify
    # when onboarding a new model template.
    if ensure_thinking and msg["role"] == "assistant" and not has_thinking:
        cleaned["reasoning_content"] = "\n"

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


# ============================================================
# 3. Filters — keep/discard predicates
# ============================================================


def _segment_has_error(messages, start, end):
    """Check if any tool message in ``messages[start:end]`` has ``is_error=True``."""
    for m in messages[start:end]:
        if m.get("role") == "tool" and m.get("is_error") is True:
            return True
    return False


def _is_empty_tool_call(msg):
    """True if assistant *msg* has no text content but has tool_calls."""
    content = msg.get("content") or ""
    return not content.strip() and bool(msg.get("tool_calls"))


def _is_bare_text_tool_call(msg):
    """True if assistant *msg* has text without ``<think>`` tags and has tool_calls."""
    content = msg.get("content") or ""
    if not content.strip() or not msg.get("tool_calls"):
        return False
    normalized = _THINK_OPEN_RE.sub("<think>", content)
    normalized = _THINK_CLOSE_RE.sub("</think>", normalized)
    match = _THINK_RE.search(normalized)
    return not (match and match.group(1).strip())


def _truncate_at_task_notification(messages):
    """Truncate messages when a ``<task-notification>`` follows a pure-text assistant.

    Claude Code emits ``<task-notification>`` as a user message when a
    background task (e.g. ``pip install``) completes.  If the model has
    already produced a text-only summary (no tool_calls), the notification
    and all subsequent messages are noise — the model just replies
    "nothing to do".  Truncating here removes that noise.

    Only triggers when the pattern is:
        assistant (text, no tool_calls) → user (<task-notification>)

    Returns:
        Truncated message list (or the original list if no truncation needed).
    """
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        if "<task-notification>" not in (m.get("content") or ""):
            continue
        # Find preceding assistant
        prev_asst = None
        for j in range(i - 1, -1, -1):
            if messages[j].get("role") == "assistant":
                prev_asst = messages[j]
                break
        if prev_asst is None:
            continue
        content = prev_asst.get("content") or ""
        if content.strip() and not prev_asst.get("tool_calls"):
            # Truncate: keep everything up to (but not including) this user msg
            return messages[:i]
    return messages


# ============================================================
# 4. Splitting — trajectory → progressive pairs
# ============================================================


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


def _split_and_filter(
    messages,
    filter_errors=True,
    strip_all_thinking=False,
    filter_empty_tool_calls=False,
    filter_bare_text_tool_calls=False,
):
    """Split trajectory into progressive pairs and optionally filter.

    By default, thinking (``<think>...</think>``) is stripped from context
    assistant turns only; the last assistant turn (training target) keeps
    its content unchanged.  Set *strip_all_thinking* to strip from every
    assistant turn including the target.

    Args:
        messages: Raw trajectory messages.
        filter_errors: If True (default), discard pairs whose current segment
            contains a tool result with ``is_error=True``.  Set to False to
            keep all pairs regardless of tool errors.
        strip_all_thinking: If True, strip ``<think>`` blocks from every
            assistant turn including the training target.
        filter_empty_tool_calls: If True, discard pairs whose training-target
            assistant turn has no text content but has tool_calls.
        filter_bare_text_tool_calls: If True, discard pairs whose
            training-target assistant turn has text content without
            ``<think>`` tags and has tool_calls.

    Returns:
        Tuple of ``(pairs, n_filtered_errors, n_filtered_empty_tc,
        n_filtered_bare_tc)``.
    """
    segments = _find_segments(messages)
    if not segments:
        return [], 0, 0, 0

    pairs = []
    n_filtered_errors = 0
    n_filtered_empty_tc = 0
    n_filtered_bare_tc = 0

    # Pre-clean all messages in context mode (thinking stripped).
    # This avoids re-cleaning the same message for every progressive pair
    # (O(N+K) instead of O(N*K) where K = number of segments).
    context_cleaned = [_clean_message(m, strip_thinking=True) for m in messages]

    # For target assistant turns, clean with thinking preserved (unless
    # strip_all_thinking is set, in which case context_cleaned is reusable).
    target_cleaned = {}
    if not strip_all_thinking:
        for asst_start, _ in segments:
            target_cleaned[asst_start] = _clean_message(
                messages[asst_start],
                strip_thinking=False,
                ensure_thinking=True,
            )

    for asst_start, seg_end in segments:
        # Check if current segment has any tool errors
        if filter_errors and _segment_has_error(messages, asst_start, seg_end):
            n_filtered_errors += 1
            continue

        # Content-type filters operate on the raw assistant message.
        asst_msg = messages[asst_start]
        if filter_empty_tool_calls and _is_empty_tool_call(asst_msg):
            n_filtered_empty_tc += 1
            continue
        if filter_bare_text_tool_calls and _is_bare_text_tool_call(asst_msg):
            n_filtered_bare_tc += 1
            continue

        # Build pair: include context up to the target assistant turn,
        # truncating tool responses that follow it.  This ensures the
        # target assistant is always the *last* message so that chat
        # templates with ``loop.last``-dependent rendering (e.g. Qwen3
        # ``<think>`` injection) behave consistently.  The tool responses
        # would have loss_mask=0 anyway and only add noise.
        pair = list(context_cleaned[: asst_start + 1])
        if not strip_all_thinking:
            pair[asst_start] = target_cleaned[asst_start]
        pairs.append(pair)

    return pairs, n_filtered_errors, n_filtered_empty_tc, n_filtered_bare_tc


# ============================================================
# 5. Tokenization — template detection, render, loss mask, dump
# ============================================================

_TEMPLATE_PATTERNS = [
    # ChatML (Qwen, etc.):  <|im_start|>assistant\n ... <|im_end|>
    (r"<\|im_start\|>assistant\n", r"<\|im_end\|>"),
    # Llama 3:  <|start_header_id|>assistant<|end_header_id|>\n\n ... <|eot_id|>
    (r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n", r"<\|eot_id\|>"),
]


def _render_tokenize_mask(messages, tokenizer, assistant_pattern, tools=None):
    """Render, tokenize, and build loss_mask for a single pair.

    Returns:
        Tuple of ``(full_text, input_ids, loss_mask, offset_mapping)``, or
        ``None`` if ``apply_chat_template`` fails.
    """
    # 1) Render the full template text.
    try:
        kwargs = {"tokenize": False}
        if tools is not None:
            kwargs["tools"] = tools
        full_text = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as e:
        logger.warning(
            "apply_chat_template failed (likely unsupported tool_calls "
            "or tool role): %s. Skipping sample.",
            e,
        )
        return None

    # 2) Tokenize with offset mapping so we can map char→token.
    encoding = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True
    )
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    # 3) Find the LAST assistant segment char range via regex.
    # In progressive SFT, only the newest (last) assistant turn is
    # the training target; earlier turns are context.
    loss_mask = [0] * len(input_ids)
    last_match = None
    for m in assistant_pattern.finditer(full_text):
        last_match = m
    if last_match is not None:
        # m.start(1) = first char of assistant content
        # m.end(0)   = char after end-of-turn token
        rs, re_ = last_match.start(1), last_match.end(0)
        for tok_idx, (cs, ce) in enumerate(offset_mapping):
            if ce > rs and cs < re_:
                loss_mask[tok_idx] = 1

    return full_text, input_ids, loss_mask, offset_mapping


class _TokenizeAndMask:
    """Picklable callable for ``Dataset.map(num_proc=N)``."""

    def __init__(self, tokenizer, assistant_pattern, tools=None, max_length=None):
        self.tokenizer = tokenizer
        self.assistant_pattern = assistant_pattern
        self.tools = tools
        self.max_length = max_length

    def __call__(self, sample):
        result = _render_tokenize_mask(
            sample["messages"], self.tokenizer, self.assistant_pattern, self.tools
        )
        if result is None:
            return {"input_ids": [], "loss_mask": []}

        _full_text, input_ids, loss_mask, _offset_mapping = result

        # Early exit: overlength or empty → return empty so a single
        # filter pass removes it together with template-failure empties.
        if self.max_length is not None and len(input_ids) > self.max_length:
            return {"input_ids": [], "loss_mask": []}

        return {"input_ids": input_ids, "loss_mask": loss_mask}


def _detect_template_pattern(tokenizer, tools=None):
    """Detect the assistant role delimiter used by this tokenizer's template.

    When *tools* is provided the probe is rendered with ``tools=`` so that
    the detected delimiters match the actual training text (some templates
    alter the system block when tools are present).

    Strategy:
        1. Try known ``_TEMPLATE_PATTERNS`` (fast, battle-tested).
        2. Fall back to double-probe diff: render the template with a known
           marker and with empty content, then diff the two strings to extract
           the exact header and end-of-turn delimiters.

    Raises:
        ValueError: If both strategies fail to detect a usable pattern.
    """
    _PROBE_CONTENT = "PROBE_MARKER"

    extra_kwargs = {}
    if tools is not None:
        extra_kwargs["tools"] = tools

    probe_msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": _PROBE_CONTENT},
    ]
    probe_text = tokenizer.apply_chat_template(
        probe_msgs, tokenize=False, **extra_kwargs
    )

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
        text_empty = tokenizer.apply_chat_template(
            probe_empty, tokenize=False, **extra_kwargs
        )

        marker_idx = probe_text.index(_PROBE_CONTENT)
        header = probe_text[:marker_idx]
        tail = probe_text[marker_idx + len(_PROBE_CONTENT) :]

        if text_empty == header + tail:
            # Extract the assistant-specific header by removing the shared
            # user-only prefix.
            user_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}],
                tokenize=False,
                **extra_kwargs,
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


def _dump_samples(pairs, tokenizer, assistant_pattern, tools, dump_dir, n_samples):
    """Dump sampled pairs as ``.txt`` + ``.json`` files for inspection.

    Args:
        pairs: List of message-list pairs (cleaned, ready for template).
        tokenizer: Tokenizer with ``apply_chat_template`` support.
        assistant_pattern: Compiled regex from ``_detect_template_pattern``.
        tools: Tool definitions list (or None).
        dump_dir: Directory to write files into (created if needed).
        n_samples: Number of random samples to dump.  ``-1`` dumps all.
    """
    import random as _random

    os.makedirs(dump_dir, exist_ok=True)

    if n_samples == -1 or n_samples >= len(pairs):
        indices = list(range(len(pairs)))
    else:
        indices = sorted(_random.sample(range(len(pairs)), n_samples))

    n_written = 0
    for i in indices:
        pair = pairs[i]

        result = _render_tokenize_mask(pair, tokenizer, assistant_pattern, tools)
        if result is None:
            continue

        full_text, input_ids, loss_mask, offset_mapping = result
        n_loss = sum(loss_mask)
        base = os.path.join(dump_dir, f"pair_{i}")

        # --- .txt ---
        with open(base + ".txt", "w", encoding="utf-8") as fout:
            fout.write(
                f"Pair {i}: {len(pair)} messages, "
                f"{len(input_ids)} tokens, loss=1: {n_loss}\n"
            )
            fout.write(f"Last msg role: {pair[-1]['role']}\n")
            fout.write(f"{'=' * 72}\n\n")

            fout.write("--- Rendered Text ---\n")
            fout.write(full_text)
            fout.write("\n\n")

            fout.write("--- Token / Loss Mask ---\n")
            fout.write(f"{'Idx':>6} | {'TokenID':>8} | Loss | Token Text\n")
            fout.write(f"{'-' * 6}-+-{'-' * 8}-+------+{'-' * 40}\n")
            for t in range(len(input_ids)):
                cs, ce = offset_mapping[t]
                tok_text = repr(full_text[cs:ce])
                fout.write(
                    f"{t:>6} | {input_ids[t]:>8} | {loss_mask[t]:>4} | {tok_text}\n"
                )

        # --- .json ---
        tokens_list = []
        for t in range(len(input_ids)):
            cs, ce = offset_mapping[t]
            tokens_list.append(
                {
                    "idx": t,
                    "token_id": input_ids[t],
                    "text": full_text[cs:ce],
                    "loss": loss_mask[t],
                }
            )
        record = {
            "pair_index": i,
            "n_messages": len(pair),
            "n_tokens": len(input_ids),
            "n_loss_tokens": n_loss,
            "rendered_text": full_text,
            "tokens": tokens_list,
        }
        with open(base + ".json", "w", encoding="utf-8") as fout:
            json.dump(record, fout, ensure_ascii=False)

        n_written += 1

    logger.info(f"Dumped {n_written} sample pairs to {dump_dir}/")


# ============================================================
# 6. Pipeline — loading, processing, distributed cache, public API
# ============================================================


def _load_trajectory_pairs(
    path: str,
    filter_errors: bool = True,
    strip_all_thinking: bool = False,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
    truncate_task_notifications: bool = False,
):
    """Load trajectory JSONL and split into progressive pairs.

    Each line has ``{"conversations": [{"messages": [...], "tools": [...]}]}``.

    Also extracts the ``tools`` list (OpenAI function-calling schema) from
    the first conversation that contains one.  All records in a dataset are
    expected to share the same tool definitions.

    Args:
        path: Path to the JSONL file.
        filter_errors: If True, discard pairs whose current segment contains
            a tool result with ``is_error=True``.
        strip_all_thinking: If True, strip thinking from all assistant turns.
        filter_empty_tool_calls: If True, discard pairs whose training-target
            assistant turn has no text content but has tool_calls.
        filter_bare_text_tool_calls: If True, discard pairs whose
            training-target assistant turn has text without ``<think>`` tags
            and has tool_calls.
        truncate_task_notifications: If True, truncate trajectories at the
            first ``<task-notification>`` that follows a pure-text assistant
            turn (no tool_calls), removing noise from background task
            completions.

    Returns:
        Tuple of ``(all_pairs, tools)`` where *tools* is ``None`` when no
        tool definitions are found.
    """
    all_pairs = []
    tools = None
    records_in = 0
    total_filtered_errors = 0
    total_filtered_empty_tc = 0
    total_filtered_bare_tc = 0
    total_truncated = 0

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

            conv = convs[-1]
            messages = conv.get("messages", [])
            if not messages:
                continue

            # Truncate noise from background task notifications.
            if truncate_task_notifications:
                truncated = _truncate_at_task_notification(messages)
                if len(truncated) < len(messages):
                    total_truncated += 1
                    messages = truncated

            # Extract tool definitions (same across all records).
            if tools is None and conv.get("tools"):
                tools = conv["tools"]
                tool_names = [t.get("function", {}).get("name", "?") for t in tools]
                logger.info(
                    f"Extracted {len(tools)} tool definitions from data: {tool_names}"
                )

            pairs, n_err, n_empty_tc, n_bare_tc = _split_and_filter(
                messages,
                filter_errors=filter_errors,
                strip_all_thinking=strip_all_thinking,
                filter_empty_tool_calls=filter_empty_tool_calls,
                filter_bare_text_tool_calls=filter_bare_text_tool_calls,
            )
            total_filtered_errors += n_err
            total_filtered_empty_tc += n_empty_tc
            total_filtered_bare_tc += n_bare_tc
            all_pairs.extend(pairs)

    filter_parts = []
    if total_truncated:
        filter_parts.append(
            f"{total_truncated} trajectories truncated at task-notification"
        )
    if total_filtered_errors:
        filter_parts.append(f"{total_filtered_errors} with tool errors")
    if total_filtered_empty_tc:
        filter_parts.append(f"{total_filtered_empty_tc} empty-content tool calls")
    if total_filtered_bare_tc:
        filter_parts.append(f"{total_filtered_bare_tc} bare-text tool calls")
    filter_msg = ", ".join(filter_parts) if filter_parts else "none"

    logger.info(
        f"Loaded {records_in} trajectories, "
        f"generated {len(all_pairs)} pairs "
        f"(filtered: {filter_msg})"
    )
    return all_pairs, tools


def _load_presplit_pairs(path: str, strip_all_thinking: bool = False):
    """Load pre-split pair JSONL where each line is ``{"messages": [...]}``.

    Messages are cleaned but no splitting or error-filtering is performed.
    By default, thinking is stripped from context assistant turns but
    preserved for the last assistant turn (the training target).  Set
    *strip_all_thinking* to strip from every assistant turn.

    Also extracts ``tools`` from the first record that contains a
    ``"tools"`` field, same as ``_load_trajectory_pairs``.

    Returns:
        Tuple of ``(all_pairs, tools)``.
    """
    all_pairs = []
    tools = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if not messages:
                continue

            if tools is None and record.get("tools"):
                tools = record["tools"]

            # Find the last assistant index so we can preserve its thinking.
            last_asst = None
            for i, m in enumerate(messages):
                if m.get("role") == "assistant":
                    last_asst = i

            pair = []
            for idx, m in enumerate(messages):
                is_target = m.get("role") == "assistant" and idx == last_asst
                strip = strip_all_thinking or not is_target
                pair.append(_clean_message(m, strip_thinking=strip))
            all_pairs.append(pair)

    if tools is not None:
        tool_names = [t.get("function", {}).get("name", "?") for t in tools]
        logger.info(f"Extracted {len(tools)} tool definitions: {tool_names}")

    logger.info(f"Loaded {len(all_pairs)} pre-split pairs from {path}")
    return all_pairs, tools


def _process_swe_sft(
    path: str,
    tokenizer,
    *,
    max_length: int | None = None,
    num_proc: int | None = None,
    pre_split: bool = False,
    filter_errors: bool = True,
    strip_all_thinking: bool = False,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
    truncate_task_notifications: bool = False,
    no_tools: bool = False,
    dump_dir: str | None = None,
    dump_n_samples: int = 0,
):
    """Core processing: load JSONL, split into pairs, tokenize, and filter.

    Extracted from ``get_swe_sft_dataset`` so that the rank-0-only path and
    the single-process path share the same logic.
    """
    tools = None

    if num_proc is None:
        num_proc = max(1, min(os.cpu_count() or 1, DATASET_NUM_PROC))

    if pre_split:
        all_pairs, tools = _load_presplit_pairs(
            path, strip_all_thinking=strip_all_thinking
        )
    else:
        all_pairs, tools = _load_trajectory_pairs(
            path,
            filter_errors=filter_errors,
            strip_all_thinking=strip_all_thinking,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
            truncate_task_notifications=truncate_task_notifications,
        )

    if no_tools:
        tools = None
        logger.info("Tool definitions disabled (no_tools=True)")
    elif tools is not None:
        tool_names = [t.get("function", {}).get("name", "?") for t in tools]
        logger.info(f"Using tools for chat template: {tool_names}")

    if not all_pairs:
        raise ValueError(f"No valid SFT pairs generated from {path}")

    dataset = Dataset.from_dict({"messages": all_pairs})

    assistant_pattern = _detect_template_pattern(tokenizer, tools=tools)

    # Dump sample pairs for inspection before the heavy map() pass.
    if dump_dir and dump_n_samples != 0:
        _dump_samples(
            all_pairs,
            tokenizer,
            assistant_pattern,
            tools,
            dump_dir,
            dump_n_samples,
        )
    # Pass max_length so overlength samples are marked empty during
    # tokenization, allowing a single filter pass instead of three.
    process_fn = _TokenizeAndMask(
        tokenizer, assistant_pattern, tools=tools, max_length=max_length
    )

    dataset = dataset.map(process_fn, num_proc=num_proc).remove_columns(["messages"])

    # Single filter pass: removes both apply_chat_template-failure empties and
    # overlength samples (which _TokenizeAndMask also marks as empty).
    before_filter = len(dataset)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0, num_proc=num_proc)
    n_filtered = before_filter - len(dataset)
    if n_filtered > 0:
        logger.info(
            f"Filtered {n_filtered} samples "
            f"(empty from template failures or exceeding max_length={max_length})"
        )

    logger.info(f"Final dataset: {len(dataset)} samples")
    return dataset


def get_swe_sft_dataset(
    path: str,
    split: str | None = None,
    tokenizer=None,
    max_length: int | None = None,
    num_proc: int | None = None,
    pre_split: bool = False,
    filter_errors: bool = True,
    strip_all_thinking: bool = False,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
    truncate_task_notifications: bool = False,
    no_tools: bool = False,
    cache_dir: str | None = None,
    dump_dir: str | None = None,
    dump_samples: int = 0,
):
    """Load SWE trajectory data and convert to SFT training pairs.

    By default, tool definitions are auto-extracted from the training data's
    ``conversations[].tools`` field and passed to ``apply_chat_template``
    so that the tokenizer renders tool definitions in the system prompt
    (e.g. Qwen3 ``# Tools`` block), matching the eval-time format.
    Set *no_tools* to skip this and render without tool definitions.

    In distributed (SPMD) mode, only rank 0 performs the heavy processing
    (JSONL loading, pair splitting, tokenization) and saves the result as
    an Arrow dataset to *cache_dir*.  Other ranks wait for rank 0 to
    finish and then load the cached dataset directly via memory-mapped I/O.

    Args:
        path: Path to the JSONL file containing SWE trajectories, or a
            directory containing a pre-tokenized Arrow dataset (saved by
            ``python -m areal.dataset.swe_sft --save-tokenized``).
        split: Unused, kept for API compatibility.
        tokenizer: Tokenizer with ``apply_chat_template`` support.
            Not required when loading a pre-tokenized dataset.
        max_length: Max token length.  Longer sequences are filtered out.
        num_proc: Number of parallel workers for tokenization.
            Defaults to ``min(os.cpu_count(), DATASET_NUM_PROC)``.
        pre_split: If True, treat input as pre-split pairs (each line is
            ``{"messages": [...]}``) instead of full trajectories.
        filter_errors: If True (default), discard pairs whose current segment
            contains a tool result with ``is_error=True``.  Set to False to
            keep all pairs regardless of tool errors.
        strip_all_thinking: If True, strip ``<think>...</think>`` from every
            assistant turn including the training target.
        filter_empty_tool_calls: If True, discard pairs whose training-target
            assistant turn has no text content but has tool_calls.
        filter_bare_text_tool_calls: If True, discard pairs whose
            training-target assistant turn has text without ``<think>``
            tags and has tool_calls.
        truncate_task_notifications: If True, truncate trajectories at the
            first ``<task-notification>`` that follows a pure-text assistant
            turn, removing noise from background task completions.
        no_tools: If True, do not pass tool definitions to
            ``apply_chat_template`` even if the data contains them.
        cache_dir: Directory to save/load the processed Arrow dataset.
            When set in distributed mode, rank 0 processes the data and
            saves here; other ranks load from this directory.  If the
            directory already contains a completed cache (``.done`` marker),
            all ranks load from it directly without reprocessing.
        dump_dir: Directory to write sample dump files (``.txt`` + ``.json``).
            Only rank 0 writes.  Set to None to disable.
        dump_samples: Number of random samples to dump.  ``-1`` = all,
            ``0`` = disabled.

    Returns:
        A HuggingFace ``Dataset`` with ``input_ids`` and ``loss_mask`` columns.
    """
    from datasets import load_from_disk

    # Pre-tokenized Arrow dataset: load directly, skip all processing.
    if os.path.isdir(path):
        logger.info(f"Loading pre-tokenized dataset from {path}")
        dataset = load_from_disk(path)

        if max_length is not None:
            before_filter = len(dataset)
            dataset = dataset.filter(
                lambda x: len(x["input_ids"]) <= max_length, num_proc=num_proc
            )
            logger.info(
                f"Filtered {before_filter - len(dataset)} samples "
                f"exceeding max_length={max_length}"
            )

        logger.info(f"Final dataset: {len(dataset)} samples")
        return dataset

    # --- Shared kwargs for _process_swe_sft ---
    process_kwargs = dict(
        max_length=max_length,
        num_proc=num_proc,
        pre_split=pre_split,
        filter_errors=filter_errors,
        strip_all_thinking=strip_all_thinking,
        filter_empty_tool_calls=filter_empty_tool_calls,
        filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        truncate_task_notifications=truncate_task_notifications,
        no_tools=no_tools,
        dump_dir=dump_dir,
        dump_n_samples=dump_samples,
    )

    # --- Distributed rank-0-only processing ---
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if cache_dir is not None and world_size > 1:
        done_marker = os.path.join(cache_dir, ".done")

        # Fast path: cache from a previous run (or rank 0 already finished).
        if os.path.exists(done_marker):
            logger.info(
                f"Rank {rank}: loading cached processed dataset from {cache_dir}"
            )
            dataset = load_from_disk(cache_dir)
            logger.info(f"Final dataset: {len(dataset)} samples")
            return dataset

        if rank == 0:
            # Rank 0: do the heavy processing and save for other ranks.
            dataset = _process_swe_sft(path, tokenizer, **process_kwargs)
            os.makedirs(cache_dir, exist_ok=True)
            dataset.save_to_disk(cache_dir)
            # Write marker AFTER save completes so readers see a consistent dir.
            with open(done_marker, "w") as f:
                f.write(str(len(dataset)))
            logger.info(
                f"Rank 0: saved processed dataset "
                f"({len(dataset)} samples) to {cache_dir}"
            )
            return dataset
        else:
            # Other ranks: wait for rank 0, then load.
            logger.info(f"Rank {rank}: waiting for rank 0 to process dataset...")
            _wait_for_marker(done_marker)
            dataset = load_from_disk(cache_dir)
            logger.info(f"Rank {rank}: loaded cached dataset ({len(dataset)} samples)")
            return dataset

    # --- Non-distributed or no cache_dir: process in current process ---
    return _process_swe_sft(path, tokenizer, **process_kwargs)


# ============================================================
# 7. CLI — ``python -m areal.dataset.swe_sft``
# ============================================================

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
    parser.add_argument(
        "--save-tokenized",
        default=None,
        metavar="DIR",
        help="Save the tokenized dataset to DIR (Arrow format). "
        "The saved directory can be used directly as the dataset path "
        "during training, skipping all processing.",
    )
    parser.add_argument(
        "--strip-all-thinking",
        action="store_true",
        help="Strip <think>...</think> from ALL assistant turns including "
        "the training target. By default only context turns are stripped.",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Do not pass tool definitions to apply_chat_template. "
        "By default, tools are auto-extracted from the data and rendered "
        "in the system prompt (e.g. Qwen3 '# Tools' block).",
    )
    parser.add_argument(
        "--filter-empty-tool-calls",
        action="store_true",
        help="Discard pairs whose training-target assistant turn has no "
        "text content but has tool_calls (silent tool invocations).",
    )
    parser.add_argument(
        "--filter-bare-text-tool-calls",
        action="store_true",
        help="Discard pairs whose training-target assistant turn has text "
        "content without <think> tags and has tool_calls.",
    )
    parser.add_argument(
        "--truncate-task-notifications",
        action="store_true",
        help="Truncate trajectories at the first <task-notification> that "
        "follows a pure-text assistant turn. Removes noise from background "
        "task completions (e.g. pip install finishing after the model's summary).",
    )
    parser.add_argument(
        "--save-trajectories",
        default=None,
        metavar="FILE",
        help="Save preprocessed trajectories to FILE (JSONL, original format) "
        "after applying trajectory-level operations (e.g. "
        "--truncate-task-notifications) but before pair splitting. "
        "Each line preserves the original record structure with the "
        "messages field updated.",
    )
    parser.add_argument(
        "--dump-samples",
        default=None,
        metavar="DIR",
        help="Save sampled pairs to DIR, one file per pair. Each file "
        "contains the rendered text and a token-by-token table with "
        "token id, decoded text, and loss_mask. Uses --num-samples "
        "to control how many pairs are dumped (default: all).",
    )
    args = parser.parse_args()

    filter_errors = not args.no_filter_errors
    strip_all_thinking = args.strip_all_thinking
    filter_empty_tool_calls = args.filter_empty_tool_calls
    filter_bare_text_tool_calls = args.filter_bare_text_tool_calls
    truncate_task_notifications = args.truncate_task_notifications

    # --- Fast path: save preprocessed trajectories ---
    if args.save_trajectories:
        records_in = 0
        records_out = 0
        n_truncated = 0
        with (
            open(args.path, encoding="utf-8") as fin,
            open(args.save_trajectories, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records_in += 1

                convs = record.get("conversations", [])
                if not convs:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_out += 1
                    continue

                conv = convs[-1]
                messages = conv.get("messages", [])

                if truncate_task_notifications and messages:
                    truncated = _truncate_at_task_notification(messages)
                    if len(truncated) < len(messages):
                        n_truncated += 1
                        messages = truncated
                    conv["messages"] = messages

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_out += 1

        parts = []
        if n_truncated:
            parts.append(f"{n_truncated} truncated at task-notification")
        op_msg = ", ".join(parts) if parts else "no changes"
        print(
            f"Saved {records_out}/{records_in} trajectories "
            f"to {args.save_trajectories} ({op_msg})"
        )
        sys.exit(0)

    # --- Load pairs ---
    if args.pre_split:
        all_pairs, tools = _load_presplit_pairs(
            args.path, strip_all_thinking=strip_all_thinking
        )
    else:
        all_pairs, tools = _load_trajectory_pairs(
            args.path,
            filter_errors=filter_errors,
            strip_all_thinking=strip_all_thinking,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
            truncate_task_notifications=truncate_task_notifications,
        )

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

    # --- Save pairs (granularity 2: cleaned pairs as JSONL) ---
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fout:
            for pair in all_pairs:
                record = {"messages": pair}
                if tools is not None:
                    record["tools"] = tools
                fout.write(json.dumps(record, ensure_ascii=False))
                fout.write("\n")
        print(f"Wrote {len(all_pairs)} pairs to {args.output}")

    if args.pairs_only:
        for i, pair in enumerate(all_pairs):
            roles = [m["role"] for m in pair]
            print(f"  Pair {i}: {len(pair)} msgs  roles={roles}")
        sys.exit(0)

    # --- Tokenize (granularity 3: full pipeline) ---
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Write pairs to temp file so _process_swe_sft can load them as
    # pre-split input, reusing the same code path as training.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as fout:
        for pair in all_pairs:
            record = {"messages": pair}
            if tools is not None:
                record["tools"] = tools
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write("\n")

    dump_dir = args.dump_samples if args.dump_samples else None
    dump_n = args.num_samples if (dump_dir and args.num_samples) else -1

    try:
        ds = get_swe_sft_dataset(
            path=tmp_path,
            tokenizer=tok,
            max_length=args.max_length,
            num_proc=args.num_proc,
            pre_split=True,
            filter_errors=filter_errors,
            no_tools=args.no_tools,
            dump_dir=dump_dir,
            dump_samples=dump_n if dump_dir else 0,
        )
    finally:
        os.unlink(tmp_path)

    print(f"\nTokenized: {len(ds)} samples")

    # --- Save tokenized dataset (Arrow format) ---
    if args.save_tokenized:
        ds.save_to_disk(args.save_tokenized)
        print(f"Saved tokenized dataset ({len(ds)} samples) to {args.save_tokenized}")
