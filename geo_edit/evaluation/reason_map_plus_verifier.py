"""Rule-based verification for ReasonMap Plus dataset.

Handles five question types:
- Counting1: Multiple-choice (ABCD), ground-truth is 0-based index into [A,B,C,D]
- Counting2/3: Open-ended integer
- TorF1/2: Yes/No, ground-truth is 1 (yes) or 0 (no)

All prompts instruct the model to place the answer in ``\\boxed{}``.
Fallback heuristics cover ``<answer>`` tags and plain text.
"""

from __future__ import annotations

import re

_BOXED_RE = re.compile(
    r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
)

_ANSWER_TAG_RE = re.compile(
    r"<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)

_BOX_TAG_RE = re.compile(
    r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>",
    re.DOTALL,
)


def extract_boxed(text: str) -> str | None:
    """Return the *last* ``\\boxed{…}`` content, or ``None``."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_answer_region(text: str) -> str:
    """Narrow *text* down to answer-tag content, falling back to full text."""
    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _BOX_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


_LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

_CHOICE_LETTER_RE = re.compile(r"\b([A-Da-d])\b")


def _parse_choice(text: str) -> int | None:
    """Parse a multiple-choice letter and return its 0-based index."""
    text = text.strip()
    if len(text) == 1 and text.upper() in _LETTER_TO_INDEX:
        return _LETTER_TO_INDEX[text.upper()]
    m = re.match(r"^([A-Da-d])\s*[).\]:]", text)
    if m:
        return _LETTER_TO_INDEX.get(m.group(1).upper())
    m = _CHOICE_LETTER_RE.search(text)
    if m:
        return _LETTER_TO_INDEX.get(m.group(1).upper())
    return None


def _parse_integer(text: str) -> int | None:
    """Parse an integer from *text*, trying direct conversion then regex."""
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    m = re.search(r"[-+]?\d+", text)
    if m:
        return int(m.group(0))
    return None


_YES_PATTERNS = re.compile(
    r"^(yes|true|correct|right|是|对)\b",
    re.IGNORECASE,
)
_NO_PATTERNS = re.compile(
    r"^(no|false|incorrect|wrong|否|错|不是)\b",
    re.IGNORECASE,
)


def _parse_yes_no(text: str) -> int | None:
    """Parse a yes/no answer.  Returns 1 for yes, 0 for no."""
    text = text.strip().lower()
    if _YES_PATTERNS.match(text):
        return 1
    if _NO_PATTERNS.match(text):
        return 0
    if re.search(r"\byes\b", text, re.IGNORECASE):
        return 1
    if re.search(r"\bno\b", text, re.IGNORECASE):
        return 0
    return None


_CHOICE_TYPES = {"Counting1"}
_INTEGER_TYPES = {"Counting2", "Counting3"}
_YESNO_TYPES = {"TorF1", "TorF2"}

VALID_TYPES = _CHOICE_TYPES | _INTEGER_TYPES | _YESNO_TYPES


def reason_map_plus_score(
    response: str,
    ground_truth: int,
    question_type: str,
) -> tuple[float, str, str | None]:
    """Score a ReasonMap Plus prediction.

    Parameters
    ----------
    response : Raw model output text.
    ground_truth : 0-3 for Counting1, integer for Counting2/3, 0 or 1 for TorF.
    question_type : One of Counting1, Counting2, Counting3, TorF1, TorF2.

    Returns
    -------
    (score, reason, extracted) — score is 1.0 or 0.0.
    """
    if question_type not in VALID_TYPES:
        return 0.0, f"unknown_type:{question_type}", None

    boxed = extract_boxed(response)

    if boxed is None:
        region = extract_answer_region(response)
        boxed = extract_boxed(region)
        if boxed is None:
            boxed = region

    extracted = boxed

    if question_type in _CHOICE_TYPES:
        parsed = _parse_choice(extracted)
        if parsed is None:
            return 0.0, "no_choice_parsed", extracted
        correct = parsed == ground_truth
        letter = chr(ord("A") + parsed)
        gt_letter = chr(ord("A") + ground_truth)
        reason = (
            f"correct:{letter}=={gt_letter}"
            if correct
            else f"wrong:{letter}!={gt_letter}"
        )
        return (1.0 if correct else 0.0), reason, extracted

    if question_type in _INTEGER_TYPES:
        parsed = _parse_integer(extracted)
        if parsed is None:
            return 0.0, "no_integer_parsed", extracted
        correct = parsed == ground_truth
        reason = (
            f"correct:{parsed}=={ground_truth}"
            if correct
            else f"wrong:{parsed}!={ground_truth}"
        )
        return (1.0 if correct else 0.0), reason, extracted

    parsed = _parse_yes_no(extracted)
    if parsed is None:
        parsed = _parse_integer(extracted)
    if parsed is None:
        return 0.0, "no_yesno_parsed", extracted
    correct = parsed == ground_truth
    gt_label = "yes" if ground_truth == 1 else "no"
    pred_label = "yes" if parsed == 1 else "no"
    reason = (
        f"correct:{pred_label}=={gt_label}"
        if correct
        else f"wrong:{pred_label}!={gt_label}"
    )
    return (1.0 if correct else 0.0), reason, extracted
