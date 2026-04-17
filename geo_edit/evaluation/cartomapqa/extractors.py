from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from geo_edit.utils.text_utils import extract_answer

_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_UNIT_RE = re.compile(
    r"([-+]?\d[\d,]*(?:\.\d+)?)\s*(m|meters?|ft|feet)",
    re.IGNORECASE,
)
_ANSWER_PREFIX_RE = re.compile(
    r"^(?:answer|final answer)\s*[:\-]\s*",
    re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    stripped = _THINK_RE.sub("", text).strip()
    return stripped if stripped else text.strip()


def _strip_answer_tag(text: str) -> str:
    extracted = extract_answer(text, "split")
    if extracted is not None:
        return extracted
    return _strip_thinking(text)


def _clean_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# ── MFS ──────────────────────────────────────────────────────────────────


def extract_mfs(text: str) -> Optional[str]:
    raw = _strip_answer_tag(text).strip()
    for char in raw:
        if char.upper() in "ABCDE":
            return char.upper()
    return None


# ── STMF-presence ────────────────────────────────────────────────────────


def extract_stmf_presence(text: str) -> Optional[str]:
    raw = _strip_answer_tag(text).strip().lower()
    raw = raw.lstrip("*").strip()
    first_word = raw.split()[0] if raw.split() else ""
    first_word = first_word.strip(".,;:!?")
    if first_word.startswith("yes"):
        return "yes"
    if first_word.startswith("no"):
        return "no"
    for word in raw.split():
        w = word.strip(".,;:!?*\"'()[]")
        if w in ("yes", "no"):
            return w
    return None


# ── STMF-counting ────────────────────────────────────────────────────────

_TEXT_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def extract_stmf_counting(text: str) -> Optional[int]:
    raw = _strip_answer_tag(text)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _ANSWER_PREFIX_RE.sub("", line)
        m = re.search(r"[-+]?\d+", line)
        if m:
            return int(m.group(0))
    for word in raw.lower().split():
        word = word.strip(".,;:!?")
        if word in _TEXT_NUMBERS:
            return _TEXT_NUMBERS[word]
    return None


# ── STMF-name_listing ────────────────────────────────────────────────────

_UNABLE_PATTERNS = [
    "there are no",
    "there is no",
    "i am unable to",
    "i'm not able",
    "none",
    "no names found",
]


def extract_stmf_name_listing(text: str) -> List[str]:
    raw = _strip_answer_tag(text).strip()
    if not raw:
        return []
    raw_lower = raw.lower().strip()
    if any(raw_lower.startswith(p) or raw_lower == p for p in _UNABLE_PATTERNS):
        return []
    names = []
    for line in raw.splitlines():
        name = re.sub(r"^\d+\.\s*", "", line).replace("*", "").strip()
        name = name.rstrip(".")
        name = name.replace("<|eot|>", "")
        if name and name != '""':
            names.append(name)
    return names


# ── MTMF ─────────────────────────────────────────────────────────────────


def extract_mtmf(text: str) -> Optional[Dict[str, Dict[str, Any]]]:
    raw = _strip_answer_tag(text).strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            pass
    return None


# ── RLE ──────────────────────────────────────────────────────────────────


def extract_rle(text: str) -> Optional[Dict[str, Any]]:
    raw = _strip_answer_tag(text).strip()
    first_line = raw.splitlines()[0].strip() if raw else ""
    m = _UNIT_RE.search(first_line)
    if m:
        value = float(m.group(1).replace(",", ""))
        unit_raw = m.group(2).lower()
        unit = "meters" if unit_raw.startswith("m") else "feet"
        return {"value": value, "unit": unit}
    m = _UNIT_RE.search(raw)
    if m:
        value = float(m.group(1).replace(",", ""))
        unit_raw = m.group(2).lower()
        unit = "meters" if unit_raw.startswith("m") else "feet"
        return {"value": value, "unit": unit}
    m = _NUMERIC_RE.search(first_line)
    if m:
        return {"value": float(m.group(0)), "unit": "unknown"}
    return None


# ── MML ──────────────────────────────────────────────────────────────────


def extract_mml(text: str) -> Optional[Dict[str, str]]:
    raw = _strip_answer_tag(text).strip()
    try:
        data = json.loads(raw)
        if "road_1" in data and "road_2" in data:
            return {"road_1": str(data["road_1"]), "road_2": str(data["road_2"])}
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if "road_1" in data and "road_2" in data:
                return {"road_1": str(data["road_1"]), "road_2": str(data["road_2"])}
        except (json.JSONDecodeError, TypeError):
            pass
    return None


# ── SRN ──────────────────────────────────────────────────────────────────


def extract_srn(text: str) -> Optional[List[str]]:
    raw = _strip_answer_tag(text).strip()
    raw = raw.replace("**Answer:**", "").replace("*", "")
    raw = raw.replace("Answer:", "")
    raw = raw.replace("<|eot|>", "")
    bracket_m = re.search(r"\[(.+?)\]", raw, re.DOTALL)
    if bracket_m:
        raw = bracket_m.group(1)
    items = [item.strip() for item in raw.split(",")]
    items = [item for item in items if item]
    if not items:
        return None
    return items


# ── Dispatch ─────────────────────────────────────────────────────────────

EXTRACTORS = {
    "cartomapqa_mfs": extract_mfs,
    "cartomapqa_stmf_presence": extract_stmf_presence,
    "cartomapqa_stmf_counting": extract_stmf_counting,
    "cartomapqa_stmf_name_listing": extract_stmf_name_listing,
    "cartomapqa_mtmf": extract_mtmf,
    "cartomapqa_rle": extract_rle,
    "cartomapqa_mml": extract_mml,
    "cartomapqa_srn": extract_srn,
}


def extract_structured(task_name: str, text: str) -> Any:
    extractor = EXTRACTORS.get(task_name)
    if extractor is None:
        raise ValueError(f"No extractor for task: {task_name}")
    return extractor(text)
