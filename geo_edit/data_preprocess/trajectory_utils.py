from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from geo_edit.utils.text_utils import extract_answer


def load_trajectory(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_meta_info(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def get_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)
    return str(content) if content else ""


def get_question_from_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    for turn in trajectory:
        if turn.get("role") == "user":
            return get_text_from_content(turn.get("content", ""))
    return ""


def extract_final_answer(trajectory: List[Dict[str, Any]]) -> Optional[str]:
    if not trajectory:
        return None
    for turn in reversed(trajectory):
        if turn.get("role") == "assistant":
            content = get_text_from_content(turn.get("content", ""))
            if "<answer>" in content:
                return extract_answer(content, mode="split")
    return None


def extract_thinking_text(trajectory: List[Dict[str, Any]]) -> str:
    texts = []
    last_answer_idx = -1
    for i in range(len(trajectory) - 1, -1, -1):
        turn = trajectory[i]
        if turn.get("role") == "assistant":
            content = get_text_from_content(turn.get("content", ""))
            if "<answer>" in content.lower():
                last_answer_idx = i
                break
    for i, turn in enumerate(trajectory):
        if turn.get("role") != "assistant":
            continue
        if i == last_answer_idx:
            continue

        content = get_text_from_content(turn.get("content", ""))
        if content.strip():
            texts.append(content)

    return "\n\n".join(texts)


def extract_answer_values(trajectory: List[Dict[str, Any]]) -> List[str]:
    answers = []
    for msg in trajectory:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        for m in re.finditer(r"<answer>(.*?)</answer>", content, re.DOTALL):
            answers.append(m.group(1).strip())
    return answers


def is_brute_force(trajectory: List[Dict[str, Any]], meta: Dict[str, Any]) -> bool:
    for msg in trajectory:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if re.search(r"[Bb]y\s+elimination", content):
            return True

    answer_values = extract_answer_values(trajectory)
    if len(set(answer_values)) > 3:
        return True

    return False
