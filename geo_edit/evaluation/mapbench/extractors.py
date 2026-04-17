from __future__ import annotations

import re
from typing import List, Optional

from geo_edit.utils.text_utils import extract_answer

_ARROW_RE = re.compile(r"->")
_STRIP_RE = re.compile(r"^[\W_]+|[\W_]+$")


def extract_navigation_steps(text: str) -> List[str]:
    raw = extract_answer(text, "split")
    if raw is None:
        raw = text.strip()

    steps = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(_ARROW_RE.findall(line)) != 1:
            continue
        line = _STRIP_RE.sub("", line)
        if "next node" in line.lower():
            return []
        line = line.replace("*", "")
        if line:
            steps.append(line)
    return steps
