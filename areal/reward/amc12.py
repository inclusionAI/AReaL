import re

from areal.reward import get_math_mc_verify_worker
from areal.utils import logging

logger = logging.getLogger("AMC12Reward")


def extract_mc_answer(text: str) -> str:
    """
    Extract a multiple-choice answer (A/B/C/D/E) from text.
    Prefers explicit answer-style patterns, then falls back to
    standalone letter detection.
    """
    if not text:
        return ""

    text = str(text).upper()

    # Pattern 1: "Answer: E", "Final answer is D", etc.
    m = re.search(r"\b(?:ANSWER|FINAL|CHOICE)\b[^A-E]*\(?\s*([A-E])\s*\)?", text)
    if m:
        return m.group(1)

    # Pattern 2: boxed format \boxed{E}
    m = re.search(r"\\BOXED\s*\{\s*([A-E])\s*\}", text)
    if m:
        return m.group(1)

    # Pattern 3: standalone letter token
    m = re.search(r"(?:^|[\s\(\[\{])([A-E])(?:$|[\s\)\]\}\.\,\;\:\!])", text)
    if m:
        return m.group(1)

    return ""


def amc12_mc_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """
    Simple 0/1 reward for AMC12 multiple-choice questions.
    """
    try:
        pred_letter = extract_mc_answer(str(completions))
        gold_letter = extract_mc_answer(str(answer)) or str(answer).strip().upper()

        if not pred_letter or not gold_letter:
            return 0.0

        worker = get_math_mc_verify_worker()
        return worker.verify(pred_letter, gold_letter)

    except Exception:
        logger.warning("Exception in amc12_mc_reward_fn", exc_info=True)
        return 0.0
