# SPDX-License-Identifier: Apache-2.0

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify.errors import TimeoutException
import re
from areal.utils import logging

logger = logging.getLogger("RewardUtils")

VALID_REWARD_FN = ["clevr_count_70k", "geometry3k"]

_BOX = re.compile(r"\\boxed\s*\{")

def _canon_gold(gt: str) -> str:
    gt = (gt or "").strip()
    if not gt:
        return gt
    # If already boxed, keep it
    if _BOX.search(gt):
        return gt
    # Put gold in a strong LaTeX “answer target”
    return f"\\boxed{{{gt}}}"

def _canon_pred(resp: str) -> str:
    resp = (resp or "").strip()
    # drop thinking
    if "</think>" in resp:
        resp = resp.split("</think>")[-1].strip()
    return resp


def get_custom_reward_fn(path: str, **kwargs):
    if "clevr_count_70k" in path:
        from .clevr_count_70k import clevr_count_70k_reward_fn

        return clevr_count_70k_reward_fn
    elif "geometry3k" in path:
        from .geometry3k import geometry3k_reward_fn

        return geometry3k_reward_fn
    else:
        raise ValueError(
            f"Reward function {path} is not supported. "
            f"Supported reward functions are: {VALID_REWARD_FN}. "
        )

class MathVerifyWorker:
    """Thin wrapper over math_verify with configurable extraction/precision.

    Args:
        try_extract_without_anchor: When False, only answers with explicit anchors
            (e.g., "answer = 1", "final answer = 1") are matched. When True,
            any numeric string in the text may be extracted.
        precision: Number of significant digits that must match.

    Notes:
        Tune these knobs based on dataset format and model output style.
    """

    def __init__(self, try_extract_without_anchor=True, precision: int = 6):
        self.verify_func = math_metric(
            gold_extraction_target=(
                ExprExtractionConfig(
                    try_extract_without_anchor=try_extract_without_anchor
                ),
                LatexExtractionConfig(),
            ),
            pred_extraction_target=(
                ExprExtractionConfig(
                    try_extract_without_anchor=try_extract_without_anchor
                ),
                LatexExtractionConfig(),
            ),
            precision=precision,
        )

    def verify_for_math500(self, response: str, ground_truth: str) -> float:
        try:
            # _canon_gold will make sure ground truth is of format: "\\boxed{" + ground_truth + "}"
            gt = _canon_gold(ground_truth)
            resp = _canon_pred(response)
            ret_score, _ = self.verify_func([gt], [resp])
            return float(ret_score)
        except (Exception, TimeoutException) as e: # TimeoutException is inherited from BaseException, instead of Exception
            logger.warning(
                f"Exception {e} in MathVerifyWorker.verify for response={response} and ground_truth={ground_truth}",
                exc_info=True,
            )
            return 0.0

    def verify(self, response: str, ground_truth: str) -> float: 
        # for gsm8k
        # assume: ground_truth_parsable = "\\boxed{" + ground_truth + "}"
        try:
            ret_score, _ = self.verify_func([ground_truth], [response])
            return float(ret_score)
        except Exception as e:
            logger.warning(
                f"Exception {e} in MathVerifyWorker.verify for response={response} and ground_truth={ground_truth}",
                exc_info=True,
            )
            return 0.0

  
# Math MC Verifier
def _canon_gold_mc(gt: str) -> str:
    gt = (gt or "").strip()
    if not gt:
        return gt
    # For MC, gold is usually A/B/C/D/E (or 5-choice variants). Keep it simple.
    gt = gt.upper()
    # If the dataset sometimes stores "(E)" or "E)" etc.
    m = re.search(r"\b([A-E])\b", gt)
    if m:
        gt = m.group(1)
    # Put in a strong LaTeX “answer target” to help extraction consistently
    if not _BOX.search(gt):
        gt = f"\\boxed{{{gt}}}"
    return gt

def _canon_pred_mc(resp: str) -> str:
    resp = (resp or "").strip()
    if not resp:
        return resp

    # If already boxed, keep as-is
    if _BOX.search(resp):
        return resp

    # Common patterns: "Answer: E", "(E)", "E.", "The answer is (E)", "choice E"
    # Prefer an explicit letter if present.
    m = re.search(r"(?i)\b(?:answer|final|choice)\b[^A-E]*\(?\s*([A-E])\s*\)?", resp)
    if m:
        return f"\\boxed{{{m.group(1).upper()}}}"

    # Otherwise look for a standalone MC letter token.
    # (Guard against matching 'A' in words by requiring boundaries.)
    m = re.search(r"(?i)(?:^|[\s\(\[\{])([A-E])(?:$|[\s\)\]\}\.\,\;\:\!])", resp)
    if m:
        return f"\\boxed{{{m.group(1).upper()}}}"

    # As a last resort, leave response unchanged (math_verify may still extract something)
    return resp


class MathMultipleChoiceVerifyWorker:
    """Verifier for multiple-choice math datasets (A/B/C/D/E style).

    This mirrors MathVerifyWorker but uses MC-specific canonicalization/extraction.
    It still leverages math_verify for robust parsing; we just normalize outputs
    so the extractor reliably finds the selected option.

    Args:
        try_extract_without_anchor: If False, requires answer anchors. For MC,
            leaving True is usually best because model outputs vary a lot.
        precision: kept for API compatibility; not important for letter matching.
        choices: string of valid choice letters.
    """

    def __init__(
        self,
        try_extract_without_anchor: bool = True,
        precision: int = 6,
        choices: str = "ABCDE",
    ):
        self.choices = "".join(sorted(set(choices.upper())))
        # We still use math_verify, but our canon_* functions steer extraction to boxed letters.
        self.verify_func = math_metric(
            gold_extraction_target=(
                ExprExtractionConfig(
                    try_extract_without_anchor=try_extract_without_anchor
                ),
                LatexExtractionConfig(),
            ),
            pred_extraction_target=(
                ExprExtractionConfig(
                    try_extract_without_anchor=try_extract_without_anchor
                ),
                LatexExtractionConfig(),
            ),
            precision=precision,
        )

    def _normalize_gold(self, ground_truth: str) -> str:
        gt = _canon_gold_mc(ground_truth)
        # If gt isn't in choices, try to recover
        m = re.search(r"\b([A-E])\b", gt.upper())
        if m and m.group(1) in self.choices:
            return f"\\boxed{{{m.group(1)}}}"
        return gt

    def _normalize_pred(self, response: str) -> str:
        resp = _canon_pred_mc(response)
        m = re.search(r"\\BOXED\{([A-E])\}", resp.upper())
        if m and m.group(1) in self.choices:
            return f"\\boxed{{{m.group(1)}}}"
        return resp

    def verify(self, response: str, ground_truth: str) -> float:
        try:
            gt = self._normalize_gold(ground_truth)
            resp = self._normalize_pred(response)

            # Primary path: use math_verify scoring
            ret_score, _ = self.verify_func([gt], [resp])
            score = float(ret_score)

            # Hard fallback: direct letter compare (useful if parsing fails)
            if score == 0.0:
                gt_letter = re.search(r"\\boxed\{([A-E])\}", gt.upper())
                pr_letter = re.search(r"\\boxed\{([A-E])\}", resp.upper())
                if gt_letter and pr_letter:
                    return 1.0 if gt_letter.group(1) == pr_letter.group(1) else 0.0

            return score

        except Exception:
            logger.warning(
                f"Exception in MathMultipleChoiceVerifyWorker.verify for response={response} and ground_truth={ground_truth}",
                exc_info=True,
            )
            return 0.0

_MATH_VERIFY_WORKER: MathVerifyWorker | None = None
_MATH_MC_VERIFY_WORKER: MathMultipleChoiceVerifyWorker | None = None


def get_math_verify_worker() -> MathVerifyWorker:
    global _MATH_VERIFY_WORKER
    if _MATH_VERIFY_WORKER is None:
        _MATH_VERIFY_WORKER = MathVerifyWorker()
    return _MATH_VERIFY_WORKER

def get_math_mc_verify_worker() -> MathMultipleChoiceVerifyWorker:
    global _MATH_MC_VERIFY_WORKER
    if _MATH_MC_VERIFY_WORKER is None:
        _MATH_MC_VERIFY_WORKER = MathMultipleChoiceVerifyWorker()
    return _MATH_MC_VERIFY_WORKER

__all__ = [
    "VALID_REWARD_FN",
    "get_custom_reward_fn",
    "MathVerifyWorker",
    "get_math_verify_worker",
    "MathMultipleChoiceVerifyWorker",
    "get_math_mc_verify_worker",
    "gsm8k_reward_fn",
    "geometry3k_reward_fn",
    "clevr_count_70k_reward_fn",
]


_LAZY_IMPORTS = {
    "gsm8k_reward_fn": "areal.reward.gsm8k",
    "geometry3k_reward_fn": "areal.reward.geometry3k",
    "clevr_count_70k_reward_fn": "areal.reward.clevr_count_70k",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
