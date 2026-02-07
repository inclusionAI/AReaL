from math_verify.grader import verify as math_verify_verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

from areal.utils import logging

logger = logging.getLogger("RewardUtils")

VALID_REWARD_FN = ["clevr_count_70k", "geometry3k"]


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

    Uses ``parse()`` + ``verify()`` directly instead of ``math_metric()``
    so that signal-based timeouts can be disabled (``parsing_timeout=None``,
    ``timeout_seconds=None``). This avoids ``signal.alarm()`` which only
    works in the main thread.

    Args:
        try_extract_without_anchor: When False, only answers with explicit anchors
            (e.g., "answer = 1", "final answer = 1") are matched. When True,
            any numeric string in the text may be extracted.
        precision: Number of significant digits that must match.

    Notes:
        Tune these knobs based on dataset format and model output style.
    """

    def __init__(self, try_extract_without_anchor=True, precision: int = 6):
        self.gold_extraction_target = (
            ExprExtractionConfig(
                try_extract_without_anchor=try_extract_without_anchor
            ),
            LatexExtractionConfig(),
        )
        self.pred_extraction_target = (
            ExprExtractionConfig(
                try_extract_without_anchor=try_extract_without_anchor
            ),
            LatexExtractionConfig(),
        )
        self.precision = precision

    def verify(self, response: str, ground_truth: str) -> float:
        try:
            gold_parsed = parse(
                ground_truth,
                extraction_config=self.gold_extraction_target,
                parsing_timeout=None,
            )
            pred_parsed = parse(
                response,
                extraction_config=self.pred_extraction_target,
                parsing_timeout=None,
            )
            if not gold_parsed or not pred_parsed:
                return 0.0
            result = math_verify_verify(
                gold_parsed,
                pred_parsed,
                float_rounding=self.precision,
                timeout_seconds=None,
            )
            return 1.0 if result else 0.0
        except Exception:
            logger.warning(
                f"Exception in MathVerifyWorker.verify for response={response} and ground_truth={ground_truth}",
                exc_info=True,
            )
            return 0.0


_MATH_VERIFY_WORKER: MathVerifyWorker | None = None


def get_math_verify_worker() -> MathVerifyWorker:
    global _MATH_VERIFY_WORKER
    if _MATH_VERIFY_WORKER is None:
        _MATH_VERIFY_WORKER = MathVerifyWorker()
    return _MATH_VERIFY_WORKER
