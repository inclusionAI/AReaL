from areal.reward import get_math_verify_worker
from areal.utils import logging

logger = logging.getLogger("GSM8KReward")


def gsm8k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        logger.warning("Exception in gsm8k_reward_fn", exc_info=True)
        return 0.0
