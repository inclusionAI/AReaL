from areal.reward import get_math_verify_worker
from areal.utils import logging

logger = logging.getLogger("MATH500_Reward")


def math500_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    try:
        worker = get_math_verify_worker()
        return worker.verify_for_math500(str(completions), str(answer))
    except Exception:
        logger.warning("Exception in MATH500_Reward", exc_info=True)
        return 0.0
