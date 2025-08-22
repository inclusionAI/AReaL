import re

from areal.reward.math_parser import math_equal


def extract_answer(pred_str, data_name, use_last_number=True):
    matches = re.findall(r"\[([^\]]+)\]", pred_str)
    if matches:
        return matches[-1]

    return ""


def geometry3k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
    sol = extract_answer(completions, data_name="")  # str number
    ans = answer
    sol = sol.replace(" ", "")
    ans = ans.replace(" ", "")
    if sol is None:
        return 0
    if ans is None:
        return 0

    if math_equal(sol, ans):
        # print(f"completions: {completions}, answer: {answer}")
        return 1
    return 0
