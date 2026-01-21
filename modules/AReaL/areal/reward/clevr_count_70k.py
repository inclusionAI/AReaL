import re


def extract_answer(pred_str, data_name, use_last_number=True):
    match = re.findall(r"\[([0-9\.]+)\]", pred_str)
    if match:
        return match[-1]

    return ""


def clevr_count_70k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
    sol = extract_answer(completions, data_name="")  # str number
    ans = answer

    if sol is None:
        return 0
    if ans is None:
        return 0

    if sol.strip() == ans.strip():
        print(f"completions: {completions}, answer: {answer}")
        return 1

    return 0
