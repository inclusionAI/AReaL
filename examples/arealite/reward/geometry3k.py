import re

def extract_answer(pred_str, data_name, use_last_number=True):
    match = re.findall(r"\[([0-9\.]+)\]", pred_str)
    if match:
        return match[-1]

    return ""


def geometry3k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
    sol = extract_answer(completions, data_name="")  # str number
    ans = answer

    if sol is None:
        return 0
    if ans is None:
        return 0

    is_numeric = sol.replace('.', '', 1).isdigit()  # Allows for decimal check
    is_latex = sol.startswith("\\frac") or '\\sqrt' in sol
    print(f"completions: {completions}, answer: {answer}")
    # Exact answer matching
    if sol == ans :
        reward = 1
    elif is_numeric and abs(float(sol) - float(ans)) < 1e-4:
        reward = 0.8  # Reward for correct numerical approximation
    elif is_latex:
        # Check if numbers in LaTeX are correct
        expected_numbers = re.findall(r'-?\d+\.?\d*', ans)  # Find all numbers in expected answer
        predicted_numbers = re.findall(r'-?\d+\.?\d*', sol)  # Find all numbers in predicted answer

        if len(expected_numbers) == len(predicted_numbers) and all(
            abs(float(pred) - float(exp)) < 1e-4 for pred, exp in zip(predicted_numbers, expected_numbers)
        ):
           reward = 0.6
    else:
        reward = 0

    return reward