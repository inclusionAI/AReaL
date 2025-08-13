import re



def extract_answer(pred_str, data_name, use_last_number=True):
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", pred_str, re.IGNORECASE)
    if match:
        return match.group(1).upper() 
    return ""

def longvideo_reason_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
    sol = extract_answer(completions, data_name="")
    ans = answer.strip().upper() if answer else ""

    if not sol or not ans:
        return 0

    if sol == ans:
        print(f"completions: {completions}, answer: {answer}")
        return 1
    return 0
