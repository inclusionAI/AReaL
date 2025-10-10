from areal.reward.math_parser import process_results


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    return int(process_results(completions, answer)[0])
