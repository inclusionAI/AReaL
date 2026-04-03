import re
import torch
import numpy as np
import logging
from collections import defaultdict
from verl import DataProto
from verl.workers.reward_manager import register

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str:
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


def compute_score(prediction: str, ground_truth) -> float:
    if isinstance(ground_truth, list):
        return max(compute_score(prediction, gt) for gt in ground_truth)

    prediction = prediction.strip().lower()
    ground_truth = str(ground_truth).strip().lower()

    if prediction == ground_truth:
        return 1.0

    try:
        if abs(float(prediction) - float(ground_truth)) < 1e-6:
            return 1.0
    except (ValueError, TypeError):
        pass

    return 0.0


@register("geo_vision_qa")
class GeoVisionQARewardManager:
    name = "geo_vision_qa"

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_printed = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")

            prediction = extract_answer(response_str)

            if not prediction:
                reward = -0.5
                accuracy = 0.0
            else:
                accuracy = compute_score(prediction, ground_truth)
                reward = 1.0 if accuracy > 0 else 0.0

            reward_extra_info["accuracy"].append(accuracy)
            reward_extra_info["score"].append(reward)
            reward_extra_info["has_answer_tag"].append(1.0 if prediction else 0.0)

            if accuracy > 0:
                reward_extra_info["correct_response_length"].append(valid_response_length)
            else:
                reward_extra_info["wrong_response_length"].append(valid_response_length)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_printed:
                already_printed[data_source] = 0
            if already_printed[data_source] < self.num_examine:
                already_printed[data_source] += 1
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[response] {response_str[:500]}...")
                print(f"[ground_truth] {ground_truth}")
                print(f"[prediction] {prediction}")
                print(f"[accuracy] {accuracy} [reward] {reward}")

        correct_len = np.mean(reward_extra_info["correct_response_length"]) if reward_extra_info["correct_response_length"] else 0.0
        wrong_len = np.mean(reward_extra_info["wrong_response_length"]) if reward_extra_info["wrong_response_length"] else 0.0
        reward_extra_info["correct_response_length"] = [correct_len] * len(reward_tensor)
        reward_extra_info["wrong_response_length"] = [wrong_len] * len(reward_tensor)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(sorted(reward_extra_info.items()))}
        return reward_tensor
