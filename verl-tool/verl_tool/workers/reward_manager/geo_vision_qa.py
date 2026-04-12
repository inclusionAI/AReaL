import re
import torch
import numpy as np
import logging
from collections import defaultdict
from verl import DataProto
from verl.workers.reward_manager import register

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"<action>.*?</action>", re.DOTALL | re.IGNORECASE)


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


def _compute_repetition_penalty(text: str) -> float:
    """R_rep ∈ {-3.0, -2.0, -1.5, 0.0} — penalise contiguous repetition."""
    if not text or len(text) < 20:
        return 0.0

    # Character-level: same char repeated 50+ times (e.g. "aaa...a")
    if re.search(r"(.)\1{49,}", text):
        return -3.0

    # Token/word-level: same word repeated 20+ times contiguously
    if re.search(r"\b(\w+)(?:\s+\1){19,}\b", text):
        return -3.0

    # Phrase-level: 4+ word phrase repeated 10+ times
    if re.search(r"(\b\w+(?:\s+\w+){3,})(?:\s+\1){9,}", text):
        return -2.0

    # Sentence-level: identical sentence repeated 5+ times
    sentences = re.split(r"[.!?\n]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if sentences:
        from collections import Counter
        counts = Counter(sentences)
        max_count = counts.most_common(1)[0][1] if counts else 0
        if max_count >= 10:
            return -2.0
        if max_count >= 5:
            return -1.5

    # Moderate: same word repeated 10+ times
    if re.search(r"\b(\w+)(?:\s+\1){9,}\b", text):
        return -1.5

    return 0.0


def _compute_format_reward(text: str) -> float:
    """R_format = (𝕀{format_ok} - 0.5) × 2 → {-1.0, +1.0}."""
    has_think = bool(_THINK_RE.search(text))
    has_answer = bool(_ANSWER_RE.search(text))

    if not has_think:
        return -1.0

    if not has_answer:
        return -1.0

    # If there are <action> blocks, they must be preceded by <think> blocks
    action_blocks = _ACTION_RE.findall(text)
    if action_blocks:
        think_positions = [m.end() for m in _THINK_RE.finditer(text)]
        action_positions = [m.start() for m in _ACTION_RE.finditer(text)]
        for apos in action_positions:
            if not any(tpos <= apos for tpos in think_positions):
                return -1.0

    return 1.0


@register("geo_vision_qa")
class GeoVisionQARewardManager:
    name = "geo_vision_qa"

    def __init__(self, config=None, tokenizer=None, num_examine=3, compute_score=None, reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.config = config

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

            # R_rep: repetition penalty
            r_rep = _compute_repetition_penalty(response_str)

            # R_format: format compliance
            r_format = _compute_format_reward(response_str)

            # R_correct: answer correctness
            prediction = extract_answer(response_str)
            accuracy = compute_score(prediction, ground_truth) if prediction else 0.0
            r_correct = 1.0 if accuracy > 0 else 0.0

            # R(U) = R_rep + R_format + R_correct
            reward = r_rep + r_format + r_correct

            reward_extra_info["accuracy"].append(accuracy)
            reward_extra_info["score"].append(reward)
            reward_extra_info["has_answer_tag"].append(1.0 if prediction else 0.0)
            reward_extra_info["r_rep"].append(r_rep)
            reward_extra_info["r_format"].append(r_format)
            reward_extra_info["r_correct"].append(r_correct)

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
                print(f"[r_rep] {r_rep} [r_format] {r_format} [r_correct] {r_correct} [reward] {reward}")

        correct_len = np.mean(reward_extra_info["correct_response_length"]) if reward_extra_info["correct_response_length"] else 0.0
        wrong_len = np.mean(reward_extra_info["wrong_response_length"]) if reward_extra_info["wrong_response_length"] else 0.0
        reward_extra_info["correct_response_length"] = [correct_len] * len(reward_tensor)
        reward_extra_info["wrong_response_length"] = [wrong_len] * len(reward_tensor)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(sorted(reward_extra_info.items()))}
        return reward_tensor
