from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from .constants import EVAL_QUERY_PROMPT, EVAL_SYSTEM_PROMPT


_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


def _parse_score(text: str) -> str:
    match = re.search(r"\bscore\s*:\s*([01])\b", text, re.IGNORECASE)
    return match.group(1) if match else ""


def _extract_answer(text: str, mode: str) -> Optional[str]:
    if mode == "split":
        parts = "<answer>{}</answer>".split("{}")
        if parts[0] not in text or parts[1] not in text:
            return None
        return text.split(parts[0])[-1].split(parts[1])[0].strip()
    if mode == "strict":
        match = _ANSWER_PATTERN.search(text)
        return match.group(1).strip() if match else None
    if mode == "strict_v2":
        match = re.search(r"<answer>(.*?)</answer>$", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    raise ValueError(f"Unsupported extract_answer_tags: {mode}")


def _is_valid_direct_answer(response: str, direct_answer_format: str) -> bool:
    if not re.match(direct_answer_format, response, re.DOTALL):
        return False
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        return False
    if response.count("<answer>") != 1 or response.count("</answer>") != 1:
        return False
    if "<action>" in response or "</action>" in response:
        return False
    return True


def _is_valid_tool_call(response: str, step_tool_call_format: str) -> bool:
    if not re.match(step_tool_call_format, response, re.DOTALL):
        return False
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        return False
    if response.count("<action>") != 1 or response.count("</action>") != 1:
        return False
    if "<answer>" in response or "</answer>" in response:
        return False
    return True


def tool_format_reward(
    predict_str_list: List[str], extra_info: Optional[Dict[str, Any]] = None
) -> tuple[float, int]:
    conv_rounds = len(predict_str_list)
    format_score = 0.0
    tool_call_count = 0
    direct_answer_format = r"^<think>.*</think>.*<answer>.*</answer>$"
    step_tool_call_format = r"^<think>.*</think>.*<action>.*</action>$"
    tool_call_pattern = _ACTION_PATTERN

    if conv_rounds == 1:
        response = predict_str_list[0].strip()
        if tool_call_pattern.findall(response):
            tool_call_count += 1
        if _is_valid_direct_answer(response, direct_answer_format):
            format_score = 1.0
    else:
        tool_call_match_flag = True
        for response in predict_str_list[:-1]:
            response = response.strip()
            if tool_call_pattern.findall(response):
                tool_call_count += 1
            if not _is_valid_tool_call(response, step_tool_call_format):
                tool_call_match_flag = False
                break
        final_answer_match_flag = _is_valid_direct_answer(
            predict_str_list[-1].strip(), direct_answer_format
        )
        if tool_call_match_flag and final_answer_match_flag:
            format_score = 1.0
    return format_score, tool_call_count


def _judge_with_openai(
    question: str, ground_truth: str, prediction: str, model: str
) -> str:
    try:
        from openai import OpenAI
    except Exception:
        return ""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""

    client = OpenAI(api_key=api_key)
    prompt = EVAL_QUERY_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    response = client.responses.create(
        model=model,
        instructions=EVAL_SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    )
    return _parse_score(response.output_text or "")


def _inner_acc_reward(
    prompt: str,
    predict_str_list: List[str],
    original_answer: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    predict_str = " ".join(predict_str_list)
    extract_answer = bool(extra_info and extra_info.get("gpt_extract_answer", False))
    if extract_answer:
        mode = extra_info.get("extract_answer_tags", "strict")
        extracted = _extract_answer(predict_str, mode)
        if extracted is None:
            return 0.0
        predict_str = extracted

    model = (
        extra_info.get("judge_model", "gpt-5-mini")
        if extra_info is not None
        else "gpt-5-mini"
    )
    score_str = _judge_with_openai(prompt, original_answer, predict_str, model)
    if not score_str:
        return 0.0
    return 1.0 if "1" in score_str else 0.0


def compute_score(
    prompt: str,
    predict_str_list: List[str],
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    acc_reward_weight = extra_info.get("acc_reward_weight", 1.0) if extra_info else 1.0
    format_reward_weight = (
        extra_info.get("format_reward_weight", 1.0) if extra_info else 1.0
    )
    tool_call_penalty = 0.1
    if extra_info is not None and "tool_call_penalty" in extra_info:
        tool_call_penalty = extra_info.get("tool_call_penalty", 0.1)

    acc = _inner_acc_reward(prompt, predict_str_list, ground_truth, extra_info)
    format_score, tool_call_count = tool_format_reward(predict_str_list, extra_info)

    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format_score

    tool_penalty_factor = (1 - tool_call_penalty) if tool_call_count > 0 else 1.0
    tool_reward = (
        extra_info.get("use_tool_reward_weight", 0.0)
        if tool_call_count > 0 and extra_info is not None
        else 0.0
    )
    score = tool_penalty_factor * acc_score + format_score + tool_reward
    return float(score)


def geo_edit_reward_fn(
    prompt: str,
    completions: Any,
    prompt_ids: List[int],
    completion_ids: List[int],
    ground_truth: str | None = None,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> float:
    if completions is None:
        predict_str_list: List[str] = []
    elif isinstance(completions, list):
        predict_str_list = [str(x) for x in completions]
    else:
        predict_str_list = [str(completions)]

    if ground_truth is None:
        ground_truth = kwargs.get("answer", "") or kwargs.get("ground_truth", "")
    if extra_info is None:
        extra_info = kwargs.get("reward_extra_info", None) or kwargs.get("extra_info", None)

    return compute_score(prompt, predict_str_list, ground_truth, extra_info)
