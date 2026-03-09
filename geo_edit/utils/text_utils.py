from __future__ import annotations

import math
import re
from typing import List, Optional

ANSWER_TEMPLATE = "<answer>{}</answer>"


def parse_score(text: str) -> str:
    match = re.search(r"\bscore\s*:\s*([01])\b", text, re.IGNORECASE)
    return match.group(1) if match else ""


def parse_leakage_score(text: str) -> str:
    """Parse leakage detection result from judge response.

    Looks for 'Leakage: 0' or 'Leakage: 1' pattern.

    Returns:
        '0' if no leakage, '1' if leakage detected, '' if not found.
    """
    match = re.search(r"\bleakage\s*:\s*([01])\b", text, re.IGNORECASE)
    return match.group(1) if match else ""


def extract_answer(text: str, mode: str) -> Optional[str]:
    parts = ANSWER_TEMPLATE.split("{}")
    if mode == "split":
        if parts[0] not in text or parts[1] not in text:
            return None
        return text.split(parts[0])[-1].split(parts[1])[0].strip()
    if mode == "strict":
        start = text.find(parts[0])
        if start == -1:
            return None
        start += len(parts[0])
        if parts[1]:
            end = text.find(parts[1], start)
            if end == -1:
                return None
            return text[start:end].strip()
        return text[start:].strip()
    raise ValueError(f"Unknown extract mode: {mode}")


def get_final_prediction(predict_str_list: List[str], extract_mode: Optional[str]) -> str:
    if not predict_str_list:
        return ""
    last = predict_str_list[-1].strip()
    if not extract_mode:
        return last
    extracted = extract_answer(last, extract_mode)
    return extracted if extracted is not None else last


def extract_choice_letter(text: str) -> Optional[str]:
    patterns = [
        r"^([A-Z])[\.\)]\s",
        r"^([A-Z])$",
        r"answer is ([A-Z])",
        r"correct answer is ([A-Z])",
        r"choose ([A-Z])",
        r"select ([A-Z])",
        r"option ([A-Z])",
        r"\(([A-Z])\)",
        r"^\s*([A-Z])\s*[\:\.\)]\s*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    for word in text.split()[:5]:
        cleaned = word.strip(".,;:()[]{}")
        if len(cleaned) == 1 and cleaned.upper() in "ABCDEFGHIJ":
            return cleaned.upper()
    return None


def clean_response(text: str, max_length: int = 200) -> str:
    prefixes_to_remove = [
        "Based on the image,",
        "Looking at the image,",
        "The image shows",
        "In the image,",
        "According to the image,",
        "From the image,",
        "I can see that",
        "The answer is:",
        "Answer:",
    ]
    text = text.strip()
    text_lower = text.lower()
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
            text_lower = text.lower()
    text = " ".join(text.split())
    if len(text) > max_length:
        sentences = re.split(r"[.!?]+", text)
        if sentences and len(sentences[0]) <= max_length:
            text = sentences[0].strip() + "."
        else:
            words = text[:max_length].split()
            text = " ".join(words[:-1]) + "..."
    return text


def parse_vlm_response(response: str, choices: Optional[List[str]] = None) -> str:
    if not response:
        return choices[0] if choices else ""
    response = response.strip()
    if not choices:
        return clean_response(response)
    answer_letter = extract_choice_letter(response)
    if answer_letter:
        idx = ord(answer_letter.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    response_lower = response.lower()
    for choice in choices:
        if choice.lower() in response_lower:
            return choice
    return choices[0]


def format_prompt_with_choices(question: str, choices: List[str]) -> str:
    question = question.strip()
    if not question.endswith(("?", ".", "!")):
        question += "?"
    formatted_choices = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    return f"""{question}

Choose from the following options:
{formatted_choices}

Please select the correct answer by stating the letter (A, B, C, etc.) of your choice."""


def calculate_confidence_score(logprobs: List[float]) -> float:
    if not logprobs:
        return 0.5
    avg_logprob = sum(logprobs) / len(logprobs)
    confidence = 1 / (1 + math.exp(-(avg_logprob + 2.5)))
    return max(0.0, min(1.0, confidence))
