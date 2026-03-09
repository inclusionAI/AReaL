"""Trajectory filtering judge using OpenAI-compatible API.

This module provides AI-powered filtering for trajectory data:
1. Wrong answer filtering - verify final answer correctness
2. Answer leakage detection - detect <answer> tags in thinking/tool-call phases

The leakage detection follows the three-phase protocol:
- Phase 1 (Reasoning): NO <answer> allowed
- Phase 2 (Tool Call): NO <answer> allowed
- Phase 3 (Final Answer): <answer> is expected here
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from openai import OpenAI

from geo_edit.prompts import (
    EVAL_QUERY_PROMPT,
    EVAL_SYSTEM_PROMPT,
    LEAKAGE_DETECTION_QUERY_PROMPT,
    LEAKAGE_DETECTION_SYSTEM_PROMPT,
)
from geo_edit.utils.text_utils import parse_leakage_score, parse_score

# Pattern to detect <answer> tags (case insensitive)
ANSWER_TAG_PATTERN = re.compile(r"<answer>", re.IGNORECASE)

# Known tool names for tool plan extraction
KNOWN_TOOL_NAMES = [
    # General tools
    "image_crop", "image_label", "draw_line", "bounding_box", "image_highlight",
    "text_ocr", "auto_segment", "bbox_segment", "grounding_dino",
    # Math tools
    "math_latex_ocr", "math_image_describe", "formula_ocr", "gllava", "multimath", "ovr",
    # Table tools
    "table_ocr",
    # Chart tools
    "chart_data_extract", "chart_trend_analysis", "chart_text_ocr", "chartmoe",
    # Map tools
    "text_spotting",
    # Document tools
    "seal_ocr",
]

# Build regex pattern for tool name extraction (case insensitive)
TOOL_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in KNOWN_TOOL_NAMES) + r")\b",
    re.IGNORECASE
)


@dataclass
class TrajectoryFilterConfig:
    """Configuration for trajectory filtering."""

    model: str = "gpt-5-mini-2025-08-07"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    filter_wrong_answers: bool = True
    filter_answer_leakage: bool = True
    filter_tool_mismatch: bool = False  # Filter if Phase 1 plan doesn't match Phase 2 tool calls
    leakage_check_mode: str = "quick"  # "quick" (regex only) or "full" (AI-based)
    max_workers: int = 16


@dataclass
class FilterStats:
    """Statistics for trajectory filtering."""

    total: int = 0
    passed: int = 0
    filtered_wrong_answer: int = 0
    filtered_leakage: int = 0
    filtered_tool_mismatch: int = 0
    failed: int = 0
    api_errors: int = 0

    def summary(self) -> str:
        """Generate a summary string of the filtering statistics."""
        lines = [
            "=== Filtering Statistics ===",
            f"Total subfolders: {self.total}",
            f"Passed: {self.passed}",
            f"Filtered (wrong answer): {self.filtered_wrong_answer}",
            f"Filtered (answer leakage): {self.filtered_leakage}",
            f"Filtered (tool mismatch): {self.filtered_tool_mismatch}",
            f"Failed to process: {self.failed}",
            f"API errors: {self.api_errors}",
        ]
        return "\n".join(lines)


class TrajectoryJudge:
    """Judge for trajectory filtering using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini-2025-08-07",
        api_base: Optional[str] = None,
    ):
        """Initialize the trajectory judge.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: Model name to use for evaluation.
            api_base: Optional custom API base URL.
        """
        client_kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY")}
        if api_base is not None:
            client_kwargs["base_url"] = api_base
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.api_mode = self._resolve_api_mode(api_base)

    @staticmethod
    def _resolve_api_mode(api_base: Optional[str]) -> str:
        """Determine API mode based on base URL."""
        if api_base and "matrixllm.alipay.com" in api_base.lower():
            return "chat"
        return "responses"

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call the API and return the response text.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.

        Returns:
            Response text from the API.
        """
        if self.api_mode == "chat":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content if resp.choices else ""
        else:
            resp = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
            )
            return resp.output_text or ""

    def judge_correctness(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
    ) -> Tuple[bool, str]:
        """Judge if the prediction is correct.

        Args:
            question: The question being answered.
            ground_truth: The ground truth answer.
            prediction: The model's predicted answer.

        Returns:
            Tuple of (is_correct, raw_response).
            is_correct is True if the prediction matches ground truth.
        """
        prompt = EVAL_QUERY_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )
        response = self._call_api(EVAL_SYSTEM_PROMPT, prompt)
        score = parse_score(response)
        is_correct = score == "1"
        return is_correct, response

    def detect_leakage(
        self,
        question: str,
        ground_truth: str,
        thinking_text: str,
        use_ai: bool = True,
    ) -> Tuple[bool, str]:
        """Detect if the thinking process contains answer leakage.

        Leakage means the model generated <answer> tags or final answers
        in the reasoning/tool-call phases (Phase 1 & 2) when it should
        only do so in Phase 3.

        Args:
            question: The question being answered.
            ground_truth: The ground truth answer.
            thinking_text: The model's thinking/reasoning text (Phase 1 & 2 only).
            use_ai: If True, use AI judge for subtle cases. If False, only use regex.

        Returns:
            Tuple of (has_leakage, reason).
            has_leakage is True if leakage is detected.
        """
        # Quick check: look for <answer> tags in thinking text
        if ANSWER_TAG_PATTERN.search(thinking_text):
            return True, "Found <answer> tag in thinking/tool-call phase"

        # If AI check is disabled, return no leakage
        if not use_ai:
            return False, "No <answer> tag found (regex only)"

        # Use AI for more subtle protocol violations
        prompt = LEAKAGE_DETECTION_QUERY_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            thinking_text=thinking_text,
        )
        response = self._call_api(LEAKAGE_DETECTION_SYSTEM_PROMPT, prompt)
        score = parse_leakage_score(response)
        has_leakage = score == "1"
        return has_leakage, response


def quick_leakage_check(thinking_text: str) -> Tuple[bool, str]:
    """Quick leakage check using regex only (no API call).

    This is a fast check that detects obvious <answer> tags in the
    thinking/tool-call phases without needing to call an external API.

    Args:
        thinking_text: The model's thinking/reasoning text (Phase 1 & 2 only).

    Returns:
        Tuple of (has_leakage, reason).
    """
    if ANSWER_TAG_PATTERN.search(thinking_text):
        return True, "Found <answer> tag in thinking/tool-call phase"
    return False, "No <answer> tag found"
