"""Unified judge interface for trajectory validation.

This module provides different judge implementations:
- LLMJudge: Uses LLM (GPT-4o-mini) for complex text comparisons
- IntegerMatchJudge: Direct integer comparison (for mapqa, visworld)
- RelaxedMatchJudge: Relaxed accuracy with tolerance (for chartqa)
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from geo_edit.evaluation.trajectory_judge import TrajectoryJudge


class BaseJudge(ABC):
    """Base class for trajectory validation judges."""

    @abstractmethod
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
            Tuple of (is_correct, reason).
        """
        pass


class LLMJudge(BaseJudge):
    """LLM-based judge using TrajectoryJudge."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        api_base: Optional[str] = None,
    ):
        self._judge = TrajectoryJudge(
            api_key=api_key,
            model=model,
            api_base=api_base,
        )

    def judge_correctness(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
    ) -> Tuple[bool, str]:
        return self._judge.judge_correctness(question, ground_truth, prediction)


class IntegerMatchJudge(BaseJudge):
    """Integer matching judge for counting tasks (mapqa, visworld)."""

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract answer from <answer> tags."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Try <|begin_of_box|>...<|end_of_box|> format
        match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try partial <answer>... (without closing tag)
        match = re.search(r"<answer>(.+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _parse_integer(text: str) -> int:
        """Parse first integer from text."""
        numbers = re.findall(r"-?\d+", text)
        if numbers:
            return int(numbers[0])
        return -999999  # Sentinel for no match

    def judge_correctness(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
    ) -> Tuple[bool, str]:
        # Extract and parse answers
        extracted = self._extract_answer(prediction)
        pred_int = self._parse_integer(extracted if extracted else prediction)
        gt_int = self._parse_integer(ground_truth)

        is_correct = pred_int == gt_int
        reason = f"pred={pred_int}, gt={gt_int}"
        return is_correct, reason


class RelaxedMatchJudge(BaseJudge):
    """Relaxed matching judge for chartqa (5% tolerance for numerical)."""

    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract answer from <answer> tags."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _normalize(answer: str) -> str:
        """Normalize answer for comparison."""
        answer = answer.lower().strip()
        answer = re.sub(r"[,\$%]", "", answer)
        return answer

    @staticmethod
    def _is_numeric(s: str) -> bool:
        """Check if string is numeric."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def judge_correctness(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
    ) -> Tuple[bool, str]:
        pred_text = self._extract_answer(prediction)
        pred_norm = self._normalize(pred_text)
        gold_norm = self._normalize(ground_truth)

        # Exact match
        if pred_norm == gold_norm:
            return True, "exact_match"

        # Numerical comparison with tolerance
        if self._is_numeric(pred_norm) and self._is_numeric(gold_norm):
            pred_val = float(pred_norm)
            gold_val = float(gold_norm)
            if gold_val == 0:
                is_correct = pred_val == 0
            else:
                is_correct = abs(pred_val - gold_val) / abs(gold_val) <= self.tolerance
            return is_correct, f"numeric: pred={pred_val}, gt={gold_val}"

        return False, f"no_match: pred={pred_norm}, gt={gold_norm}"


# Dataset to judge type mapping
DATASET_JUDGE_MAP = {
    # Integer matching datasets
    "mapqa": IntegerMatchJudge,
    "mapeval": IntegerMatchJudge,
    "mapeval_visual": IntegerMatchJudge,
    "visworld": IntegerMatchJudge,
    "stmf_counting": IntegerMatchJudge,
    "shortest_path": IntegerMatchJudge,
    # Relaxed matching datasets
    "chartqa": RelaxedMatchJudge,
    # Default to LLM for others
}


def create_judge(
    dataset_name: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
    force_llm: bool = False,
) -> BaseJudge:
    """Create appropriate judge for dataset.

    Args:
        dataset_name: Name of the dataset.
        api_key: API key for LLM judge.
        model: Model name for LLM judge.
        api_base: API base URL for LLM judge.
        force_llm: If True, always use LLM judge.

    Returns:
        Appropriate judge instance.
    """
    if force_llm:
        return LLMJudge(api_key=api_key, model=model, api_base=api_base)

    # Check for dataset-specific judge
    dataset_lower = dataset_name.lower()
    for key, judge_class in DATASET_JUDGE_MAP.items():
        if key in dataset_lower:
            return judge_class()

    # Default to LLM judge
    return LLMJudge(api_key=api_key, model=model, api_base=api_base)
