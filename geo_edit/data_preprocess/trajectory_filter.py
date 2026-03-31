from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from geo_edit.data_preprocess.trajectory_utils import (
    extract_final_answer,
    extract_thinking_text,
    get_question_from_trajectory,
    is_brute_force,
    load_meta_info,
    load_trajectory,
)
from geo_edit.evaluation.trajectory_judge import (
    TrajectoryFilterConfig,
    TrajectoryJudge,
    check_tool_plan_mismatch,
    quick_leakage_check,
)
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class AugmentStats:
    total: int = 0
    passed_filter: int = 0
    filtered_wrong_answer: int = 0
    filtered_leakage: int = 0
    filtered_tool_mismatch: int = 0
    filtered_brute_force: int = 0
    filtered_structure_error: int = 0
    api_errors: int = 0
    total_think_blocks: int = 0
    blocks_diversified: int = 0
    blocks_skipped_short: int = 0
    blocks_failed: int = 0
    output_subfolders: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  Augmentation Statistics",
            "=" * 55,
            f"  Total subfolders:            {self.total}",
            f"  Passed filter:               {self.passed_filter}",
            f"  Filtered (wrong answer):     {self.filtered_wrong_answer}",
            f"  Filtered (leakage):          {self.filtered_leakage}",
            f"  Filtered (tool mismatch):    {self.filtered_tool_mismatch}",
            f"  Filtered (brute force):      {self.filtered_brute_force}",
            f"  Filtered (structure error):  {self.filtered_structure_error}",
            f"  API errors:                  {self.api_errors}",
            "-" * 55,
            f"  Total think blocks:          {self.total_think_blocks}",
            f"  Blocks diversified:          {self.blocks_diversified}",
            f"  Blocks skipped (< 20 chars): {self.blocks_skipped_short}",
            f"  Blocks failed (kept orig):   {self.blocks_failed}",
            "-" * 55,
            f"  Output subfolders:           {self.output_subfolders}",
            "=" * 55,
        ]
        return "\n".join(lines)


def filter_subfolder(
    subfolder: Path,
    judge: Optional[TrajectoryJudge],
    config: TrajectoryFilterConfig,
    filter_brute_force: bool = True,
) -> Tuple[bool, str]:
    trajectory_path = subfolder / "trajectory.json"
    meta_path = subfolder / "meta_info.jsonl"

    if not trajectory_path.exists() or not meta_path.exists():
        return False, "structure_error"

    try:
        trajectory = load_trajectory(trajectory_path)
        meta_info = load_meta_info(meta_path)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to load data in %s: %s", subfolder.name, e)
        return False, "structure_error"

    if not trajectory or not meta_info:
        return False, "structure_error"

    if filter_brute_force:
        if is_brute_force(trajectory, meta_info):
            logger.debug("Filtered %s: brute force", subfolder.name)
            return False, "brute_force"

    if config.filter_wrong_answers and judge:
        question = get_question_from_trajectory(trajectory)
        ground_truth = meta_info.get("answer", "")
        if isinstance(ground_truth, list):
            ground_truth = ", ".join(str(x) for x in ground_truth)

        prediction = extract_final_answer(trajectory)
        if not prediction:
            logger.info("Filtered %s: no final answer found", subfolder.name)
            return False, "wrong_answer"

        try:
            is_correct, _ = judge.judge_correctness(question, str(ground_truth), prediction)
            if not is_correct:
                logger.info(
                    "Filtered %s: wrong answer (pred=%s, gt=%s)",
                    subfolder.name,
                    prediction[:50],
                    str(ground_truth)[:50],
                )
                return False, "wrong_answer"
        except Exception as e:
            logger.warning("API error checking correctness for %s: %s", subfolder.name, e)
            return False, "api_error"

    if config.filter_answer_leakage:
        thinking_text = extract_thinking_text(trajectory)
        if thinking_text:
            if config.leakage_check_mode == "quick":
                has_leakage, reason = quick_leakage_check(thinking_text)
                if has_leakage:
                    logger.info("Filtered %s: %s", subfolder.name, reason)
                    return False, "leakage"
            elif judge:
                question = get_question_from_trajectory(trajectory)
                ground_truth = meta_info.get("answer", "")
                if isinstance(ground_truth, list):
                    ground_truth = ", ".join(str(x) for x in ground_truth)
                try:
                    has_leakage, reason = judge.detect_leakage(
                        question, str(ground_truth), thinking_text, use_ai=True
                    )
                    if has_leakage:
                        logger.info(
                            "Filtered %s: leakage — %s",
                            subfolder.name,
                            reason[:100],
                        )
                        return False, "leakage"
                except Exception as e:
                    logger.warning("API error checking leakage for %s: %s", subfolder.name, e)
                    return False, "api_error"

    if config.filter_tool_mismatch:
        has_mismatch, reason = check_tool_plan_mismatch(trajectory)
        if has_mismatch:
            logger.info("Filtered %s: tool mismatch — %s", subfolder.name, reason)
            return False, "tool_mismatch"

    return True, ""
