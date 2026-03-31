"""Augment SFT trajectory data with LLM-driven think block diversification and filtering.

Pipeline:
1. Filter: Validate structure, check brute-force / correctness / leakage / tool-mismatch
2. Diversify: Extract <think> blocks, classify, LLM-rephrase in global batch
3. Output: Copy filtered + diversified trajectories preserving directory structure

The output directory mirrors the source layout so that
``convert_trajectory_to_sft.py`` can read it directly.

Usage (filter only):
    python -m geo_edit.data_preprocess.augment_sft_data \\
        --src-dir /path/to/trajectories \\
        --dst-dir /path/to/output \\
        --skip-diversify

Usage (full pipeline):
    python -m geo_edit.data_preprocess.augment_sft_data \\
        --src-dir /path/to/trajectories \\
        --dst-dir /path/to/output \\
        --api-base https://matrixllm.alipay.com/v1 \\
        --model kimi-k2.5 \\
        --api-key $API_KEY \\
        --judge-api-base https://matrixllm.alipay.com/v1 \\
        --judge-model gpt-5-mini-2025-08-07 \\
        --judge-api-key $JUDGE_KEY \\
        --filter-wrong-answers \\
        --filter-answer-leakage \\
        --leakage-check-mode quick \\
        --max-concurrent 16 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from geo_edit.data_preprocess.convert_trajectory_to_sft import (
    is_brute_force,
)
from geo_edit.data_preprocess.package_trajectory import (
    _extract_final_answer_from_trajectory,
    _extract_thinking_text,
    _get_question_from_trajectory,
    _load_meta_info,
    _load_trajectory,
)
from geo_edit.evaluation.trajectory_judge import (
    TrajectoryFilterConfig,
    TrajectoryJudge,
    check_tool_plan_mismatch,
    quick_leakage_check,
)
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────


class PromptType(Enum):
    """Classification of think blocks for diversification."""

    A = "first_tool"  # First tool-call reasoning
    B = "subsequent_tool"  # Subsequent tool-call reasoning
    C = "final_reasoning"  # Final reasoning / synthesis


@dataclass
class ThinkBlock:
    """A single ``<think>...</think>`` block extracted from a trajectory."""

    msg_index: int  # Index in the trajectory message list
    part_index: Optional[int]  # None if content is str; int if content is list
    start_pos: int  # Char offset of ``<think>`` in the text
    end_pos: int  # Char offset right after ``</think>``
    text: str  # Inner text (between the tags)
    prompt_type: Optional[PromptType] = None
    tool_name: Optional[str] = None  # Extracted from "Tool: xxx"


@dataclass
class AugmentStats:
    """Statistics for the full augmentation pipeline."""

    # Filter phase
    total: int = 0
    passed_filter: int = 0
    filtered_wrong_answer: int = 0
    filtered_leakage: int = 0
    filtered_tool_mismatch: int = 0
    filtered_brute_force: int = 0
    filtered_structure_error: int = 0
    api_errors: int = 0

    # Diversification phase
    total_think_blocks: int = 0
    blocks_diversified: int = 0
    blocks_skipped_short: int = 0
    blocks_failed: int = 0

    # Output phase
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


# ── Prompt Templates ─────────────────────────────────────────────────────────

DIVERSIFY_SYSTEM_PROMPT = (
    "You are a text rephrasing assistant. Rephrase the given reasoning text "
    "while preserving its core meaning and key technical information. "
    "Output ONLY the rephrased text, nothing else."
)

PROMPT_FIRST_TOOL = """\
Rephrase the following reasoning text where a model decides to use the tool \
"{tool_name}" as its first analysis step.

Requirements:
- Keep the tool name "{tool_name}" exactly as-is in the output
- Vary how and why this tool is chosen as the first step
- Change sentence structure, word choice, and perspective
- Maximum output length: {max_len} characters
- Do NOT include any XML tags like <think>, </think>, <answer>, or <action>
- Output ONLY the rephrased reasoning text, no explanations

Original text:
{text}"""

PROMPT_SUBSEQUENT_TOOL = """\
Rephrase the following reasoning text where a model decides to call the tool \
"{tool_name}" after analyzing previous results.

Requirements:
- Keep the tool name "{tool_name}" exactly as-is in the output
- Diversify the opening transition phrase (avoid repetitive patterns like \
"Wait, I think my previous analysis might be incomplete—")
- Preserve the core logic: why previous results were insufficient and why \
this new tool is needed
- Maximum output length: {max_len} characters
- Do NOT include any XML tags like <think>, </think>, <answer>, or <action>
- Output ONLY the rephrased reasoning text, no explanations

Original text:
{text}"""

PROMPT_FINAL_REASONING = """\
Rephrase the following final reasoning text that synthesizes analysis results.

Requirements:
- Preserve ALL factual data, numbers, measurements, and logical conclusions EXACTLY
- Diversify transition words, sentence structure, and phrasing
- Do NOT change any numerical values or the final conclusion
- Maximum output length: {max_len} characters
- Do NOT include any XML tags like <think>, </think>, <answer>, or <action>
- Output ONLY the rephrased reasoning text, no explanations

Original text:
{text}"""


# ── Core Functions ───────────────────────────────────────────────────────────


def extract_think_blocks(trajectory: List[Dict[str, Any]]) -> List[ThinkBlock]:
    """Extract all ``<think>...</think>`` blocks from assistant messages.

    Handles both string content and list-of-parts content formats.
    """
    blocks: List[ThinkBlock] = []
    think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    for msg_idx, msg in enumerate(trajectory):
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content")
        if content is None:
            continue

        if isinstance(content, str):
            for m in think_re.finditer(content):
                blocks.append(
                    ThinkBlock(
                        msg_index=msg_idx,
                        part_index=None,
                        start_pos=m.start(),
                        end_pos=m.end(),
                        text=m.group(1),
                    )
                )

        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                elif isinstance(part, str):
                    text = part
                else:
                    continue

                for m in think_re.finditer(text):
                    blocks.append(
                        ThinkBlock(
                            msg_index=msg_idx,
                            part_index=part_idx,
                            start_pos=m.start(),
                            end_pos=m.end(),
                            text=m.group(1),
                        )
                    )

    return blocks


def classify_think_blocks(blocks: List[ThinkBlock]) -> None:
    """Classify think blocks into prompt types A / B / C **in-place**.

    - Blocks containing a ``Tool:`` pattern → A (first) or B (subsequent)
    - Other blocks → C (final reasoning)
    """
    tool_re = re.compile(r"Tool:\s*(?:functions\.)?(\w+)")
    tool_think_count = 0

    for block in blocks:
        match = tool_re.search(block.text)
        if match:
            tool_think_count += 1
            block.tool_name = match.group(1)
            block.prompt_type = PromptType.A if tool_think_count == 1 else PromptType.B
        else:
            block.prompt_type = PromptType.C


def filter_subfolder(
    subfolder: Path,
    judge: Optional[TrajectoryJudge],
    config: TrajectoryFilterConfig,
    filter_brute_force: bool = True,
) -> Tuple[bool, str]:
    """Run all configured filters on a single trajectory subfolder.

    Returns:
        ``(passed, reason)`` — *reason* is an empty string when passed.
    """
    trajectory_path = subfolder / "trajectory.json"
    meta_path = subfolder / "meta_info.jsonl"

    # 1. Structure validation
    if not trajectory_path.exists() or not meta_path.exists():
        return False, "structure_error"

    try:
        trajectory = _load_trajectory(trajectory_path)
        meta_info = _load_meta_info(meta_path)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to load data in %s: %s", subfolder.name, e)
        return False, "structure_error"

    if not trajectory or not meta_info:
        return False, "structure_error"

    # 2. Brute-force check
    if filter_brute_force:
        if is_brute_force(trajectory, meta_info):
            logger.debug("Filtered %s: brute force", subfolder.name)
            return False, "brute_force"

    # 3. Correctness check (requires judge + API)
    if config.filter_wrong_answers and judge:
        question = _get_question_from_trajectory(trajectory)
        ground_truth = meta_info.get("answer", "")
        if isinstance(ground_truth, list):
            ground_truth = ", ".join(str(x) for x in ground_truth)

        prediction = _extract_final_answer_from_trajectory(trajectory)
        if not prediction:
            logger.info(
                "Filtered %s: no final answer found", subfolder.name
            )
            return False, "wrong_answer"

        try:
            is_correct, _ = judge.judge_correctness(
                question, str(ground_truth), prediction
            )
            if not is_correct:
                logger.info(
                    "Filtered %s: wrong answer (pred=%s, gt=%s)",
                    subfolder.name,
                    prediction[:50],
                    str(ground_truth)[:50],
                )
                return False, "wrong_answer"
        except Exception as e:
            logger.warning(
                "API error checking correctness for %s: %s", subfolder.name, e
            )
            return False, "api_error"

    # 4. Leakage check
    if config.filter_answer_leakage:
        thinking_text = _extract_thinking_text(trajectory)
        if thinking_text:
            if config.leakage_check_mode == "quick":
                has_leakage, reason = quick_leakage_check(thinking_text)
                if has_leakage:
                    logger.info("Filtered %s: %s", subfolder.name, reason)
                    return False, "leakage"
            elif judge:
                question = _get_question_from_trajectory(trajectory)
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
                    logger.warning(
                        "API error checking leakage for %s: %s", subfolder.name, e
                    )
                    return False, "api_error"

    # 5. Tool-plan mismatch check
    if config.filter_tool_mismatch:
        has_mismatch, reason = check_tool_plan_mismatch(trajectory)
        if has_mismatch:
            logger.info(
                "Filtered %s: tool mismatch — %s", subfolder.name, reason
            )
            return False, "tool_mismatch"

    return True, ""


# ── Diversification Client ───────────────────────────────────────────────────


class DiversificationClient:
    """Client for LLM-driven think block diversification."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_retries: int = 3,
        temperature: float = 0.7,
    ):
        from openai import OpenAI  # lazy: allow --skip-diversify without openai

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature

    # -- prompt construction --------------------------------------------------

    def _build_prompt(self, block: ThinkBlock) -> str:
        max_len = len(block.text) + 100
        if block.prompt_type == PromptType.A:
            return PROMPT_FIRST_TOOL.format(
                tool_name=block.tool_name, max_len=max_len, text=block.text
            )
        if block.prompt_type == PromptType.B:
            return PROMPT_SUBSEQUENT_TOOL.format(
                tool_name=block.tool_name, max_len=max_len, text=block.text
            )
        return PROMPT_FINAL_REASONING.format(max_len=max_len, text=block.text)

    # -- validation -----------------------------------------------------------

    @staticmethod
    def _validate_result(block: ThinkBlock, rephrased: str) -> bool:
        # Must be non-empty and substantial
        if not rephrased or len(rephrased) <= 10:
            return False
        # Length constraint
        if len(rephrased) > len(block.text) + 100:
            return False
        # Tool name must survive for A / B types
        if block.prompt_type in (PromptType.A, PromptType.B):
            if block.tool_name and block.tool_name not in rephrased:
                return False
        # No forbidden tags
        rephrased_lower = rephrased.lower()
        for tag in ("<think>", "</think>", "<answer>", "</answer>", "<action>", "</action>"):
            if tag in rephrased_lower:
                return False
        return True

    # -- single block ---------------------------------------------------------

    def diversify_block(self, block: ThinkBlock) -> str:
        """Diversify one think block with exponential-backoff retries."""
        prompt = self._build_prompt(block)

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": DIVERSIFY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )
                rephrased = (resp.choices[0].message.content or "").strip()

                if self._validate_result(block, rephrased):
                    return rephrased

                logger.debug(
                    "Validation failed (attempt %d/%d, len=%d)",
                    attempt + 1,
                    self.max_retries,
                    len(rephrased),
                )
            except Exception as e:
                logger.warning(
                    "API error diversifying block (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )

            # Exponential back-off: 1 s → 2 s → 4 s
            time.sleep(2**attempt)

        logger.warning("All retries exhausted for block, keeping original text")
        return block.text

    # -- batch ----------------------------------------------------------------

    def diversify_blocks_batch(
        self,
        blocks: List[ThinkBlock],
        max_workers: int = 8,
    ) -> Dict[int, str]:
        """Diversify multiple blocks in parallel.

        Blocks shorter than 20 characters are skipped (kept as-is).

        Returns:
            Mapping of block index → new text.
        """
        results: Dict[int, str] = {}
        eligible: List[Tuple[int, ThinkBlock]] = []

        for i, b in enumerate(blocks):
            if len(b.text) < 20:
                results[i] = b.text  # keep original
            else:
                eligible.append((i, b))

        if not eligible:
            return results

        logger.info(
            "Diversifying %d / %d eligible think blocks (%d workers)…",
            len(eligible),
            len(blocks),
            max_workers,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.diversify_block, block): idx
                for idx, block in eligible
            }

            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                completed += 1
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("Failed to diversify block %d: %s", idx, e)
                    results[idx] = blocks[idx].text

                if completed % 50 == 0:
                    logger.info(
                        "  diversification progress: %d / %d", completed, len(eligible)
                    )

        return results


# ── Apply diversified blocks back ────────────────────────────────────────────


def apply_diversified_blocks(
    trajectory: List[Dict[str, Any]],
    blocks: List[ThinkBlock],
    new_texts: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Splice diversified text back into the trajectory.

    Blocks are grouped by ``(msg_index, part_index)`` and replaced in
    **reverse** start-position order so that character offsets stay valid.
    """
    traj = copy.deepcopy(trajectory)

    # Group by location
    groups: Dict[Tuple[int, Optional[int]], List[Tuple[ThinkBlock, str]]] = {}
    for block_idx, block in enumerate(blocks):
        if block_idx not in new_texts:
            continue
        key = (block.msg_index, block.part_index)
        groups.setdefault(key, []).append((block, new_texts[block_idx]))

    for (msg_idx, part_idx), pairs in groups.items():
        # Reverse order → highest start_pos first
        pairs.sort(key=lambda p: p[0].start_pos, reverse=True)

        for block, new_text in pairs:
            replacement = f"<think>{new_text}</think>"

            if part_idx is None:
                # Content is a plain string
                content = traj[msg_idx].get("content", "")
                if isinstance(content, str):
                    traj[msg_idx]["content"] = (
                        content[: block.start_pos]
                        + replacement
                        + content[block.end_pos :]
                    )
            else:
                # Content is a list of parts
                parts = traj[msg_idx].get("content")
                if not isinstance(parts, list) or part_idx >= len(parts):
                    continue

                part = parts[part_idx]
                if isinstance(part, dict) and part.get("type") == "text":
                    old = part.get("text", "")
                    part["text"] = (
                        old[: block.start_pos] + replacement + old[block.end_pos :]
                    )
                elif isinstance(part, str):
                    parts[part_idx] = (
                        part[: block.start_pos] + replacement + part[block.end_pos :]
                    )

    return traj


# ── File copying ─────────────────────────────────────────────────────────────


def copy_subfolder(
    src_sub: Path,
    dst_sub: Path,
    modified_trajectory: Optional[List[Dict[str, Any]]] = None,
    use_symlink: bool = False,
) -> None:
    """Copy a trajectory subfolder to *dst_sub*, optionally injecting a
    modified ``trajectory.json``.

    Images can be symlinked instead of copied to save disk space.
    ``output.jsonl`` is copied as-is (it contains conversation_history
    records without ``<think>`` blocks, so no sync is needed).
    """
    os.makedirs(dst_sub, exist_ok=True)

    # 1. trajectory.json — write modified or copy original
    if modified_trajectory is not None:
        with open(dst_sub / "trajectory.json", "w", encoding="utf-8") as f:
            json.dump(modified_trajectory, f, ensure_ascii=False, indent=2)
    else:
        src_traj = src_sub / "trajectory.json"
        if src_traj.exists():
            shutil.copy2(src_traj, dst_sub / "trajectory.json")

    # 2. Flat files — copy as-is
    for filename in ("meta_info.jsonl", "output.jsonl", "extra_info.jsonl"):
        src_file = src_sub / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_sub / filename)

    # 3. input_image.png — copy or symlink
    src_img = src_sub / "input_image.png"
    dst_img = dst_sub / "input_image.png"
    if src_img.exists():
        if use_symlink:
            if not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)
        else:
            shutil.copy2(src_img, dst_img)

    # 4. images/ directory — copytree or symlink
    src_images = src_sub / "images"
    dst_images = dst_sub / "images"
    if src_images.exists() and src_images.is_dir():
        if use_symlink:
            if not dst_images.exists():
                os.symlink(src_images.resolve(), dst_images)
        else:
            if dst_images.exists():
                shutil.rmtree(dst_images)
            shutil.copytree(src_images, dst_images)

    # 5. Any other files not yet handled
    _KNOWN_NAMES = {
        "trajectory.json",
        "meta_info.jsonl",
        "output.jsonl",
        "extra_info.jsonl",
        "input_image.png",
        "images",
    }
    for item in src_sub.iterdir():
        if item.name in _KNOWN_NAMES:
            continue
        dst_item = dst_sub / item.name
        if item.is_file():
            shutil.copy2(item, dst_item)
        elif item.is_dir() and not dst_item.exists():
            shutil.copytree(item, dst_item)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Augment SFT trajectory data: filter bad trajectories and "
            "diversify <think> blocks via LLM rephrasing."
        ),
    )

    # -- paths ----------------------------------------------------------------
    p.add_argument("--src-dir", type=str, required=True, help="Source trajectory directory")
    p.add_argument("--dst-dir", type=str, required=True, help="Output directory")

    # -- diversification LLM --------------------------------------------------
    p.add_argument(
        "--api-base",
        type=str,
        default="https://matrixllm.alipay.com/v1",
        help="Diversification LLM API base URL",
    )
    p.add_argument("--model", type=str, default="kimi-k2.5", help="Diversification model")
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Diversification API key (default: env LLM_API_KEY)",
    )

    # -- judge LLM ------------------------------------------------------------
    p.add_argument(
        "--judge-api-base",
        type=str,
        default=None,
        help="Judge API base URL (default: same as --api-base)",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Judge model name",
    )
    p.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help="Judge API key (default: same as --api-key)",
    )

    # -- filter flags ---------------------------------------------------------
    p.add_argument(
        "--filter-wrong-answers",
        action="store_true",
        help="Enable wrong-answer filtering (requires judge)",
    )
    p.add_argument(
        "--filter-answer-leakage",
        action="store_true",
        help="Enable answer-leakage filtering",
    )
    p.add_argument(
        "--filter-brute-force",
        action="store_true",
        default=True,
        help="Enable brute-force filtering (default: True)",
    )
    p.add_argument(
        "--no-filter-brute-force",
        dest="filter_brute_force",
        action="store_false",
        help="Disable brute-force filtering",
    )
    p.add_argument(
        "--filter-tool-mismatch",
        action="store_true",
        help="Enable tool-plan mismatch filtering",
    )
    p.add_argument(
        "--leakage-check-mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Leakage check mode: 'quick' (regex) or 'full' (AI judge)",
    )

    # -- concurrency ----------------------------------------------------------
    p.add_argument(
        "--max-concurrent", type=int, default=16, help="Workers for parallel filtering"
    )
    p.add_argument(
        "--max-llm-workers", type=int, default=8, help="Workers for LLM diversification"
    )

    # -- misc -----------------------------------------------------------------
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--skip-diversify",
        action="store_true",
        help="Only filter — skip LLM diversification",
    )
    p.add_argument(
        "--use-symlink",
        action="store_true",
        help="Symlink images instead of copying (saves disk)",
    )

    return p


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    random.seed(args.seed)

    # -- resolve API keys -----------------------------------------------------
    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    judge_api_base = args.judge_api_base or args.api_base
    judge_api_key = args.judge_api_key or api_key

    # -- initialise judge (if any filter needs it) ----------------------------
    needs_judge = args.filter_wrong_answers or (
        args.filter_answer_leakage and args.leakage_check_mode == "full"
    )

    judge: Optional[TrajectoryJudge] = None
    if needs_judge:
        if judge_api_key:
            judge = TrajectoryJudge(
                api_key=judge_api_key,
                model=args.judge_model,
                api_base=judge_api_base,
            )
            logger.info(
                "Judge initialised: model=%s, api_base=%s",
                args.judge_model,
                judge_api_base,
            )
        else:
            logger.warning(
                "Judge requested but no API key — disabling judge-based filters"
            )
            args.filter_wrong_answers = False
            if args.leakage_check_mode == "full":
                args.leakage_check_mode = "quick"

    # -- build filter config --------------------------------------------------
    config = TrajectoryFilterConfig(
        model=args.judge_model,
        api_key=judge_api_key,
        api_base=judge_api_base,
        filter_wrong_answers=args.filter_wrong_answers,
        filter_answer_leakage=args.filter_answer_leakage,
        filter_tool_mismatch=args.filter_tool_mismatch,
        leakage_check_mode=args.leakage_check_mode,
        max_workers=args.max_concurrent,
    )

    # -- initialise diversification client ------------------------------------
    client: Optional[DiversificationClient] = None
    if not args.skip_diversify:
        if api_key:
            client = DiversificationClient(
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
            )
            logger.info("Diversification client initialised: model=%s", args.model)
        else:
            logger.warning(
                "No API key for diversification — will copy without diversifying"
            )

    # -- enumerate subfolders -------------------------------------------------
    subfolders = sorted(d for d in src_dir.iterdir() if d.is_dir())
    stats = AugmentStats(total=len(subfolders))

    logger.info("Found %d subfolders in %s", len(subfolders), src_dir)
    logger.info(
        "Filters: brute_force=%s  wrong_answer=%s  leakage=%s (%s)  tool_mismatch=%s",
        args.filter_brute_force,
        args.filter_wrong_answers,
        args.filter_answer_leakage,
        args.leakage_check_mode,
        args.filter_tool_mismatch,
    )
    logger.info(
        "Diversification: %s",
        "disabled" if args.skip_diversify or client is None else f"model={args.model}",
    )

    # ── Phase 1: Filter ─────────────────────────────────────────────────
    logger.info("Phase 1 — Filtering %d subfolders …", len(subfolders))
    passed_subfolders: List[Path] = []

    def _safe_filter(sf: Path) -> Tuple[Path, bool, str]:
        try:
            passed, reason = filter_subfolder(
                sf, judge, config, args.filter_brute_force
            )
            return sf, passed, reason
        except Exception as exc:
            logger.error("Unexpected error filtering %s: %s", sf.name, exc)
            return sf, False, "structure_error"

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = [pool.submit(_safe_filter, sf) for sf in subfolders]
        for future in as_completed(futures):
            sf, passed, reason = future.result()
            if passed:
                passed_subfolders.append(sf)
                stats.passed_filter += 1
            elif reason == "wrong_answer":
                stats.filtered_wrong_answer += 1
            elif reason == "leakage":
                stats.filtered_leakage += 1
            elif reason == "tool_mismatch":
                stats.filtered_tool_mismatch += 1
            elif reason == "brute_force":
                stats.filtered_brute_force += 1
            elif reason == "api_error":
                stats.api_errors += 1
            else:  # structure_error or unknown
                stats.filtered_structure_error += 1

    logger.info(
        "Phase 1 complete: %d / %d passed", stats.passed_filter, stats.total
    )

    if not passed_subfolders:
        logger.error("No subfolders passed filtering — nothing to output")
        print(stats.summary())
        return

    # Deterministic ordering
    passed_subfolders.sort()

    # ── Phase 2: Diversify ──────────────────────────────────────────────
    modified_trajectories: Dict[Path, List[Dict[str, Any]]] = {}

    if client is not None and not args.skip_diversify:
        logger.info("Phase 2 — Extracting and diversifying think blocks …")

        # 2a. Load trajectories & extract blocks
        all_blocks: List[ThinkBlock] = []
        block_origins: List[Tuple[Path, int]] = []  # (subfolder, local_idx)
        subfolder_data: Dict[Path, Tuple[List[Dict[str, Any]], List[ThinkBlock]]] = {}

        for sf in passed_subfolders:
            traj = _load_trajectory(sf / "trajectory.json")
            blocks = extract_think_blocks(traj)
            classify_think_blocks(blocks)
            subfolder_data[sf] = (traj, blocks)

            for local_idx, block in enumerate(blocks):
                all_blocks.append(block)
                block_origins.append((sf, local_idx))

        stats.total_think_blocks = len(all_blocks)
        stats.blocks_skipped_short = sum(1 for b in all_blocks if len(b.text) < 20)

        logger.info(
            "Extracted %d think blocks (%d eligible for diversification)",
            len(all_blocks),
            len(all_blocks) - stats.blocks_skipped_short,
        )

        # 2b. Diversify in one global batch
        if all_blocks:
            global_results = client.diversify_blocks_batch(
                all_blocks, max_workers=args.max_llm_workers
            )

            # Count outcomes
            for idx, new_text in global_results.items():
                if len(all_blocks[idx].text) < 20:
                    continue  # already counted as skipped
                if new_text != all_blocks[idx].text:
                    stats.blocks_diversified += 1
                else:
                    stats.blocks_failed += 1

            # 2c. Distribute results back per subfolder
            sf_new_texts: Dict[Path, Dict[int, str]] = {}
            for global_idx, new_text in global_results.items():
                sf, local_idx = block_origins[global_idx]
                sf_new_texts.setdefault(sf, {})[local_idx] = new_text

            # 2d. Apply to each trajectory
            for sf in passed_subfolders:
                if sf in subfolder_data and sf in sf_new_texts:
                    traj, blocks = subfolder_data[sf]
                    modified_trajectories[sf] = apply_diversified_blocks(
                        traj, blocks, sf_new_texts[sf]
                    )

        logger.info(
            "Phase 2 complete: %d diversified, %d failed, %d skipped",
            stats.blocks_diversified,
            stats.blocks_failed,
            stats.blocks_skipped_short,
        )
    else:
        logger.info("Phase 2 — Skipped (diversification disabled)")

    # ── Phase 3: Copy ───────────────────────────────────────────────────
    logger.info("Phase 3 — Copying %d subfolders to %s …", len(passed_subfolders), dst_dir)

    for sf in passed_subfolders:
        dst_sub = dst_dir / sf.name
        mod_traj = modified_trajectories.get(sf)
        copy_subfolder(sf, dst_sub, mod_traj, args.use_symlink)
        stats.output_subfolders += 1

    # ── Summary ─────────────────────────────────────────────────────────
    summary = stats.summary()
    logger.info("\n%s", summary)
    print(summary)


if __name__ == "__main__":
    main()
