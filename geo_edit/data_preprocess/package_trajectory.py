"""Package multi-turn trajectory data into HuggingFace parquet dataset.

This module reads trajectory.json files from subfolders and packages them into
a parquet dataset for testing tool call effectiveness.

Features:
- Packages trajectory data with images into HuggingFace parquet format
- Optional filtering:
  - Wrong answer filtering: removes trajectories with incorrect final answers (requires API)
  - Answer leakage filtering: removes trajectories with <answer> tags in thinking/tool-call phases
    (follows three-phase protocol: Phase 1=reasoning, Phase 2=tool call, Phase 3=final answer)
  - Tool mismatch filtering: removes trajectories where Phase 1 tool plan doesn't match Phase 2 calls

Usage (basic, no filtering):
    python -m geo_edit.data_preprocess.package_trajectory \
        --data_dir /path/to/trajectories \
        --out_path /path/to/output.parquet

Usage (with leakage and tool mismatch filtering - fast, no API needed):
    python -m geo_edit.data_preprocess.package_trajectory \
        --data_dir /path/to/trajectories \
        --out_path /path/to/output.parquet \
        --filter_answer_leakage \
        --filter_tool_mismatch \
        --leakage_check_mode quick

Usage (with all filtering - requires API):
    python -m geo_edit.data_preprocess.package_trajectory \
        --data_dir /path/to/trajectories \
        --out_path /path/to/output.parquet \
        --api_base "https://matrixllm.alipay.com/v1" \
        --api_key YOUR_API_KEY \
        --filter_wrong_answers \
        --filter_answer_leakage \
        --filter_tool_mismatch \
        --leakage_check_mode full \
        --max_workers 32
"""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, Features, Image as HFImage, Sequence, Value
from geo_edit.data_preprocess.trajectory_utils import (
    extract_final_answer,
    extract_thinking_text,
    get_question_from_trajectory,
    get_text_from_content,
    load_meta_info,
    load_trajectory,
)

from geo_edit.evaluation.trajectory_judge import (
    FilterStats,
    TrajectoryFilterConfig,
    TrajectoryJudge,
    check_tool_plan_mismatch,
    quick_leakage_check,
)
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


def _extract_turns_except_last(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract all turns except the last assistant turn containing the answer.

    The last turn is identified as:
    - role == "assistant"
    - contains "<answer>...</answer>" pattern
    """
    if not trajectory:
        return []

    last_turn = trajectory[-1]
    if (last_turn.get("role") == "assistant" and
        "<answer>" in str(last_turn.get("content", ""))):
        return trajectory[:-1]

    return trajectory


def _resolve_image_path(url: str, subfolder: Path) -> Optional[Path]:
    """Resolve image path from file:// URL.

    Args:
        url: File URL like "file:///storage/.../image.png"
        subfolder: Current subfolder path for relative resolution

    Returns:
        Resolved Path object or None if not resolvable
    """
    if not url.startswith("file://"):
        return None

    # Remove file:// prefix
    path_str = url[7:]

    # Handle Windows paths that might start with /C:/
    if path_str.startswith("/") and len(path_str) > 2 and path_str[2] == ":":
        path_str = path_str[1:]

    path = Path(path_str)

    if path.exists():
        return path

    # Try relative to subfolder
    relative_path = subfolder / path.name
    if relative_path.exists():
        return relative_path

    return None


def _extract_and_replace_images(
    trajectory: List[Dict[str, Any]],
    subfolder: Path
) -> Tuple[List[Dict[str, Any]], List[bytes]]:
    """Extract images from trajectory and replace URLs with placeholders.

    Traverses messages, extracts file:// images, and replaces with [IMAGE_N] placeholders.

    Args:
        trajectory: List of conversation turns
        subfolder: Path to the subfolder containing images

    Returns:
        Tuple of (modified trajectory with placeholders, list of image bytes)
    """
    # Deep copy to avoid modifying original
    trajectory = copy.deepcopy(trajectory)
    images_bytes: List[bytes] = []
    image_idx = 0

    for turn in trajectory:
        content = turn.get("content", [])
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue

                if part.get("type") == "image_url":
                    image_url_data = part.get("image_url", {})
                    if isinstance(image_url_data, dict):
                        url = image_url_data.get("url", "")
                    else:
                        url = str(image_url_data)

                    if url.startswith("file://"):
                        image_path = _resolve_image_path(url, subfolder)
                        if image_path and image_path.exists():
                            images_bytes.append(image_path.read_bytes())
                            # Replace URL with placeholder
                            if isinstance(image_url_data, dict):
                                part["image_url"]["url"] = f"[IMAGE_{image_idx}]"
                            else:
                                part["image_url"] = {"url": f"[IMAGE_{image_idx}]"}
                            image_idx += 1
                        else:
                            logger.warning("Image not found: %s", url)

    return trajectory, images_bytes


def _count_tool_calls(trajectory: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    """Count tool calls and extract tool names from trajectory.

    Returns:
        Tuple of (total tool call count, list of unique tool names)
    """
    count = 0
    names: List[str] = []

    for turn in trajectory:
        tool_calls = turn.get("tool_calls", [])
        for tc in tool_calls:
            count += 1
            func = tc.get("function", {})
            name = func.get("name", "")
            if name and name not in names:
                names.append(name)

    return count, names




# Filter result constants for tracking
FILTER_PASSED = "passed"
FILTER_WRONG_ANSWER = "wrong_answer"
FILTER_LEAKAGE = "leakage"
FILTER_TOOL_MISMATCH = "tool_mismatch"
FILTER_FAILED = "failed"
FILTER_API_ERROR = "api_error"


def _process_subfolder(
    subfolder: Path,
    judge: Optional[TrajectoryJudge] = None,
    config: Optional[TrajectoryFilterConfig] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Process a single subfolder and return a dataset record.

    Args:
        subfolder: Path to subfolder containing trajectory.json and meta_info.jsonl
        judge: Optional TrajectoryJudge for filtering
        config: Optional TrajectoryFilterConfig for filtering settings

    Returns:
        Tuple of (record, filter_reason):
        - record: Dataset record dict or None if processing failed or filtered
        - filter_reason: One of FILTER_* constants indicating result
    """
    trajectory_path = subfolder / "trajectory.json"
    meta_path = subfolder / "meta_info.jsonl"

    if not trajectory_path.exists():
        logger.warning("Missing trajectory.json in %s", subfolder.name)
        return None, FILTER_FAILED

    if not meta_path.exists():
        logger.warning("Missing meta_info.jsonl in %s", subfolder.name)
        return None, FILTER_FAILED

    try:
        # Load data
        trajectory = load_trajectory(trajectory_path)
        meta_info = load_meta_info(meta_path)

        # Apply filtering if config is provided
        if config:
            # Get ground truth from meta_info
            ground_truth = meta_info.get("answer", "")
            if isinstance(ground_truth, list):
                ground_truth = ", ".join(str(x) for x in ground_truth)

            question = get_question_from_trajectory(trajectory)

            # Filter 1: Wrong answer check (requires judge/API)
            if config.filter_wrong_answers and judge:
                prediction = extract_final_answer(trajectory)
                if prediction:
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
                            return None, FILTER_WRONG_ANSWER
                    except Exception as e:
                        logger.warning(
                            "API error checking correctness for %s: %s",
                            subfolder.name,
                            e,
                        )
                        # Fail-open: continue processing on API error
                        # but track API error for statistics
                        # We continue instead of returning to allow other filters to run

            # Filter 2: Answer leakage check
            # Leakage = <answer> tags appearing in Phase 1/2 (thinking/tool-call)
            if config.filter_answer_leakage:
                thinking_text = extract_thinking_text(trajectory)
                if thinking_text:
                    # Quick mode: use regex only (fast, no API call needed)
                    if config.leakage_check_mode == "quick":
                        has_leakage, reason = quick_leakage_check(thinking_text)
                        if has_leakage:
                            # Find the problematic part
                            answer_idx = thinking_text.lower().find("<answer>")
                            if answer_idx >= 0:
                                start = max(0, answer_idx - 50)
                                end = min(len(thinking_text), answer_idx + 100)
                                problematic_snippet = thinking_text[start:end]
                                logger.warning(
                                    "Filtered %s: %s\n  Problematic content: ...%s...",
                                    subfolder.name,
                                    reason,
                                    problematic_snippet,
                                )
                            else:
                                logger.info("Filtered %s: %s", subfolder.name, reason)
                            return None, FILTER_LEAKAGE
                    elif judge:
                        # Full mode: use AI judge for more subtle detection
                        try:
                            has_leakage, reason = judge.detect_leakage(
                                question, str(ground_truth), thinking_text, use_ai=True
                            )
                            if has_leakage:
                                # Show snippet of thinking text for debugging
                                snippet = thinking_text[:200] + "..." if len(thinking_text) > 200 else thinking_text
                                logger.warning(
                                    "Filtered %s: answer leakage detected\n"
                                    "  AI reason: %s\n"
                                    "  Ground truth: %s\n"
                                    "  Thinking snippet: %s",
                                    subfolder.name,
                                    reason[:200] if len(reason) > 200 else reason,
                                    str(ground_truth)[:100],
                                    snippet,
                                )
                                return None, FILTER_LEAKAGE
                        except Exception as e:
                            logger.warning(
                                "API error checking leakage for %s: %s",
                                subfolder.name,
                                e,
                            )
                            # Fail-open: continue processing on API error

            # Filter 3: Tool plan mismatch check
            # Check if Phase 1 reasoning declares tools that match Phase 2 actual calls
            if config.filter_tool_mismatch:
                has_mismatch, reason = check_tool_plan_mismatch(trajectory)
                if has_mismatch:
                    logger.info(
                        "Filtered %s: tool plan mismatch - %s",
                        subfolder.name,
                        reason,
                    )
                    return None, FILTER_TOOL_MISMATCH

        # Extract turns (excluding final answer)
        turns = _extract_turns_except_last(trajectory)

        # Extract and replace images
        processed_turns, images_bytes = _extract_and_replace_images(turns, subfolder)

        # Count tool calls
        num_tool_calls, tool_names = _count_tool_calls(turns)

        # Get answer from meta_info
        answer = meta_info.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(x) for x in answer)

        # Build record
        record = {
            "id": subfolder.name,
            "messages": json.dumps(processed_turns, ensure_ascii=False),
            "images": [{"bytes": b, "path": None} for b in images_bytes],
            "num_turns": len(processed_turns),
            "num_tool_calls": num_tool_calls,
            "tool_names": tool_names,
            "answer": str(answer),
            "meta_info": json.dumps(meta_info, ensure_ascii=False),
        }

        return record, FILTER_PASSED

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", subfolder.name, e)
        return None, FILTER_FAILED
    except Exception as e:
        logger.error("Error processing %s: %s", subfolder.name, e)
        return None, FILTER_FAILED


def package_trajectory_dataset(
    data_dir: Path,
    out_path: Path,
    config: Optional[TrajectoryFilterConfig] = None,
) -> None:
    """Package trajectory data from multiple subfolders into parquet dataset.

    Args:
        data_dir: Parent directory containing trajectory subfolders
        out_path: Output parquet file path
        config: Optional filtering configuration
    """
    records: List[Dict[str, Any]] = []
    stats = FilterStats()

    # Initialize judge if API-based filtering is needed
    judge = None
    needs_api = config and (
        config.filter_wrong_answers or
        (config.filter_answer_leakage and config.leakage_check_mode == "full")
    )

    if needs_api:
        assert config is not None
        if config.api_key:
            judge = TrajectoryJudge(
                api_key=config.api_key,
                model=config.model,
                api_base=config.api_base,
            )
            logger.info(
                "API-based filtering enabled - wrong_answers=%s, leakage=%s (mode=%s), model=%s",
                config.filter_wrong_answers,
                config.filter_answer_leakage,
                config.leakage_check_mode,
                config.model,
            )
        else:
            logger.warning(
                "API-based filtering requested but no API key provided, "
                "skipping wrong_answer filter and using quick leakage check"
            )
            # Disable wrong answer filtering, keep quick leakage check
            if config.filter_wrong_answers:
                config.filter_wrong_answers = False

    if config and config.filter_answer_leakage and config.leakage_check_mode == "quick":
        logger.info("Quick leakage filtering enabled (regex only, no API needed)")

    if config and config.filter_tool_mismatch:
        logger.info("Tool mismatch filtering enabled (no API needed)")

    # Traverse all subfolders
    subfolders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    stats.total = len(subfolders)
    logger.info("Found %d subfolders in %s", len(subfolders), data_dir)

    # Use ThreadPoolExecutor for parallel processing when filtering is enabled
    max_workers = config.max_workers if config else 1
    use_parallel = max_workers > 1 and config and (
        config.filter_wrong_answers or config.filter_answer_leakage or config.filter_tool_mismatch
    )

    if use_parallel:
        # Parallel processing with filtering
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_subfolder, subfolder, judge, config
                ): subfolder
                for subfolder in subfolders
            }

            for future in as_completed(futures):
                subfolder = futures[future]
                try:
                    record, filter_reason = future.result()
                    if record:
                        records.append(record)
                    # Aggregate stats based on filter_reason
                    if filter_reason == FILTER_PASSED:
                        stats.passed += 1
                    elif filter_reason == FILTER_WRONG_ANSWER:
                        stats.filtered_wrong_answer += 1
                    elif filter_reason == FILTER_LEAKAGE:
                        stats.filtered_leakage += 1
                    elif filter_reason == FILTER_TOOL_MISMATCH:
                        stats.filtered_tool_mismatch += 1
                    elif filter_reason == FILTER_FAILED:
                        stats.failed += 1
                except Exception as e:
                    logger.error("Error processing %s: %s", subfolder.name, e)
                    stats.failed += 1
    else:
        # Sequential processing (original behavior or no filtering)
        for subfolder in subfolders:
            record, filter_reason = _process_subfolder(subfolder, judge, config)
            if record:
                records.append(record)
            # Aggregate stats based on filter_reason
            if filter_reason == FILTER_PASSED:
                stats.passed += 1
            elif filter_reason == FILTER_WRONG_ANSWER:
                stats.filtered_wrong_answer += 1
            elif filter_reason == FILTER_LEAKAGE:
                stats.filtered_leakage += 1
            elif filter_reason == FILTER_TOOL_MISMATCH:
                stats.filtered_tool_mismatch += 1
            elif filter_reason == FILTER_FAILED:
                stats.failed += 1

    if not records:
        logger.error("No valid records found")
        return

    # Define features
    features = Features({
        "id": Value("string"),
        "messages": Value("string"),
        "images": Sequence(HFImage()),
        "num_turns": Value("int64"),
        "num_tool_calls": Value("int64"),
        "tool_names": Sequence(Value("string")),
        "answer": Value("string"),
        "meta_info": Value("string"),
    })

    # Create and save dataset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(records, features=features)
    ds.to_parquet(str(out_path))

    logger.info("Saved parquet: %s (%d records)", out_path, len(ds))
    print(f"Saved parquet: {out_path} ({len(ds)} records)")

    # Log filtering statistics if filtering was enabled
    if config and (config.filter_wrong_answers or config.filter_answer_leakage or config.filter_tool_mismatch):
        logger.info("\n%s", stats.summary())
        print(stats.summary())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package multi-turn trajectory data into HuggingFace parquet dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Parent directory containing trajectory subfolders.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output parquet file path.",
    )
    # Filtering arguments
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for AI judge (enables filtering).",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL (e.g., https://matrixllm.alipay.com/v1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Model to use for AI judge.",
    )
    parser.add_argument(
        "--filter_wrong_answers",
        action="store_true",
        help="Filter trajectories with incorrect final answers.",
    )
    parser.add_argument(
        "--filter_answer_leakage",
        action="store_true",
        help="Filter trajectories where thinking contains <answer> tags (protocol violation).",
    )
    parser.add_argument(
        "--filter_tool_mismatch",
        action="store_true",
        help="Filter trajectories where Phase 1 tool plan doesn't match Phase 2 tool calls.",
    )
    parser.add_argument(
        "--leakage_check_mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Leakage check mode: 'quick' (regex only, fast) or 'full' (AI-based, thorough).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Max parallel workers for AI evaluation.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out_path).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Build filter config if filtering is requested
    config = None
    if args.filter_wrong_answers or args.filter_answer_leakage or args.filter_tool_mismatch:
        config = TrajectoryFilterConfig(
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            filter_wrong_answers=args.filter_wrong_answers,
            filter_answer_leakage=args.filter_answer_leakage,
            filter_tool_mismatch=args.filter_tool_mismatch,
            leakage_check_mode=args.leakage_check_mode,
            max_workers=args.max_workers,
        )

    package_trajectory_dataset(data_dir, out_path, config)


if __name__ == "__main__":
    main()
