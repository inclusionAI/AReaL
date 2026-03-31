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
        --max-concurrent 16 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from geo_edit.data_preprocess.diversification import (
    DiversificationClient,
    ThinkBlock,
    apply_diversified_blocks,
    classify_think_blocks,
    extract_think_blocks,
)
from geo_edit.data_preprocess.trajectory_filter import AugmentStats, filter_subfolder
from geo_edit.data_preprocess.trajectory_utils import (
    get_text_from_content,
    load_meta_info,
    load_trajectory,
)
from geo_edit.evaluation.trajectory_judge import (
    TrajectoryFilterConfig,
    TrajectoryJudge,
)
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

_THINK_TOOL_RE = re.compile(
    r"<think>.*?Tool:\s*(?:functions\.)?(\w+).*?</think>", re.DOTALL | re.IGNORECASE
)


def _discover_subfolders(src_dir: Path) -> List[Tuple[Path, Path]]:
    """Find trajectory subfolders, handling both flat and nested layouts.

    Returns (subfolder, image_dir) pairs where *subfolder* contains
    ``trajectory.json`` + ``meta_info.jsonl`` and *image_dir* holds
    ``input_image.png`` (same as subfolder for flat layout, parent
    directory for nested ``task_id/traj_N/`` layout).
    """
    results: List[Tuple[Path, Path]] = []
    for child in sorted(src_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "trajectory.json").exists() and (
            child / "meta_info.jsonl"
        ).exists():
            results.append((child, child))
        else:
            for grandchild in sorted(child.iterdir()):
                if not grandchild.is_dir():
                    continue
                if (grandchild / "trajectory.json").exists() and (
                    grandchild / "meta_info.jsonl"
                ).exists():
                    results.append((grandchild, child))
    return results


def _local_correctness_check(meta: Dict[str, Any]) -> Tuple[bool, str]:
    """Lightweight correctness check using string comparison (no API needed).

    Catches obviously wrong trajectories where output_text != answer.
    For semantic equivalence checking, use --filter-wrong-answers with a judge.
    """
    output = str(meta.get("output_text", "")).strip()
    answer = str(meta.get("answer", "")).strip()
    if not output:
        return False, "no output_text in meta"
    if output.lower() == answer.lower():
        return True, ""
    return False, f"output '{output[:50]}' != answer '{answer[:50]}'"


def _strict_tool_match(trajectory: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Check that each Phase1 ``<think>Tool: X</think>`` matches the
    immediately following Phase2 ``tool_calls[].function.name`` 1:1.

    Returns ``(has_mismatch, reason)``.
    """
    i = 0
    step = 0
    while i < len(trajectory):
        msg = trajectory[i]
        if msg.get("role") != "assistant" or msg.get("tool_calls"):
            i += 1
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            content = get_text_from_content(content)
        if not isinstance(content, str):
            i += 1
            continue

        match = _THINK_TOOL_RE.search(content)
        if not match:
            i += 1
            continue

        planned_tool = match.group(1).lower()
        step += 1

        j = i + 1
        while j < len(trajectory):
            next_msg = trajectory[j]
            if next_msg.get("role") == "assistant" and next_msg.get("tool_calls"):
                tcs = next_msg["tool_calls"]
                actual_name = (
                    tcs[0].get("function", {}).get("name", "").lower() if tcs else ""
                )
                if actual_name and actual_name != planned_tool:
                    return True, (
                        f"Step {step}: planned '{planned_tool}' "
                        f"but called '{actual_name}'"
                    )
                break
            elif next_msg.get("role") != "assistant":
                break
            j += 1

        i = j + 1 if j < len(trajectory) else i + 1

    return False, ""


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

    if modified_trajectory is not None:
        with open(dst_sub / "trajectory.json", "w", encoding="utf-8") as f:
            json.dump(modified_trajectory, f, ensure_ascii=False, indent=2)
    else:
        src_traj = src_sub / "trajectory.json"
        if src_traj.exists():
            shutil.copy2(src_traj, dst_sub / "trajectory.json")

    for filename in ("meta_info.jsonl", "output.jsonl", "extra_info.jsonl"):
        src_file = src_sub / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_sub / filename)

    _INPUT_IMAGE_RE = re.compile(r"^input_image(?:_\d+)?\.png$")

    src_img = src_sub / "input_image.png"
    dst_img = dst_sub / "input_image.png"
    if src_img.exists():
        if use_symlink:
            if not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)
        else:
            shutil.copy2(src_img, dst_img)

    for src_file in src_sub.iterdir():
        if (
            src_file.is_file()
            and _INPUT_IMAGE_RE.match(src_file.name)
            and src_file.name != "input_image.png"
        ):
            dst_file = dst_sub / src_file.name
            if not dst_file.exists():
                if use_symlink:
                    os.symlink(src_file.resolve(), dst_file)
                else:
                    shutil.copy2(src_file, dst_file)

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
        if _INPUT_IMAGE_RE.match(item.name):
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

    p.add_argument(
        "--src-dir", type=str, required=True, help="Source trajectory directory"
    )
    p.add_argument("--dst-dir", type=str, required=True, help="Output directory")

    p.add_argument(
        "--api-base",
        type=str,
        default="https://matrixllm.alipay.com/v1",
        help="Diversification LLM API base URL",
    )
    p.add_argument(
        "--model", type=str, default="kimi-k2.5", help="Diversification model"
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Diversification API key (default: env LLM_API_KEY)",
    )

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

    p.add_argument(
        "--filter-wrong-answers",
        action="store_true",
        help="Enable wrong-answer filtering (requires judge)",
    )
    p.add_argument(
        "--filter-answer-leakage",
        action="store_true",
        default=True,
        help="Enable answer-leakage filtering (default: True, quick regex mode)",
    )
    p.add_argument(
        "--no-filter-answer-leakage",
        dest="filter_answer_leakage",
        action="store_false",
        help="Disable answer-leakage filtering",
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
        default=True,
        help="Enable tool-plan mismatch filtering (default: True)",
    )
    p.add_argument(
        "--no-filter-tool-mismatch",
        dest="filter_tool_mismatch",
        action="store_false",
        help="Disable tool-plan mismatch filtering",
    )
    p.add_argument(
        "--leakage-check-mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Leakage check mode: 'quick' (regex) or 'full' (AI judge)",
    )

    p.add_argument(
        "--max-concurrent", type=int, default=16, help="Workers for parallel filtering"
    )
    p.add_argument(
        "--max-llm-workers", type=int, default=8, help="Workers for LLM diversification"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for diversification LLM (use 1.0 for kimi-k2.5)",
    )
    p.add_argument(
        "--requests-per-minute",
        type=int,
        default=60,
        help="Rate limit for diversification API (default: 60, 0 = no limit)",
    )

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

    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    judge_api_base = args.judge_api_base or args.api_base
    judge_api_key = args.judge_api_key or api_key

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

    client: Optional[DiversificationClient] = None
    if not args.skip_diversify:
        if api_key:
            client = DiversificationClient(
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                requests_per_minute=args.requests_per_minute,
            )
            logger.info("Diversification client initialised: model=%s", args.model)
        else:
            logger.warning(
                "No API key for diversification — will copy without diversifying"
            )

    subfolders_with_images = _discover_subfolders(src_dir)
    stats = AugmentStats(total=len(subfolders_with_images))
    use_local_correctness = not args.filter_wrong_answers

    logger.info("Found %d subfolders in %s", len(subfolders_with_images), src_dir)
    logger.info(
        "Filters: brute_force=%s  wrong_answer=%s (local_fallback=%s)  leakage=%s (%s)  tool_mismatch=%s",
        args.filter_brute_force,
        args.filter_wrong_answers,
        use_local_correctness,
        args.filter_answer_leakage,
        args.leakage_check_mode,
        args.filter_tool_mismatch,
    )
    logger.info(
        "Diversification: %s",
        "disabled" if args.skip_diversify or client is None else f"model={args.model}",
    )

    # ── Phase 1: Filter ─────────────────────────────────────────────────
    logger.info("Phase 1 — Filtering %d subfolders …", len(subfolders_with_images))
    passed_subfolders: List[Tuple[Path, Path]] = []

    def _safe_filter(sf: Path, img_dir: Path) -> Tuple[Path, Path, bool, str]:
        try:
            meta = load_meta_info(sf / "meta_info.jsonl")
            if not meta:
                return sf, img_dir, False, "structure_error"
            if use_local_correctness:
                is_correct, reason = _local_correctness_check(meta)
                if not is_correct:
                    logger.info("Filtered %s: %s", sf.name, reason)
                    return sf, img_dir, False, "wrong_answer"

            passed, reason = filter_subfolder(
                sf, judge, config, args.filter_brute_force
            )
            if passed and config.filter_tool_mismatch:
                traj = load_trajectory(sf / "trajectory.json")
                has_mismatch, mismatch_reason = _strict_tool_match(traj)
                if has_mismatch:
                    logger.info(
                        "Filtered %s: strict tool mismatch — %s",
                        sf.name,
                        mismatch_reason,
                    )
                    return sf, img_dir, False, "tool_mismatch"
            return sf, img_dir, passed, reason
        except Exception as exc:
            logger.error("Unexpected error filtering %s: %s", sf.name, exc)
            return sf, img_dir, False, "structure_error"

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = [
            pool.submit(_safe_filter, sf, img_dir)
            for sf, img_dir in subfolders_with_images
        ]
        for future in as_completed(futures):
            sf, img_dir, passed, reason = future.result()
            if passed:
                passed_subfolders.append((sf, img_dir))
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
            else:
                stats.filtered_structure_error += 1

    logger.info("Phase 1 complete: %d / %d passed", stats.passed_filter, stats.total)

    if not passed_subfolders:
        logger.error("No subfolders passed filtering — nothing to output")
        print(stats.summary())
        return

    passed_subfolders.sort(key=lambda t: t[0])

    # ── Phase 2: Diversify ──────────────────────────────────────────────
    modified_trajectories: Dict[Path, List[Dict[str, Any]]] = {}

    if client is not None and not args.skip_diversify:
        logger.info("Phase 2 — Extracting and diversifying think blocks …")

        all_blocks: List[ThinkBlock] = []
        block_origins: List[Tuple[Path, int]] = []
        subfolder_data: Dict[Path, Tuple[List[Dict[str, Any]], List[ThinkBlock]]] = {}

        for sf, _img_dir in passed_subfolders:
            traj = load_trajectory(sf / "trajectory.json")
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

        if all_blocks:
            global_results = client.diversify_blocks_batch(
                all_blocks, max_workers=args.max_llm_workers
            )

            for idx, new_text in global_results.items():
                if len(all_blocks[idx].text) < 20:
                    continue
                if new_text != all_blocks[idx].text:
                    stats.blocks_diversified += 1
                else:
                    stats.blocks_failed += 1

            sf_new_texts: Dict[Path, Dict[int, str]] = {}
            for global_idx, new_text in global_results.items():
                sf, local_idx = block_origins[global_idx]
                sf_new_texts.setdefault(sf, {})[local_idx] = new_text

            for sf, _img_dir in passed_subfolders:
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
    logger.info(
        "Phase 3 — Copying %d subfolders to %s …", len(passed_subfolders), dst_dir
    )

    for sf, img_dir in passed_subfolders:
        if sf == img_dir:
            dst_sub = dst_dir / sf.name
        else:
            dst_sub = dst_dir / f"{img_dir.name}_{sf.name}"
        mod_traj = modified_trajectories.get(sf)
        copy_subfolder(sf, dst_sub, mod_traj, args.use_symlink)
        if img_dir != sf:
            _img_re = re.compile(r"^input_image(?:_\d+)?\.png$")
            for src_file in img_dir.iterdir():
                if src_file.is_file() and _img_re.match(src_file.name):
                    dst_file = dst_sub / src_file.name
                    if not dst_file.exists():
                        if args.use_symlink:
                            os.symlink(src_file.resolve(), dst_file)
                        else:
                            shutil.copy2(src_file, dst_file)
        stats.output_subfolders += 1

    summary = stats.summary()
    logger.info("\n%s", summary)
    print(summary)


if __name__ == "__main__":
    main()
