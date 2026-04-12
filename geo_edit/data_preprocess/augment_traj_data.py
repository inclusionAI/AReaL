"""Augment SFT trajectory data with LLM-driven think block diversification and filtering.

Pipeline:
1. Filter: Validate structure, check brute-force / correctness / leakage / tool-mismatch
2. Diversify: Extract <think> blocks, classify, LLM-rephrase in global batch
3. Output: Copy filtered + diversified trajectories preserving directory structure

The output directory mirrors the source layout so that
``convert_trajectory_to_sft.py`` can read it directly.

Usage (filter only):
    python -m geo_edit.data_preprocess.augment_traj_data \\
        --src-dir /path/to/trajectories \\
        --dst-dir /path/to/output \\
        --skip-diversify

Usage (full pipeline, all options):
    python -m geo_edit.data_preprocess.augment_traj_data \\
        --src-dir /path/to/trajectories \\
        --dst-dir /path/to/output \\
        --api-base https://matrixllm.alipay.com/v1 \\
        --model GLM-4.7 \\
        --api-key $API_KEY \\
        --judge-api-base https://matrixllm.alipay.com/v1 \\
        --judge-model gpt-5-mini-2025-08-07 \\
        --judge-api-key $JUDGE_KEY \\
        --filter-wrong-answers \\
        --filter-answer-leakage \\
        --leakage-check-mode full \\
        --filter-brute-force \\
        --filter-tool-mismatch \\
        --max-concurrent 16 \\
        --max-llm-workers 8 \\
        --temperature 0.7 \\
        --requests-per-minute 60

All arguments:
    --src-dir               Source trajectory directory (required)
    --dst-dir               Output directory (required)
    --api-base              Diversification LLM API base URL (default: https://matrixllm.alipay.com/v1)
    --model                 Diversification model (default: GLM-4.7)
    --api-key               Diversification API key (default: env LLM_API_KEY)
    --judge-api-base        Judge API base URL (default: same as --api-base)
    --judge-model           Judge model name (default: gpt-5-mini-2025-08-07)
    --judge-api-key         Judge API key (default: same as --api-key)
    --filter-wrong-answers  Enable wrong-answer filtering via judge
    --filter-answer-leakage / --no-filter-answer-leakage
                            Enable/disable answer-leakage filtering (default: on)
    --filter-brute-force / --no-filter-brute-force
                            Enable/disable brute-force filtering (default: on)
    --filter-tool-mismatch / --no-filter-tool-mismatch
                            Enable/disable tool-plan mismatch filtering (default: on)
    --leakage-check-mode    Leakage check: 'quick' (regex) or 'full' (AI judge) (default: full)
    --max-concurrent        Workers for parallel filtering (default: 16)
    --max-llm-workers       Workers for LLM diversification (default: 8)
    --temperature           Temperature for diversification LLM (default: 0.7)
    --requests-per-minute   Rate limit for diversification API (default: 60, 0=no limit)
    --skip-diversify        Only filter, skip LLM diversification
    --reuse-filter          Reuse cached Phase 1 filter results if available
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

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

# Suppress httpx INFO logs (OpenAI SDK HTTP request/response noise)
logging.getLogger("httpx").setLevel(logging.WARNING)

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


def _rewrite_image_paths(trajectory: List[Dict[str, Any]], src_sub: Path, dst_sub: Path) -> List[Dict[str, Any]]:
    """Rewrite file:// image URLs and image_path fields in trajectory to point to dst."""
    src_str = str(src_sub.resolve())
    dst_str = str(dst_sub.resolve())

    raw = json.dumps(trajectory, ensure_ascii=False)
    raw = raw.replace(f"file://{src_str}", f"file://{dst_str}")
    raw = raw.replace(src_str.replace("\\", "/"), dst_str.replace("\\", "/"))
    raw = raw.replace(src_str.replace("/", "\\\\"), dst_str.replace("/", "\\\\"))
    return json.loads(raw)


def copy_subfolder(
    src_sub: Path,
    dst_sub: Path,
    modified_trajectory: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Copy a trajectory subfolder to *dst_sub*, optionally injecting a
    modified ``trajectory.json``.

    Image paths inside the trajectory are rewritten to point to *dst_sub*.
    """
    os.makedirs(dst_sub, exist_ok=True)

    # Load trajectory (modified or from disk), rewrite paths, then save
    if modified_trajectory is not None:
        traj = _rewrite_image_paths(modified_trajectory, src_sub, dst_sub)
    else:
        src_traj = src_sub / "trajectory.json"
        if src_traj.exists():
            with open(src_traj, "r", encoding="utf-8") as f:
                traj = json.load(f)
            traj = _rewrite_image_paths(traj, src_sub, dst_sub)
        else:
            traj = None

    if traj is not None:
        with open(dst_sub / "trajectory.json", "w", encoding="utf-8") as f:
            json.dump(traj, f, ensure_ascii=False, indent=2)

    for filename in ("meta_info.jsonl", "output.jsonl", "extra_info.jsonl"):
        src_file = src_sub / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_sub / filename)

    _INPUT_IMAGE_RE = re.compile(r"^input_image(?:_\d+)?\.png$")

    src_img = src_sub / "input_image.png"
    dst_img = dst_sub / "input_image.png"
    if src_img.exists():
        shutil.copy2(src_img, dst_img)

    for src_file in src_sub.iterdir():
        if (
            src_file.is_file()
            and _INPUT_IMAGE_RE.match(src_file.name)
            and src_file.name != "input_image.png"
        ):
            dst_file = dst_sub / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)

    src_images = src_sub / "images"
    dst_images = dst_sub / "images"
    if src_images.exists() and src_images.is_dir():
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


# ── Filter Cache ─────────────────────────────────────────────────────────────

_FILTER_CACHE_NAME = ".filter_cache.json"


def _filter_config_fingerprint(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a dict capturing the filter-relevant CLI settings."""
    return {
        "filter_wrong_answers": args.filter_wrong_answers,
        "filter_answer_leakage": args.filter_answer_leakage,
        "filter_brute_force": args.filter_brute_force,
        "filter_tool_mismatch": args.filter_tool_mismatch,
        "leakage_check_mode": args.leakage_check_mode,
    }


def _save_filter_cache(
    dst_dir: Path,
    src_dir: Path,
    args: argparse.Namespace,
    passed: List[Tuple[Path, Path]],
    stats: AugmentStats,
) -> None:
    """Persist Phase 1 filter results so subsequent runs can skip filtering."""
    os.makedirs(dst_dir, exist_ok=True)
    cache = {
        "src_dir": str(src_dir),
        "filter_config": _filter_config_fingerprint(args),
        "stats": {
            "total": stats.total,
            "passed_filter": stats.passed_filter,
            "filtered_wrong_answer": stats.filtered_wrong_answer,
            "filtered_leakage": stats.filtered_leakage,
            "filtered_tool_mismatch": stats.filtered_tool_mismatch,
            "filtered_brute_force": stats.filtered_brute_force,
            "filtered_structure_error": stats.filtered_structure_error,
            "api_errors": stats.api_errors,
        },
        "passed_subfolders": [[str(sf), str(img)] for sf, img in passed],
    }
    cache_path = dst_dir / _FILTER_CACHE_NAME
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info("Filter cache saved to %s (%d entries)", cache_path, len(passed))


def _load_filter_cache(
    dst_dir: Path,
    src_dir: Path,
    args: argparse.Namespace,
    current_total: int,
) -> Optional[Tuple[List[Tuple[Path, Path]], AugmentStats]]:
    """Load cached Phase 1 results if cache exists and settings match."""
    cache_path = dst_dir / _FILTER_CACHE_NAME
    if not cache_path.exists():
        logger.info("No filter cache found at %s", cache_path)
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read filter cache: %s", e)
        return None

    if cache.get("src_dir") != str(src_dir):
        logger.info("Filter cache src_dir mismatch — re-filtering")
        return None

    if cache.get("filter_config") != _filter_config_fingerprint(args):
        logger.info("Filter cache config mismatch — re-filtering")
        return None

    s = cache.get("stats", {})
    if s.get("total", -1) != current_total:
        logger.info(
            "Filter cache total mismatch (cached=%d, current=%d) — re-filtering",
            s.get("total", -1),
            current_total,
        )
        return None

    passed_raw = cache.get("passed_subfolders", [])
    passed = [(Path(sf), Path(img)) for sf, img in passed_raw]

    missing = [(sf, img) for sf, img in passed if not sf.exists()]
    if missing:
        logger.warning(
            "Filter cache: %d / %d cached paths missing on disk — re-filtering",
            len(missing),
            len(passed),
        )
        return None

    stats = AugmentStats(
        total=s.get("total", 0),
        passed_filter=s.get("passed_filter", 0),
        filtered_wrong_answer=s.get("filtered_wrong_answer", 0),
        filtered_leakage=s.get("filtered_leakage", 0),
        filtered_tool_mismatch=s.get("filtered_tool_mismatch", 0),
        filtered_brute_force=s.get("filtered_brute_force", 0),
        filtered_structure_error=s.get("filtered_structure_error", 0),
        api_errors=s.get("api_errors", 0),
    )
    logger.info(
        "Loaded filter cache: %d / %d passed (%s)",
        stats.passed_filter,
        stats.total,
        cache_path,
    )
    return passed, stats


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
        "--model", type=str, default="GLM-4.7", help="Diversification model"
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
        default="full",
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

    p.add_argument(
        "--skip-diversify",
        action="store_true",
        help="Only filter — skip LLM diversification",
    )
    p.add_argument(
        "--reuse-filter",
        action="store_true",
        help="Reuse cached Phase 1 filter results from a previous run (skips re-filtering when src_dir, filter config, and subfolder count all match)",
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
        client = DiversificationClient(
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            requests_per_minute=args.requests_per_minute,
        )
        logger.info("Diversification client initialised: model=%s", args.model)

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
    passed_subfolders: List[Tuple[Path, Path]] = []
    _phase1_from_cache = False

    if args.reuse_filter:
        _cached = _load_filter_cache(dst_dir, src_dir, args, current_total=stats.total)
        if _cached is not None:
            passed_subfolders, stats = _cached
            _phase1_from_cache = True

    if not _phase1_from_cache:
        def _safe_filter(sf: Path, img_dir: Path) -> Tuple[Path, Path, bool, str]:
            try:
                meta = load_meta_info(sf / "meta_info.jsonl")
                if not meta:
                    return sf, img_dir, False, "structure_error"
                if use_local_correctness:
                    is_correct, reason = _local_correctness_check(meta)
                    if not is_correct:
                        logger.debug("Filtered %s: %s", sf.name, reason)
                        return sf, img_dir, False, "wrong_answer"

                passed, reason = filter_subfolder(
                    sf, judge, config, args.filter_brute_force
                )
                if passed and config.filter_tool_mismatch:
                    traj = load_trajectory(sf / "trajectory.json")
                    has_mismatch, mismatch_reason = _strict_tool_match(traj)
                    if has_mismatch:
                        logger.debug(
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
            pbar = tqdm(
                total=len(futures), desc="Phase 1: Filtering", unit="traj"
            )
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
                pbar.update(1)
            pbar.close()

        logger.info("Phase 1 complete: %d / %d passed filter", stats.passed_filter, stats.total)
        _save_filter_cache(dst_dir, src_dir, args, passed_subfolders, stats)

    if not passed_subfolders:
        logger.error("No subfolders passed filtering — nothing to output")
        print(stats.summary())
        return

    passed_subfolders.sort(key=lambda t: t[0])

    # ── Phase 2: Diversify ──────────────────────────────────────────────
    modified_trajectories: Dict[Path, List[Dict[str, Any]]] = {}

    if client is not None and not args.skip_diversify:
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
            "Phase 2 complete: %d diversified, %d failed, %d skipped (short)",
            stats.blocks_diversified,
            stats.blocks_failed,
            stats.blocks_skipped_short,
        )

    # ── Phase 3: Copy ───────────────────────────────────────────────────
    for sf, img_dir in tqdm(passed_subfolders, desc="Phase 3: Copying", unit="traj"):
        if sf == img_dir:
            dst_sub = dst_dir / sf.name
        else:
            dst_sub = dst_dir / f"{img_dir.name}_{sf.name}"
        mod_traj = modified_trajectories.get(sf)
        copy_subfolder(sf, dst_sub, mod_traj)
        if img_dir != sf:
            _img_re = re.compile(r"^input_image(?:_\d+)?\.png$")
            for src_file in img_dir.iterdir():
                if src_file.is_file() and _img_re.match(src_file.name):
                    dst_file = dst_sub / src_file.name
                    if not dst_file.exists():
                        shutil.copy2(src_file, dst_file)
        stats.output_subfolders += 1

    summary = stats.summary()
    logger.info("\n%s", summary)
    print(summary)


if __name__ == "__main__":
    main()
