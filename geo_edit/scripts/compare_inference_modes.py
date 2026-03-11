"""Compare tool-augmented vs direct inference modes.

This script compares two inference approaches on the same dataset samples:
1. Tool-Augmented Mode: Uses trajectory parquet with multi-turn conversation history
2. Direct Mode: Direct single-turn inference without tools

Usage:
    python -m geo_edit.scripts.compare_inference_modes \
        --trajectory_parquet /path/to/filtered_trajectory.parquet \
        --trajectory_results /path/to/trajectory_results \
        --raw_parquet /path/to/raw_dataset.parquet \
        --dataset_name cartomapqa_en \
        --output_dir /path/to/comparison_output \
        --model_name_or_path /path/to/model \
        --api_base http://127.0.0.1:8000 \
        --eval_api_key $EVAL_API_KEY \
        --max_concurrent_requests 32
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm

from geo_edit.agents.api_agent import APIBasedAgent, AgentConfig
from geo_edit.config import build_api_agent_configs
from geo_edit.datasets.task_registry import get_dataset_spec
from geo_edit.environment.task.openai_compatible_vision_qa_task import OpenAICompatibleVisionQATask
from geo_edit.evaluation.openai_as_judge import EvalConfig, evaluate_record
from geo_edit.tool_definitions import ToolRouter
from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

DIRECT_SYSTEM_PROMPT = """
You are an advanced AI assistant capable of complex reasoning.
You must strictly adhere to the following protocol:

1. Reasoning Process: Before providing your answer, analyze the
problem step by step. Output your reasoning inside <think> and </think> tags.

2. Final Output: When you have formulated your conclusion,
wrap your final answer in <answer> and </answer> tags.
"""

TEMPERATURE = 1.0
N_RETRY = 3
SEED = 42


# =============================================================================
# ID Extraction and Dataset Filtering
# =============================================================================

def extract_sample_ids_from_trajectory(parquet_path: Path, dataset_name: str) -> Dict[str, str]:
    """Extract mapping from trajectory ID to raw dataset ID.

    The trajectory parquet ID is the subfolder name from inference output.
    The raw ID is extracted from meta_info field based on dataset spec.

    Args:
        parquet_path: Path to trajectory parquet file.
        dataset_name: Dataset name to determine ID key.

    Returns:
        Dict mapping parquet_id -> raw_id
    """
    ds = Dataset.from_parquet(str(parquet_path))
    spec = get_dataset_spec(dataset_name)
    id_mapping = {}

    for sample in ds:
        parquet_id = sample["id"]
        meta_info = json.loads(sample["meta_info"]) if sample.get("meta_info") else {}

        # Try to get raw ID from meta_info using dataset spec's id_key
        raw_id = meta_info.get(spec.id_key)
        if raw_id is None:
            # Fallback: try common ID keys
            raw_id = meta_info.get("id") or meta_info.get("case_id") or meta_info.get("index") or parquet_id

        id_mapping[str(parquet_id)] = str(raw_id)

    logger.info("Extracted %d sample IDs from trajectory parquet", len(id_mapping))
    return id_mapping


def filter_raw_dataset(
    raw_parquet_path: Path,
    target_ids: Set[str],
    dataset_name: str,
) -> Dataset:
    """Filter raw dataset to only include target sample IDs.

    Args:
        raw_parquet_path: Path to raw dataset parquet.
        target_ids: Set of sample IDs to include.
        dataset_name: Dataset name to determine ID key.

    Returns:
        Filtered dataset containing only target samples.
    """
    ds = load_dataset("parquet", data_files=str(raw_parquet_path))["train"]
    spec = get_dataset_spec(dataset_name)

    original_count = len(ds)
    filtered = ds.filter(lambda x: str(x[spec.id_key]) in target_ids)
    logger.info("Filtered dataset: %d -> %d samples", original_count, len(filtered))

    return filtered


# =============================================================================
# Direct Inference Worker
# =============================================================================

_WORKER_AGENT: Optional[APIBasedAgent] = None
_WORKER_MODEL_TYPE: Optional[str] = None
_WORKER_API_MODE: Optional[str] = None


def _init_worker(
    model_name_or_path: str,
    model_type: str,
    api_base: str,
    api_mode: str,
    api_key: Optional[str],
):
    """Initialize worker process with agent."""
    global _WORKER_AGENT, _WORKER_MODEL_TYPE, _WORKER_API_MODE

    tool_router = ToolRouter(tool_mode="direct")
    agent_configs = build_api_agent_configs(
        tool_router,
        api_mode=api_mode,
        temperature=TEMPERATURE,
        system_prompt=DIRECT_SYSTEM_PROMPT.strip(),
        reasoning_level="low"
    )

    config = AgentConfig(
        model_type=model_type,
        model_name=model_name_or_path,
        api_key=api_key,
        api_base=api_base,
        generate_config=agent_configs.generate_config,
        n_retry=N_RETRY,
        api_mode=api_mode,
    )
    _WORKER_AGENT = APIBasedAgent(config)
    _WORKER_MODEL_TYPE = model_type
    _WORKER_API_MODE = api_mode


def _run_one_task(task_payload: dict) -> Tuple[bool, Optional[dict]]:
    """Run a single direct inference task."""
    assert _WORKER_AGENT is not None

    task_id = task_payload["id"]
    task_save_dir = task_payload["task_save_dir"]
    answer = task_payload["answer"]
    image_path = task_payload["image_path"]
    text_prompt = task_payload["prompt"]
    text_only = task_payload.get("text_only", False)

    meta_path = os.path.join(task_save_dir, "meta_info.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return True, json.loads(f.readline().strip())

    model_type_map = {"vLLM": "vllm", "SGLang": "sglang", "OpenAI": "openai"}
    model_type = model_type_map.get(_WORKER_MODEL_TYPE, "vllm")

    task_kwargs = {"model_type": model_type, "api_mode": _WORKER_API_MODE}
    extra_kwargs = task_payload.get("task_kwargs")
    if isinstance(extra_kwargs, dict):
        task_kwargs.update(extra_kwargs)
    if text_only:
        task_kwargs["text_only"] = True

    task = OpenAICompatibleVisionQATask(
        task_id=task_id,
        task_prompt=text_prompt,
        task_answer=answer,
        task_image_path=image_path,
        tool_functions={},
        save_dir=task_save_dir,
        **task_kwargs,
    )

    _WORKER_AGENT.reset()
    try:
        action, extra_info = _WORKER_AGENT.act(task.contents)
        _ = task.parse_action(step=1, action=action, extra_info=extra_info)

        if task.state:
            return True, task.save_trajectory()

        shutil.rmtree(task_save_dir, ignore_errors=True)
        return False, None

    except Exception as e:
        logger.error("[%s] worker failed: %s", task_id, e)
        shutil.rmtree(task_save_dir, ignore_errors=True)
        return False, None


def run_direct_inference(
    filtered_dataset: Dataset,
    output_dir: Path,
    dataset_name: str,
    model_name_or_path: str,
    model_type: str,
    api_base: str,
    api_mode: str,
    api_key: Optional[str],
    max_workers: int,
) -> List[Dict[str, Any]]:
    """Run direct inference on filtered dataset.

    Args:
        filtered_dataset: Pre-filtered dataset.
        output_dir: Output directory for results.
        dataset_name: Dataset name for prompt building.
        model_name_or_path: Model name or path.
        model_type: Model type (vLLM, SGLang, OpenAI).
        api_base: API base URL.
        api_mode: API mode (chat_completions, responses).
        api_key: Optional API key.
        max_workers: Number of worker processes.

    Returns:
        List of meta info dicts for completed samples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = get_dataset_spec(dataset_name)

    # Collect pending items
    meta_info_list: List[Dict[str, Any]] = []
    pending_items = []

    for item in filtered_dataset:
        task_id = str(item[spec.id_key])
        task_save_dir = output_dir / task_id
        meta_path = task_save_dir / "meta_info.jsonl"

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_info_list.append(json.loads(f.readline().strip()))
        else:
            pending_items.append(item)

    logger.info("Direct inference: Already done: %d, Pending: %d", len(meta_info_list), len(pending_items))

    if not pending_items:
        return meta_info_list

    # Multiprocessing pool
    ctx = mp.get_context("spawn")
    n_workers = max(1, max_workers)

    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(model_name_or_path, model_type, api_base, api_mode, api_key),
    ) as pool:
        inflight = []
        submit_idx = 0
        pbar = tqdm(total=len(pending_items), desc="Direct inference")

        while submit_idx < len(pending_items) or inflight:
            # Submit tasks
            while submit_idx < len(pending_items) and len(inflight) < n_workers:
                item = pending_items[submit_idx]
                submit_idx += 1

                task_id = str(item[spec.id_key])
                task_save_dir = output_dir / task_id
                meta_path = task_save_dir / "meta_info.jsonl"

                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_info_list.append(json.loads(f.readline().strip()))
                    continue

                task_save_dir.mkdir(parents=True, exist_ok=True)

                # Handle image
                image_path = None
                text_only = spec.image_key is None
                if spec.image_key:
                    image_path = str(task_save_dir / "input_image.png")
                    if not os.path.exists(image_path):
                        image = item.get(spec.image_key)
                        if isinstance(image, Image.Image):
                            image.save(image_path)
                        elif isinstance(image, dict) and "bytes" in image:
                            image = Image.open(BytesIO(image["bytes"]))
                            image.save(image_path)
                        elif isinstance(image, bytes):
                            image = Image.open(BytesIO(image))
                            image.save(image_path)
                        else:
                            logger.warning("Invalid image type for %s: %s", task_id, type(image))
                            text_only = True
                            image_path = None
                else:
                    text_only = True

                payload = {
                    "id": task_id,
                    "task_save_dir": str(task_save_dir),
                    "prompt": spec.build_prompt(item, use_tools=False),
                    "answer": spec.get_answer(item),
                    "image_path": image_path,
                    "text_only": text_only,
                    "task_kwargs": spec.build_task_kwargs(item),
                }

                ar = pool.apply_async(_run_one_task, (payload,))
                inflight.append((task_id, ar))

            # Harvest finished tasks
            any_done = False
            still_inflight = []
            for task_id, ar in inflight:
                if ar.ready():
                    ok, meta_info = ar.get()
                    if ok and meta_info is not None:
                        meta_info_list.append(meta_info)
                    pbar.update(1)
                    any_done = True
                else:
                    still_inflight.append((task_id, ar))
            inflight = still_inflight

            if not any_done:
                time.sleep(0.05)

        pbar.close()

    return meta_info_list


# =============================================================================
# Evaluation
# =============================================================================

def run_evaluation(
    result_path: Path,
    output_path: Path,
    eval_api_key: str,
    eval_api_base: Optional[str],
    eval_model: str,
) -> Dict[str, Any]:
    """Run evaluation on result directory using openai_as_judge.

    Args:
        result_path: Path to results directory.
        output_path: Path to save evaluation results.
        eval_api_key: API key for evaluation.
        eval_api_base: Optional API base URL for evaluation.
        eval_model: Model name for evaluation.

    Returns:
        Dict with evaluation metrics.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig(model=eval_model, api_key=eval_api_key, api_base=eval_api_base)
    eval_output_path = output_path / "eval_result.jsonl"

    total = 0
    correct = 0
    filtered = 0
    eval_results = []

    max_workers = 32
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for meta_path in iter_meta_info_files(str(result_path)):
            record_id = os.path.basename(os.path.dirname(meta_path))
            for record in load_records(meta_path):
                futures.append(executor.submit(evaluate_record, record, cfg, record_id))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            eval_item = future.result()
            eval_results.append(eval_item)
            result = eval_item["result"]
            is_filter = isinstance(result, dict) and result.get("is_filter")
            if is_filter:
                filtered += 1
            else:
                total += 1
                if result == 1.0:
                    correct += 1

    # Write results
    with open(eval_output_path, "w", encoding="utf-8") as f:
        for item in eval_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    accuracy = (correct / total) if total else 0.0
    summary_path = output_path / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"evaluated={total}\n")
        f.write(f"correct={correct}\n")
        f.write(f"filtered={filtered}\n")
        f.write(f"accuracy={accuracy:.6f}\n")

    return {
        "total": total,
        "correct": correct,
        "filtered": filtered,
        "accuracy": accuracy,
        "results": eval_results,
    }


# =============================================================================
# Comparison
# =============================================================================

def generate_comparison_report(
    trajectory_metrics: Dict[str, Any],
    direct_metrics: Dict[str, Any],
    dataset_name: str,
    model_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Generate comparison report between two inference modes.

    Args:
        trajectory_metrics: Evaluation metrics for trajectory mode.
        direct_metrics: Evaluation metrics for direct mode.
        dataset_name: Dataset name.
        model_name: Model name.
        output_path: Path to save report.

    Returns:
        Comparison report dict.
    """
    # Build per-sample comparison
    traj_results_by_id = {r["id"]: r for r in trajectory_metrics["results"]}
    direct_results_by_id = {r["id"]: r for r in direct_metrics["results"]}

    per_sample_diff = []
    both_correct = 0
    both_wrong = 0
    traj_only_correct = 0
    direct_only_correct = 0

    all_ids = set(traj_results_by_id.keys()) | set(direct_results_by_id.keys())
    for sample_id in sorted(all_ids):
        traj_result = traj_results_by_id.get(sample_id, {}).get("result", 0)
        direct_result = direct_results_by_id.get(sample_id, {}).get("result", 0)

        # Handle filtered results
        traj_score = 1 if traj_result == 1.0 else 0
        direct_score = 1 if direct_result == 1.0 else 0

        per_sample_diff.append({
            "id": sample_id,
            "trajectory": traj_score,
            "direct": direct_score,
        })

        if traj_score == 1 and direct_score == 1:
            both_correct += 1
        elif traj_score == 0 and direct_score == 0:
            both_wrong += 1
        elif traj_score == 1:
            traj_only_correct += 1
        else:
            direct_only_correct += 1

    improvement = trajectory_metrics["accuracy"] - direct_metrics["accuracy"]
    improvement_pct = (improvement / direct_metrics["accuracy"] * 100) if direct_metrics["accuracy"] > 0 else 0

    report = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "total_samples": len(all_ids),
        "trajectory_mode": {
            "accuracy": trajectory_metrics["accuracy"],
            "correct": trajectory_metrics["correct"],
            "incorrect": trajectory_metrics["total"] - trajectory_metrics["correct"],
            "filtered": trajectory_metrics["filtered"],
        },
        "direct_mode": {
            "accuracy": direct_metrics["accuracy"],
            "correct": direct_metrics["correct"],
            "incorrect": direct_metrics["total"] - direct_metrics["correct"],
            "filtered": direct_metrics["filtered"],
        },
        "comparison": {
            "accuracy_improvement": improvement,
            "improvement_percentage": f"{improvement_pct:.1f}%",
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "trajectory_only_correct": traj_only_correct,
            "direct_only_correct": direct_only_correct,
        },
        "per_sample_diff": per_sample_diff,
    }

    # Save report
    report_path = output_path / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Comparison report saved to %s", report_path)
    return report


def print_comparison_summary(report: Dict[str, Any]) -> None:
    """Print comparison summary to console."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Dataset: {report['dataset_name']}")
    print(f"Model: {report['model_name']}")
    print(f"Total Samples: {report['total_samples']}")
    print()
    print("Trajectory Mode (with tools):")
    print(f"  Accuracy: {report['trajectory_mode']['accuracy']:.4f}")
    print(f"  Correct: {report['trajectory_mode']['correct']}")
    print(f"  Incorrect: {report['trajectory_mode']['incorrect']}")
    print()
    print("Direct Mode (no tools):")
    print(f"  Accuracy: {report['direct_mode']['accuracy']:.4f}")
    print(f"  Correct: {report['direct_mode']['correct']}")
    print(f"  Incorrect: {report['direct_mode']['incorrect']}")
    print()
    print("Comparison:")
    print(f"  Accuracy Improvement: {report['comparison']['accuracy_improvement']:.4f} ({report['comparison']['improvement_percentage']})")
    print(f"  Both Correct: {report['comparison']['both_correct']}")
    print(f"  Both Wrong: {report['comparison']['both_wrong']}")
    print(f"  Trajectory Only Correct: {report['comparison']['trajectory_only_correct']}")
    print(f"  Direct Only Correct: {report['comparison']['direct_only_correct']}")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare tool-augmented vs direct inference modes."
    )

    # Input paths
    parser.add_argument(
        "--trajectory_parquet",
        type=str,
        required=True,
        help="Path to filtered trajectory parquet file.",
    )
    parser.add_argument(
        "--trajectory_results",
        type=str,
        required=True,
        help="Path to trajectory inference results directory.",
    )
    parser.add_argument(
        "--raw_parquet",
        type=str,
        required=True,
        help="Path to raw dataset parquet file.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (must match DATASET_SPECS).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for comparison results.",
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model name or path for inference.",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        required=True,
        help="API base URL for inference.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vLLM",
        choices=["vLLM", "SGLang", "OpenAI"],
        help="Model type.",
    )
    parser.add_argument(
        "--api_mode",
        type=str,
        default="chat_completions",
        choices=["chat_completions", "responses"],
        help="API mode.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for inference (optional for vLLM/SGLang).",
    )

    # Evaluation configuration
    parser.add_argument(
        "--eval_api_key",
        type=str,
        required=True,
        help="API key for evaluation judge.",
    )
    parser.add_argument(
        "--eval_api_base",
        type=str,
        default=None,
        help="API base URL for evaluation judge.",
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Model name for evaluation judge.",
    )

    # Processing configuration
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=8,
        help="Number of concurrent requests for inference.",
    )
    parser.add_argument(
        "--skip_direct_inference",
        action="store_true",
        help="Skip direct inference (use existing results).",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation (use existing eval results).",
    )

    args = parser.parse_args()

    # Setup paths
    trajectory_parquet_path = Path(args.trajectory_parquet)
    trajectory_results_path = Path(args.trajectory_results)
    raw_parquet_path = Path(args.raw_parquet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_results_path = output_dir / "direct_results"
    trajectory_eval_path = output_dir / "trajectory_eval"
    direct_eval_path = output_dir / "direct_eval"

    # Step 1: Extract sample IDs from trajectory parquet
    logger.info("Step 1: Extracting sample IDs from trajectory parquet...")
    id_mapping = extract_sample_ids_from_trajectory(trajectory_parquet_path, args.dataset_name)
    target_ids = set(id_mapping.values())

    # Step 2: Filter raw dataset
    logger.info("Step 2: Filtering raw dataset...")
    filtered_dataset = filter_raw_dataset(raw_parquet_path, target_ids, args.dataset_name)

    # Step 3: Run direct inference
    if not args.skip_direct_inference:
        logger.info("Step 3: Running direct inference...")
        run_direct_inference(
            filtered_dataset=filtered_dataset,
            output_dir=direct_results_path,
            dataset_name=args.dataset_name,
            model_name_or_path=args.model_name_or_path,
            model_type=args.model_type,
            api_base=args.api_base,
            api_mode=args.api_mode,
            api_key=args.api_key,
            max_workers=args.max_concurrent_requests,
        )
    else:
        logger.info("Step 3: Skipping direct inference (using existing results)")

    # Step 4: Evaluate both result sets
    if not args.skip_evaluation:
        logger.info("Step 4a: Evaluating trajectory results...")
        trajectory_metrics = run_evaluation(
            result_path=trajectory_results_path,
            output_path=trajectory_eval_path,
            eval_api_key=args.eval_api_key,
            eval_api_base=args.eval_api_base,
            eval_model=args.eval_model,
        )

        logger.info("Step 4b: Evaluating direct results...")
        direct_metrics = run_evaluation(
            result_path=direct_results_path,
            output_path=direct_eval_path,
            eval_api_key=args.eval_api_key,
            eval_api_base=args.eval_api_base,
            eval_model=args.eval_model,
        )
    else:
        logger.info("Step 4: Skipping evaluation (loading existing results)")
        # Load existing evaluation results
        trajectory_metrics = _load_existing_eval(trajectory_eval_path)
        direct_metrics = _load_existing_eval(direct_eval_path)

    # Step 5: Generate comparison report
    logger.info("Step 5: Generating comparison report...")
    report = generate_comparison_report(
        trajectory_metrics=trajectory_metrics,
        direct_metrics=direct_metrics,
        dataset_name=args.dataset_name,
        model_name=args.model_name_or_path,
        output_path=output_dir,
    )

    # Print summary
    print_comparison_summary(report)
    logger.info("Comparison complete. Results saved to %s", output_dir)


def _load_existing_eval(eval_path: Path) -> Dict[str, Any]:
    """Load existing evaluation results from directory."""
    eval_result_path = eval_path / "eval_result.jsonl"
    summary_path = eval_path / "summary.txt"

    results = []
    if eval_result_path.exists():
        with open(eval_result_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    metrics = {"total": 0, "correct": 0, "filtered": 0, "accuracy": 0.0, "results": results}

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    if key in ("evaluated", "total"):
                        metrics["total"] = int(value)
                    elif key == "correct":
                        metrics["correct"] = int(value)
                    elif key == "filtered":
                        metrics["filtered"] = int(value)
                    elif key == "accuracy":
                        metrics["accuracy"] = float(value)

    return metrics


if __name__ == "__main__":
    main()
