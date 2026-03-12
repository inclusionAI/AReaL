"""Iterative sampling script for generating valid trajectories.

Based on separated_reasoning_generate.py, this script adds iterative sampling
to retry and extend tool rounds until a valid trajectory is generated.

Valid trajectory definition:
1. Answer is correct (verified by TrajectoryJudge)
2. No answer leakage in Phase 1/2 (optional check)

Sampling strategy:
- For each round, do Phase 1 (reasoning) + Phase 2 (tool call)
- Then try Phase 3 (final answer) up to N times
- If all Phase 3 attempts fail, extend to next round with more tool calls
- Maximum rounds: 5
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from io import BytesIO
from typing import Optional, Tuple

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from geo_edit.prompts.system_prompts import (
    SEPARATED_USER_PROMPT,
    ITERATIVE_EXTENDED_REASONING_PROMPT,
)
from geo_edit.datasets.task_registry import DATASET_SPECS, get_dataset_spec
from geo_edit.tool_definitions import ToolRouter
from geo_edit.evaluation.trajectory_judge import TrajectoryJudge
from geo_edit.utils.logger import setup_logger
from geo_edit.utils.stats import save_global_meta_info
from geo_edit.utils.text_utils import extract_response_text
from geo_edit.utils.worker_utils import init_worker_base, WorkerContext

logger = setup_logger(__name__)

# =============================================================================
# Worker globals (one per process)
# =============================================================================
_WORKER_CTX: "WorkerContext | None" = None
_WORKER_JUDGE: "TrajectoryJudge | None" = None
_WORKER_MAX_ITERATIVE_ROUNDS: int = 5
_WORKER_ATTEMPTS_PER_ROUND: int = 2
_WORKER_SKIP_LEAKAGE_CHECK: bool = False


# =============================================================================
# Worker initialization
# =============================================================================
def _init_worker(
    api_key: str,
    model_name_or_path: str,
    model_type: str,
    api_base: str,
    port: int,
    output_path: str,
    enabled_agent_names: list,
    enable_tools: list,
    # Params for iterative sampling
    judge_api_key: str,
    judge_model: str,
    judge_api_base: Optional[str],
    max_iterative_rounds: int,
    attempts_per_round: int,
    skip_leakage_check: bool,
):
    """Initialize worker for iterative sampling."""
    global _WORKER_CTX, _WORKER_JUDGE
    global _WORKER_MAX_ITERATIVE_ROUNDS, _WORKER_ATTEMPTS_PER_ROUND, _WORKER_SKIP_LEAKAGE_CHECK

    # Set iterative sampling params
    _WORKER_MAX_ITERATIVE_ROUNDS = max_iterative_rounds
    _WORKER_ATTEMPTS_PER_ROUND = attempts_per_round
    _WORKER_SKIP_LEAKAGE_CHECK = skip_leakage_check

    # Use shared worker initialization
    _WORKER_CTX = init_worker_base(
        api_key=api_key,
        model_name_or_path=model_name_or_path,
        model_type=model_type,
        api_base=api_base,
        port=port,
        output_path=output_path,
        enabled_agent_names=enabled_agent_names,
        enable_tools=enable_tools,
    )

    # Initialize TrajectoryJudge for validation
    _WORKER_JUDGE = TrajectoryJudge(
        api_key=judge_api_key,
        model=judge_model,
        api_base=judge_api_base,
    )

    logger.info(f"Worker initialized with TrajectoryJudge (PID: {os.getpid()})")


# =============================================================================
# Trajectory validation
# =============================================================================
def _validate_trajectory(
    question: str,
    ground_truth: str,
    prediction: str,
    reasoning_text: str,
    actual_tools: set,
) -> Tuple[bool, str]:
    """Validate trajectory using combined LLM check (single API call).

    Checks correctness, leakage, and tool match in one call.
    """
    assert _WORKER_JUDGE is not None, "Judge not initialized"

    return _WORKER_JUDGE.validate_trajectory(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
        reasoning_text=reasoning_text,
        actual_tools=actual_tools,
    )


# =============================================================================
# Iterative sampling main loop
# =============================================================================
def _run_one_task_iterative(task_payload: dict) -> Tuple[bool, Optional[dict]]:
    """Run iterative sampling for a single task.

    Strategy:
    - For each round, do Phase 1 + Phase 2 (reasoning + tool call)
    - Then try Phase 3 (final answer) up to attempts_per_round times
    - If all Phase 3 attempts fail, extend to next round
    """
    assert _WORKER_CTX is not None, "Worker not initialized"
    ctx = _WORKER_CTX
    agent = ctx.agent
    api_mode = ctx.api_mode
    phase_configs = ctx.phase_configs

    task_id = task_payload["id"]
    task_save_dir = task_payload["task_save_dir"]
    question = task_payload["prompt"]
    ground_truth = task_payload["answer"]
    image_path = task_payload["image_path"]
    text_only = task_payload.get("text_only", False)
    answer_format = task_payload.get("answer_format")

    # Check if already completed
    meta_path = os.path.join(task_save_dir, "meta_info.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_info = json.loads(f.readline().strip())
        return True, meta_info

    # Create task object ONCE
    formatted_text_prompt = SEPARATED_USER_PROMPT.format(Question=question)
    task_kwargs = {"model_type": "google" if api_mode == "google" else "sglang"}
    if api_mode != "google":
        task_kwargs["api_mode"] = api_mode
    extra_kwargs = task_payload.get("task_kwargs")
    if isinstance(extra_kwargs, dict):
        task_kwargs.update(extra_kwargs)
    if text_only:
        task_kwargs["text_only"] = True

    tool_functions = ctx.tool_router.get_available_tools()
    tool_return_types = ctx.tool_router.get_tool_return_types()

    os.makedirs(task_save_dir, exist_ok=True)
    task = ctx.task_class(
        task_id=task_id,
        task_prompt=formatted_text_prompt,
        task_answer=ground_truth,
        task_image_path=image_path,
        tool_functions=tool_functions,
        tool_return_types=tool_return_types,
        save_dir=task_save_dir,
        **task_kwargs,
    )

    agent.reset()
    original_generate_config = agent.config.generate_config
    answer_pattern = re.compile(r"<answer>", re.IGNORECASE)
    think_pattern = re.compile(r"<think>.*?Tool:\s*(\w+).*?</think>", re.DOTALL | re.IGNORECASE)

    all_thinking_text = []
    all_actual_tools = set()
    total_attempts = 0

    # Iterate through rounds (each round adds one more tool call)
    for current_round in range(1, _WORKER_MAX_ITERATIVE_ROUNDS + 1):
        logger.info(f"[{task_id}] Starting Round {current_round}")

        try:
            # ===== Phase 1: Generate reasoning =====
            agent.config.generate_config = phase_configs.reasoning_only

            # For Round 2+, temporarily add extended reasoning prompt (not saved to trajectory)
            if current_round > 1:
                contents_before_prompt = copy.deepcopy(task.contents)
                task.append_system_prompt(ITERATIVE_EXTENDED_REASONING_PROMPT)

            reasoning_action, reasoning_extra = agent.act(task.contents)
            logger.warning(reasoning_action)  # Display model's tool call plan

            # Restore contents (remove the temporary prompt) before adding reasoning
            if current_round > 1:
                task.contents = contents_before_prompt

            reasoning_text = extract_response_text(reasoning_action, api_mode)
            all_thinking_text.append(reasoning_text)

            # Validate format
            if answer_pattern.search(reasoning_text):
                raise ValueError("Reasoning phase should not generate <answer>")
            if not think_pattern.search(reasoning_text):
                raise ValueError("Invalid format - missing <think>Tool: ...</think>")

            # ===== Phase 2: Generate tool call =====
            task.append_assistant_message(reasoning_text)
            agent.config.generate_config = phase_configs.tool_call_only
            tool_action, tool_extra = agent.act(task.contents)
            agent.config.generate_config = original_generate_config

            merged_extra = {"reasoning_" + k: v for k, v in reasoning_extra.items()}
            merged_extra.update(tool_extra)

            function_call_part_list = task.parse_action(
                step=current_round, action=tool_action, extra_info=merged_extra
            )

            if not function_call_part_list:
                logger.warning(f"[{task_id}] Round {current_round}: No tool calls generated")
                break

            # Collect actual tool names for validation
            for tool_call in function_call_part_list:
                all_actual_tools.add(tool_call.name.lower())

            task.update_observation_from_action(function_call_part_list)

        except Exception as e:
            logger.warning(f"[{task_id}] Round {current_round} Phase 1/2 failed: {e}")
            continue

        # ===== Phase 3: Try final answer multiple times =====
        for attempt in range(_WORKER_ATTEMPTS_PER_ROUND):
            total_attempts += 1
            logger.info(f"[{task_id}] Round {current_round} Attempt {attempt + 1}: Generating final answer")

            try:
                # Save contents state before Phase 3 (for retry) - deep copy needed
                contents_before_phase3 = copy.deepcopy(task.contents)
                conv_history_len = len(task.conversation_history)

                if answer_format:
                    task.append_prompt(answer_format)

                agent.config.generate_config = phase_configs.final_answer
                action, extra_info = agent.act(task.contents)
                agent.config.generate_config = original_generate_config
                task.parse_action(step=current_round + 1, action=action, extra_info=extra_info)

                if not task.state:
                    logger.warning(f"[{task_id}] Round {current_round} Attempt {attempt + 1}: task.state is False")
                    # Restore state for retry
                    task.contents = contents_before_phase3
                    task.conversation_history = task.conversation_history[:conv_history_len]
                    continue

                # Save and validate
                meta_info = task.save_trajectory()
                final_answer = meta_info.get("output_text", "")

                is_valid, reason = _validate_trajectory(
                    question=question,
                    ground_truth=ground_truth,
                    prediction=final_answer,
                    reasoning_text="\n".join(all_thinking_text),
                    actual_tools=all_actual_tools,
                )

                if is_valid:
                    logger.info(f"[{task_id}] Valid trajectory found after {total_attempts} attempts")
                    return True, meta_info

                logger.warning(f"[{task_id}] Round {current_round} Attempt {attempt + 1} invalid: {reason}")

                # Always restore conversation_history (don't save failed attempts to trajectory)
                task.conversation_history = task.conversation_history[:conv_history_len]

                # Check if this is the last attempt and we're going to next round
                is_last_attempt = (attempt == _WORKER_ATTEMPTS_PER_ROUND - 1)
                is_going_to_next_round = (current_round < _WORKER_MAX_ITERATIVE_ROUNDS)

                # Only restore contents if NOT last attempt going to next round
                # (Keep Phase 3 in contents so model sees wrong answer in Round 2+)
                if not (is_last_attempt and is_going_to_next_round):
                    task.contents = contents_before_phase3

                # Clean up saved files for retry
                for item in os.listdir(task_save_dir):
                    item_path = os.path.join(task_save_dir, item)
                    if item == "input_image.png":
                        continue
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        os.remove(item_path)

            except Exception as e:
                logger.warning(f"[{task_id}] Round {current_round} Attempt {attempt + 1} failed: {e}")
                # Always restore conversation_history
                task.conversation_history = task.conversation_history[:conv_history_len]
                # Restore contents for retry within same round
                task.contents = contents_before_phase3
                continue

        # All Phase 3 attempts failed for this round, extend to next round
        if current_round < _WORKER_MAX_ITERATIVE_ROUNDS:
            logger.info(f"[{task_id}] Extending to Round {current_round + 1}")

    logger.warning(f"[{task_id}] Max attempts reached without valid trajectory")
    # Clean up
    if os.path.exists(task_save_dir):
        for item in os.listdir(task_save_dir):
            item_path = os.path.join(task_save_dir, item)
            if item == "input_image.png":
                continue
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                os.remove(item_path)
    return False, None


# =============================================================================
# Main function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Iterative sampling for valid trajectory generation.")
    # Original params
    parser.add_argument("--api_key", type=str, default=None, help="API key for model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--dataset_split", type=str, default=None, help="Dataset split name.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--model_name_or_path", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--model_type", type=str, default="OpenAI", choices=["Google", "SGLang", "OpenAI"])
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for API.")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max_concurrent_requests", type=int, default=16)
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--node_resource", type=str, default=None)
    parser.add_argument("--enable_tools", type=str, nargs="+", default=None)

    # New params for iterative sampling
    parser.add_argument("--max_iterative_rounds", type=int, default=5,
                        help="Maximum tool call rounds for iterative sampling")
    parser.add_argument("--attempts_per_round", type=int, default=2,
                        help="Number of Phase 3 attempts before extending rounds")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini",
                        help="Model for trajectory validation")
    parser.add_argument("--judge_api_key", type=str, default=None,
                        help="API key for judge (defaults to OPENAI_API_KEY)")
    parser.add_argument("--judge_api_base", type=str, default=None,
                        help="API base URL for judge model")
    parser.add_argument("--skip_leakage_check", action="store_true",
                        help="Skip answer leakage detection")

    args = parser.parse_args()

    if args.model_type == "Google" and not args.api_key:
        raise ValueError("API key must be provided for Google models.")

    # Use OPENAI_API_KEY if judge_api_key not provided
    judge_api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY")
    if not judge_api_key:
        raise ValueError("Judge API key must be provided via --judge_api_key or OPENAI_API_KEY env")

    # Initialize Ray tool agents
    tool_router = ToolRouter(
        tool_mode="force",
        enable_tools=args.enable_tools,
        node_resource=args.node_resource or "tool_agent"
    )
    enabled_agent_names = tool_router.get_enabled_agents() if tool_router.is_agent_enabled() else []

    if enabled_agent_names:
        logger.info(f"Initialized {len(enabled_agent_names)} shared Ray tool agents")

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    # Load dataset
    if args.dataset_path.startswith("thuml/") or "/" in args.dataset_path and not os.path.exists(args.dataset_path):
        split = args.dataset_split or "ballgame"
        dataset = load_dataset(args.dataset_path, split=split)
    else:
        dataset = load_dataset("parquet", data_files=args.dataset_path)["train"]

    logger.info(f"Dataset size: {len(dataset)}")

    if args.sample_rate < 1.0:
        sample_size = int(len(dataset) * args.sample_rate)
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        logger.info(f"Sampled {sample_size} examples")

    dataset_spec = get_dataset_spec(args.dataset_name)
    n_trajectories = args.n_trajectories
    meta_info_list = []
    pending_items = []

    for item in dataset:
        task_id = str(item[dataset_spec.id_key])
        task_base_dir = os.path.join(output_path, task_id)

        for traj_id in range(n_trajectories):
            traj_save_dir = task_base_dir if n_trajectories == 1 else os.path.join(task_base_dir, f"traj_{traj_id}")
            meta_path = os.path.join(traj_save_dir, "meta_info.jsonl")

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_info_list.append(json.loads(f.readline().strip()))
            else:
                pending_items.append((item, traj_id))

    logger.info(f"Already done: {len(meta_info_list)}, Pending: {len(pending_items)}")

    ctx = mp.get_context("spawn")
    n_workers = max(1, int(args.max_concurrent_requests))
    logger.info(f"Starting {n_workers} worker processes")

    pool = ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(
            args.api_key,
            args.model_name_or_path,
            args.model_type,
            args.api_base,
            args.port,
            output_path,
            enabled_agent_names,
            args.enable_tools,
            judge_api_key,
            args.judge_model,
            args.judge_api_base,
            args.max_iterative_rounds,
            args.attempts_per_round,
            args.skip_leakage_check,
        ),
    )

    interrupted = False
    try:
        inflight = []
        submit_idx = 0
        pbar = tqdm(total=len(pending_items), desc="processing")

        while submit_idx < len(pending_items) or inflight:
            while submit_idx < len(pending_items) and len(inflight) < n_workers:
                item, traj_id = pending_items[submit_idx]
                submit_idx += 1

                task_id = str(item[dataset_spec.id_key])
                task_base_dir = os.path.join(output_path, task_id)
                traj_save_dir = task_base_dir if n_trajectories == 1 else os.path.join(task_base_dir, f"traj_{traj_id}")
                meta_path = os.path.join(traj_save_dir, "meta_info.jsonl")

                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_info_list.append(json.loads(f.readline().strip()))
                    continue

                os.makedirs(task_base_dir, exist_ok=True)
                os.makedirs(traj_save_dir, exist_ok=True)

                image_path = None
                text_only = dataset_spec.image_key is None
                if dataset_spec.image_key:
                    image_path = os.path.join(task_base_dir, "input_image.png")
                    if not os.path.exists(image_path):
                        image = item.get(dataset_spec.image_key)
                        if isinstance(image, list):
                            image = image[0] if image else None
                        if isinstance(image, Image.Image):
                            image.save(image_path)
                        elif isinstance(image, dict) and "bytes" in image:
                            Image.open(BytesIO(image["bytes"])).save(image_path)
                        elif isinstance(image, bytes):
                            Image.open(BytesIO(image)).save(image_path)

                payload = {
                    "id": task_id,
                    "traj_id": traj_id,
                    "task_base_dir": task_base_dir,
                    "task_save_dir": traj_save_dir,
                    "prompt": dataset_spec.build_prompt(item, True, separated=True),
                    "answer": dataset_spec.get_answer(item),
                    "image_path": image_path,
                    "text_only": text_only,
                    "task_kwargs": dataset_spec.build_task_kwargs(item),
                    "answer_format": dataset_spec.answer_format,
                }

                ar = pool.apply_async(_run_one_task_iterative, (payload,))
                inflight.append((f"{task_id}_traj{traj_id}", ar))

            still_inflight = []
            for tid, ar in inflight:
                if ar.ready():
                    ok, meta_info = ar.get()
                    if ok and meta_info is not None:
                        meta_info_list.append(meta_info)
                    pbar.update(1)
                else:
                    still_inflight.append((tid, ar))
            inflight = still_inflight

            if inflight and not any(ar.ready() for _, ar in inflight):
                time.sleep(0.05)

        pbar.close()
    except KeyboardInterrupt:
        interrupted = True
        pbar.close()
        logger.info("\nInterrupted. Terminating workers...")
        pool.terminate()
        pool.join()
    finally:
        if not interrupted:
            pool.close()
            pool.join()

    if meta_info_list:
        save_global_meta_info(output_path, meta_info_list)

    logger.info(f"Completed. Total valid: {len(meta_info_list)}")

    if tool_router.is_agent_enabled():
        tool_router.shutdown_agents()

    if interrupted:
        sys.exit(1)


if __name__ == "__main__":
    main()
