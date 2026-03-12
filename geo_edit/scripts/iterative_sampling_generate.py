"""Iterative sampling script for generating valid trajectories.

Based on separated_reasoning_generate.py, this script adds iterative sampling
to retry and extend tool rounds until a valid trajectory is generated.

Valid trajectory definition:
1. Answer is correct (verified by TrajectoryJudge)
2. No answer leakage in Phase 1/2 (optional check)

Sampling strategy:
- Round 1: Try up to 2 times with tool_rounds=1
- Round 2+: If failed, extend tool_rounds and retry up to 2 times
- Maximum rounds: 5
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional, Tuple

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from geo_edit.prompts.system_prompts import (
    SEPARATED_USER_PROMPT,
    ITERATIVE_EXTENDED_REASONING_PROMPT,
)
from geo_edit.datasets.task_registry import DATASET_SPECS, get_dataset_spec
from geo_edit.tool_definitions import ToolRouter
from geo_edit.evaluation.trajectory_judge import TrajectoryJudge, quick_leakage_check
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
# State management for iterative sampling
# =============================================================================
@dataclass
class IterativeSamplingState:
    """State management for iterative sampling."""
    task_id: str
    current_tool_rounds: int = 1
    attempt_count: int = 0  # Attempts within current round
    total_attempts: int = 0
    previous_answer: Optional[str] = None

    def should_retry(self) -> bool:
        """Check if should retry within current round."""
        return self.attempt_count < _WORKER_ATTEMPTS_PER_ROUND

    def should_extend(self) -> bool:
        """Check if should extend to more tool rounds."""
        return self.current_tool_rounds < _WORKER_MAX_ITERATIVE_ROUNDS

    def increment_attempt(self) -> None:
        """Increment attempt counters."""
        self.attempt_count += 1
        self.total_attempts += 1

    def extend_rounds(self) -> None:
        """Extend to next round with more tool calls."""
        self.current_tool_rounds += 1
        self.attempt_count = 0


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
        extra_prompts={"extended_reasoning": ITERATIVE_EXTENDED_REASONING_PROMPT},
    )

    # Initialize TrajectoryJudge for validation
    _WORKER_JUDGE = TrajectoryJudge(
        api_key=judge_api_key,
        model=judge_model,
        api_base=judge_api_base,
    )

    logger.info(f"Worker initialized with TrajectoryJudge (PID: {os.getpid()})")


# =============================================================================
# Single attempt execution (three-phase reasoning)
# =============================================================================
def _execute_single_attempt(
    task_payload: dict,
    tool_rounds: int,
    extended_reasoning_config: Any = None,
    previous_task_state: Optional[dict] = None,
) -> Tuple[bool, Optional[dict], Optional[str], str, Optional[dict]]:
    """Execute a single attempt of three-phase reasoning.

    Args:
        task_payload: Task data including prompt, answer, image_path, etc.
        tool_rounds: Maximum number of tool call rounds.
        extended_reasoning_config: Config for extended rounds (Round 2+).
        previous_task_state: State from previous attempt to continue from (for Round 2+).

    Returns:
        (success, meta_info, final_answer, thinking_text, task_state_for_next)
    """
    assert _WORKER_CTX is not None, "Worker not initialized"
    ctx = _WORKER_CTX
    agent = ctx.agent
    api_mode = ctx.api_mode
    phase_configs = ctx.phase_configs

    task_id = task_payload["id"]
    task_save_dir = task_payload["task_save_dir"]
    answer = task_payload["answer"]
    image_path = task_payload["image_path"]
    text_prompt = task_payload["prompt"]
    text_only = task_payload.get("text_only", False)
    answer_format = task_payload.get("answer_format")

    formatted_text_prompt = SEPARATED_USER_PROMPT.format(Question=text_prompt)

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

    task = ctx.task_class(
        task_id=task_id,
        task_prompt=formatted_text_prompt,
        task_answer=answer,
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

    all_thinking_text = []  # Collect all thinking for leakage check
    start_step = 1  # Default start from step 1

    # Restore previous state if provided (for Round 2+)
    if previous_task_state is not None:
        task.contents = previous_task_state["contents"]
        task.conversation_history = previous_task_state["conversation_history"]
        task.image_list = previous_task_state["image_list"]
        task.image_path_map = previous_task_state["image_path_map"]
        if hasattr(task, "image_url_map"):
            task.image_url_map = previous_task_state.get("image_url_map", {})
        all_thinking_text = previous_task_state.get("thinking_text", [])
        start_step = previous_task_state.get("completed_steps", 0) + 1
        logger.info(f"[{task_id}] Restored state from previous attempt, continuing from step {start_step}")

    try:
        for i in range(start_step - 1, tool_rounds):
            step = i + 1

            # ===== Phase 1: Generate reasoning =====
            # Use extended config for extra rounds (step > 1 when tool_rounds > 1)
            if extended_reasoning_config is not None and step > 1:
                agent.config.generate_config = extended_reasoning_config
            elif step > 1:
                agent.config.generate_config = phase_configs.multi_round_reasoning
            else:
                agent.config.generate_config = phase_configs.reasoning_only

            reasoning_action, reasoning_extra = agent.act(task.contents)

            # Extract reasoning text
            reasoning_text = extract_response_text(reasoning_action, api_mode)

            all_thinking_text.append(reasoning_text)

            # Check for early answer in multi-round
            if step > 1 and answer_pattern.search(reasoning_text):
                answer_match = re.search(r"<answer>(.*?)</answer>", reasoning_text, re.DOTALL | re.IGNORECASE)
                output_text = answer_match.group(1).strip() if answer_match else reasoning_text
                think_match = re.search(r"<think>(.*?)</think>", reasoning_text, re.DOTALL | re.IGNORECASE)
                thinking = think_match.group(1).strip() if think_match else ""

                task.append_assistant_message(reasoning_text)
                contents_for_save = [task._stringify_observation_item(item) for item in (task.contents if isinstance(task.contents, list) else task.contents.get("input", []))]
                task._record_conversation_history(
                    step=step,
                    contents_for_save=contents_for_save,
                    action_record={"text": output_text, "tool_calls": []},
                    thinking_process=thinking,
                    output_text=output_text,
                    tool_calls=[],
                    extra_info=reasoning_extra,
                )
                meta_info = task.save_trajectory()
                return True, meta_info, output_text, "\n".join(all_thinking_text), None

            # Validate format for step 1
            if step == 1 and answer_pattern.search(reasoning_text):
                raise ValueError("First round should not generate <answer>")

            if not think_pattern.search(reasoning_text):
                raise ValueError(f"Invalid format - missing <think>Tool: ...</think>")

            # ===== Phase 2: Generate tool call =====
            task.append_assistant_message(reasoning_text)
            agent.config.generate_config = phase_configs.tool_call_only
            tool_action, tool_extra = agent.act(task.contents)
            agent.config.generate_config = original_generate_config

            merged_extra = {"reasoning_" + k: v for k, v in reasoning_extra.items()}
            merged_extra.update(tool_extra)

            function_call_part_list = task.parse_action(
                step=step, action=tool_action, extra_info=merged_extra
            )

            if not function_call_part_list:
                break

            task.update_observation_from_action(function_call_part_list)

        # Build task state for potential continuation (before Phase 3)
        task_state_for_next = {
            "contents": task.contents,
            "conversation_history": task.conversation_history,
            "image_list": task.image_list,
            "image_path_map": task.image_path_map,
            "image_url_map": getattr(task, "image_url_map", {}),
            "thinking_text": all_thinking_text,
            "completed_steps": tool_rounds,
        }

        # ===== Phase 3: Generate final answer =====
        if task.state:
            if answer_format:
                task.append_prompt(answer_format)
            agent.config.generate_config = phase_configs.final_answer
            action, extra_info = agent.act(task.contents)
            agent.config.generate_config = original_generate_config
            task.parse_action(step=tool_rounds + 1, action=action, extra_info=extra_info)

        if task.state:
            meta_info = task.save_trajectory()
            final_answer = meta_info.get("output_text", "")
            return True, meta_info, final_answer, "\n".join(all_thinking_text), task_state_for_next

        return False, None, None, "\n".join(all_thinking_text), task_state_for_next

    except Exception as e:
        logger.warning(f"[{task_id}] Attempt failed: {e}")
        return False, None, None, "\n".join(all_thinking_text), None


# =============================================================================
# Trajectory validation
# =============================================================================
def _validate_trajectory(
    question: str,
    ground_truth: str,
    prediction: str,
    thinking_text: str,
) -> Tuple[bool, str]:
    """Validate if the trajectory is valid.

    Args:
        question: The question.
        ground_truth: Ground truth answer.
        prediction: Model's prediction.
        thinking_text: Combined thinking text from Phase 1&2.

    Returns:
        (is_valid, reason)
    """
    # 1. Check answer correctness
    is_correct, _ = _WORKER_JUDGE.judge_correctness(question, ground_truth, prediction)
    if not is_correct:
        return False, "wrong_answer"

    # 2. Check answer leakage (optional)
    if not _WORKER_SKIP_LEAKAGE_CHECK:
        has_leakage, _ = quick_leakage_check(thinking_text)
        if has_leakage:
            return False, "answer_leakage"

    return True, "valid"


# =============================================================================
# Iterative sampling main loop
# =============================================================================
def _run_one_task_iterative(task_payload: dict) -> Tuple[bool, Optional[dict]]:
    """Run iterative sampling for a single task.

    Strategy:
    - Round 1: Try up to attempts_per_round times with tool_rounds=1
    - If failed, extend to Round 2 (tool_rounds=2) and retry
    - Continue until max_iterative_rounds or success
    """
    assert _WORKER_CTX is not None, "Worker not initialized"
    task_id = task_payload["id"]
    task_save_dir = task_payload["task_save_dir"]
    question = task_payload["prompt"]
    ground_truth = task_payload["answer"]

    # Check if already completed
    meta_path = os.path.join(task_save_dir, "meta_info.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_info = json.loads(f.readline().strip())
        return True, meta_info

    state = IterativeSamplingState(task_id=task_id)
    # State from the last attempt of previous round (Round N-1's last attempt)
    # All attempts in Round N should start from this state
    previous_round_state = None
    # State from current round's last attempt (for passing to next round)
    current_round_last_state = None

    while state.current_tool_rounds <= _WORKER_MAX_ITERATIVE_ROUNDS:
        # Determine if we should continue from previous round's state
        # Round 2+ should always start from previous round's last state
        should_continue_from_previous = (
            state.current_tool_rounds > 1 and
            previous_round_state is not None
        )

        # Clean up previous attempt's files
        if state.total_attempts > 0 and os.path.exists(task_save_dir):
            shutil.rmtree(task_save_dir, ignore_errors=True)
        os.makedirs(task_save_dir, exist_ok=True)

        # Select reasoning config
        if state.current_tool_rounds > 1:
            extended_config = _WORKER_CTX.phase_configs.extended_reasoning
        else:
            extended_config = None

        # Pass previous round's state for Round 2+ (all attempts in the same round use the same base state)
        previous_state = previous_round_state if should_continue_from_previous else None

        logger.info(f"[{task_id}] Attempt {state.total_attempts + 1}: "
                   f"tool_rounds={state.current_tool_rounds}, "
                   f"attempt={state.attempt_count + 1}/{_WORKER_ATTEMPTS_PER_ROUND}, "
                   f"continue_from_previous={should_continue_from_previous}")

        # Execute single attempt
        success, meta_info, final_answer, thinking_text, task_state = _execute_single_attempt(
            task_payload,
            tool_rounds=state.current_tool_rounds,
            extended_reasoning_config=extended_config,
            previous_task_state=previous_state,
        )

        if not success or final_answer is None:
            logger.warning(f"[{task_id}] Attempt {state.total_attempts + 1} execution failed")
            state.increment_attempt()
            if not state.should_retry():
                if state.should_extend():
                    # Moving to next round: use current round's last successful state (if any)
                    previous_round_state = current_round_last_state
                    current_round_last_state = None
                    state.extend_rounds()
                    logger.info(f"[{task_id}] Extending to {state.current_tool_rounds} tool rounds")
                else:
                    break
            continue

        # Validate trajectory
        is_valid, reason = _validate_trajectory(
            question=question,
            ground_truth=ground_truth,
            prediction=final_answer,
            thinking_text=thinking_text,
        )

        if is_valid:
            logger.info(f"[{task_id}] Valid trajectory found after {state.total_attempts + 1} attempts")
            return True, meta_info

        logger.info(f"[{task_id}] Attempt {state.total_attempts + 1} invalid: {reason}")
        state.previous_answer = final_answer
        state.increment_attempt()

        # Save task state from current attempt (for potential use by next round)
        if task_state is not None:
            current_round_last_state = task_state

        if not state.should_retry():
            if state.should_extend():
                # Moving to next round: save current round's last state as previous_round_state
                # This will be the base state for ALL attempts in the next round
                previous_round_state = current_round_last_state
                current_round_last_state = None
                state.extend_rounds()
                logger.info(f"[{task_id}] Extending to {state.current_tool_rounds} tool rounds")
            else:
                break

    logger.warning(f"[{task_id}] Max attempts reached without valid trajectory")
    shutil.rmtree(task_save_dir, ignore_errors=True)
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
                        help="Number of attempts before extending rounds")
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
