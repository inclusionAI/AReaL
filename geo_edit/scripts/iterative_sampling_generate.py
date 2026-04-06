"""Iterative sampling script for generating valid trajectories.

Based on separated_reasoning_generate.py, this script adds iterative sampling
to retry and extend tool rounds until a valid trajectory is generated.

Valid trajectory definition:
1. Answer is correct (verified by TrajectoryJudge)
2. No answer leakage in Phase 1/2 (optional check)

Sampling strategy:
- Each round does one of: Phase 1 (reasoning) + Phase 2 (tool call), or answer
- Round 1 forces a tool call; subsequent rounds let the model decide
- If model outputs <answer>, validate with judge; if wrong, inject reflection and continue
- Maximum rounds controlled by --max_iterative_rounds
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import re
import sys
import time
from io import BytesIO
from typing import Optional, Tuple
import PIL
from datasets import load_dataset
import PIL
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
PIL.IMAGE_MAX_IMAGE_PIXELS = None  # Disable PIL DecompressionBombError for large images
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
    skip_leakage_check: bool,
):
    """Initialize worker for iterative sampling."""
    global _WORKER_CTX, _WORKER_JUDGE
    global _WORKER_MAX_ITERATIVE_ROUNDS, _WORKER_SKIP_LEAKAGE_CHECK

    # Set iterative sampling params
    _WORKER_MAX_ITERATIVE_ROUNDS = max_iterative_rounds
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
    prediction: str = "",
    reasoning_text: str = "",
    actual_tools: set = None,
    check_correctness: bool = True,
    task_category: str = "",
    image_path: str = None,
    judge_prompt: Optional[str] = None,
) -> Tuple[bool, str]:
    """Validate trajectory using judge.

    Args:
        check_correctness: If True, checks answer correctness (requires prediction).
            If False, only checks leakage + tool consistency (for tool call rounds
            where no answer exists yet).
        task_category: Task category string. When "maze", uses algorithmic verification.
        image_path: Path to task image, required for maze verification.
        judge_prompt: Optional additional prompt for LLM judge (task-specific hints).
    """
    # Maze tasks: use algorithmic wall-collision verification instead of LLM judge
    if task_category == "maze" and image_path and check_correctness:
        from geo_edit.evaluation.maze_verifier import maze_judge

        logger.warning(f"Using maze algorithmic verifier instead of LLM judge (category={task_category})")
        return maze_judge(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
            image_path=image_path,
        )

    assert _WORKER_JUDGE is not None, "Judge not initialized"

    if check_correctness:
        is_correct, reason = _WORKER_JUDGE.judge_correctness(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
            additional_prompt=judge_prompt,
        )
        if is_correct:
            return True, "valid"
        return (
            False,
            f"wrong_answer (gt={ground_truth}, pred={prediction}) reason={reason}",
        )

    is_valid, reason = _WORKER_JUDGE.validate_trajectory(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction or "[not yet answered]",
        reasoning_text=reasoning_text,
        actual_tools=actual_tools or set(),
    )

    if not is_valid and reason.startswith("wrong_answer"):
        # No answer yet — ignore correctness failure, only leakage/tool_match matter
        return True, "valid"

    return is_valid, reason


# =============================================================================
# Iterative sampling main loop
# =============================================================================
def _run_one_task_iterative(task_payload: dict) -> Tuple[bool, Optional[dict]]:
    """Run iterative sampling for a single task.

    Strategy:
    - Each round either calls one tool OR produces an answer
    - Round 1 forces a tool call; subsequent rounds use CHAIN_TOOL_SELECTION_PROMPT
    - If model outputs <answer>, validate with judge; if wrong, continue with reflection
    - Maximum rounds: _WORKER_MAX_ITERATIVE_ROUNDS
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
    tool_guidance = task_payload.get("tool_guidance")
    judge_prompt = task_payload.get("judge_prompt")
    task_category = (
        task_payload.get("task_kwargs", {})
        .get("meta_info_extra", {})
        .get("category", "")
    )

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
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    think_pattern = re.compile(
        r"<think>.*?Tool:\s*(\w+).*?</think>", re.DOTALL | re.IGNORECASE
    )

    all_thinking_text = []
    all_actual_tools = set()
    judge_failed = False  # True after judge rejects an answer
    has_answered = False  # True if model ever attempted an answer
    max_phase_retries = 3  # Max retries for Phase 1/2 exceptions within a round

    current_round = 1
    while current_round <= _WORKER_MAX_ITERATIVE_ROUNDS:
        logger.info(f"[{task_id}] Starting Round {current_round}")

        # Save state for exception recovery
        task_state_before_round = task.save_state()
        agent_state_before_round = agent.save_state()
        thinking_text_before_round = list(all_thinking_text)
        actual_tools_before_round = set(all_actual_tools)

        retry_count = 0
        round_success = False

        while retry_count < max_phase_retries:
            try:
                # ===== Phase 1: Reasoning (select tool or answer) =====
                if current_round == 1 or judge_failed:
                    # First round or after wrong answer: force tool selection
                    agent.config.generate_config = phase_configs.reasoning_only
                else:
                    # Subsequent rounds: chain reasoning (select tool or answer)
                    agent.config.generate_config = phase_configs.chain_reasoning

                # Inject temporary prompts
                contents_before_prompt = None
                if judge_failed and current_round > 1:
                    contents_before_prompt = copy.deepcopy(task.contents)
                    task.append_system_prompt(
                        ITERATIVE_EXTENDED_REASONING_PROMPT.format(
                            used_tools=", ".join(all_actual_tools)
                            if all_actual_tools
                            else "None"
                        )
                    )
                elif tool_guidance:
                    # Inject per-dataset tool guidance as temporary prompt
                    contents_before_prompt = copy.deepcopy(task.contents)
                    task.append_system_prompt(tool_guidance)

                reasoning_action, reasoning_extra = agent.act(task.contents)
                agent.config.generate_config = original_generate_config
                logger.warning(reasoning_action)

                # Restore contents (remove temporary prompt)
                if contents_before_prompt is not None:
                    task.contents = contents_before_prompt

                reasoning_text = extract_response_text(reasoning_action, api_mode)
                all_thinking_text.append(reasoning_text)

                # Check if model decided to answer
                answer_match = answer_pattern.search(reasoning_text)
                if answer_match and current_round > 1:
                    if judge_failed:
                        # After wrong answer, must call tool first — reject answer attempt
                        raise ValueError(
                            "Model tried to answer immediately after wrong answer; forcing tool call"
                        )
                    # ===== Answer path =====
                    answer_text = answer_match.group(1).strip()
                    logger.info(
                        f"[{task_id}] Round {current_round}: Model produced answer: {answer_text}"
                    )

                    # Validate with judge BEFORE committing to contents/history
                    # Model self-terminated with <answer>: only check correctness
                    is_valid, reason = _validate_trajectory(
                        question=question,
                        ground_truth=ground_truth,
                        prediction=answer_text,
                        check_correctness=True,
                        task_category=task_category,
                        image_path=image_path,
                        judge_prompt=judge_prompt,
                    )

                    if is_valid:
                        # Commit answer to contents and history, then save
                        task.append_assistant_message(reasoning_text)
                        think_match = re.search(
                            r"<think>(.*?)</think>", reasoning_text, re.DOTALL
                        )
                        thinking_process = (
                            think_match.group(1).strip() if think_match else ""
                        )
                        task._record_conversation_history(
                            step=current_round,
                            contents_for_save=[],
                            action_record={"text": answer_text, "tool_calls": []},
                            thinking_process=thinking_process,
                            output_text=answer_text,
                            tool_calls=[],
                            extra_info=reasoning_extra,
                        )
                        meta_info = task.save_trajectory()
                        logger.info(
                            f"[{task_id}] Valid trajectory at Round {current_round}"
                        )
                        return True, meta_info

                    # Judge rejected: do NOT append to contents/history
                    # Wrong answer does NOT consume a round — immediately force tool call
                    logger.warning(
                        f"[{task_id}] Round {current_round} answer rejected: {reason}"
                    )
                    judge_failed = True
                    has_answered = True
                    # Reset state to before this attempt, then retry within same round
                    task.restore_state(task_state_before_round)
                    agent.restore_state(agent_state_before_round)
                    all_thinking_text = thinking_text_before_round
                    all_actual_tools = actual_tools_before_round
                    retry_count += 1
                    continue  # Retry — next iteration will use reasoning_only (judge_failed=True)

                # ===== Tool call path =====
                # Validate Phase 1 format (must have Tool: in think tags)
                if not think_pattern.search(reasoning_text):
                    raise ValueError(
                        f"Invalid Phase 1 format - missing <think>Tool: ...</think>"
                    )

                judge_failed = False  # Reset judge flag on tool call rounds
                task.append_assistant_message(reasoning_text)

                # ===== Phase 2: Execute tool call =====
                agent.config.generate_config = phase_configs.tool_call_only
                tool_action, tool_extra = agent.act(task.contents)
                agent.config.generate_config = original_generate_config

                merged_extra = {"reasoning_" + k: v for k, v in reasoning_extra.items()}
                merged_extra.update(tool_extra)

                function_call_part_list = task.parse_action(
                    step=current_round, action=tool_action, extra_info=merged_extra
                )

                if not function_call_part_list:
                    logger.warning(
                        f"[{task_id}] Round {current_round}: No tool calls generated"
                    )
                    round_success = True
                    break

                # Collect tool names for this round
                round_tool_names = {tc.name.lower() for tc in function_call_part_list}

                # Validate leakage + tool consistency before executing tools
                is_valid, reason = _validate_trajectory(
                    question=question,
                    ground_truth=ground_truth,
                    reasoning_text=reasoning_text,
                    actual_tools=round_tool_names,
                    check_correctness=False,
                )
                if not is_valid:
                    logger.warning(
                        f"[{task_id}] Round {current_round} tool-call rejected: {reason}"
                    )
                    raise ValueError(f"Tool-call validation failed: {reason}")

                all_actual_tools.update(round_tool_names)

                task.update_observation_from_action(function_call_part_list)
                round_success = True
                break  # Phase 1+2 completed successfully

            except Exception as e:
                logger.warning(
                    f"[{task_id}] Round {current_round} failed (retry {retry_count}): {e}"
                )
                task.restore_state(task_state_before_round)
                agent.restore_state(agent_state_before_round)
                all_thinking_text = thinking_text_before_round
                all_actual_tools = actual_tools_before_round
                retry_count += 1
                continue

        if not round_success:
            logger.warning(f"[{task_id}] Round {current_round}: exhausted retries")
            break

        current_round += 1

    # Reached max rounds without valid answer — force final answer if never answered
    if not has_answered:
        logger.info(f"[{task_id}] Max rounds reached, forcing final answer")
        try:
            contents_before = copy.deepcopy(task.contents)
            if answer_format:
                task.append_prompt(answer_format)

            agent.config.generate_config = phase_configs.final_answer
            action, extra_info = agent.act(task.contents)
            agent.config.generate_config = original_generate_config

            task.contents = contents_before
            task.parse_action(step=current_round, action=action, extra_info=extra_info)
        except Exception as e:
            logger.warning(f"[{task_id}] Forced final answer failed: {e}")

    logger.warning(f"[{task_id}] Max rounds reached without valid trajectory")
    meta_info = task.save_trajectory()
    return False, meta_info


# =============================================================================
# Main function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Iterative sampling for valid trajectory generation."
    )
    # Original params
    parser.add_argument("--api_key", type=str, default=None, help="API key for model.")
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--dataset_split", type=str, default=None, help="Dataset split name."
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, choices=sorted(DATASET_SPECS.keys())
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument("--model_name_or_path", type=str, default="gpt-5-2025-08-07")
    parser.add_argument(
        "--model_type",
        type=str,
        default="OpenAI",
        choices=["Google", "SGLang", "OpenAI"],
    )
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for API.")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max_concurrent_requests", type=int, default=16)
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--node_resource", type=str, default=None)
    parser.add_argument("--enable_tools", type=str, nargs="+", default=None)

    # New params for iterative sampling
    parser.add_argument(
        "--max_iterative_rounds",
        type=int,
        default=5,
        help="Maximum tool call rounds for iterative sampling",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="Model for trajectory validation",
    )
    parser.add_argument(
        "--judge_api_key",
        type=str,
        default=None,
        help="API key for judge (defaults to OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--judge_api_base", type=str, default=None, help="API base URL for judge model"
    )
    parser.add_argument(
        "--skip_leakage_check",
        action="store_true",
        help="Skip answer leakage detection",
    )

    args = parser.parse_args()

    if args.model_type == "Google" and not args.api_key:
        raise ValueError("API key must be provided for Google models.")

    # Use OPENAI_API_KEY if judge_api_key not provided
    judge_api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY")
    if not judge_api_key:
        raise ValueError(
            "Judge API key must be provided via --judge_api_key or OPENAI_API_KEY env"
        )

    # Initialize Ray tool agents
    tool_router = ToolRouter(
        tool_mode="force",
        enable_tools=args.enable_tools,
        node_resource=args.node_resource or "tool_agent",
    )
    if tool_router.is_agent_enabled():
        from geo_edit.environment.tool_agents import get_manager
        manager = get_manager()
        enabled_agent_names = manager.get_all_actor_names()
    else:
        enabled_agent_names = []

    if enabled_agent_names:
        logger.info(f"Initialized {len(enabled_agent_names)} shared Ray tool agents")

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    # Load dataset
    if (
        args.dataset_path.startswith("thuml/")
        or "/" in args.dataset_path
        and not os.path.exists(args.dataset_path)
    ):
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
    dataset, _pre_saved_images = dataset_spec.prepare_images(dataset, output_path)
    n_trajectories = args.n_trajectories
    meta_info_list = []
    pending_items = []

    for item in dataset:
        task_id = str(item[dataset_spec.id_key])
        task_base_dir = os.path.join(output_path, task_id)

        for traj_id in range(n_trajectories):
            traj_save_dir = (
                task_base_dir
                if n_trajectories == 1
                else os.path.join(task_base_dir, f"traj_{traj_id}")
            )
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
                traj_save_dir = (
                    task_base_dir
                    if n_trajectories == 1
                    else os.path.join(task_base_dir, f"traj_{traj_id}")
                )
                meta_path = os.path.join(traj_save_dir, "meta_info.jsonl")

                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_info_list.append(json.loads(f.readline().strip()))
                    continue

                os.makedirs(task_base_dir, exist_ok=True)
                os.makedirs(traj_save_dir, exist_ok=True)

                image_path = None
                text_only = dataset_spec.image_key is None
                if _pre_saved_images:
                    image_path = _pre_saved_images.get(task_id)
                elif dataset_spec.image_key:
                    raw_image = item.get(dataset_spec.image_key)
                    images = (
                        raw_image
                        if isinstance(raw_image, list)
                        else [raw_image]
                        if raw_image is not None
                        else []
                    )

                    def _save_one(img, path):
                        if isinstance(img, Image.Image):
                            img.save(path)
                        elif isinstance(img, dict) and "bytes" in img:
                            Image.open(BytesIO(img["bytes"])).save(path)
                        elif isinstance(img, bytes):
                            Image.open(BytesIO(img)).save(path)

                    if len(images) == 1:
                        image_path = os.path.join(task_base_dir, "input_image.png")
                        if not os.path.exists(image_path):
                            _save_one(images[0], image_path)
                    elif len(images) > 1:
                        image_path = []
                        for img_idx, img in enumerate(images):
                            p = os.path.join(
                                task_base_dir, f"input_image_{img_idx}.png"
                            )
                            if not os.path.exists(p):
                                _save_one(img, p)
                            image_path.append(p)

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
                    "tool_guidance": dataset_spec.get_tool_guidance(item),
                    "judge_prompt": dataset_spec.get_judge_prompt(item),
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
