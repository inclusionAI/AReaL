"""Separated reasoning script for three-phase tool call generation.

Supports Gemini and GPT models (Google API and matrixllm chat_completions).
Only supports force tool mode.

Phase 1: Generate reasoning (can see tools, but cannot execute)
Phase 2: Generate tool call based on reasoning (no additional reasoning)
Phase 3: Generate final answer (no tool execution)
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import time
from io import BytesIO

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from geo_edit.agents.api_agent import AgentConfig, APIBasedAgent
from geo_edit.config import (
    build_google_agent_configs,
    build_api_agent_configs,
    derive_google_config,
    derive_api_config,
)
from geo_edit.constants import MAX_TOOL_CALLS
from geo_edit.prompts import get_system_prompt
from geo_edit.prompts.system_prompts import (
    SIMPLIFIED_TOOL_SELECTION_PROMPT,
    SEPARATED_TOOL_CALL_ONLY_PROMPT,
    SEPARATED_FINAL_ANSWER_PROMPT,
    SEPARATED_USER_PROMPT,
    MULTI_ROUND_TOOL_SELECTION_PROMPT,
)

from geo_edit.datasets.task_registry import DATASET_SPECS, get_dataset_spec
from geo_edit.tool_definitions import ToolRouter
from geo_edit.environment.task.google_vision_qa_task import GoogleVisionQATask
from geo_edit.environment.task.openai_compatible_vision_qa_task import OpenAICompatibleVisionQATask
from geo_edit.utils.logger import setup_logger
from geo_edit.utils.stats import save_global_meta_info

logger = setup_logger(__name__)

# Worker globals (one per process)
_WORKER_AGENT: "APIBasedAgent | None" = None
_WORKER_AGENT_CONFIGS = None
_WORKER_OUTPUT_PATH: "str | None" = None
_WORKER_MAX_TOOL_CALLS: "int | None" = None
_WORKER_TASK_CLASS = None
_WORKER_API_MODE: "str | None" = None
_WORKER_TOOL_ROUTER: "ToolRouter | None" = None
_WORKER_REASONING_ONLY_CONFIG = None
_WORKER_MULTI_ROUND_REASONING_CONFIG = None
_WORKER_TOOL_CALL_ONLY_CONFIG = None
_WORKER_FINAL_ANSWER_CONFIG = None


def _init_worker(
    api_key: str,
    model_name_or_path: str,
    model_type: str,
    api_base: str,
    port: int,
    output_path: str,
    max_tool_calls: int,
    enabled_agent_names: list,
    enable_tools: list,
):
    """Initialize worker for Gemini/GPT models (Google API or matrixllm).

    Agent instance is created once per worker and reused for all tasks.
    Ray tool agents are shared across all workers (initialized in main process).
    Workers connect to existing Ray actors by name.
    """
    global _WORKER_AGENT, _WORKER_AGENT_CONFIGS, _WORKER_OUTPUT_PATH, _WORKER_MAX_TOOL_CALLS
    global _WORKER_TASK_CLASS, _WORKER_API_MODE
    global _WORKER_TOOL_ROUTER, _WORKER_REASONING_ONLY_CONFIG, _WORKER_MULTI_ROUND_REASONING_CONFIG
    global _WORKER_TOOL_CALL_ONLY_CONFIG, _WORKER_FINAL_ANSWER_CONFIG

    # Create ToolRouter WITHOUT initializing Ray actors
    # Pass enable_tools to override config.yaml (same as main process)
    _WORKER_TOOL_ROUTER = ToolRouter(tool_mode="force", enable_tools=enable_tools, skip_agent_init=True)

    # Connect to existing Ray actors created in main process
    if enabled_agent_names:
        from geo_edit.environment.tool_agents import get_manager
        manager = get_manager()
        # Get agent configs to pass to connect_to_existing_agents
        agent_configs = _WORKER_TOOL_ROUTER.get_enabled_agent_configs()
        manager.connect_to_existing_agents(enabled_agent_names, configs=agent_configs)
        logger.info(f"Worker (PID: {os.getpid()}) connected to {len(enabled_agent_names)} Ray actors: {enabled_agent_names}")

    if model_type == "Google" and not api_key:
        raise ValueError("API key must be provided for Google models.")

    system_prompt = get_system_prompt(model_type, "force")

    # Determine api_mode: Google API or matrixllm (chat_completions)
    if model_type == "Google":
        _WORKER_API_MODE = "google"
        agent_configs = build_google_agent_configs(
            _WORKER_TOOL_ROUTER,
            thinking_level="low",
            include_thoughts=True,
            temperature=1.0,
            system_prompt=system_prompt,
        )
        _WORKER_TASK_CLASS = GoogleVisionQATask
        base = agent_configs.generate_config
        _WORKER_REASONING_ONLY_CONFIG = derive_google_config(
            base, system_prompt=SIMPLIFIED_TOOL_SELECTION_PROMPT, tool_mode="NONE"
        )
        _WORKER_MULTI_ROUND_REASONING_CONFIG = derive_google_config(
            base, system_prompt=MULTI_ROUND_TOOL_SELECTION_PROMPT, tool_mode="NONE"
        )
        _WORKER_TOOL_CALL_ONLY_CONFIG = derive_google_config(
            base, system_prompt=SEPARATED_TOOL_CALL_ONLY_PROMPT
        )
        _WORKER_FINAL_ANSWER_CONFIG = derive_google_config(
            base, system_prompt=SEPARATED_FINAL_ANSWER_PROMPT, tool_mode="NONE"
        )
    else:
        # SGLang/OpenAI via matrixllm uses chat_completions
        _WORKER_API_MODE = "chat_completions"
        agent_configs = build_api_agent_configs(
            _WORKER_TOOL_ROUTER,
            api_mode="chat_completions",
            temperature=1.0,
            reasoning_level="low",
            system_prompt=system_prompt,
        )
        _WORKER_TASK_CLASS = OpenAICompatibleVisionQATask
        base = agent_configs.generate_config
        _WORKER_REASONING_ONLY_CONFIG = derive_api_config(
            base, api_mode="chat_completions", system_prompt=SIMPLIFIED_TOOL_SELECTION_PROMPT, tool_choice="none"
        )
        _WORKER_MULTI_ROUND_REASONING_CONFIG = derive_api_config(
            base, api_mode="chat_completions", system_prompt=MULTI_ROUND_TOOL_SELECTION_PROMPT, tool_choice="none"
        )
        _WORKER_TOOL_CALL_ONLY_CONFIG = derive_api_config(
            base, api_mode="chat_completions", system_prompt=SEPARATED_TOOL_CALL_ONLY_PROMPT
        )
        _WORKER_FINAL_ANSWER_CONFIG = derive_api_config(
            base, api_mode="chat_completions", system_prompt=SEPARATED_FINAL_ANSWER_PROMPT, tool_choice="none"
        )

    config = AgentConfig(
        model_type=model_type,
        model_name=model_name_or_path,
        api_key=api_key,
        api_base=api_base,
        port=port,
        generate_config=agent_configs.generate_config,
        n_retry=3,
        api_mode=_WORKER_API_MODE,
    )
    _WORKER_AGENT_CONFIGS = agent_configs
    _WORKER_AGENT = APIBasedAgent(config)
    _WORKER_OUTPUT_PATH = output_path
    _WORKER_MAX_TOOL_CALLS = max_tool_calls

    logger.info(f"Worker initialized with {model_type} agent (PID: {os.getpid()})")


def _run_one_task(task_payload: dict):
    """
    Worker: run separated three-phase generation.
    Phase 1: Generate reasoning (no tool execution)
    Phase 2: Generate tool call (no additional reasoning)
    Phase 3: Generate final answer (no tool execution)
    """
    assert _WORKER_AGENT is not None
    assert _WORKER_AGENT_CONFIGS is not None
    assert _WORKER_MAX_TOOL_CALLS is not None
    assert _WORKER_TOOL_ROUTER is not None
    assert _WORKER_REASONING_ONLY_CONFIG is not None
    assert _WORKER_TOOL_CALL_ONLY_CONFIG is not None
    assert _WORKER_FINAL_ANSWER_CONFIG is not None

    task_id = task_payload["id"]
    task_save_dir = task_payload["task_save_dir"]
    answer = task_payload["answer"]
    image_path = task_payload["image_path"]
    text_prompt = task_payload["prompt"]
    text_only = task_payload.get("text_only", False)
    answer_format = task_payload.get("answer_format")  # Answer format instruction for Phase 3

    formatted_text_prompt = SEPARATED_USER_PROMPT.format(Question=text_prompt)

    meta_path = os.path.join(task_save_dir, "meta_info.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_info = json.loads(f.readline().strip())
        return True, meta_info

    task_kwargs = {"model_type": "google" if _WORKER_API_MODE == "google" else "sglang"}
    if _WORKER_API_MODE != "google":
        task_kwargs["api_mode"] = _WORKER_API_MODE
    extra_kwargs = task_payload.get("task_kwargs")
    if isinstance(extra_kwargs, dict):
        task_kwargs.update(extra_kwargs)
    if text_only:
        logger.info(f"[{task_id}] running text-only task.")
        task_kwargs["text_only"] = True

    tool_functions = _WORKER_TOOL_ROUTER.get_available_tools()
    tool_return_types = _WORKER_TOOL_ROUTER.get_tool_return_types()

    task = _WORKER_TASK_CLASS(
        task_id=task_id,
        task_prompt=formatted_text_prompt,
        task_answer=answer,
        task_image_path=image_path,
        tool_functions=tool_functions,
        tool_return_types=tool_return_types,
        save_dir=task_save_dir,
        **task_kwargs,
    )

    # Reset agent state for new task (reuses same agent instance)
    _WORKER_AGENT.reset()
    original_generate_config = _WORKER_AGENT.config.generate_config
    answer_pattern = re.compile(r"<answer>", re.IGNORECASE)
    think_pattern = re.compile(r"<think>.*?Tool:\s*(\w+).*?</think>", re.DOTALL | re.IGNORECASE)

    try:
        for i in range(_WORKER_MAX_TOOL_CALLS):
            step = i + 1

            # ===== Phase 1: Generate reasoning (no tool call, no answer) =====
            logger.info(f"[{task_id}] Step {step} Phase 1: Generating reasoning...")
            # Use multi-round prompt after first round
            if step > 1:
                _WORKER_AGENT.config.generate_config = _WORKER_MULTI_ROUND_REASONING_CONFIG
            else:
                _WORKER_AGENT.config.generate_config = _WORKER_REASONING_ONLY_CONFIG
            reasoning_action, reasoning_extra = _WORKER_AGENT.act(task.contents)
            logger.warning(reasoning_action)
            # Extract reasoning text from action
            if _WORKER_API_MODE == "google":
                text_parts = [p.text for p in reasoning_action.parts if p.text and not p.thought]
                reasoning_text = "\n".join(text_parts)
            else:
                reasoning_text = reasoning_action.choices[0].message.content or ""

            logger.info(f"[{task_id}] Step {step} Phase 1: Reasoning generated ({len(reasoning_text)} chars)")

            # For step > 1: check if model wants to provide final answer (contains <answer>)
            if step > 1 and answer_pattern.search(reasoning_text):
                logger.info(f"[{task_id}] Step {step} Phase 1: Model chose to provide final answer directly")
                # Parse answer from reasoning_text
                answer_match = re.search(r"<answer>(.*?)</answer>", reasoning_text, re.DOTALL | re.IGNORECASE)
                output_text = answer_match.group(1).strip() if answer_match else reasoning_text
                think_match = re.search(r"<think>(.*?)</think>", reasoning_text, re.DOTALL | re.IGNORECASE)
                thinking = think_match.group(1).strip() if think_match else ""
                # Stringify contents for saving
                contents_for_save = [task._stringify_observation_item(item) for item in (task.contents if isinstance(task.contents, list) else task.contents.get("input", []))]
                # Record final step
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
                logger.info(f"[{task_id}] Task completed (early answer) - Total steps: {meta_info.get('total_steps', 'N/A')}")
                return True, meta_info

            # Check for <answer> in step 1 - error if found
            if step == 1 and answer_pattern.search(reasoning_text):
                logger.warning(reasoning_text)
                raise ValueError("First round reasoning phase should not generate <answer>. Model violated the protocol.")

            # Validate format: must contain <think>Tool: [name]</think>
            if not think_pattern.search(reasoning_text):
                logger.warning(f"[{task_id}] Step {step} Phase 1: Invalid format - missing <think>Tool: ...</think>")
                logger.warning(f"Raw output: {reasoning_text}")
                raise ValueError(f"Reasoning phase must output <think>Tool: [name]\\nReason: ...</think> format. Got: {reasoning_text[:200]}")

            # ===== Phase 2: Generate tool call =====
            logger.info(f"[{task_id}] Step {step} Phase 2: Generating tool call...")
            task.append_assistant_message(reasoning_text)

            _WORKER_AGENT.config.generate_config = _WORKER_TOOL_CALL_ONLY_CONFIG
            tool_action, tool_extra = _WORKER_AGENT.act(task.contents)

            _WORKER_AGENT.config.generate_config = original_generate_config

            # Merge reasoning_extra into tool_extra for recording
            merged_extra = {
                "reasoning_" + k: v for k, v in reasoning_extra.items()
            }
            merged_extra.update(tool_extra)

            # Task parses tool call or answer
            function_call_part_list = task.parse_action(
                step=step,
                action=tool_action,
                extra_info=merged_extra
            )

            if not function_call_part_list:
                logger.error(f"[{task_id}] Step {step} Phase 2: No tool calls, final answer detected")
                break

            # Log tool calls
            tool_names = [tc.name for tc in function_call_part_list]
            logger.info(f"[{task_id}] Step {step} Phase 2: Tool calls generated: {tool_names}")

            task.update_observation_from_action(function_call_part_list)

        # ===== Phase 3: Generate final answer =====
        if task.state:
            logger.info(f"[{task_id}] Phase 3: Generating final answer...")
            # Inject answer format instruction before generating final answer
            if answer_format:
                task.append_prompt(answer_format)
            _WORKER_AGENT.config.generate_config = _WORKER_FINAL_ANSWER_CONFIG
            action, extra_info = _WORKER_AGENT.act(task.contents)
            _WORKER_AGENT.config.generate_config = original_generate_config
            task.parse_action(step=_WORKER_MAX_TOOL_CALLS + 1, action=action, extra_info=extra_info)

        if task.state:
            meta_info = task.save_trajectory()
            logger.info(f"[{task_id}] Task completed successfully - Total steps: {meta_info.get('total_steps', 'N/A')}, "
                       f"Total tokens: {meta_info.get('tokens_used_total', 'N/A')}")
            return True, meta_info

        logger.warning(f"[{task_id}] Task failed - no valid trajectory")
        shutil.rmtree(task_save_dir, ignore_errors=True)
        return False, None

    except Exception as e:
        logger.error(f"[{task_id}] Worker failed with exception: {e}", exc_info=True)
        shutil.rmtree(task_save_dir, ignore_errors=True)
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Separated reasoning generation for Gemini/GPT models.")
    parser.add_argument("--api_key", type=str, default=None, help="API key for Google API.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--dataset_split", type=str, default=None, help="Dataset split name (for HuggingFace datasets with named splits).")
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_SPECS.keys()), help="Dataset adapter name.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output.")
    parser.add_argument("--model_name_or_path", type=str, default="gemini-2.5-pro-preview-05-06", help="Model name.")
    parser.add_argument("--model_type", type=str, default="Google", choices=["Google", "SGLang", "OpenAI"], help="Model provider.")
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for matrixllm server.")
    parser.add_argument("--port", type=int, default=None, help="Port for server.")
    parser.add_argument("--max_concurrent_requests", type=int, default=16, help="Number of worker processes.")
    parser.add_argument("--sample_rate", type=float, default=0.1, help="Sampling rate for the dataset.")
    parser.add_argument("--n_trajectories", type=int, default=1, help="Number of trajectories per task.")
    parser.add_argument("--node_resource", type=str, default=None, help="Ray custom resource name (default: 'tool_agent').")
    parser.add_argument("--enable_tools", type=str, nargs="+", default=None,
                        help="Tool names or categories to enable (overrides config.yaml). "
                             "Categories: general, math, table, chart, map, document, ocr, segment. "
                             "Examples: --enable_tools math chart, --enable_tools text_ocr formula_ocr")
    parser.add_argument("--max_tool_calls", type=int, default=None,
                        help="Max tool calls per task (default: constants.MAX_TOOL_CALLS)")
    args = parser.parse_args()

    if args.model_type == "Google" and not args.api_key:
        raise ValueError("API key must be provided for Google models.")

    # Initialize Ray tool agents in main process (shared by all workers)
    # Ray will auto-connect to local cluster or start one if needed
    tool_router = ToolRouter(
        tool_mode="force",
        enable_tools=args.enable_tools,
        node_resource=args.node_resource or "tool_agent"
    )
    enabled_agent_names = tool_router.get_enabled_agents() if tool_router.is_agent_enabled() else []

    if args.enable_tools:
        logger.info(f"Tool override from command line: {args.enable_tools}")

    if enabled_agent_names:
        logger.info(f"Initialized {len(enabled_agent_names)} shared Ray tool agents in main process: {enabled_agent_names}")
    else:
        logger.info("No tool agents enabled (check config.yaml)")

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    # Load dataset - support both HuggingFace datasets and local parquet files
    if args.dataset_path.startswith("thuml/") or "/" in args.dataset_path and not os.path.exists(args.dataset_path):
        # HuggingFace dataset
        split = args.dataset_split or "ballgame"  # Default to first split for VisWorld-Eval
        dataset = load_dataset(args.dataset_path, split=split)
        logger.info(f"Loaded HuggingFace dataset {args.dataset_path} (split: {split})")
    else:
        # Local parquet file
        dataset = load_dataset("parquet", data_files=args.dataset_path)["train"]
        logger.info(f"Loaded local parquet file {args.dataset_path}")

    logger.info(f"Dataset size: {len(dataset)}")

    if args.sample_rate < 1.0:
        sample_size = int(len(dataset) * args.sample_rate)
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        logger.info(f"Sampled {sample_size} examples from the dataset.")

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
    logger.info(f"Starting {n_workers} worker processes (each reuses Agent for all tasks)")

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
            args.max_tool_calls if args.max_tool_calls is not None else MAX_TOOL_CALLS,
            enabled_agent_names,
            args.enable_tools,  # Pass enable_tools to workers
        ),
    )

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

                        # Handle list of images (e.g., VisWorld-Eval)
                        if isinstance(image, list):
                            if image:  # Non-empty list
                                image = image[0]  # Take first image
                            else:
                                raise ValueError("Empty image list provided")

                        # Existing handlers
                        if isinstance(image, Image.Image):
                            image.save(image_path)
                        elif isinstance(image, dict) and "bytes" in image:
                            Image.open(BytesIO(image["bytes"])).save(image_path)
                        else:
                            raise ValueError(f"Invalid image type: {type(image)}")

                payload = {
                    "id": task_id,
                    "traj_id": traj_id,
                    "task_save_dir": traj_save_dir,
                    "prompt": dataset_spec.build_prompt(item, True, separated=True),  # Use separated prompt (no role/answer format)
                    "answer": dataset_spec.get_answer(item),
                    "image_path": image_path,
                    "text_only": text_only,
                    "task_kwargs": dataset_spec.build_task_kwargs(item),
                    "answer_format": dataset_spec.answer_format,  # Added only in final answer phase
                }

                ar = pool.apply_async(_run_one_task, (payload,))
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
    finally:
        # Close pool cleanly BEFORE shutting down Ray - even if errors occurred
        logger.info("Closing worker pool...")
        pool.close()
        pool.join()
        logger.info("Worker pool closed successfully")

    save_global_meta_info(output_path, meta_info_list)
    logger.info(f"All tasks completed. Total successful: {len(meta_info_list)}")

    # Shutdown Ray tool agents AFTER pool is closed to avoid SIGTERM conflicts
    # Note: We only shutdown tool agents, NOT Ray itself (Ray cluster remains running)
    if tool_router.is_agent_enabled():
        logger.info("Shutting down shared Ray tool agents...")
        tool_router.shutdown_agents()
        logger.info("Ray tool agents shutdown complete")

    logger.info("Script completed. Ray cluster remains running.")


if __name__ == "__main__":
    main()
