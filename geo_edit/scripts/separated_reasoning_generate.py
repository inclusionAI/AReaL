"""Separated reasoning script for two-phase tool call generation.

Supports Gemini and GPT models (Google API and matrixllm chat_completions).
Only supports force tool mode.

Phase 1: Generate reasoning (can see tools, but cannot execute)
Phase 2: Generate tool call based on reasoning (no additional reasoning)
"""
import argparse
import json
import logging
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
    build_google_reasoning_only_config,
    build_google_tool_call_only_config,
    build_api_reasoning_only_config,
    build_api_tool_call_only_config,
)
from geo_edit.constants import MAX_TOOL_CALLS
from geo_edit.prompts import get_system_prompt
from geo_edit.prompts.system_prompts import SEPARATED_TOOL_CALL_ONLY_PROMPT
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
_WORKER_TOOL_CALL_ONLY_CONFIG = None


def _init_worker(
    api_key: str,
    model_name_or_path: str,
    model_type: str,
    api_base: str,
    port: int,
    output_path: str,
    max_tool_calls: int,
):
    """Initialize worker for Gemini/GPT models (Google API or matrixllm)."""
    global _WORKER_AGENT, _WORKER_AGENT_CONFIGS, _WORKER_OUTPUT_PATH, _WORKER_MAX_TOOL_CALLS
    global _WORKER_TASK_CLASS, _WORKER_API_MODE
    global _WORKER_TOOL_ROUTER, _WORKER_REASONING_ONLY_CONFIG, _WORKER_TOOL_CALL_ONLY_CONFIG

    # Force mode only
    _WORKER_TOOL_ROUTER = ToolRouter(tool_mode="force")

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
        _WORKER_REASONING_ONLY_CONFIG = build_google_reasoning_only_config(agent_configs.generate_config)
        _WORKER_TOOL_CALL_ONLY_CONFIG = build_google_tool_call_only_config(agent_configs.generate_config)
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
        _WORKER_REASONING_ONLY_CONFIG = build_api_reasoning_only_config(agent_configs.generate_config)
        _WORKER_TOOL_CALL_ONLY_CONFIG = build_api_tool_call_only_config(agent_configs.generate_config)

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


def _run_one_task(task_payload: dict):
    """
    Worker: run separated two-phase generation.
    Phase 1: Generate reasoning (no tool execution)
    Phase 2: Generate tool call (no additional reasoning)
    """
    assert _WORKER_AGENT is not None
    assert _WORKER_AGENT_CONFIGS is not None
    assert _WORKER_MAX_TOOL_CALLS is not None
    assert _WORKER_TOOL_ROUTER is not None
    assert _WORKER_REASONING_ONLY_CONFIG is not None
    assert _WORKER_TOOL_CALL_ONLY_CONFIG is not None

    task_id = task_payload["id"]
    task_save_dir = task_payload["task_save_dir"]
    answer = task_payload["answer"]
    image_path = task_payload["image_path"]
    text_prompt = task_payload["prompt"]
    text_only = task_payload.get("text_only", False)

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
        task_prompt=text_prompt,
        task_answer=answer,
        task_image_path=image_path,
        tool_functions=tool_functions,
        tool_return_types=tool_return_types,
        save_dir=task_save_dir,
        **task_kwargs,
    )

    _WORKER_AGENT.reset()
    original_generate_config = _WORKER_AGENT.config.generate_config
    answer_pattern = re.compile(r"<answer>", re.IGNORECASE)

    try:
        for i in range(_WORKER_MAX_TOOL_CALLS):
            # ===== Phase 1: Generate reasoning (no tool call, no answer) =====
            _WORKER_AGENT.config.generate_config = _WORKER_REASONING_ONLY_CONFIG
            reasoning_action, reasoning_extra = _WORKER_AGENT.act(task.contents)

            # Extract reasoning text from action
            if _WORKER_API_MODE == "google":
                text_parts = [p.text for p in reasoning_action.parts if p.text and not p.thought]
                reasoning_text = "\n".join(text_parts)
            else:
                reasoning_text = reasoning_action.choices[0].message.content or ""

            # Check for <answer> - error if found
            if answer_pattern.search(reasoning_text):
                raise ValueError("Reasoning phase should not generate <answer>. Model violated the protocol.")

            # ===== Phase 2: Generate tool call =====
            task.append_assistant_message(reasoning_text)
            task.append_prompt(SEPARATED_TOOL_CALL_ONLY_PROMPT)

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
                step=i + 1,
                action=tool_action,
                extra_info=merged_extra
            )

            if not function_call_part_list:
                break

            task.update_observation_from_action(function_call_part_list)

        # Force final answer if max tool calls reached
        if task.state:
            logger.info(f"[{task_id}] reached max tool calls {_WORKER_MAX_TOOL_CALLS}, forcing final answer.")
            task.append_prompt("Max tool calls reached. Please provide the final answer based on the information gathered so far.")
            _WORKER_AGENT.config.generate_config = _WORKER_AGENT_CONFIGS.force_final_generate_config
            action, extra_info = _WORKER_AGENT.act(task.contents)
            _WORKER_AGENT.config.generate_config = original_generate_config
            task.parse_action(step=_WORKER_MAX_TOOL_CALLS + 1, action=action, extra_info=extra_info)

        if task.state:
            meta_info = task.save_trajectory()
            return True, meta_info

        shutil.rmtree(task_save_dir, ignore_errors=True)
        return False, None

    except Exception as e:
        logging.error(f"[{task_id}] worker failed: {e}")
        shutil.rmtree(task_save_dir, ignore_errors=True)
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Separated reasoning generation for Gemini/GPT models.")
    parser.add_argument("--api_key", type=str, default=None, help="API key for Google API.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_SPECS.keys()), help="Dataset adapter name.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output.")
    parser.add_argument("--model_name_or_path", type=str, default="gemini-2.5-pro-preview-05-06", help="Model name.")
    parser.add_argument("--model_type", type=str, default="Google", choices=["Google", "SGLang", "OpenAI"], help="Model provider.")
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for matrixllm server.")
    parser.add_argument("--port", type=int, default=None, help="Port for server.")
    parser.add_argument("--max_concurrent_requests", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--sample_rate", type=float, default=0.1, help="Sampling rate for the dataset.")
    parser.add_argument("--n_trajectories", type=int, default=1, help="Number of trajectories per task.")
    parser.add_argument("--node_resource", type=str, default=None, help="Ray custom resource name.")
    args = parser.parse_args()

    if args.model_type == "Google" and not args.api_key:
        raise ValueError("API key must be provided for Google models.")

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    dataset = load_dataset("parquet", data_files=args.dataset_path)["train"]
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

    tool_router = ToolRouter(tool_mode="force", node_resource=args.node_resource)

    ctx = mp.get_context("spawn")
    n_workers = max(1, int(args.max_concurrent_requests))

    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(
            args.api_key,
            args.model_name_or_path,
            args.model_type,
            args.api_base,
            args.port,
            output_path,
            MAX_TOOL_CALLS,
        ),
    ) as pool:
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
                    "prompt": dataset_spec.build_prompt(item, True),  # Always use tool prompt for force mode
                    "answer": item[dataset_spec.answer_key],
                    "image_path": image_path,
                    "text_only": text_only,
                    "task_kwargs": dataset_spec.build_task_kwargs(item),
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

    save_global_meta_info(output_path, meta_info_list)
    tool_router.shutdown_agents()


if __name__ == "__main__":
    main()
