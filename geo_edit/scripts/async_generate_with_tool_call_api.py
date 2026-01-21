import json
import os
import argparse
import logging
import shutil
import time
import multiprocessing as mp

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

from ..agents.api_agent import APIBasedAgent, AgentConfig
from ..agents.vllm_agent import VLLMBasedAgent
from ..environment.action import TOOL_FUNCTIONS
from ..environment.task.google_vision_qa_task import GoogleVisionQATask
from ..environment.task.openai_vision_qa_task import OpenAIVisionQATask
from ..environment.task.vllm_vision_qa_task import VLLMVisionQATask
from ..config import (
    build_agent_configs,
    build_openai_agent_configs,
    build_vllm_agent_configs,
)
from ..constants import SYSTEM_PROMPT, MAX_TOOL_CALLS, SUDOKU_TOOL_CALL_INPUT_TEMPLATE, MATHVISION_INPUT_TEMPLATE, NOTOOL_INPUT_TEMPLATE, SUDOKU_TEXT_INPUT_TEMPLATE
from ..utils.logger import setup_logger

logger = setup_logger(__name__)
# ---------------------------
# Worker globals (one per process)
# ----------------------------
_WORKER_AGENT = None
_WORKER_AGENT_CONFIGS = None
_WORKER_INPUT_TEMPLATE = None
_WORKER_OUTPUT_PATH = None
_WORKER_MAX_TOOL_CALLS = None
_WORKER_TASK_CLASS = None
_WORKER_MODEL_TYPE = None


def _init_worker(
    api_key: str,
    model_name_or_path: str,
    model_type: str,
    api_base: str,
    port: int,
    output_path: str,
    input_template: str,
    max_tool_calls: int,
):
    global _WORKER_AGENT, _WORKER_AGENT_CONFIGS, _WORKER_INPUT_TEMPLATE, _WORKER_OUTPUT_PATH, _WORKER_MAX_TOOL_CALLS, _WORKER_TASK_CLASS, _WORKER_MODEL_TYPE

    max_output_tokens = None
    if model_type in {"Google", "OpenAI"} and not api_key:
        raise ValueError("API key must be provided for Google/OpenAI models.")

    if model_type == "Google":
        agent_configs = build_agent_configs(
            max_output_tokens=max_output_tokens,
            thinking_level="low",
            include_thoughts=True,
            temperature=1.0,
            system_prompt=SYSTEM_PROMPT,
            candidate_count=1,
            tool_mode="AUTO",
            disable_automatic_function_calling=True,
        )
        _WORKER_TASK_CLASS = GoogleVisionQATask
    elif model_type == "OpenAI":
        agent_configs = build_openai_agent_configs(
            max_output_tokens=max_output_tokens,
            temperature=1.0,
            system_prompt=SYSTEM_PROMPT,
            tool_mode="AUTO",
            reasoning_level="medium",
        )
        _WORKER_TASK_CLASS = OpenAIVisionQATask
    else:
        agent_configs = build_vllm_agent_configs(
            max_output_tokens=max_output_tokens,
            temperature=1.0,
            tool_mode="AUTO",
        )
        _WORKER_TASK_CLASS = VLLMVisionQATask

    config = AgentConfig(
        model_type=model_type,
        model_name=model_name_or_path,
        api_key=api_key,
        api_base=api_base,
        port=port,
        generate_config=agent_configs.generate_config,
        n_retry=3,
    )

    _WORKER_AGENT_CONFIGS = agent_configs
    agent_cls = VLLMBasedAgent if model_type == "vLLM" else APIBasedAgent
    _WORKER_AGENT = agent_cls(config)
    _WORKER_INPUT_TEMPLATE = input_template
    _WORKER_OUTPUT_PATH = output_path
    _WORKER_MAX_TOOL_CALLS = max_tool_calls
    _WORKER_MODEL_TYPE = model_type


def _run_one_task(task_payload: dict):
    """
    Worker: do NOT mkdir here, do NOT save image here.
    Image must already be saved and passed in as a path.
    Returns:
      (ok: bool, meta_info: dict|None)
    """
    global _WORKER_AGENT, _WORKER_AGENT_CONFIGS, _WORKER_MAX_TOOL_CALLS, _WORKER_INPUT_TEMPLATE

    task_id = task_payload["id"]
    task_save_dir = task_payload["task_save_dir"]
    question = task_payload["question"]
    options = task_payload["options"]
    answer = task_payload["answer"]
    image_path = task_payload["image_path"]
    rows = task_payload["rows"]
    cols = task_payload["cols"]
    initial_board = task_payload["initial_board"]
    visual_elements = task_payload["visual_elements"]

    meta_path = os.path.join(task_save_dir, "meta_info.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_info = json.loads(f.readline().strip())
        return True, meta_info

    text_prompt = _WORKER_INPUT_TEMPLATE.format(rules=question, answer=answer, rows=rows, cols=cols, total_cells=rows*cols, initial_board=initial_board, visual_elements=visual_elements)

    task_kwargs = {"text_only": False}
    if _WORKER_MODEL_TYPE == "vLLM":
        task_kwargs["system_prompt"] = SYSTEM_PROMPT
    task = _WORKER_TASK_CLASS(
        task_id=task_id,
        task_prompt=text_prompt,
        task_answer=answer,
        task_image_path=image_path,
        tool_functions=TOOL_FUNCTIONS,
        save_dir=task_save_dir,
        **task_kwargs,
    )

    _WORKER_AGENT.reset()

    try:
        for i in range(_WORKER_MAX_TOOL_CALLS):
            action, extra_info = _WORKER_AGENT.act(task.contents)
            function_call_part_list = task.parse_action(step=i + 1, action=action, extra_info=extra_info)

            if not function_call_part_list:
                break

            task.update_observation_from_action(function_call_part_list)
            
            if i>=4:
                _WORKER_AGENT.config.generate_config = _WORKER_AGENT_CONFIGS.generate_config

        if task.state and _WORKER_AGENT.step_count >= _WORKER_MAX_TOOL_CALLS:
            force_prompt = "Max tool calls reached. Please provide the final answer without further tool calls."
            if _WORKER_MODEL_TYPE == "Google":
                task.contents.append(force_prompt)
            else:
                task.append_prompt(force_prompt)
            original_generate_config = _WORKER_AGENT.config.generate_config
            _WORKER_AGENT.config.generate_config = _WORKER_AGENT_CONFIGS.force_generate_config
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
    parser = argparse.ArgumentParser(description="Generate content with tool calls using API models (multiprocess).")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the selected provider.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--model_name_or_path", type=str, default="gemini-3-pro-preview", help="Model name or path.")
    parser.add_argument("--model_type", type=str, default="Google", choices=["Google", "OpenAI", "vLLM"], help="Model provider.")
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for vLLM OpenAI-compatible server.")
    parser.add_argument("--port", type=int, default=None, help="Port for vLLM OpenAI-compatible server.")
    parser.add_argument("--max_concurrent_requests", type=int, default=8, help="Number of worker processes (agent pool).")
    parser.add_argument("--sample_rate", type=float, default=0.1, help="Sampling rate for the dataset.")
    args = parser.parse_args()
    if args.model_type in {"Google", "OpenAI"} and not args.api_key:
        raise ValueError("API key must be provided for Google/OpenAI models.")

    seed = 42
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    dataset = load_dataset("parquet", data_files=args.dataset_path)["train"]
    # dataset = dataset.filter(lambda x: x["image_preview"] is not None)
    logger.info(f"Dataset size after filtering: {len(dataset)}")

    if args.sample_rate < 1.0:
        sample_size = int(len(dataset) * args.sample_rate)
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
        logger.info(f"Sampled {sample_size} examples from the dataset.")

    # input_template = MATHVISION_INPUT_TEMPLATE
    input_template =  SUDOKU_TOOL_CALL_INPUT_TEMPLATE

    # 1) main process scan: collect done meta_info + pending items
    meta_info_list = []
    pending_items = []

    for item in dataset:
        task_id = item["puzzle_id"]
        task_save_dir = os.path.join(output_path, task_id)
        meta_path = os.path.join(task_save_dir, "meta_info.jsonl")

        if os.path.exists(task_save_dir) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_info = json.loads(f.readline().strip())
            meta_info_list.append(meta_info)
        else:
            pending_items.append(item)

    logger.info(f"Already done: {len(meta_info_list)}")
    logger.info(f"Pending: {len(pending_items)}")

    # 2) multiprocessing pool, create folder+save image ONLY when submitting task
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
            input_template,
            MAX_TOOL_CALLS,
        ),
    ) as pool:

        inflight = []  # list[(task_id, AsyncResult)]
        submit_idx = 0

        pbar = tqdm(total=len(pending_items), desc="processing")

        while submit_idx < len(pending_items) or inflight:
            # submit up to n_workers tasks; mkdir+save image just before submit
            while submit_idx < len(pending_items) and len(inflight) < n_workers:
                item = pending_items[submit_idx]
                submit_idx += 1

                task_id = item["puzzle_id"]
                task_save_dir = os.path.join(output_path, task_id)
                meta_path = os.path.join(task_save_dir, "meta_info.jsonl")

                # if completed by other runs, load and skip
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_info = json.loads(f.readline().strip())
                    meta_info_list.append(meta_info)
                    continue

                os.makedirs(task_save_dir, exist_ok=True)

                image = item["board_image"]
                if isinstance(image, Image.Image):
                    image_path = os.path.join(task_save_dir, "input_image.png")
                    image.save(image_path)
                else:
                    image_path = image

                payload = {
                    "id": task_id,
                    "task_save_dir": task_save_dir,
                    "question": item["rules"],
                    "answer": item["solution"],
                    "image_path": image_path,
                    "options": item.get("options", ""),
                    "rows": item["rows"],
                    "cols": item["cols"],
                    "initial_board": item["initial_board"],
                    "visual_elements": str(item["visual_elements"]) if "visual_elements" in item else "",
                }

                ar = pool.apply_async(_run_one_task, (payload,))
                inflight.append((task_id, ar))

            # harvest finished tasks
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

    # 3) aggregate global stats
    total_tool_calls = 0
    total_tokens = 0
    tool_usage_counts = {}
    reach_max_tool_call_count = 0
    direct_answer_count = 0

    for info in meta_info_list:
        total_tool_calls += info["function_call_total_count"]
        total_tokens += info["tokens_used_total"]
        if info["total_steps"] >= MAX_TOOL_CALLS:
            reach_max_tool_call_count += 1
        if info["function_call_total_count"] == 0:
            direct_answer_count += 1
        for tool_name, count in info["function_call_each_count"].items():
            tool_usage_counts[tool_name] = tool_usage_counts.get(tool_name, 0) + count

    global_meta_info = {
        "total_examples": len(meta_info_list),
        "total_tool_calls": total_tool_calls,
        "total_tokens": total_tokens,
        "tool_usage_counts": tool_usage_counts,
        "reach_max_tool_call_count": reach_max_tool_call_count,
        "direct_answer_count": direct_answer_count,
    }

    global_meta_info_jsonl_path = os.path.join(output_path, "global_meta_info.jsonl")
    with open(global_meta_info_jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(global_meta_info) + "\n")


if __name__ == "__main__":
    main()
