import json
import os
import argparse
import logging
import shutil
import time
import multiprocessing as mp
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

from geo_edit.agents.api_agent import APIBasedAgent, AgentConfig
from geo_edit.agents.vllm_agent import VLLMBasedAgent
from geo_edit.environment.action import TOOL_FUNCTIONS
from geo_edit.environment.task.google_vision_qa_task import GoogleVisionQATask
from geo_edit.environment.task.openai_vision_qa_task import OpenAIVisionQATask
from geo_edit.environment.task.vllm_vision_qa_task import VLLMVisionQATask
from geo_edit.config import (
    build_google_agent_configs,
    build_openai_agent_configs,
    build_vllm_agent_configs,
)
from geo_edit.scripts.script_utils import save_global_meta_info
from geo_edit.constants import MAX_TOOL_CALLS, get_system_prompt
from geo_edit.datasets.task_registry import DATASET_SPECS, get_dataset_spec
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)
# ---------------------------
# Worker globals (one per process)
# ----------------------------
_WORKER_AGENT = None
_WORKER_AGENT_CONFIGS = None
_WORKER_OUTPUT_PATH = None
_WORKER_MAX_TOOL_CALLS = None
_WORKER_TASK_CLASS = None
_WORKER_MODEL_TYPE = None
_WORKER_SYSTEM_PROMPT = None

def _init_worker(
    api_key: str,
    model_name_or_path: str,
    model_type: str,
    api_base: str,
    port: int,
    output_path: str,
    max_tool_calls: int,
    use_tools: str,
):
    global _WORKER_AGENT, _WORKER_AGENT_CONFIGS, _WORKER_OUTPUT_PATH, _WORKER_MAX_TOOL_CALLS, _WORKER_TASK_CLASS, _WORKER_MODEL_TYPE, _WORKER_SYSTEM_PROMPT

    max_output_tokens = None
    if model_type in {"Google", "OpenAI"} and not api_key:
        raise ValueError("API key must be provided for Google/OpenAI models.")
    system_prompt = (
        get_system_prompt(model_type, use_tools) if use_tools != "direct" else None
    )

    if model_type == "Google":
        agent_configs = build_google_agent_configs(
            max_output_tokens=max_output_tokens,
            thinking_level="low",
            include_thoughts=True,
            temperature=1.0,
            system_prompt=system_prompt,
            tool_mode=use_tools,
        )
        _WORKER_TASK_CLASS = GoogleVisionQATask
    elif model_type == "OpenAI":
        agent_configs = build_openai_agent_configs(
            max_output_tokens=max_output_tokens,
            temperature=1.0,
            tool_mode=use_tools,
            reasoning_level="minimal",
        )
        _WORKER_TASK_CLASS = OpenAIVisionQATask
    else:
        agent_configs = build_vllm_agent_configs(
            max_output_tokens=max_output_tokens,
            temperature=1.0,
            system_prompt=system_prompt,
            tool_mode=use_tools,
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
    _WORKER_OUTPUT_PATH = output_path
    _WORKER_MAX_TOOL_CALLS = max_tool_calls
    _WORKER_MODEL_TYPE = model_type
    _WORKER_SYSTEM_PROMPT = system_prompt


def _run_one_task(task_payload: dict):
    """
    Worker: do NOT mkdir here, do NOT save image here.
    Image must already be saved and passed in as a path.
    Returns:
      (ok: bool, meta_info: dict|None)
    """
    global _WORKER_AGENT, _WORKER_AGENT_CONFIGS, _WORKER_MAX_TOOL_CALLS, _WORKER_SYSTEM_PROMPT, _WORKER_MODEL_TYPE

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

    task_kwargs = {}
    if text_only:
        logger.info(f"[{task_id}] running text-only task.")
        task_kwargs["text_only"] = True
    if _WORKER_MODEL_TYPE == "vLLM":
        task_kwargs["system_prompt"] = _WORKER_SYSTEM_PROMPT
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
    original_generate_config = _WORKER_AGENT.config.generate_config.copy()
    try:
        for i in range(_WORKER_MAX_TOOL_CALLS):
            action, extra_info = _WORKER_AGENT.act(task.contents)
            function_call_part_list = task.parse_action(step=i + 1, action=action, extra_info=extra_info)

            if not function_call_part_list:
                break

            task.update_observation_from_action(function_call_part_list)
            
        if task.state and _WORKER_AGENT.step_count >= _WORKER_MAX_TOOL_CALLS:
                logger.info(f"[{task_id}] reached max tool calls {_WORKER_MAX_TOOL_CALLS}, forcing final answer.")
                force_prompt = "Max tool calls reached. Please provide the final answer based on the information gathered so far."
                task.append_prompt(force_prompt)
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
    parser = argparse.ArgumentParser(description="Generate content with tool calls using API models (multiprocess).")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the selected provider.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_SPECS.keys()), help="Dataset adapter name.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--model_name_or_path", type=str, default="gemini-3-pro-preview", help="Model name or path.")
    parser.add_argument("--model_type", type=str, default="Google", choices=["Google", "OpenAI", "vLLM"], help="Model provider.")
    parser.add_argument("--use_tools", type=str, default="auto", choices=["direct", "auto", "force"])
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for vLLM OpenAI-compatible server.")
    parser.add_argument("--port", type=int, default=None, help="Port for vLLM OpenAI-compatible server.")
    parser.add_argument("--max_concurrent_requests", type=int, default=8, help="Number of worker processes (agent pool).")
    parser.add_argument("--sample_rate", type=float, default=0.1, help="Sampling rate for the dataset.")
    parser.add_argument("--n_trajectories", type=int, default=1, help="Number of trajectories to generate per task.")
    args = parser.parse_args()
    if args.model_type in {"Google", "OpenAI"} and not args.api_key:
        raise ValueError("API key must be provided for Google/OpenAI models.")

    seed = 42   
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    dataset = load_dataset("parquet", data_files=args.dataset_path)["train"]
    logger.info(f"Dataset size after filtering: {len(dataset)}")

    if args.sample_rate < 1.0:
        sample_size = int(len(dataset) * args.sample_rate)
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
        logger.info(f"Sampled {sample_size} examples from the dataset.")

    dataset_spec = get_dataset_spec(args.dataset_name)
    tool_mode = args.use_tools
    if tool_mode == "direct" and dataset_spec.notool_prompt_template is None:
        logger.warning("Dataset %s has no no-tool template; using tool template.", dataset_spec.name)

    # 1 main process scan: collect done meta_info + pending (task, traj_id) pairs
    n_trajectories = args.n_trajectories
    meta_info_list = []
    pending_items = []  # list of (item, traj_id)

    for item in dataset:
        task_id = str(item[dataset_spec.id_key])
        task_base_dir = os.path.join(output_path, task_id)

        for traj_id in range(n_trajectories):
            if n_trajectories == 1:
                # Backward compatible: single trajectory uses task_base_dir directly
                traj_save_dir = task_base_dir
            else:
                traj_save_dir = os.path.join(task_base_dir, f"traj_{traj_id}")
            meta_path = os.path.join(traj_save_dir, "meta_info.jsonl")

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_info = json.loads(f.readline().strip())
                meta_info_list.append(meta_info)
            else:
                pending_items.append((item, traj_id))

    logger.info(f"Already done: {len(meta_info_list)}")
    logger.info(f"Pending: {len(pending_items)} (tasks x trajectories)")

    # 2multiprocessing pool, create folder+save image ONLY when submitting task
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
            tool_mode,
        ),
    ) as pool:

        inflight = []  # list[(task_id, AsyncResult)]
        submit_idx = 0

        pbar = tqdm(total=len(pending_items), desc="processing")

        while submit_idx < len(pending_items) or inflight:
            # submit up to n_workers tasks; mkdir+save image just before submit
            while submit_idx < len(pending_items) and len(inflight) < n_workers:
                item, traj_id = pending_items[submit_idx]
                submit_idx += 1

                task_id = str(item[dataset_spec.id_key])
                task_base_dir = os.path.join(output_path, task_id)
                if n_trajectories == 1:
                    traj_save_dir = task_base_dir
                else:
                    traj_save_dir = os.path.join(task_base_dir, f"traj_{traj_id}")
                meta_path = os.path.join(traj_save_dir, "meta_info.jsonl")

                # if completed by other runs, load and skip
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_info = json.loads(f.readline().strip())
                    meta_info_list.append(meta_info)
                    continue

                os.makedirs(task_base_dir, exist_ok=True)
                os.makedirs(traj_save_dir, exist_ok=True)

                # Save input image to task_base_dir (shared across trajectories)
                image_path = None
                text_only = dataset_spec.image_key is None
                if dataset_spec.image_key:
                    image_path = os.path.join(task_base_dir, "input_image.png")
                    if not os.path.exists(image_path):
                        image = item.get(dataset_spec.image_key)
                        if isinstance(image, Image.Image):
                            image.save(image_path)
                        elif isinstance(image, dict) and "bytes" in image and isinstance(image["bytes"], (bytes, bytearray)):
                            image = Image.open(BytesIO(image["bytes"]))
                            image.save(image_path)
                        else:
                            raise ValueError(f"Invalid image type: {type(image)}")
                else:
                    text_only = True

                payload = {
                    "id": task_id,
                    "traj_id": traj_id,
                    "task_save_dir": traj_save_dir,
                    "prompt": dataset_spec.build_prompt(item, tool_mode != "direct"),
                    "answer": item[dataset_spec.answer_key],
                    "image_path": image_path,
                    "text_only": text_only,
                }

                ar = pool.apply_async(_run_one_task, (payload,))
                inflight.append((f"{task_id}_traj{traj_id}", ar))

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

    save_global_meta_info(output_path, meta_info_list)


if __name__ == "__main__":
    main()
