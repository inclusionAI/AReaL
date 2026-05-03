from __future__ import annotations
"""Async direct generation script without tool calls.

Supports vLLM/SGLang via OpenAI-compatible API with custom api_base.
"""
"""
python geo_edit/scripts/direct_generate.py \
    --dataset_path /path/to/data.parquet \
    --output_dir /path/to/output \
    --dataset_name shortest_path_image \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --api_base http://localhost:8000 \
    --model_type vLLM \
    --api_mode chat_completions \
    --max_concurrent_requests 8
"""

import json
import multiprocessing as mp
import os
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from geo_edit.agents.api_agent import APIBasedAgent, AgentConfig
from geo_edit.config import build_api_agent_configs
from geo_edit.datasets.task_registry import DATASET_SPECS, get_dataset_spec
from geo_edit.environment.task.openai_compatible_vision_qa_task import OpenAICompatibleVisionQATask
from geo_edit.tool_definitions import ToolRouter
from geo_edit.prompts.system_prompts import VLLM_NO_TOOL_SYSTEM_PROMPT
from geo_edit.utils.logger import setup_logger
from geo_edit.utils.stats import save_global_meta_info

logger = setup_logger(__name__)

SYSTEM_PROMPT = VLLM_NO_TOOL_SYSTEM_PROMPT

# Fixed parameters
TEMPERATURE = 1.0
N_RETRY = 3
SEED = 42

# =============================================================================
# Worker globals
# =============================================================================

_WORKER_AGENT: APIBasedAgent | None = None
_WORKER_OUTPUT_PATH: str | None = None
_WORKER_MODEL_TYPE: str | None = None
_WORKER_API_MODE: str | None = None
_WORKER_NO_IMAGE_COMPRESSION: bool = False


def _init_worker(
    model_name_or_path: str,
    model_type: str,
    api_base: str,
    api_mode: str,
    api_key: str | None,
    no_image_compression: bool = False,
    temperature: float = TEMPERATURE,
    max_output_tokens: int | None = None,
):
    global _WORKER_AGENT, _WORKER_OUTPUT_PATH, _WORKER_MODEL_TYPE, _WORKER_API_MODE, _WORKER_NO_IMAGE_COMPRESSION

    _WORKER_NO_IMAGE_COMPRESSION = no_image_compression

    tool_router = ToolRouter(tool_mode="direct")
    agent_configs = build_api_agent_configs(
        tool_router,
        api_mode=api_mode,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_prompt=SYSTEM_PROMPT.strip(),
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


def _run_one_task(task_payload: dict):
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
    if _WORKER_NO_IMAGE_COMPRESSION:
        task_kwargs["max_image_base64_bytes"] = None
    if text_only:
        task_kwargs["text_only"] = True

    _WORKER_AGENT.reset()
    response_validator = task_payload.get("response_validator")
    max_attempts = 5 if response_validator else 1

    for attempt in range(max_attempts):
        if attempt > 0:
            shutil.rmtree(task_save_dir, ignore_errors=True)
            os.makedirs(task_save_dir, exist_ok=True)

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
                if response_validator:
                    output = ""
                    if hasattr(action, "choices") and action.choices:
                        output = action.choices[0].message.content or ""
                    else:
                        output = getattr(action, "output_text", "") or ""
                    if not response_validator(output):
                        logger.warning(f"[{task_id}] Attempt {attempt + 1}/{max_attempts}: response_validator rejected, retrying")
                        continue
                return True, task.save_trajectory()

            if attempt < max_attempts - 1:
                continue
            shutil.rmtree(task_save_dir, ignore_errors=True)
            return False, None

        except Exception as e:
            logger.error(f"[{task_id}] worker failed (attempt {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                continue
            shutil.rmtree(task_save_dir, ignore_errors=True)
            return False, None

    shutil.rmtree(task_save_dir, ignore_errors=True)
    return False, None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Async direct generation without tool calls.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset parquet file path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model name.")
    parser.add_argument("--api_base", type=str, required=True, help="API base URL.")
    parser.add_argument("--model_type", type=str, default="vLLM", choices=["vLLM", "SGLang", "OpenAI"])
    parser.add_argument("--api_mode", type=str, default="chat_completions", choices=["chat_completions", "responses"])
    parser.add_argument("--api_key", type=str, default=None, help="API key (optional for vLLM/SGLang).")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Dataset sampling rate.")
    parser.add_argument("--max_concurrent_requests", type=int, default=8, help="Number of worker processes.")
    parser.add_argument(
        "--no_image_compression",
        action="store_true",
        help="Disable image compression (send original quality to API). Default: compress to 4MB base64.",
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--max_output_tokens", type=int, default=None, help="Max completion tokens per request (default: server/model default).")
    args = parser.parse_args()

    if args.model_type == "OpenAI" and not args.api_key:
        raise ValueError("API key is required for OpenAI model type.")

    logger.info("System prompt:\n%s", SYSTEM_PROMPT.strip())

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    dataset = load_dataset("parquet", data_files=args.dataset_path)["train"]
    dataset_spec = get_dataset_spec(args.dataset_name)

    if args.sample_rate < 1.0:
        sample_size = int(len(dataset) * args.sample_rate)
        dataset = dataset.shuffle(seed=SEED).select(range(sample_size))
        logger.info("Sampled %d examples.", sample_size)

    # Collect pending items
    meta_info_list: List[Dict[str, Any]] = []
    pending_items = []

    for item in dataset:
        task_id = str(item[dataset_spec.id_key])
        task_save_dir = os.path.join(output_path, task_id)
        meta_path = os.path.join(task_save_dir, "meta_info.jsonl")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_info_list.append(json.loads(f.readline().strip()))
        else:
            pending_items.append(item)

    logger.info("Already done: %d, Pending: %d", len(meta_info_list), len(pending_items))

    # Multiprocessing pool
    ctx = mp.get_context("spawn")
    n_workers = max(1, args.max_concurrent_requests)

    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(args.model_name_or_path, args.model_type, args.api_base, args.api_mode, args.api_key, args.no_image_compression, args.temperature, args.max_output_tokens),
    ) as pool:
        inflight = []
        submit_idx = 0
        pbar = tqdm(total=len(pending_items), desc="Processing")

        while submit_idx < len(pending_items) or inflight:
            # Submit tasks
            while submit_idx < len(pending_items) and len(inflight) < n_workers:
                item = pending_items[submit_idx]
                submit_idx += 1

                task_id = str(item[dataset_spec.id_key])
                task_save_dir = os.path.join(output_path, task_id)
                meta_path = os.path.join(task_save_dir, "meta_info.jsonl")

                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_info_list.append(json.loads(f.readline().strip()))
                    continue

                os.makedirs(task_save_dir, exist_ok=True)

                # Handle image (single or multi-image)
                image_path = None
                text_only = dataset_spec.image_key is None
                if dataset_spec.image_key:
                    image = item.get(dataset_spec.image_key)

                    # Normalize image container:
                    #   - numpy.ndarray / list -> treat as multi-image list
                    #   - single object (PIL/dict/bytes/str) -> wrap into 1-element list
                    try:
                        import numpy as _np
                        is_array = isinstance(image, _np.ndarray)
                    except ImportError:
                        is_array = False
                    is_multi = (isinstance(image, list) or is_array) and (
                        not (isinstance(image, str) or isinstance(image, bytes))
                    )
                    images = list(image) if is_multi else [image]

                    saved_paths: list[str] = []
                    for idx, sub in enumerate(images):
                        if isinstance(sub, str) and os.path.isfile(sub):
                            saved_paths.append(sub)
                            continue
                        sub_path = os.path.join(
                            task_save_dir,
                            "input_image.png" if len(images) == 1 else f"input_image_{idx:02d}.png",
                        )
                        if not os.path.exists(sub_path):
                            if isinstance(sub, Image.Image):
                                sub.save(sub_path)
                            elif (
                                isinstance(sub, dict)
                                and isinstance(sub.get("bytes"), (bytes, bytearray))
                            ):
                                Image.open(BytesIO(sub["bytes"])).save(sub_path)
                            elif (
                                isinstance(sub, dict)
                                and isinstance(sub.get("path"), str)
                                and os.path.exists(sub["path"])
                            ):
                                import shutil
                                shutil.copy2(sub["path"], sub_path)
                            elif isinstance(sub, bytes):
                                Image.open(BytesIO(sub)).save(sub_path)
                            else:
                                raise ValueError(f"Invalid image item type: {type(sub)}")
                        saved_paths.append(sub_path)

                    image_path = saved_paths[0] if len(saved_paths) == 1 else saved_paths
                else:
                    text_only = True

                payload = {
                    "id": task_id,
                    "task_save_dir": task_save_dir,
                    "prompt": dataset_spec.build_prompt(item, use_tools=False),
                    "answer": dataset_spec.get_answer(item),
                    "image_path": image_path,
                    "text_only": text_only,
                    "task_kwargs": dataset_spec.build_task_kwargs(item),
                    "response_validator": dataset_spec.response_validator,
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

    save_global_meta_info(output_path, meta_info_list)
    logger.info("Done. Processed %d examples.", len(meta_info_list))


if __name__ == "__main__":
    main()
