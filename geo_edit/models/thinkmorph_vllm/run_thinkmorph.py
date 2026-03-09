#!/usr/bin/env python
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# ThinkMorph generic inference script with task registry support
# Outputs results in openai_as_judge compatible format
#
# Usage:
#   python -m geo_edit.models.thinkmorph_vllm.run_thinkmorph \
#       --model_path /path/to/ThinkMorph-7B \
#       --task mathvisionqa \
#       --dataset_path ./dataset.parquet \
#       --output_dir ./outputs \
#       --num_gpus 8 \
#       --models_per_gpu 4

import os
import json
import argparse
import logging
from typing import Any, Dict, List, Optional

from PIL import Image
from datasets import load_dataset, Dataset
from tqdm import tqdm

from geo_edit.datasets.task_registry import DATASET_SPECS, get_dataset_spec
from geo_edit.models.thinkmorph_vllm import ThinkMorphDP
from geo_edit.models.thinkmorph_vllm.configs import (
    DEFAULT_CONFIG,
    FAST_CONFIG,
    HIGH_QUALITY_CONFIG,
    REASONING_CONFIG,
    EDITING_CONFIG,
)

# Available config presets
CONFIG_PRESETS = {
    "default": DEFAULT_CONFIG,
    "fast": FAST_CONFIG,
    "high_quality": HIGH_QUALITY_CONFIG,
    "reasoning": REASONING_CONFIG,
    "editing": EDITING_CONFIG,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_inference(
    model_path: str,
    task: str,
    dataset_path: Optional[str] = None,
    output_dir: str = "./outputs",
    num_gpus: int = 8,
    models_per_gpu: int = 1,
    max_samples: Optional[int] = None,
    split: str = "test",
    think: bool = True,
    understanding_output: bool = False,
    max_mem_per_gpu: str = "140GiB",
    use_tools: bool = False,
    config_preset: str = "default",
):
    """
    Run ThinkMorph inference on a registered task.
    Saves results in openai_as_judge compatible format.

    Args:
        model_path: Path to ThinkMorph model
        task: Task name from task_registry (e.g., "mathvisionqa", "cartomapqa_srn")
        dataset_path: Path to dataset (optional, uses default if not specified)
        output_dir: Directory to save results
        num_gpus: Number of GPUs to use
        models_per_gpu: Number of model instances per GPU
        max_samples: Maximum samples to process
        split: Dataset split to use
        think: Enable thinking mode
        understanding_output: Text-only output (disable visual thinking)
        max_mem_per_gpu: Maximum GPU memory per device
        use_tools: Use tool-enabled prompt template
        config_preset: Inference config preset (default, fast, high_quality, reasoning, editing)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get inference config
    if config_preset not in CONFIG_PRESETS:
        logger.error(f"Unknown config preset: {config_preset}")
        logger.info(f"Available presets: {list(CONFIG_PRESETS.keys())}")
        return None
    inference_config = CONFIG_PRESETS[config_preset]
    logger.info(f"Using config preset: {config_preset}")
    logger.info(f"  num_timesteps: {inference_config.get('num_timesteps')}")
    logger.info(f"  cfg_text_scale: {inference_config.get('cfg_text_scale')}")
    logger.info(f"  cfg_img_scale: {inference_config.get('cfg_img_scale')}")

    # Get dataset spec from registry
    try:
        spec = get_dataset_spec(task)
        logger.info(f"Using registered task: {task}")
    except KeyError:
        logger.error(f"Unknown task: {task}")
        logger.info(f"Available tasks: {list(DATASET_SPECS.keys())}")
        return None

    # Load dataset
    logger.info(f"Loading dataset...")
    if dataset_path:
        if dataset_path.endswith('.parquet'):
            dataset = Dataset.from_parquet(dataset_path)
        elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            dataset = Dataset.from_json(dataset_path)
        elif os.path.isdir(dataset_path):
            dataset = load_dataset(dataset_path, split=split)
        else:
            dataset = load_dataset(dataset_path, split=split)
    else:
        logger.error("No dataset_path specified")
        return None

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"Loaded {len(dataset)} samples")

    # Prepare samples using DatasetSpec
    samples = []
    for idx, item in enumerate(dataset):
        # Build prompt from spec
        prompt = spec.build_prompt(item, use_tools=use_tools)

        # Get image
        image = None
        if spec.image_key and spec.image_key in item:
            image = item[spec.image_key]

        # Get ID
        sample_id = item.get(spec.id_key, str(idx))

        # Get ground truth answer
        gt_answer = spec.get_answer(item)

        samples.append({
            'image': image,
            'text': prompt,
            'id': sample_id,
            'gt_answer': gt_answer,
            'original_item': dict(item),
        })

    # Initialize DP inferencer
    total_parallelism = num_gpus * models_per_gpu
    logger.info(f"Initializing ThinkMorphDP:")
    logger.info(f"  GPUs: {num_gpus}")
    logger.info(f"  Models per GPU: {models_per_gpu}")
    logger.info(f"  Total parallelism: {total_parallelism}")

    dp = ThinkMorphDP(
        model_path=model_path,
        num_gpus=num_gpus,
        max_mem_per_gpu=max_mem_per_gpu,
        models_per_gpu=models_per_gpu,
        inference_config=inference_config,
    )

    # Results container
    results = []

    # Callback to save each result immediately when received
    def on_result(dp_result: Dict[str, Any]):
        idx = dp_result['original_idx']
        sample = samples[idx]
        outputs = dp_result.get('output', [])

        # Extract text outputs
        text_outputs = [o for o in outputs if isinstance(o, str)]
        sample_id = str(sample.get('id', idx))
        gt_answer = sample.get('gt_answer', '')

        # Create result in openai_as_judge compatible format
        result = {
            "id": sample_id,
            "question": sample.get('text'),
            "answer": str(gt_answer),
            "output_text": text_outputs,
        }

        # Add task-specific metadata
        task_kwargs = spec.build_task_kwargs(sample.get('original_item', {}))
        if task_kwargs:
            result["metadata"] = task_kwargs

        # Save to subdirectory format for openai_as_judge (immediately!)
        sample_dir = os.path.join(output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # Save generated images immediately
        img_outputs = [o for o in outputs if isinstance(o, Image.Image)]
        img_paths = []
        for i, img in enumerate(img_outputs):
            img_filename = f"gen_{i}.png"
            img_path = os.path.join(sample_dir, img_filename)
            img.save(img_path)
            img_paths.append(img_filename)
        if img_paths:
            result["generated_images"] = img_paths

        # Save meta_info.jsonl immediately
        meta_info_path = os.path.join(sample_dir, "meta_info.jsonl")
        with open(meta_info_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        results.append(result)

    # Run inference with on_result callback for immediate saving
    logger.info(f"Running inference (think={think}, understanding_output={understanding_output})...")
    dp.infer_samples(
        samples,
        think=think,
        understanding_output=understanding_output,
        on_result=on_result,  # Save immediately when each result is ready
    )

    # Save summary
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": task,
            "total": len(results),
            "num_gpus": num_gpus,
            "models_per_gpu": models_per_gpu,
            "total_parallelism": total_parallelism,
            "think": think,
            "understanding_output": understanding_output,
            "config_preset": config_preset,
            "inference_config": inference_config,
        }, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"Config: {config_preset}")
    print(f"Samples: {len(results)}")
    print(f"GPUs: {num_gpus}, Models/GPU: {models_per_gpu}, Total parallelism: {total_parallelism}")
    print(f"Visual thinking: {not understanding_output}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ThinkMorph inference with task registry support"
    )

    # Required arguments
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to ThinkMorph model'
    )
    parser.add_argument(
        '--task', type=str, required=True,
        choices=list(DATASET_SPECS.keys()),
        help=f'Task name from registry: {list(DATASET_SPECS.keys())}'
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='Path to dataset (parquet, json, or HuggingFace dataset)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for results'
    )

    # GPU configuration
    parser.add_argument(
        '--num_gpus', type=int, default=8,
        help='Number of GPUs (default: 8)'
    )
    parser.add_argument(
        '--models_per_gpu', type=int, default=1,
        help='Number of model instances per GPU (default: 1, max ~4 for 140GB GPU)'
    )
    parser.add_argument(
        '--max_mem_per_gpu', type=str, default='140GiB',
        help='Maximum memory per GPU (default: 140GiB)'
    )

    # Dataset options
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Maximum samples to process (default: all)'
    )
    parser.add_argument(
        '--split', type=str, default='test',
        help='Dataset split (default: test)'
    )

    # Inference options
    parser.add_argument(
        '--think', action='store_true', default=True,
        help='Enable thinking mode (default: True)'
    )
    parser.add_argument(
        '--no_think', action='store_true',
        help='Disable thinking mode'
    )
    parser.add_argument(
        '--understanding_output', action='store_true',
        help='Text-only output (disable visual thinking)'
    )
    parser.add_argument(
        '--use_tools', action='store_true',
        help='Use tool-enabled prompt template'
    )
    parser.add_argument(
        '--config', type=str, default='default',
        choices=list(CONFIG_PRESETS.keys()),
        help=f'Inference config preset: {list(CONFIG_PRESETS.keys())} (default: default)'
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        task=args.task,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        models_per_gpu=args.models_per_gpu,
        max_samples=args.max_samples,
        split=args.split,
        think=args.think and not args.no_think,
        understanding_output=args.understanding_output,
        max_mem_per_gpu=args.max_mem_per_gpu,
        use_tools=args.use_tools,
        config_preset=args.config,
    )


if __name__ == '__main__':
    main()
