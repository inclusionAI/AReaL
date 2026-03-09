#!/usr/bin/env python
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# ThinkMorph generic inference script with task registry support
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
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from PIL import Image
from datasets import load_dataset, Dataset
from tqdm import tqdm

from geo_edit.datasets.task_registry import DATASET_SPECS, get_dataset_spec, DatasetSpec
from geo_edit.models.thinkmorph_vllm import ThinkMorphDP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_answer(text: str, answer_type: str = "text") -> str:
    """
    Extract answer from model output.

    Supports multiple formats:
    - <answer>...</answer> tags
    - \\boxed{...} format
    - Direction sequences (UDLR)
    - Multiple choice (A-D)
    """
    result = text

    # Try to find answer in <answer>...</answer> tags
    answer_tag = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL | re.IGNORECASE)
    if answer_tag:
        result = answer_tag.group(1).strip()

    # Try to find answer in \boxed{} format
    boxed = re.search(r'\\boxed\{([^}]+)\}', result)
    if boxed:
        result = boxed.group(1).strip()
        if answer_type == "direction":
            dirs = re.findall(r'[UDLR]', result.upper())
            return ','.join(dirs)
        return result

    if answer_tag:
        return result

    # Fallback extraction based on answer type
    if answer_type == "direction":
        dir_pattern = re.search(r'([UDLR](?:[,\s]*[UDLR])*)', text.upper())
        if dir_pattern:
            dirs = re.findall(r'[UDLR]', dir_pattern.group(1))
            return ','.join(dirs)
    elif answer_type == "choice":
        match = re.search(r'\b([A-D])\b', text.split('\n')[-1])
        if match:
            return match.group(1)
    elif answer_type == "number":
        # Try to extract a number
        match = re.search(r'[-+]?\d*\.?\d+', result)
        if match:
            return match.group(0)

    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def normalize_answer(answer: str, answer_type: str = "text") -> str:
    """Normalize answer for comparison."""
    answer = str(answer).strip()

    if answer_type == "direction":
        dirs = re.findall(r'[UDLRudlr]', answer)
        return ','.join(d.upper() for d in dirs)
    elif answer_type == "choice":
        return answer.upper()
    elif answer_type == "number":
        try:
            return str(float(answer))
        except ValueError:
            return answer
    else:
        return answer.lower()


def check_answer(pred: str, gt: str, answer_type: str = "text") -> bool:
    """Check if prediction matches ground truth."""
    pred_norm = normalize_answer(pred, answer_type)
    gt_norm = normalize_answer(gt, answer_type)

    if answer_type == "number":
        try:
            return abs(float(pred_norm) - float(gt_norm)) < 1e-6
        except ValueError:
            return pred_norm == gt_norm

    return pred_norm == gt_norm


def infer_answer_type(task_name: str) -> str:
    """Infer answer type from task name."""
    task_lower = task_name.lower()

    if "srn" in task_lower or "navigation" in task_lower or "path" in task_lower:
        return "direction"
    elif "choice" in task_lower or "qa" in task_lower:
        return "choice"
    elif "counting" in task_lower:
        return "number"
    else:
        return "text"


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
):
    """
    Run ThinkMorph inference on a registered task.

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
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get dataset spec from registry
    try:
        spec = get_dataset_spec(task)
        logger.info(f"Using registered task: {task}")
    except KeyError:
        logger.error(f"Unknown task: {task}")
        logger.info(f"Available tasks: {list(DATASET_SPECS.keys())}")
        return None

    # Infer answer type
    answer_type = infer_answer_type(task)
    logger.info(f"Answer type: {answer_type}")

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
            'original_item': dict(item),  # Keep original for metadata
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
    )

    # Run inference
    logger.info(f"Running inference (think={think}, understanding_output={understanding_output})...")
    dp_results = dp.infer_samples(
        samples,
        think=think,
        understanding_output=understanding_output,
    )

    # Process results
    results = []
    correct = 0
    total = 0

    for dp_result in tqdm(dp_results, desc="Processing results"):
        idx = dp_result['original_idx']
        sample = samples[idx]
        outputs = dp_result.get('output', [])

        # Extract text outputs
        text_outputs = [o for o in outputs if isinstance(o, str)]
        full_response = "\n".join(text_outputs)
        pred_answer = extract_answer(full_response, answer_type=answer_type)

        # Check correctness
        gt_answer = sample.get('gt_answer', '')
        is_correct = check_answer(pred_answer, str(gt_answer), answer_type)

        if is_correct:
            correct += 1
        total += 1

        result = {
            "id": sample.get('id'),
            "question": sample.get('text'),
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
            "correct": is_correct,
            "full_response": full_response,
        }

        # Add task-specific metadata
        task_kwargs = spec.build_task_kwargs(sample.get('original_item', {}))
        if task_kwargs:
            result["metadata"] = task_kwargs

        results.append(result)

        # Save generated images
        img_outputs = [o for o in outputs if isinstance(o, Image.Image)]
        for i, img in enumerate(img_outputs):
            img_path = os.path.join(output_dir, f"{sample.get('id')}_gen_{i}.png")
            img.save(img_path)

    # Calculate accuracy
    accuracy = correct / total * 100 if total > 0 else 0

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": task,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "num_gpus": num_gpus,
            "models_per_gpu": models_per_gpu,
            "total_parallelism": total_parallelism,
            "think": think,
            "understanding_output": understanding_output,
            "answer_type": answer_type,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"GPUs: {num_gpus}, Models/GPU: {models_per_gpu}, Total parallelism: {total_parallelism}")
    print(f"Visual thinking: {not understanding_output}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")

    return accuracy, results


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
    )


if __name__ == '__main__':
    main()
