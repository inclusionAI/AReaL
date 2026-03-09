#!/usr/bin/env python
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# ThinkMorph multi-GPU Data Parallel inference command line script
#
# Usage:
#   python -m geo_edit.models.thinkmorph_vllm.run_dp \
#       --model_path /path/to/ThinkMorph-7B \
#       --dataset_path ./test_dataset.parquet \
#       --output_dir ./outputs \
#       --num_gpus 8 \
#       --batch_size 16

import os
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="ThinkMorph multi-GPU Data Parallel inference"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to ThinkMorph model directory'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to HuggingFace dataset (parquet file or dataset name)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=8,
        help='Number of GPUs to use (default: 8)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Total batch size across all GPUs (default: 8)'
    )
    parser.add_argument(
        '--image_field',
        type=str,
        default='image',
        help='Image field name in dataset (default: image)'
    )
    parser.add_argument(
        '--text_field',
        type=str,
        default='text',
        help='Text field name in dataset (default: text)'
    )
    parser.add_argument(
        '--id_field',
        type=str,
        default='id',
        help='ID field name in dataset (default: id)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to use (default: test)'
    )
    parser.add_argument(
        '--think',
        action='store_true',
        default=True,
        help='Enable thinking mode (default: True)'
    )
    parser.add_argument(
        '--no_think',
        action='store_true',
        help='Disable thinking mode'
    )
    parser.add_argument(
        '--understanding_output',
        action='store_true',
        default=True,
        help='Text-only output mode (default: True)'
    )
    parser.add_argument(
        '--max_mem_per_gpu',
        type=str,
        default='140GiB',
        help='Maximum memory per GPU (default: 140GiB)'
    )

    args = parser.parse_args()

    # Handle think flag
    think = args.think and not args.no_think

    # Load dataset
    logger.info(f"Loading dataset from: {args.dataset_path}")

    from datasets import load_dataset, Dataset

    if args.dataset_path.endswith('.parquet'):
        dataset = Dataset.from_parquet(args.dataset_path)
    elif args.dataset_path.endswith('.json') or args.dataset_path.endswith('.jsonl'):
        dataset = Dataset.from_json(args.dataset_path)
    elif os.path.isdir(args.dataset_path):
        # Load from local directory
        dataset = load_dataset(args.dataset_path, split=args.split)
    else:
        # Load from HuggingFace Hub
        dataset = load_dataset(args.dataset_path, split=args.split)

    # Limit samples if specified
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Initialize DP inferencer
    from .inference_dp import ThinkMorphDP

    dp = ThinkMorphDP(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_mem_per_gpu=args.max_mem_per_gpu,
    )

    # Run inference
    logger.info("Starting inference...")
    results = dp.infer_dataset(
        dataset,
        image_field=args.image_field,
        text_field=args.text_field,
        id_field=args.id_field,
        think=think,
        understanding_output=args.understanding_output,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results
    output_path = os.path.join(args.output_dir, 'results.json')

    # Convert outputs to JSON-serializable format
    json_results = []
    for r in results:
        json_result = {
            'id': r.get('id'),
            'original_idx': r.get('original_idx'),
        }
        # Handle output (may contain strings or images)
        output = r.get('output', [])
        if isinstance(output, list):
            json_output = []
            for item in output:
                if isinstance(item, str):
                    json_output.append({'type': 'text', 'content': item})
                else:
                    # PIL Image - save to file
                    img_filename = f"{r.get('id', r.get('original_idx'))}_img.png"
                    img_path = os.path.join(args.output_dir, img_filename)
                    try:
                        item.save(img_path)
                        json_output.append({'type': 'image', 'path': img_filename})
                    except Exception as e:
                        json_output.append({'type': 'error', 'message': str(e)})
            json_result['output'] = json_output
        else:
            json_result['output'] = str(output)

        json_results.append(json_result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Processed {len(results)} samples")


if __name__ == '__main__':
    main()
