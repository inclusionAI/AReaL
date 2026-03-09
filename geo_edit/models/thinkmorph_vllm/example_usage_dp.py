"""
ThinkMorph Multi-GPU DP Inference and Evaluation

Example usage for testing batch inference with Data Parallel.
Similar to example_usage.py but uses ThinkMorphDP for multi-GPU inference.
"""

import os
import json
import re
from typing import Optional
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from geo_edit.models.thinkmorph_vllm import ThinkMorphDP, ThinkMorphBatchInference, ThinkMorphInference


def extract_answer(text: str, answer_type: str = "choice") -> str:
    """Extract answer from model output."""
    # Try to find answer in <answer>...</answer> tags
    answer_tag = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_tag:
        return answer_tag.group(1).strip()

    # Try to find answer in \boxed{} format
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()

    if answer_type == "direction":
        dir_pattern = re.search(r'([UDLR](?:[,\s]*[UDLR])*)', text.upper())
        if dir_pattern:
            dirs = re.findall(r'[UDLR]', dir_pattern.group(1))
            return ','.join(dirs)
    else:
        match = re.search(r'\b([A-D])\b', text.split('\n')[-1])
        if match:
            return match.group(1)

    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def normalize_direction_answer(answer: str) -> str:
    """Normalize direction answer for comparison."""
    dirs = re.findall(r'[UDLRudlr]', answer)
    return ','.join(d.upper() for d in dirs)


def evaluate_with_dp(
    model_path: str = "ThinkMorph/ThinkMorph-7B",
    dataset_path: Optional[str] = None,
    dataset_name: str = "vispuzzle",
    output_dir: str = "./dp_results",
    num_gpus: int = 8,
    max_mem_per_gpu: str = "140GiB",
    max_samples: Optional[int] = None,
    split: str = "test",
    understanding_output: bool = False,
):
    """
    Run inference and evaluation using multi-GPU Data Parallel.

    Each GPU runs one sample at a time with full interleaved text-image
    generation support (visual thinking).

    Args:
        model_path: Path to ThinkMorph model
        dataset_path: Path to dataset (local path, or None to use HuggingFace)
        dataset_name: Dataset type ("vispuzzle" or "spatial_navigation")
        output_dir: Directory to save results
        num_gpus: Number of GPUs to use
        max_mem_per_gpu: Maximum GPU memory per device
        max_samples: Limit number of samples (None for all)
        split: Dataset split to use
        understanding_output: If True, only generate text; if False, enable visual thinking
    """
    os.makedirs(output_dir, exist_ok=True)

    # Dataset field mapping
    FIELD_MAP = {
        "vispuzzle": {
            "image": "image",
            "question": ["question", "prompt"],
            "answer": ["answer", "label"],
            "id": ["id"],
            "answer_type": "choice",
        },
        "spatial_navigation": {
            "image": "problem_image_0",
            "question": ["question"],
            "answer": ["answer"],
            "id": ["pid"],
            "answer_type": "direction",
        },
    }

    field_map = FIELD_MAP.get(dataset_name.lower(), FIELD_MAP["vispuzzle"])
    answer_type = field_map["answer_type"]

    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_path:
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_path.endswith('.parquet'):
            dataset = load_dataset("parquet", data_files=dataset_path, split="train")
        else:
            dataset = load_dataset(dataset_path, split=split)
    else:
        if dataset_name.lower() == "spatial_navigation":
            dataset = load_dataset("/storage/openpsi/data/lcy_image_edit/Spatial_Navigation", split="train")
        else:
            dataset = load_dataset("ThinkMorph/VisPuzzle", split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Loaded {len(dataset)} samples")

    # Helper to get field value
    def get_field(sample, field_keys):
        if isinstance(field_keys, str):
            return sample.get(field_keys)
        for key in field_keys:
            if key in sample:
                return sample.get(key)
        return None

    # Prepare samples for DP inference
    samples = []
    for idx, sample in enumerate(dataset):
        image = get_field(sample, field_map["image"])
        question = get_field(sample, field_map["question"]) or ""
        gt_answer = get_field(sample, field_map["answer"]) or ""
        sample_id = get_field(sample, field_map["id"]) or str(idx)

        samples.append({
            'image': image,
            'text': question,
            'id': sample_id,
            'gt_answer': gt_answer,
        })

    # Initialize DP inferencer
    print(f"Initializing ThinkMorphDP with {num_gpus} GPUs...")
    dp = ThinkMorphDP(
        model_path=model_path,
        num_gpus=num_gpus,
        max_mem_per_gpu=max_mem_per_gpu,
    )

    # Run DP inference with full interleaved generation
    print(f"Running DP inference (understanding_output={understanding_output})...")
    dp_results = dp.infer_samples(
        samples,
        think=True,
        understanding_output=understanding_output,
    )

    # Process results
    results = []
    correct = 0

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
        if answer_type == "direction":
            pred_normalized = normalize_direction_answer(pred_answer)
            gt_normalized = normalize_direction_answer(str(gt_answer))
            is_correct = pred_normalized == gt_normalized
        else:
            is_correct = pred_answer.upper() == str(gt_answer).upper()

        if is_correct:
            correct += 1

        result = {
            "id": sample.get('id'),
            "question": sample.get('text'),
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
            "correct": is_correct,
            "full_response": full_response,
        }
        results.append(result)

        # Save generated images
        img_outputs = [o for o in outputs if isinstance(o, Image.Image)]
        for i, img in enumerate(img_outputs):
            img.save(os.path.join(output_dir, f"{sample.get('id')}_gen_{i}.png"))

    # Calculate accuracy
    accuracy = correct / len(results) * 100 if results else 0

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(results),
            "num_gpus": num_gpus,
            "understanding_output": understanding_output,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"{dataset_name} DP Evaluation Complete")
    print(f"GPUs: {num_gpus}, Visual thinking: {not understanding_output}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"Results saved to: {results_path}")
    print(f"{'='*50}")

    return accuracy, results


def evaluate_single_gpu_batch(
    model_path: str = "ThinkMorph/ThinkMorph-7B",
    dataset_path: Optional[str] = None,
    dataset_name: str = "vispuzzle",
    output_dir: str = "./batch_results",
    max_mem_per_gpu: str = "140GiB",
    max_samples: Optional[int] = None,
    split: str = "test",
):
    """
    Run inference using single GPU with batch processing.

    For testing batch inference without multi-process DP.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Dataset field mapping (same as above)
    FIELD_MAP = {
        "vispuzzle": {
            "image": "image",
            "question": ["question", "prompt"],
            "answer": ["answer", "label"],
            "id": ["id"],
            "answer_type": "choice",
        },
        "spatial_navigation": {
            "image": "problem_image_0",
            "question": ["question"],
            "answer": ["answer"],
            "id": ["pid"],
            "answer_type": "direction",
        },
    }

    field_map = FIELD_MAP.get(dataset_name.lower(), FIELD_MAP["vispuzzle"])
    answer_type = field_map["answer_type"]

    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_path:
        if dataset_path.endswith('.parquet'):
            dataset = load_dataset("parquet", data_files=dataset_path, split="train")
        else:
            dataset = load_dataset(dataset_path, split=split)
    else:
        if dataset_name.lower() == "spatial_navigation":
            dataset = load_dataset("/storage/openpsi/data/lcy_image_edit/Spatial_Navigation", split="train")
        else:
            dataset = load_dataset("ThinkMorph/VisPuzzle", split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Loaded {len(dataset)} samples")

    def get_field(sample, field_keys):
        if isinstance(field_keys, str):
            return sample.get(field_keys)
        for key in field_keys:
            if key in sample:
                return sample.get(key)
        return None

    # Prepare samples
    samples = []
    for idx, sample in enumerate(dataset):
        image = get_field(sample, field_map["image"])
        question = get_field(sample, field_map["question"]) or ""
        gt_answer = get_field(sample, field_map["answer"]) or ""
        sample_id = get_field(sample, field_map["id"]) or str(idx)

        samples.append({
            'image': image,
            'text': question,
            'id': sample_id,
            'gt_answer': gt_answer,
        })

    # Initialize batch inferencer (single GPU)
    print("Initializing ThinkMorphBatchInference (single GPU)...")
    inferencer = ThinkMorphBatchInference(
        model_path=model_path,
        max_mem_per_gpu=max_mem_per_gpu,
    )

    # Run batch inference
    print("Running batch inference...")
    batch_results = inferencer.infer_batch_parallel(
        samples,
        think=True,
        understanding_output=True,
    )

    # Process results
    results = []
    correct = 0

    for idx, (sample, outputs) in enumerate(zip(samples, batch_results)):
        text_outputs = [o for o in outputs if isinstance(o, str)]
        full_response = "\n".join(text_outputs)
        pred_answer = extract_answer(full_response, answer_type=answer_type)

        gt_answer = sample.get('gt_answer', '')
        if answer_type == "direction":
            pred_normalized = normalize_direction_answer(pred_answer)
            gt_normalized = normalize_direction_answer(str(gt_answer))
            is_correct = pred_normalized == gt_normalized
        else:
            is_correct = pred_answer.upper() == str(gt_answer).upper()

        if is_correct:
            correct += 1

        result = {
            "id": sample.get('id'),
            "question": sample.get('text'),
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
            "correct": is_correct,
            "full_response": full_response,
        }
        results.append(result)

    accuracy = correct / len(results) * 100 if results else 0

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"{dataset_name} Batch Evaluation Complete")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"Results saved to: {results_path}")
    print(f"{'='*50}")

    return accuracy, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ThinkMorph DP/Batch Inference Test")
    parser.add_argument('--mode', type=str, default='dp', choices=['dp', 'batch'],
                        help='Inference mode: dp (multi-GPU with visual thinking) or batch (single GPU text-only)')
    parser.add_argument('--model_path', type=str,
                        default='/storage/openpsi/models/lcy_image_edit/ThinkMorph-7B',
                        help='Path to ThinkMorph model')
    parser.add_argument('--dataset_name', type=str, default='spatial_navigation',
                        choices=['vispuzzle', 'spatial_navigation'],
                        help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./dp_results',
                        help='Output directory')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs (for DP mode)')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='Maximum samples to process')
    parser.add_argument('--max_mem_per_gpu', type=str, default='140GiB',
                        help='Maximum memory per GPU')
    parser.add_argument('--understanding_output', action='store_true',
                        help='Text-only output (disable visual thinking)')

    args = parser.parse_args()

    if args.mode == 'dp':
        # Multi-GPU DP inference with full interleaved generation
        evaluate_with_dp(
            model_path=args.model_path,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            num_gpus=args.num_gpus,
            max_mem_per_gpu=args.max_mem_per_gpu,
            max_samples=args.max_samples,
            split="train" if args.dataset_name == "spatial_navigation" else "test",
            understanding_output=args.understanding_output,
        )
    else:
        # Single GPU batch inference (text-only)
        evaluate_single_gpu_batch(
            model_path=args.model_path,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            max_mem_per_gpu=args.max_mem_per_gpu,
            max_samples=args.max_samples,
            split="train" if args.dataset_name == "spatial_navigation" else "test",
        )
