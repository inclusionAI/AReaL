"""
VisPuzzle / Spatial_Navigation Benchmark Inference and Evaluation with ThinkMorph vLLM
"""

import os
import json
import re
from typing import Optional
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from thinkmorph_vllm import VLLMInterleavedInference


def extract_answer(text: str, answer_type: str = "choice") -> str:
    """Extract answer from model output.

    Args:
        text: Model output text
        answer_type: "choice" for A/B/C/D, "direction" for direction sequences like D,R,L,U
    """
    # Try to find answer in \boxed{} format
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()

    # Try to find answer in <answer>...</answer> tags
    answer_tag = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_tag:
        return answer_tag.group(1).strip()

    if answer_type == "direction":
        # Extract direction sequence (e.g., "D,R,L,U" or "D R L U" or "DRLU")
        # Look for patterns like D,R or D R or consecutive UDLR
        dir_pattern = re.search(r'([UDLR](?:[,\s]*[UDLR])*)', text.upper())
        if dir_pattern:
            # Normalize to comma-separated format
            dirs = re.findall(r'[UDLR]', dir_pattern.group(1))
            return ','.join(dirs)
    else:
        # Try to find single letter answer (A/B/C/D)
        match = re.search(r'\b([A-D])\b', text.split('\n')[-1])
        if match:
            return match.group(1)

    # Return last non-empty line as fallback
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def normalize_direction_answer(answer: str) -> str:
    """Normalize direction answer for comparison."""
    # Extract only direction characters and uppercase
    dirs = re.findall(r'[UDLRudlr]', answer)
    return ','.join(d.upper() for d in dirs)


def evaluate_vispuzzle(
    model_path: str = "ThinkMorph/ThinkMorph-7B",
    dataset_path: Optional[str] = None,
    dataset_name: str = "vispuzzle",
    output_dir: str = "./vispuzzle_results",
    max_mem_per_gpu: str = "40GiB",
    max_samples: Optional[int] = None,
    split: str = "test",
):
    """
    Run inference and evaluation on VisPuzzle or Spatial_Navigation benchmark.

    Args:
        model_path: Path to ThinkMorph model (local path or HuggingFace model ID)
        dataset_path: Path to dataset (local path, or None to use HuggingFace)
        dataset_name: Dataset type ("vispuzzle" or "spatial_navigation")
        output_dir: Directory to save results
        max_mem_per_gpu: Maximum GPU memory per device (e.g., "40GiB", "80GiB")
        max_samples: Limit number of samples (None for all)
        split: Dataset split to use ("train" or "test")
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
        # Load from local path
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_path.endswith('.parquet'):
            dataset = load_dataset("parquet", data_files=dataset_path, split="train")
        else:
            # Assume it's a HuggingFace dataset ID or local directory
            dataset = load_dataset(dataset_path, split=split)
    else:
        if dataset_name.lower() == "spatial_navigation":
            dataset = load_dataset("/storage/openpsi/data/lcy_image_edit/Spatial_Navigation", split="train")
        else:
            dataset = load_dataset("ThinkMorph/VisPuzzle", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Loaded {len(dataset)} samples")

    # Initialize inferencer (manual loading, not vLLM)
    print("Initializing ThinkMorph...")
    inferencer = VLLMInterleavedInference(
        model_path=model_path,
        max_mem_per_gpu=max_mem_per_gpu,
    )

    # Helper to get field value with fallback
    def get_field(sample, field_keys):
        if isinstance(field_keys, str):
            return sample.get(field_keys)
        for key in field_keys:
            if key in sample:
                return sample.get(key)
        return None

    # Run inference
    results = []
    correct = 0

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        # Get image and question using field mapping
        image = get_field(sample, field_map["image"])
        question = get_field(sample, field_map["question"]) or ""
        gt_answer = get_field(sample, field_map["answer"]) or ""
        sample_id = get_field(sample, field_map["id"]) or str(idx)

        # Run inference with thinking
        outputs = inferencer.infer_single(
            image=image,
            text=question,
            think=True,
            understanding_output=False
        )

        # Extract text outputs
        text_outputs = [o for o in outputs if isinstance(o, str)]
        full_response = "\n".join(text_outputs)
        pred_answer = extract_answer(full_response, answer_type=answer_type)

        # Check correctness
        if answer_type == "direction":
            # Normalize both answers for direction comparison
            pred_normalized = normalize_direction_answer(pred_answer)
            gt_normalized = normalize_direction_answer(str(gt_answer))
            is_correct = pred_normalized == gt_normalized
        else:
            is_correct = pred_answer.upper() == str(gt_answer).upper()
        if is_correct:
            correct += 1

        # Save result
        result = {
            "id": sample_id,
            "question": question,
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
            "correct": is_correct,
            "full_response": full_response,
        }
        results.append(result)

        # Save generated images
        img_outputs = [o for o in outputs if isinstance(o, Image.Image)]
        for i, img in enumerate(img_outputs):
            img.save(os.path.join(output_dir, f"{sample_id}_gen_{i}.png"))

    # Calculate accuracy
    accuracy = correct / len(results) * 100

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"{dataset_name} Evaluation Complete")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"Results saved to: {results_path}")
    print(f"{'='*50}")

    return accuracy, results


if __name__ == "__main__":
    # Example 1: VisPuzzle from HuggingFace
    # evaluate_vispuzzle(
    #     model_path="ThinkMorph/ThinkMorph-7B",
    #     dataset_name="vispuzzle",
    #     output_dir="./vispuzzle_results",
    #     max_mem_per_gpu="40GiB",
    #     max_samples=None,
    # )

    # Example 2: VisPuzzle from local paths
    # evaluate_vispuzzle(
    #     model_path="D:/models/ThinkMorph-7B",           # 本地模型路径
    #     dataset_path="D:/datasets/VisPuzzle",           # 本地数据集路径 (目录、.json、.jsonl 或 .parquet)
    #     dataset_name="vispuzzle",
    #     output_dir="./vispuzzle_results",
    #     max_mem_per_gpu="40GiB",
    #     max_samples=10,  # Set to e.g. 10 for quick test
    # )

    # Example 3: Spatial_Navigation from HuggingFace
    evaluate_vispuzzle(
        model_path="/storage/openpsi/models/lcy_image_edit/ThinkMorph-7B",           # 本地模型路径
        dataset_name="spatial_navigation",              # 使用 Spatial_Navigation 数据集
        output_dir="/storage/openpsi/data/lcy_image_edit/spatial_nav_results",
        max_mem_per_gpu="40GiB",
        max_samples=10,  # Set to e.g. 10 for quick test
        split="train",                                  # Spatial_Navigation 只有 train split
    )

    # Example 4: Spatial_Navigation from local path
    # evaluate_vispuzzle(
    #     model_path="D:/models/ThinkMorph-7B",
    #     dataset_path="D:/datasets/Spatial_Navigation",  # 或者 HuggingFace ID: "ThinkMorph/Spatial_Navigation"
    #     dataset_name="spatial_navigation",
    #     output_dir="./spatial_nav_results",
    #     max_mem_per_gpu="40GiB",
    #     max_samples=10,
    #     split="train",
    # )
