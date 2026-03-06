"""
VisPuzzle Benchmark Inference and Evaluation with ThinkMorph vLLM
"""

import os
import json
import re
from typing import Optional
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from thinkmorph_vllm import VLLMInterleavedInference


def extract_answer(text: str) -> str:
    """Extract answer from model output (e.g., A, B, C, D)."""
    # Try to find answer in \boxed{} format
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()

    # Try to find single letter answer (A/B/C/D)
    match = re.search(r'\b([A-D])\b', text.split('\n')[-1])
    if match:
        return match.group(1)

    # Return last non-empty line as fallback
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def evaluate_vispuzzle(
    model_path: str = "ThinkMorph/ThinkMorph-7B",
    dataset_path: Optional[str] = None,
    output_dir: str = "./vispuzzle_results",
    tensor_parallel_size: int = 1,
    max_samples: Optional[int] = None,
):
    """
    Run inference and evaluation on VisPuzzle benchmark.

    Args:
        model_path: Path to ThinkMorph model (local path or HuggingFace model ID)
        dataset_path: Path to dataset (local path, or None to use HuggingFace)
        output_dir: Directory to save results
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_samples: Limit number of samples (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load VisPuzzle dataset
    print("Loading VisPuzzle dataset...")
    if dataset_path:
        # Load from local path
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_path.endswith('.parquet'):
            dataset = load_dataset("parquet", data_files=dataset_path, split="train")
        else:
            # Assume it's a local directory in HuggingFace format
            dataset = load_dataset(dataset_path, split="test")
    else:
        dataset = load_dataset("ThinkMorph/VisPuzzle", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"Loaded {len(dataset)} samples")

    # Initialize inferencer
    print("Initializing ThinkMorph vLLM...")
    inferencer = VLLMInterleavedInference(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
    )

    # Run inference
    results = []
    correct = 0

    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        # Get image and question
        image = sample.get("image")
        question = sample.get("question", sample.get("prompt", ""))
        gt_answer = sample.get("answer", sample.get("label", ""))
        sample_id = sample.get("id", str(idx))

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
        pred_answer = extract_answer(full_response)

        # Check correctness
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
    print(f"VisPuzzle Evaluation Complete")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"Results saved to: {results_path}")
    print(f"{'='*50}")

    return accuracy, results


if __name__ == "__main__":
    # Example 1: Load from HuggingFace (default)
    # evaluate_vispuzzle(
    #     model_path="ThinkMorph/ThinkMorph-7B",
    #     output_dir="./vispuzzle_results",
    #     tensor_parallel_size=1,
    #     max_samples=None,
    # )

    # Example 2: Load from local paths
    evaluate_vispuzzle(
        model_path="D:/models/ThinkMorph-7B",           # 本地模型路径
        dataset_path="D:/datasets/VisPuzzle",           # 本地数据集路径 (目录、.json、.jsonl 或 .parquet)
        output_dir="./vispuzzle_results",
        tensor_parallel_size=1,
        max_samples=10,  # Set to e.g. 10 for quick test
    )
