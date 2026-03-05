"""
VisPuzzle Benchmark Inference and Evaluation with ThinkMorph vLLM
"""

import os
import json
import re
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
    output_dir: str = "./vispuzzle_results",
    tensor_parallel_size: int = 1,
    max_samples: int = None,
):
    """
    Run inference and evaluation on VisPuzzle benchmark.

    Args:
        model_path: Path to ThinkMorph model
        output_dir: Directory to save results
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_samples: Limit number of samples (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load VisPuzzle dataset
    print("Loading VisPuzzle dataset...")
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
    evaluate_vispuzzle(
        model_path="ThinkMorph/ThinkMorph-7B",
        output_dir="./vispuzzle_results",
        tensor_parallel_size=1,
        max_samples=None,  # Set to e.g. 10 for quick test
    )
