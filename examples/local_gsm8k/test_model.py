#!/usr/bin/env python3
"""
Test and compare model performance before and after training.
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import from AReaL, fallback to simple version
try:
    from areal.reward.math_parser import process_results
except ImportError:
    # Simple fallback parser
    import re
    
    def process_results(completions, answer):
        """Simple parser to extract final answer."""
        # Extract answer from ground truth
        try:
            gt_answer = float(re.findall(r'-?\d+\.?\d*', answer.split("####")[-1])[-1])
        except:
            return [False]
        
        # Check completions
        results = []
        for completion in completions:
            try:
                pred_answer = float(re.findall(r'-?\d+\.?\d*', completion.split("####")[-1])[-1])
                results.append(pred_answer == gt_answer)
            except:
                results.append(False)
        
        return results if len(results) > 0 else [False]


def test_model(model_path: str, max_samples: int = 10, max_new_tokens: int = 512):
    """Test the model on GSM8K samples."""
    
    print(f"\n{'='*60}")
    print(f"Testing model: {model_path}")
    print(f"{'='*60}\n")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use CPU for more stable inference
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    torch_dtype = torch.float32  # Use float32 for CPU
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    
    # Load GSM8K test set
    from datasets import load_dataset
    
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    results = []
    correct = 0
    
    for i, sample in enumerate(dataset.select(range(min(max_samples, len(dataset))))):
        question = sample["question"]
        correct_answer = sample["answer"]
        
        # Format prompt
        prompt = f"{question}\nPlease provide your final answer within \\boxed{{}}."
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate with greedy decoding for stability
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Use greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Check answer
        is_correct = process_results([generated_text], correct_answer)[0]
        
        if is_correct:
            correct += 1
        
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "generated": generated_text,
            "correct": is_correct,
        })
        
        print(f"\n--- Question {i+1} ---")
        print(f"Question: {question[:100]}...")
        print(f"Generated: {generated_text[:200]}...")
        print(f"Correct Answer: {correct_answer[:100]}...")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    
    accuracy = correct / len(results) * 100
    print(f"\n{'='*60}")
    print(f"ACCURACY: {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"{'='*60}\n")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "results": results,
    }


def compare_models(base_model: str, trained_model: str, max_samples: int = 10):
    """Compare base model and trained model."""
    
    print(f"\n{'#'*60}")
    print("MODEL COMPARISON")
    print(f"{'#'*60}\n")
    
    # Test base model
    base_results = test_model(base_model, max_samples=max_samples)
    
    # Test trained model
    trained_results = test_model(trained_model, max_samples=max_samples)
    
    # Print comparison
    print(f"\n{'#'*60}")
    print("COMPARISON SUMMARY")
    print(f"{'#'*60}")
    print(f"Base Model Accuracy:    {base_results['accuracy']:.2f}%")
    print(f"Trained Model Accuracy: {trained_results['accuracy']:.2f}%")
    improvement = trained_results['accuracy'] - base_results['accuracy']
    print(f"Improvement:            {improvement:+.2f}%")
    print(f"{'#'*60}\n")
    
    # Save results
    comparison = {
        "base_model": base_model,
        "trained_model": trained_model,
        "base_results": base_results,
        "trained_results": trained_results,
        "improvement": improvement,
    }
    
    with open("model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("Results saved to model_comparison.json")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Test and compare models")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path to test",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        default="./outputs/gsm8k-local",
        help="Trained model path",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base and trained models",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of samples to test",
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.base_model, args.trained_model, max_samples=args.max_samples)
    elif args.model:
        test_model(args.model, max_samples=args.max_samples)
    else:
        # Default: test trained model
        test_model(args.trained_model, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

