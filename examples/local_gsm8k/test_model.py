#!/usr/bin/env python3
"""
Test and compare model performance before and after training.
"""

import argparse
import json
import os
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from areal.reward.math_parser import process_results, parse_digits

def test_model(
    model_path: str,
    max_samples: int = 10,
    max_new_tokens: int = 1024,
    log_dir: str | None = None,
    test_all: bool = False,
):
    """Test the model on GSM8K samples."""
    
    # Prepare logging
    if log_dir is None:
        log_dir = os.path.join("examples", "local_gsm8k", "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_{ts}.log")

    def _log(msg: str):
        with open(log_path, "a") as lf:
            lf.write(msg + "\n")

    _log(f"\n{'='*60}")
    _log(f"Testing model: {model_path}")
    _log(f"Log file: {log_path}")
    _log(f"{'='*60}\n")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use CPU for more stable inference
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    _log(f"Using device: {device}")
    
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
    
    # Determine how many samples to test
    if test_all or max_samples == -1:
        num_samples = len(dataset)
        _log(f"Testing on FULL dataset: {num_samples} samples")
    else:
        num_samples = min(max_samples, len(dataset))
        _log(f"Testing on {num_samples} samples (out of {len(dataset)} total)")
    
    results = []
    correct = 0
    
    pbar = tqdm(enumerate(dataset.select(range(num_samples))), total=num_samples)
    for i, sample in pbar:
        question = sample["question"]
        
        correct_answer = sample["answer"]
        hashes_idx = correct_answer.find("#### ")
        if hashes_idx != -1:
            correct_answer = correct_answer[:hashes_idx] + "\\boxed{" + correct_answer[hashes_idx + 5 :] + "}"
        
        # Format prompt - Use GSM8K format (no \boxed{} prompt)
        # This matches the training format
        prompt = f"{question}\n"
        
        # Tokenize
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate with greedy decoding for stability
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            outputs = model.generate(inputs, **gen_kwargs)
        
        # Get the generated token IDs (excluding the prompt)
        input_len = inputs.shape[-1]
        generated_token_ids = outputs[0][input_len:]
        
        # Check if EOS was generated
        eos_was_generated = tokenizer.eos_token_id in generated_token_ids.tolist()
        eos_position = None
        if eos_was_generated:
            eos_positions = [i for i, tok_id in enumerate(generated_token_ids.tolist()) if tok_id == tokenizer.eos_token_id]
            if eos_positions:
                eos_position = eos_positions[0]  # First EOS position
        
        # Decode with skipping special tokens for final output
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        # Determine stop reason
        new_tokens = len(generated_token_ids)
        if new_tokens >= max_new_tokens:
            stop_reason = "max_new_tokens"
        elif eos_was_generated:
            stop_reason = f"eos_token (at position {eos_position})"
        else:
            stop_reason = "unknown"

        parser_result, extracted_answers = process_results(generated_text, correct_answer)
        gt_extracted, sol_extracted = [parse_digits(ans) for ans in extracted_answers]

        is_correct = bool(parser_result)

        if is_correct:
            correct += 1
        
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "generated": generated_text,
            "gt_extracted": gt_extracted,
            "sol_extracted": sol_extracted,
            "correct": is_correct,
        })
        
        _log(f"\n--- Question {i+1} ---")
        _log(f"Question: {question}")
        _log(f"Stop reason: {stop_reason}; new_tokens: {new_tokens}/{max_new_tokens}")
        _log(f"EOS detected: {eos_was_generated}")
        
        log_ready_generated_text = generated_text.replace("\n", "\n\t").strip()
        log_ready_correct_answer = correct_answer.replace("\n", "\n\t").strip()
        _log(f"Generated Answer:\n\t{log_ready_generated_text}")
        _log(f"Correct Answer (full):\n\t{log_ready_correct_answer}")
        
        _log(f"Extracted -> GT: {gt_extracted} | Sol: {sol_extracted}")
        _log(f"Result: {'[CORRECT]' if is_correct else '[INCORRECT]'}")

        pbar.set_postfix({
            "Correct": correct,
            "Total": i + 1,
            "Accuracy (%)": f"{(correct / (i + 1) * 100):.2f}",
        })
    
    accuracy = correct / len(results) * 100
    _log(f"\n{'='*60}")
    _log(f"ACCURACY: {accuracy:.2f}% ({correct}/{len(results)})")
    _log(f"Log saved to: {log_path}")
    _log(f"{'='*60}\n")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "results": results,
    }


def compare_models(base_model: str, trained_model: str, max_samples: int = 10, test_all: bool = False):
    """Compare base model and trained model."""
    
    print(f"\n{'#'*60}")
    print("MODEL COMPARISON")
    print(f"{'#'*60}\n")
    
    # Test base model
    base_results = test_model(base_model, max_samples=max_samples, test_all=test_all)
    
    # Test trained model
    trained_results = test_model(trained_model, max_samples=max_samples, test_all=test_all)
    
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
        default="./outputs/gsm8k-training",
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
        help="Maximum number of samples to test (use -1 or --all for full test set)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test on full GSM8K test set (all 1319 samples)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to write timestamped log files",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    
    args = parser.parse_args()
    
    # Determine if testing all samples
    test_all = args.all or args.max_samples == -1
    
    if args.compare:
        compare_models(
            args.base_model, 
            args.trained_model, 
            max_samples=args.max_samples if not test_all else -1
        )
    elif args.model:
        test_model(
            args.model,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            log_dir=args.log_dir,
            test_all=test_all,
        )
    else:
        # Default: test trained model
        test_model(
            args.trained_model,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            log_dir=args.log_dir,
            test_all=test_all,
        )


if __name__ == "__main__":
    main()

