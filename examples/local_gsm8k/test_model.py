#!/usr/bin/env python3
"""
Test and compare model performance before and after training.
"""

import argparse
import json
import os
import re
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    # Newer versions
    from transformers.generation.stopping_criteria import (
        StoppingCriteria,
        StoppingCriteriaList,
    )
except Exception:
    # Fallback import path
    from transformers import StoppingCriteria, StoppingCriteriaList

# Try to import from AReaL, fallback to simple version
try:
    from areal.reward.math_parser import process_results
except ImportError:
    # Simple fallback parser
    import re
    
    def process_results(completions, answer):
        """Improved parser to extract final answer from GSM8K format."""
        # Extract answer from ground truth (GSM8K format: ... #### number.)
        try:
            # Look for "#### number" pattern
            gt_answer = float(re.findall(r'-?\d+\.?\d*', answer.split("####")[-1])[-1])
        except:
            return [False]
        
        # Check completions
        results = []
        for completion in completions:
            pred_answer = None
            try:
                # Method 1: Look for #### pattern (GSM8K standard)
                if "####" in completion:
                    numbers_after_hash = re.findall(r'-?\d+\.?\d*', completion.split("####")[-1])
                    if numbers_after_hash:
                        pred_answer = float(numbers_after_hash[-1])
                
                # Method 2: Look for <<...>>=X pattern (training format)
                # Extract the value X from patterns like "<<computation=X>>X"
                if pred_answer is None:
                    # Find all <<...=X>> patterns
                    pattern_matches = re.findall(r'<<[^>]*=(-?\d+\.?\d*)>>', completion)
                    if pattern_matches:
                        # Take the last matched value (most likely the final answer)
                        pred_answer = float(pattern_matches[-1])
                    
                    # Also check for "= <<X>>" or "<<X>>" standalone patterns
                    if pred_answer is None:
                        # Look for "<<number>>" patterns and take the last one
                        bracket_nums = re.findall(r'<<(-?\d+\.?\d*)>>', completion)
                        if bracket_nums:
                            pred_answer = float(bracket_nums[-1])
                
                # Method 3: Look for sentences ending with numbers (likely final answer)
                # Find sentences that end with a number followed by period/newline
                if pred_answer is None:
                    # Pattern: sentence ending with "number." or "number\n"
                    end_patterns = re.findall(r'(\d+\.?\d*)\s*[.\n]', completion)
                    if end_patterns:
                        # Take last number that appears near end
                        pred_answer = float(end_patterns[-1])
                
                # Method 4: Fallback - just take the last number
                if pred_answer is None:
                    numbers = re.findall(r'-?\d+\.?\d*', completion)
                    if numbers:
                        # Skip very large numbers (likely intermediate calculations)
                        # Prioritize reasonable numbers
                        candidates = [float(n) for n in numbers[-5:]]  # Last 5 numbers
                        # Filter out unrealistic values (like 1000000)
                        reasonable = [n for n in candidates if abs(n) < 1000000 and (abs(n) > 0.001 or n == 0)]
                        if reasonable:
                            pred_answer = reasonable[-1]
                        else:
                            pred_answer = float(numbers[-1])
                
                # Compare answers (within small tolerance for floating point)
                if pred_answer is not None:
                    results.append(abs(pred_answer - gt_answer) < 0.01)
                else:
                    results.append(False)
            except Exception as e:
                results.append(False)
        
        return results if len(results) > 0 else [False]


class StopOnHash(StoppingCriteria):
    """Stop generation when the tokenizer sequence for '####' appears."""

    def __init__(self, hash_ids):
        super().__init__()
        self.hash_ids = hash_ids

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # input_ids: [batch, seq]
        if input_ids is None or input_ids.shape[0] == 0:
            return False
        seq = input_ids[0].tolist()
        hlen = len(self.hash_ids)
        if hlen == 0 or len(seq) < hlen:
            return False
        # Check if the end of sequence matches '####'
        return seq[-hlen:] == self.hash_ids


def test_model(
    model_path: str,
    max_samples: int = 10,
    max_new_tokens: int = 1024,
    stop_on_hash: bool = True,
    log_dir: str | None = None,
):
    """Test the model on GSM8K samples."""
    
    # Prepare logging
    if log_dir is None:
        log_dir = os.path.join("examples", "local_gsm8k", "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_{ts}.log")

    def _log(msg: str):
        print(msg)
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
    _log(f"Using device: {device}")
    
    torch_dtype = torch.float32  # Use float32 for CPU
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    
    # Prepare stopping criteria (optional)
    stopping_criteria = None
    if stop_on_hash:
        hash_ids = tokenizer.encode("####", add_special_tokens=False)
        stopping_criteria = StoppingCriteriaList([StopOnHash(hash_ids)])

    # Load GSM8K test set
    from datasets import load_dataset
    
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    results = []
    correct = 0
    
    for i, sample in enumerate(dataset.select(range(min(max_samples, len(dataset))))):
        question = sample["question"]
        correct_answer = sample["answer"]
        
        # Format prompt - Use GSM8K format (no \boxed{} prompt)
        # This matches the training format
        prompt = f"{question}\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate with greedy decoding for stability
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if stopping_criteria is not None:
                gen_kwargs["stopping_criteria"] = stopping_criteria
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Get the generated token IDs (excluding the prompt)
        input_len = inputs.input_ids.shape[-1]
        generated_token_ids = outputs[0][input_len:]
        
        # Check if EOS was generated
        eos_was_generated = tokenizer.eos_token_id in generated_token_ids.tolist()
        eos_position = None
        if eos_was_generated:
            eos_positions = [i for i, tok_id in enumerate(generated_token_ids.tolist()) if tok_id == tokenizer.eos_token_id]
            if eos_positions:
                eos_position = eos_positions[0]  # First EOS position
        
        # Decode without skipping special tokens first to check for EOS
        full_output_raw = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Decode with skipping special tokens for final output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_output[len(prompt):]
        
        # Also decode just the generated tokens to verify
        generated_text_from_tokens = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        # Determine stop reason
        new_tokens = len(generated_token_ids)
        if stop_on_hash and "####" in generated_text:
            stop_reason = "hash"
        elif new_tokens >= max_new_tokens:
            stop_reason = "max_new_tokens"
        elif eos_was_generated:
            stop_reason = f"eos_token (at position {eos_position})"
        else:
            stop_reason = "unknown"
        
        # Extract numeric answers (GT and prediction)
        def _extract_gt(ans_text: str):
            try:
                return float(re.findall(r'-?\d+\.?\d*', ans_text.split("####")[-1])[-1])
            except Exception:
                return None

        def _extract_pred(pred_text: str):
            # Method 1: after ####
            if "####" in pred_text:
                try:
                    return float(re.findall(r'-?\d+\.?\d*', pred_text.split("####")[-1])[-1])
                except Exception:
                    pass
            # Method 2: <<...=X>> or <<X>>
            try:
                matches = re.findall(r'<<[^>]*=(-?\d+\.?\d*)>>', pred_text)
                if matches:
                    return float(matches[-1])
                bracket_nums = re.findall(r'<<(-?\d+\.?\d*)>>', pred_text)
                if bracket_nums:
                    return float(bracket_nums[-1])
            except Exception:
                pass
            # Method 3: sentence ending with number
            try:
                end_patterns = re.findall(r'(\d+\.?\d*)\s*[.\n]', pred_text)
                if end_patterns:
                    return float(end_patterns[-1])
            except Exception:
                pass
            # Method 4: fallback last number
            try:
                numbers = re.findall(r'-?\d+\.?\d*', pred_text)
                if numbers:
                    candidates = [float(n) for n in numbers[-5:]]
                    reasonable = [n for n in candidates if abs(n) < 1000000 and (abs(n) > 0.001 or n == 0)]
                    return reasonable[-1] if reasonable else float(numbers[-1])
            except Exception:
                pass
            return None

        gt_extracted = _extract_gt(correct_answer)
        pred_extracted = _extract_pred(generated_text)

        # Check answer - use both methods for robustness
        is_correct = False
        parser_result = None
        try:
            # Primary correctness via parser
            parser_result = process_results([generated_text], correct_answer)[0]
            is_correct = bool(parser_result)
        except Exception as e:
            # If parser fails, fall back to direct numeric comparison
            pass
        
        # If parser says incorrect or failed, double-check with direct comparison
        if not is_correct and gt_extracted is not None and pred_extracted is not None:
            # Use same tolerance as process_results (0.01)
            is_correct = abs(pred_extracted - gt_extracted) < 0.01
        
        if is_correct:
            correct += 1
        
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "generated": generated_text,
            "gt_extracted": gt_extracted,
            "pred_extracted": pred_extracted,
            "correct": is_correct,
        })
        
        _log(f"\n--- Question {i+1} ---")
        _log(f"Question: {question}")
        _log(f"Stop reason: {stop_reason}; new_tokens: {new_tokens}/{max_new_tokens}")
        _log(f"EOS detected: {eos_was_generated}")
        if eos_was_generated and eos_position is not None:
            _log(f"EOS token position in generated sequence: {eos_position}")
            # Show what was decoded up to EOS vs full sequence
            generated_up_to_eos = tokenizer.decode(generated_token_ids[:eos_position], skip_special_tokens=True)
            _log(f"Generated up to EOS (before truncation):\n{generated_up_to_eos}")
        # Verify decoding consistency
        if generated_text != generated_text_from_tokens:
            _log(f"WARNING: Decoding mismatch detected!")
            _log(f"From full output: {repr(generated_text[:100])}")
            _log(f"From tokens only: {repr(generated_text_from_tokens[:100])}")
        _log(f"Generated (full, from full_output):\n{generated_text}")
        _log(f"Generated (from tokens only, for verification):\n{generated_text_from_tokens}")
        _log(f"Correct Answer (full):\n{correct_answer}")
        _log(f"Extracted -> GT: {gt_extracted} | Pred: {pred_extracted}")
        _log(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    
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
    parser.add_argument(
        "--no-stop-on-hash",
        action="store_true",
        help="Disable stopping generation when '####' is produced",
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.base_model, args.trained_model, max_samples=args.max_samples)
    elif args.model:
        test_model(
            args.model,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            stop_on_hash=not args.no_stop_on_hash,
            log_dir=args.log_dir,
        )
    else:
        # Default: test trained model
        test_model(
            args.trained_model,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            stop_on_hash=not args.no_stop_on_hash,
            log_dir=args.log_dir,
        )


if __name__ == "__main__":
    main()

