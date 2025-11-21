#!/usr/bin/env python3
"""
Test script for trained reasoning GRPO model on GSM8K dataset (Cloud version).

This script tests reasoning models that were trained with XML format:
<reasoning>...</reasoning><answer>...</answer>

It uses the same system prompt format as training to ensure consistency.
Uses direct model loading (no SGLang server required) for simplicity.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure AReaL is in the Python path for math_parser
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import AReaL's math parser for consistent answer extraction
from areal.reward.math_parser import process_results


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from GRPO checkpoint."""
    if os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            print(f"[INFO] Loading model from checkpoint directory: {checkpoint_path}")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            return model, tokenizer
        else:
            raise ValueError(
                f"Checkpoint directory {checkpoint_path} does not contain config.json. "
                f"Expected a HuggingFace-format checkpoint."
            )
    else:
        print(f"[INFO] Loading model from HuggingFace: {checkpoint_path}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        return model, tokenizer


def test_reasoning_model(
    model_path: str,
    max_samples: int = 10,
    max_new_tokens: int = 1024,  # Increased for reasoning chains
    log_dir: str | None = None,
    test_all: bool = False,
    temperature: float = 0.0,
):
    """Test the reasoning model on GSM8K samples using XML reasoning format."""
    
    # Prepare logging - save to network volume so it persists after pod stops
    if log_dir is None:
        # Default to network volume location (persists after pod stops)
        log_dir = os.path.join("/workspace", "outputs", "grpo", "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_reasoning_{ts}.log")
    
    def _log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
    
    _log(f"\n{'='*60}")
    _log(f"Testing REASONING model: {model_path}")
    _log(f"Log file: {log_path}")
    _log(f"{'='*60}\n")
    
    # Determine device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    _log(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model_from_checkpoint(model_path, device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.eval()
    _log(f"Model loaded and set to eval mode")
    
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
    
    # System prompt matching training format
    SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
[Your step-by-step reasoning process here]
</reasoning>
<answer>
[Your final numerical answer here]
</answer>"""
    
    results = []
    correct = 0
    
    for i, sample in enumerate(dataset.select(range(num_samples))):
        question = sample["question"]
        correct_answer = sample["answer"]
        
        # Format prompt with reasoning system prompt (matching training format)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        # Tokenize
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "temperature": temperature if temperature > 0.0 else None,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            # Remove None values
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            outputs = model.generate(inputs, **gen_kwargs)
        
        # Decode
        input_len = inputs.shape[-1]
        generated_token_ids = outputs[0][input_len:]
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        # Check correctness using AReaL's math parser (handles XML format)
        is_correct = False
        parser_error = None
        try:
            parser_result = process_results([generated_text], correct_answer)[0]
            is_correct = bool(parser_result)
        except Exception as e:
            parser_error = str(e)
            _log(f"Warning: Parser failed for sample {i+1}: {e}")
        
        if is_correct:
            correct += 1
        
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "generated": generated_text,
            "correct": is_correct,
            "parser_error": parser_error,
        })
        
        # Log detailed info for first few samples or when incorrect
        if i < 3 or not is_correct:
            _log(f"\n--- Sample {i+1} ---")
            _log(f"Question: {question[:100]}...")
            _log(f"Generated: {generated_text[:500]}...")  # Show more for reasoning
            _log(f"Correct Answer: {correct_answer[:100]}...")
            _log(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
            if parser_error:
                _log(f"Parser Error: {parser_error}")
        
        # Log progress every 10 samples or for first/last
        if (i + 1) % 10 == 0 or i == 0 or i == num_samples - 1:
            _log(f"Progress: {i+1}/{num_samples} | Correct: {correct}/{i+1} | Accuracy: {correct/(i+1)*100:.2f}%")
    
    accuracy = correct / len(results) * 100
    _log(f"\n{'='*60}")
    _log(f"FINAL ACCURACY: {accuracy:.2f}% ({correct}/{len(results)})")
    _log(f"Log saved to: {log_path}")
    _log(f"{'='*60}\n")
    
    print(f"\n{'='*80}")
    print(f"REASONING MODEL ACCURACY: {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"Log saved to: {log_path}")
    print(f"{'='*80}\n")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test trained reasoning GRPO model on GSM8K")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint (HuggingFace-format directory or HuggingFace model identifier)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of samples to test (use -1 for full test set)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test on full GSM8K test set (all 1319 samples)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate (default: 1024 for reasoning chains)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0.0 = greedy, >0.0 = sampling)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to write timestamped log files",
    )
    
    args = parser.parse_args()
    
    test_all = args.all or args.max_samples == -1
    
    test_reasoning_model(
        args.model_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        log_dir=args.log_dir,
        test_all=test_all,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

