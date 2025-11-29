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
    model_name: str = "Model",  # Name for logging (e.g., "Baseline" or "Trained")
    batch_size: int = 8,
):
    """Test the reasoning model on GSM8K samples using XML reasoning format.
    
    Args:
        batch_size: Number of samples to process in parallel (default: 8). 
                   Larger batches are faster but use more GPU memory.
    """
    
    # Prepare logging - save to network volume so it persists after pod stops
    if log_dir is None:
        # Default to network volume location (persists after pod stops)
        log_dir = os.path.join("/workspace", "outputs", "grpo", "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include model_name in filename to distinguish baseline vs trained (like GRPO test script)
    log_path = os.path.join(log_dir, f"test_reasoning_{model_name.lower()}_{ts}.log")
    
    def _log(msg: str):
        # Print directly to stdout/stderr (unbuffered) so it shows in terminal immediately
        print(msg, flush=True)
        # Also write to log file
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
            lf.flush()  # Ensure it's written immediately
    
    _log(f"\n{'='*80}")
    _log(f"Testing {model_name} REASONING model: {model_path}")
    _log(f"Max new tokens: {max_new_tokens}")
    _log(f"Log file: {log_path}")
    _log(f"{'='*80}\n")
    
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
    
    # Determine if this is a baseline model (not trained with XML format)
    # Baseline models should be tested WITHOUT the XML system prompt for fair comparison
    is_baseline = (
        "BASELINE" in model_name.upper() or 
        "baseline" in model_name.lower() or
        (not os.path.isdir(model_path) and "/" in model_path)  # HuggingFace model identifier
    )
    
    if is_baseline:
        _log(f"⚠️  Detected BASELINE model - testing WITHOUT XML system prompt for fair comparison")
        _log(f"   (Baseline models were not trained with XML reasoning format)")
    else:
        _log(f"✓ Testing TRAINED model - using XML reasoning format (matching training)")
    
    # System prompt matching training format (only for trained models)
    SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
[Your step-by-step reasoning process here]
</reasoning>
<answer>
[Your final numerical answer here]
</answer>"""
    
    results = []
    correct = 0
    
    # Process samples in batches for faster inference
    dataset_subset = list(dataset.select(range(num_samples)))
    _log(f"Processing {num_samples} samples in batches of {batch_size}...")
    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_samples = dataset_subset[batch_start:batch_end]
        batch_size_actual = len(batch_samples)
        
        # Prepare batch data
        batch_questions = []
        batch_correct_answers = []
        batch_messages = []
        
        for sample in batch_samples:
            question = sample["question"]
            correct_answer = sample["answer"]
            
            # Process correct_answer: convert #### to \boxed{} format (matching local_gsm8k/test_model.py)
            hashes_idx = correct_answer.find("#### ")
            if hashes_idx != -1:
                correct_answer = correct_answer[:hashes_idx] + "\\boxed{" + correct_answer[hashes_idx + 5 :] + "}"
            
            batch_questions.append(question)
            batch_correct_answers.append(correct_answer)
            
            # Format prompt: baseline models get simple format, trained models get XML system prompt
            if is_baseline:
                # Baseline: Use simple format (no system prompt) - matches local_gsm8k/test_model.py
                messages = [
                    {"role": "user", "content": f"{question}\n"}
                ]
            else:
                # Trained: Use XML reasoning system prompt (matching training format)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]
            batch_messages.append(messages)
        
        # Tokenize batch (with padding)
        tokenizer.padding_side = "left"  # Left padding for generation
        batch_inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        # Store input lengths for decoding
        input_lengths = (batch_inputs != tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
        
        # Generate in batch
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
            batch_outputs = model.generate(batch_inputs, **gen_kwargs)
        
        # Decode each output separately
        for batch_idx, (sample_idx, sample) in enumerate(zip(range(batch_start, batch_end), batch_samples)):
            i = sample_idx
            question = batch_questions[batch_idx]
            correct_answer = batch_correct_answers[batch_idx]
            
            # Extract generated tokens (skip input tokens)
            input_len = input_lengths[batch_idx]
            generated_token_ids = batch_outputs[batch_idx][input_len:]
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            
            # Check correctness using AReaL's math parser (handles XML format)
            is_correct = False
            parser_error = None
            try:
                parser_result, extracted_answers = process_results(correct_answer, generated_text)
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
            
            # Log detailed info for all samples with correct answer comparison
            _log(f"\n{'='*80}")
            _log(f"Sample {i+1}/{num_samples}")
            _log(f"{'='*80}")
            _log(f"Question:\n{question}")
            _log(f"\n{'─'*80}")
            _log(f"Generated Response:\n{generated_text}")
            _log(f"\n{'─'*80}")
            _log(f"Correct Answer:\n{correct_answer}")
            _log(f"\n{'─'*80}")
            _log(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
            if parser_error:
                _log(f"Parser Error: {parser_error}")
            _log(f"{'='*80}")
        
        # Log progress every batch or for first/last
        if (batch_end) % 10 == 0 or batch_start == 0 or batch_end >= num_samples:
            _log(f"Progress: {batch_end}/{num_samples} | Correct: {correct}/{batch_end} | Accuracy: {correct/batch_end*100:.2f}%")
    
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
        default=1024,  # Keep 1024 for reasoning chains (already sufficient)
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
    parser.add_argument(
        "--model-name",
        type=str,
        default="Model",
        help="Name for logging (e.g., 'Baseline' or 'Trained')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for faster inference (default: 8). Larger batches are faster but use more GPU memory.",
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
        model_name=args.model_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

