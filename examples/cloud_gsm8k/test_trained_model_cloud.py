#!/usr/bin/env python3
"""
Test script for trained GRPO model on GSM8K dataset (Cloud version).
This script loads a trained model checkpoint and evaluates it on the GSM8K test set.
Supports both config-based and direct model-path testing.
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", message=".*Gloo.*Rank.*connected.*")

# Suppress Gloo messages
os.environ["GLOG_minloglevel"] = "2"

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


def test_model(
    model_path: str,
    max_samples: int = 50,
    max_new_tokens: int = 256,
    log_dir: str | None = None,
    test_all: bool = False,
    temperature: float = 0.0,
    model_name: str = "Model",
):
    """Test the model on GSM8K samples."""
    
    # Prepare logging - save to network volume so it persists after pod stops
    if log_dir is None:
        log_dir = os.path.join("/workspace", "outputs", "grpo", "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_model_{model_name.lower()}_{ts}.log")
    
    def _log(msg: str):
        # Print directly to stdout/stderr (unbuffered) so it shows in terminal immediately
        print(msg, flush=True)
        # Also write to log file
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
            lf.flush()  # Ensure it's written immediately
    
    _log(f"\n{'='*80}")
    _log(f"Testing {model_name} model: {model_path}")
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
    
    results = []
    correct = 0
    
    for i, sample in enumerate(dataset.select(range(num_samples))):
        question = sample["question"]
        correct_answer = sample["answer"]
        
        # Note: process_results expects the answer in its original format from the dataset
        # It will extract the answer from both the ground truth and generated text
        # So we should NOT modify the correct_answer format - let process_results handle it
        # The dataset's answer field contains the full solution with #### prefix, which process_results can handle
        
        # Format prompt: simple user message format (no system prompt for standard GRPO)
        messages = [
            {"role": "user", "content": f"{question}\nPlease put your final answer within \\boxed{{}}."}
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
        
        # Check correctness using AReaL's math parser
        # Note: process_results(answer, solution) where answer is ground truth, solution is generated
        is_correct = False
        parser_error = None
        extracted_answers = (None, None)
        try:
            # Debug: log what we're passing to process_results
            _log(f"[DEBUG] Sample {i+1}: correct_answer length={len(correct_answer)}, generated_text length={len(generated_text)}")
            _log(f"[DEBUG] Sample {i+1}: correct_answer preview={correct_answer[:100]}...")
            _log(f"[DEBUG] Sample {i+1}: generated_text preview={generated_text[:100]}...")
            
            parser_result, extracted_answers = process_results(correct_answer, generated_text)
            is_correct = bool(parser_result)
            
            _log(f"[DEBUG] Sample {i+1}: parser_result={parser_result}, extracted_answers={extracted_answers}")
        except Exception as e:
            parser_error = str(e)
            _log(f"Warning: Parser failed for sample {i+1}: {e}")
            import traceback
            _log(f"Traceback: {traceback.format_exc()}")
        
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
        
        # Log progress every 10 samples or for first/last
        if (i + 1) % 10 == 0 or i == 0 or i == num_samples - 1:
            _log(f"Progress: {i+1}/{num_samples} | Correct: {correct}/{i+1} | Accuracy: {correct/(i+1)*100:.2f}%")
    
    accuracy = correct / len(results) * 100
    _log(f"\n{'='*80}")
    _log(f"FINAL ACCURACY: {accuracy:.2f}% ({correct}/{len(results)})")
    _log(f"Log saved to: {log_path}")
    _log(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} MODEL ACCURACY: {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"Log saved to: {log_path}")
    print(f"{'='*80}\n")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "results": results,
        "log_path": log_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Test trained GRPO model on GSM8K")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (HuggingFace-format directory or HuggingFace model identifier). If not provided, will try to use --config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (alternative to --model-path). Will extract model path from config.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
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
        default=512,  # Increased from 256 to allow longer reasoning chains
        help="Maximum new tokens to generate (default: 512)",
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
    
    args = parser.parse_args()
    
    # Determine model path
    model_path = args.model_path
    if model_path is None and args.config is not None:
        # Extract model path from config
        from omegaconf import OmegaConf
        from pathlib import Path
        
        config_file = Path(args.config)
        if not config_file.is_absolute():
            config_file = Path.cwd() / config_file
        config = OmegaConf.load(config_file)
        model_path = config.get("actor", {}).get("path", None)
        if model_path is None:
            print("ERROR: Could not extract model path from config. Please use --model-path instead.")
            sys.exit(1)
    elif model_path is None:
        print("ERROR: Must provide either --model-path or --config")
        sys.exit(1)
    
    test_all = args.all or args.max_samples == -1
    
    return test_model(
        model_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        log_dir=args.log_dir,
        test_all=test_all,
        temperature=args.temperature,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
