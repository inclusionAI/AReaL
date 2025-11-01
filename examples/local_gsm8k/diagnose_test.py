"""Diagnostic script to compare base vs trained model outputs."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from datasets import load_dataset

def extract_answer(text, method="standard"):
    """Extract answer from text using various methods."""
    results = {}
    
    # Method 1: Look for #### pattern (GSM8K standard)
    if "####" in text:
        try:
            # Get everything after last ####
            after_hash = text.split("####")[-1]
            # Extract numbers
            numbers = re.findall(r'-?\d+\.?\d*', after_hash)
            if numbers:
                results["hash_pattern"] = float(numbers[-1])
        except:
            pass
    
    # Method 2: Look for final number anywhere in text
    try:
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            results["last_number"] = float(numbers[-1])
    except:
        pass
    
    # Method 3: Look for patterns like "The answer is X" or "= X"
    try:
        # Pattern: "= X" or "=X" at end
        equals_pattern = re.findall(r'=\s*(-?\d+\.?\d*)', text)
        if equals_pattern:
            results["equals_pattern"] = float(equals_pattern[-1])
    except:
        pass
    
    return results

def test_model_detailed(model_path: str, max_samples: int = 5, max_new_tokens: int = 512):
    """Test model with detailed output analysis."""
    
    print(f"\n{'='*60}")
    print(f"Testing model: {model_path}")
    print(f"{'='*60}\n")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cpu")
    torch_dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    
    # Load dataset
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    stats = {
        "total_length": 0,
        "has_hash": 0,
        "answers_extracted": 0,
        "correct": 0,
        "truncated": 0,
    }
    
    for i, sample in enumerate(dataset.select(range(min(max_samples, len(dataset))))):
        question = sample["question"]
        correct_answer = sample["answer"]
        
        # Extract ground truth
        try:
            gt_answer = float(re.findall(r'-?\d+\.?\d*', correct_answer.split("####")[-1])[-1])
        except:
            print(f"Warning: Could not extract GT answer for question {i+1}")
            continue
        
        prompt = f"{question}\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_output[len(prompt):]
        
        # Check if truncated
        generated_tokens = len(outputs[0]) - len(inputs.input_ids[0])
        is_truncated = generated_tokens >= max_new_tokens
        
        # Extract answers
        extracted = extract_answer(generated_text)
        
        # Stats
        stats["total_length"] += len(generated_text)
        if "####" in generated_text:
            stats["has_hash"] += 1
        
        # Check correctness with each method
        correct_by_method = {}
        for method, value in extracted.items():
            if value is not None:
                stats["answers_extracted"] += 1
                correct_by_method[method] = abs(value - gt_answer) < 0.01
        
        if is_truncated:
            stats["truncated"] += 1
        
        # Determine if any method got it right
        is_correct = any(correct_by_method.values()) if correct_by_method else False
        if is_correct:
            stats["correct"] += 1
        
        # Print detailed info for first few
        if i < 3:
            print(f"\n{'='*60}")
            print(f"Question {i+1}")
            print(f"{'='*60}")
            print(f"GT Answer: {gt_answer}")
            print(f"Generated length: {len(generated_text)} chars, {generated_tokens} tokens")
            print(f"Truncated: {is_truncated}")
            print(f"Has ####: {'####' in generated_text}")
            print(f"Extracted: {extracted}")
            print(f"Correct: {is_correct}")
            print(f"\nGenerated text (first 300 chars):")
            print(generated_text[:300] + ("..." if len(generated_text) > 300 else ""))
            if "####" in generated_text:
                # Show context around ####
                hash_idx = generated_text.find("####")
                context = generated_text[max(0, hash_idx-50):hash_idx+100]
                print(f"\nContext around ####:")
                print(context)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Avg length: {stats['total_length'] / max_samples:.0f} chars")
    print(f"Has #### pattern: {stats['has_hash']}/{max_samples} ({100*stats['has_hash']/max_samples:.1f}%)")
    print(f"Answers extracted: {stats['answers_extracted']}")
    print(f"Truncated responses: {stats['truncated']}/{max_samples} ({100*stats['truncated']/max_samples:.1f}%)")
    print(f"Correct: {stats['correct']}/{max_samples} ({100*stats['correct']/max_samples:.1f}%)")
    
    return stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    
    test_model_detailed(args.model, args.max_samples, args.max_new_tokens)


