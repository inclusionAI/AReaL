"""
Evaluation Script for AIME'24 and AIME'25 using SGLang (Offline Engine).
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
import sglang as sgl

# Default Model
DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B" 

SYSTEM_PROMPT = """You are a helpful assistant. Solve the math problem step by step. The final answer should be formatted as \\boxed{answer}."""

def extract_boxed_answer(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None

def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    normalized = answer.strip()
    normalized = normalized.replace("\\text{", "").replace("}", "")
    normalized = normalized.replace(" ", "").replace("$", "")
    return normalized.lower()

def compare_answers(predicted: str, ground_truth: str) -> bool:
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    if pred_norm == gt_norm:
        return True
    try:
        pred_int = int(re.search(r'-?\d+', predicted).group())
        gt_int = int(re.search(r'-?\d+', ground_truth).group())
        return pred_int == gt_int
    except:
        pass
    return False

def load_problems(file_path: str) -> List[Dict]:
    problems = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    problems.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return problems

def main():
    parser = argparse.ArgumentParser(description="Eval AIME 24/25 with SGLang Engine")
    parser.add_argument("--aime24-path", default="s1-parallel/AIME24.jsonl")
    parser.add_argument("--aime25-path", default="s1-parallel/AIME25.jsonl")
    parser.add_argument("--output-file", default="qwen3-aime.md")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    
    args = parser.parse_args()
    
    print(f"Initializing SGLang Engine with model {args.model_path} (TP={args.tp_size})...")
    engine = sgl.Engine(model_path=args.model_path, tp_size=args.tp_size, trust_remote_code=True)
    
    results_summary = ""
    
    for year, path in [("AIME 24", args.aime24_path), ("AIME 25", args.aime25_path)]:
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
            
        print(f"Evaluatiing {year}...")
        problems = load_problems(path)
        
        # Batch generation
        prompts = []
        for p in problems:
            # Construct chat-like prompt manually or use tokenizer if available. 
            # SGLang Engine supports chat template if implemented, but safer to use raw string or simple format.
            # Qwen uses ChatML usually.
            content = p.get("problem", "")
            # full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(content)
            
        print(f"Generating responses for {len(prompts)} problems...")
        sampling_params = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": args.max_tokens}
        outputs = engine.generate(prompts, sampling_params)
        
        correct = 0
        total = 0
        details = []
        
        for i, (p, output) in enumerate(zip(problems, outputs)):
            gt_answer = str(p.get("answer", ""))
            pred_text = output["text"]
            pred_answer = extract_boxed_answer(pred_text)
            
            is_correct = False
            if pred_answer:
                is_correct = compare_answers(pred_answer, gt_answer)
            
            if is_correct:
                correct += 1
            total += 1
            
            status_icon = "✅" if is_correct else "❌"
            details.append(f"| {i+1} | {gt_answer} | {pred_answer or 'N/A'} | {status_icon} |")
            print(f"[{year}] Problem {i+1}: {status_icon} (Pred: {pred_answer}, GT: {gt_answer})")
            
        accuracy = (correct / total * 100) if total > 0 else 0
        summary_line = f"## {year} Results\n**Accuracy**: {correct}/{total} ({accuracy:.2f}%)\n\n"
        table_header = "| ID | Ground Truth | Predicted | Result |\n|---|---|---|---|\n"
        results_summary += summary_line + table_header + "\n".join(details) + "\n\n"

    with open(args.output_file, "w") as f:
        f.write(f"# Qwen3-8B AIME Evaluation Results\n")
        f.write(f"**Date**: {datetime.now()}\n")
        f.write(f"**Model**: {args.model_path}\n\n")
        f.write(results_summary)
        
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
