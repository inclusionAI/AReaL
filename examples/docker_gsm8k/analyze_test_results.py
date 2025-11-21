#!/usr/bin/env python3
"""Analyze test results to understand why answers aren't being extracted correctly."""

import re
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from areal.reward.math_parser import extract_answer, process_results

# Read the log file
log_file = "examples/docker_gsm8k/logs/test_reasoning_20251121_150506.log"

if not os.path.exists(log_file):
    print(f"Log file not found: {log_file}")
    sys.exit(1)

with open(log_file, 'r') as f:
    content = f.read()

# Extract sample data
samples = []
pattern = r'--- Sample (\d+) ---\nQuestion: (.*?)\nGenerated: (.*?)\nCorrect Answer: (.*?)\nResult: (.*?)\n'
matches = re.findall(pattern, content, re.DOTALL)

print(f"Found {len(matches)} samples in log\n")
print("="*80)

# Analyze first 5 samples
for i, (sample_num, question, generated, correct_answer, result) in enumerate(matches[:5]):
    print(f"\n--- Sample {sample_num} ---")
    print(f"Question: {question[:100]}...")
    print(f"\nGenerated (first 300 chars): {generated[:300]}...")
    print(f"\nCorrect Answer: {correct_answer[:100]}...")
    print(f"Result: {result}")
    
    # Check if XML format is present
    has_reasoning = '<reasoning>' in generated.lower()
    has_answer_tag = '<answer>' in generated.lower()
    
    print(f"\nFormat Analysis:")
    print(f"  Contains <reasoning>: {has_reasoning}")
    print(f"  Contains <answer>: {has_answer_tag}")
    
    # Try to extract answer
    try:
        extracted = extract_answer(generated, "gsm8k")
        print(f"  Extracted answer: {extracted}")
        
        # Try to extract from ground truth
        gt_extracted = extract_answer(correct_answer, "gsm8k", use_last_number=True)
        print(f"  Ground truth extracted: {gt_extracted}")
        
        # Check if they match
        from areal.reward.math_parser import math_equal
        try:
            match = math_equal(str(extracted), str(gt_extracted), timeout=False) if extracted and gt_extracted else False
            print(f"  Match: {match}")
        except:
            print(f"  Match: Could not compare")
    except Exception as e:
        print(f"  Extraction error: {e}")
    
    print("-"*80)

# Check overall format usage
all_generated = [m[2] for m in matches]
has_reasoning_count = sum(1 for g in all_generated if '<reasoning>' in g.lower())
has_answer_count = sum(1 for g in all_generated if '<answer>' in g.lower())

print(f"\n{'='*80}")
print(f"Overall Statistics:")
print(f"  Total samples analyzed: {len(matches)}")
print(f"  Samples with <reasoning> tag: {has_reasoning_count} ({has_reasoning_count/len(matches)*100:.1f}%)")
print(f"  Samples with <answer> tag: {has_answer_count} ({has_answer_count/len(matches)*100:.1f}%)")
print(f"{'='*80}")

