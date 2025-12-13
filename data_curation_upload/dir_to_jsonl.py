import os
import json
import random

# Set your paths here
TXT_DIR = '/Users/zzy/Desktop/data_curation_new/new_result_with_conclusions_new/'
PROBLEM_JSONL = '/Users/zzy/Downloads/out_open_math_reasoning_small.jsonl'
OUTPUT_JSONL = 'output.jsonl'

# Load all problems into memory
cot_to_problem = {}
problem_to_cot = {}
with open(PROBLEM_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        cot = obj.get('CoT', '').strip()
        problem = obj.get('Problem', '').strip()
        cot_to_problem[cot] = problem
        problem_to_cot[problem] = cot

def find_problem_and_cot(first_line):
    for cot, problem in cot_to_problem.items():
        if first_line in cot:
            return problem, cot
    return None, None

# Collect all formatted entries
all_entries = []

for filename in os.listdir(TXT_DIR):
    if filename.endswith('.txt'):
        txt_path = os.path.join(TXT_DIR, filename)
        with open(txt_path, 'r', encoding='utf-8') as txt_f:
            lines = txt_f.readlines()
            if not lines:
                continue
            first_line = lines[0].strip()
            problem, cot = find_problem_and_cot(first_line)
            if not problem:
                continue  # skip if no matching problem found
            
            # Create entry from directory content
            content = ''.join(lines).strip()
            formatted_dir = (
                "<|im_start|>system\n"
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{problem}<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>\n"
                f"{content}\n"
                "<|im_end|>"
            )
            all_entries.append({"text": formatted_dir})
            
            # Create entry from original CoT
            formatted_cot = (
                "<|im_start|>system\n"
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{problem}<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>\n"
                f"{cot}\n"
                "<|im_end|>"
            )
            all_entries.append({"text": formatted_cot})

# Duplicate all entries 4 times
all_entries = all_entries * 4

# Randomly permute
random.shuffle(all_entries)

# Write to output file
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
    for entry in all_entries:
        out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')