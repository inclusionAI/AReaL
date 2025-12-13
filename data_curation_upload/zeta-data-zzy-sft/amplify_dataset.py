#!/usr/bin/env python3
"""
Amplify JSONL dataset by duplicating lines whose 'input' field ends with '<|im_start|>assistant\n'
Each qualifying line is copied 3 times and inserted at random positions in the output.
"""

import json
import random
from typing import List, Dict, Any


def amplify_jsonl(input_file: str, output_file: str, num_copies: int = 3):
    """
    Amplify JSONL dataset by copying qualifying lines.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        num_copies: Number of times to copy each qualifying line (default: 3)
    """
    # Read all lines from input file
    print(f"Reading from {input_file}...")
    lines = []
    lines_to_amplify = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                lines.append(data)
                
                # Check if input ends with "<|im_start|>assistant\n"
                if 'input' in data and data['input'].endswith("<|im_start|>assistant\n"):
                    # Store the line and its copies
                    for _ in range(num_copies):
                        lines_to_amplify.append(data.copy())
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {idx + 1}: {e}")
                continue
    
    print(f"Original lines: {len(lines)}")
    print(f"Lines to amplify: {len(lines_to_amplify) // num_copies}")
    print(f"Total copies to insert: {len(lines_to_amplify)}")
    
    # Create a list of all lines including the amplified ones
    all_lines = lines.copy()
    
    # Insert amplified lines at random positions
    for amplified_line in lines_to_amplify:
        # Generate a random position to insert
        random_pos = random.randint(0, len(all_lines))
        all_lines.insert(random_pos, amplified_line)
    
    print(f"Total lines after amplification: {len(all_lines)}")
    
    # Write to output file
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in all_lines:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    
    print("Done!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python amplify_dataset.py <input_file> <output_file> [num_copies]")
        print("Example: python amplify_dataset.py input.jsonl output.jsonl 3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_copies = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    # Set random seed for reproducibility (optional - remove if you want different results each time)
    # random.seed(42)
    
    amplify_jsonl(input_file, output_file, num_copies)
