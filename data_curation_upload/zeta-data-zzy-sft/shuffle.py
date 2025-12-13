import json
import random

def shuffle_jsonl(input_file, output_file):
    """
    Shuffle a JSONL file randomly.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output shuffled JSONL file
    """
    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle the lines randomly
    random.shuffle(lines)
    
    # Write shuffled lines to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Shuffled {len(lines)} lines from {input_file} to {output_file}")

if __name__ == "__main__":
    # Example usage
    input_file = "conversations_output.jsonl"
    output_file = "conversations_output_shuffled.jsonl"
    
    shuffle_jsonl(input_file, output_file)