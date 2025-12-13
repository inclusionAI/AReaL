import json
import sys

def filter_jsonl(input_file, output_file):
    """
    Filter JSONL dataset to keep only lines where input ends with '<|im_start|>assistant'
    """
    kept_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_count += 1
            try:
                data = json.loads(line.strip())
                
                # Check if 'input' field exists and ends with '<|im_start|>assistant' (with or without newline)
                if 'input' in data and (data['input'].endswith('<|im_start|>assistant') or 
                                       data['input'].endswith('<|im_start|>assistant\n')):
                    kept_count += 1
                    outfile.write(line)  # Keep this line
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {total_count}: {e}", file=sys.stderr)
                continue
    
    print(f"Total lines processed: {total_count}")
    print(f"Lines kept (ending with '<|im_start|>assistant'): {kept_count}")
    print(f"Lines filtered out: {total_count - kept_count}")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_assistant_ending.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    filter_jsonl(input_file, output_file)
