import json
import re

def process_jsonl_file(input_file, output_file):
    """
    Process JSONL file to ensure '<|im_start|>assistant' is followed by a newline.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    processed_count = 0
    unchanged_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Check if 'input' field exists
                if 'input' in data:
                    original_input = data['input']
                    
                    # Replace '<|im_start|>assistant' with '<|im_start|>assistant\n'
                    # but only if it's not already followed by a newline
                    # Pattern: '<|im_start|>assistant' NOT followed by '\n'
                    pattern = r'<\|im_start\|>assistant(?!\n)'
                    replacement = r'<|im_start|>assistant\n'
                    
                    modified_input = re.sub(pattern, replacement, original_input)
                    
                    # Update the data
                    if modified_input != original_input:
                        data['input'] = modified_input
                        processed_count += 1
                    else:
                        unchanged_count += 1
                
                # Write the (possibly modified) line to output file
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                # Optionally write the problematic line as-is
                outfile.write(line)
    
    print(f"Processing complete!")
    print(f"Lines processed (modified): {processed_count}")
    print(f"Lines unchanged: {unchanged_count}")
    print(f"Total lines: {processed_count + unchanged_count}")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    # Example usage
    input_file = "conversations_output_shuffled.jsonl"
    output_file = "conversations_output_shuffled_processed.jsonl"
    
    # You can change these file paths as needed
    # input_file = "conversations_output_shuffled.jsonl"
    # output_file = "conversations_output_shuffled_processed.jsonl"
    
    process_jsonl_file(input_file, output_file)
