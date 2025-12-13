import json

def modify_dataset(input_file, output_file):
    """
    Modify a JSONL dataset based on the following rules:
    - If input doesn't end with "<|im_start|>assistant\n", then:
      - new_input = from start to "<|im_start|>assistant\n" (including it)
      - new_output = remaining part of original input + original output
    - Otherwise, keep the line as is
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    modified_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_count += 1
            
            # Parse the JSON line
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error parsing line {total_count}: {e}")
                continue
            
            input_text = data.get('input', '')
            output_text = data.get('output', '')
            
            # Check if input ends with "<|im_start|>assistant\n"
            marker = "<|im_start|>assistant\n"
            
            if not input_text.endswith(marker):
                # Find the position of the marker
                marker_pos = input_text.find(marker)
                
                if marker_pos != -1:
                    # Split at the marker position
                    # new_input includes the marker
                    new_input = input_text[:marker_pos + len(marker)]
                    # new_output is the remaining part + original output
                    remaining = input_text[marker_pos + len(marker):]
                    new_output = remaining + output_text
                    
                    # Update the data
                    data['input'] = new_input
                    data['output'] = new_output
                    modified_count += 1
                else:
                    # Marker not found, keep as is
                    print(f"Warning: Line {total_count} doesn't contain '{marker}'")
            
            # Write the (possibly modified) line to output
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Processing complete!")
    print(f"Total lines processed: {total_count}")
    print(f"Lines modified: {modified_count}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Example usage
    input_file = "conversations_output_shuffled_processed.jsonl"
    output_file = "conversations_output_shuffled_processed_all_train.jsonl"
    
    modify_dataset(input_file, output_file)
