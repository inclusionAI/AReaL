import json

def remove_error_lines(input_file, output_file):
    """
    Remove lines that contain error information from a JSONL file
    and save the cleaned data to a new file.
    """
    cleaned_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse the JSON line
                data = json.loads(line.strip())
                
                # Check if this line contains error information
                if 'error' in data:
                    print(f"Removing line {line_num} with error: {data.get('error', 'Unknown error')}")
                    continue
                
                # If no error, keep this line
                cleaned_lines.append(line)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON, skipping: {e}")
                continue
    
    # Write cleaned data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    print(f"\nProcessing complete!")
    print(f"Removed {len([line for line in open(input_file, 'r').readlines()]) - len(cleaned_lines)} error lines")
    print(f"Cleaned data saved to: {output_file}")

def main():
    input_file = "output_1001_to_1500_batch80_delay1min.jsonl"
    output_file = "output_1001_to_1500_batch80_delay1min_cleaned.jsonl"
    
    try:
        remove_error_lines(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()