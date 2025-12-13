import json
import argparse
from pathlib import Path


def format_text_with_template(problem, text):
    """
    Format the text with the Qwen template.
    
    Args:
        problem: The problem text to be inserted
        text: The original text content
        
    Returns:
        Formatted text with template
    """
    template = (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{problem}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{text}<|im_end|>"
    )
    return template


def process_jsonl_file(input_file, output_file):
    """
    Process a JSONL file and add the template to the text field.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the JSON line
                data = json.loads(line.strip())
                
                # Check if required fields exist
                if 'problem' not in data or 'text' not in data:
                    print(f"Warning: Line {line_num} missing 'problem' or 'text' field. Skipping.")
                    error_count += 1
                    continue
                
                # Format the text with the template
                formatted_text = format_text_with_template(data['problem'], data['text'])
                
                # Update the text field
                data['text'] = formatted_text
                
                # Write the modified line to output file
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} is not valid JSON: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} lines")
    print(f"Errors encountered: {error_count} lines")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Process JSONL file and add Qwen template to text field'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to output JSONL file (default: input_file with _formatted suffix)'
    )
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.parent / f"{input_path.stem}_formatted{input_path.suffix}")
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    # Process the file
    process_jsonl_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
