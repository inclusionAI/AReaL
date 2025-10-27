import json
import argparse
import os
from typing import Dict, List, Any

def filter_conversations_by_gpt_length(input_file: str, output_file: str, min_length: int = 50) -> None:
    """
    Filter conversations by removing those with GPT responses shorter than min_length
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file  
        min_length: Minimum length for GPT responses (default: 50)
    """
    filtered_count = 0
    total_count = 0
    removed_count = 0
    
    print(f"Processing {input_file}...")
    print(f"Minimum GPT response length: {min_length} characters")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                total_count += 1
                
                # Check if this conversation should be kept
                keep_conversation = True
                short_gpt_responses = []
                
                if 'conversations' in data:
                    for i, message in enumerate(data['conversations']):
                        if message.get('from') == 'gpt':
                            gpt_value = message.get('value', '')
                            if len(gpt_value) < min_length:
                                keep_conversation = False
                                short_gpt_responses.append({
                                    'index': i,
                                    'length': len(gpt_value),
                                    'preview': gpt_value[:50] + '...' if len(gpt_value) > 50 else gpt_value
                                })
                
                if keep_conversation:
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    filtered_count += 1
                else:
                    removed_count += 1
                    if len(short_gpt_responses) > 0:
                        print(f"Line {line_num}: Removed conversation with {len(short_gpt_responses)} short GPT response(s)")
                        for resp in short_gpt_responses[:2]:  # Show first 2 short responses
                            print(f"  - Message {resp['index']}: {resp['length']} chars - '{resp['preview']}'")
                        if len(short_gpt_responses) > 2:
                            print(f"  - ... and {len(short_gpt_responses) - 2} more short responses")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue
    
    print(f"\n=== Processing Complete ===")
    print(f"Total conversations processed: {total_count}")
    print(f"Conversations kept: {filtered_count}")
    print(f"Conversations removed: {removed_count}")
    print(f"Removal rate: {removed_count/total_count*100:.1f}%")
    print(f"Output written to: {output_file}")

def simple_filter(input_file: str, min_length: int = 50) -> None:
    """
    Simple version that creates output file with _filtered suffix
    """
    base_name = os.path.splitext(input_file)[0]
    extension = os.path.splitext(input_file)[1]
    output_file = f"{base_name}_filtered{extension}"
    
    filter_conversations_by_gpt_length(input_file, output_file, min_length)

def main():
    parser = argparse.ArgumentParser(
        description='Filter JSONL conversations by removing those with short GPT responses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter with default 50 character minimum
  python filter_gpt_responses.py input.jsonl output.jsonl
  
  # Filter with custom minimum length
  python filter_gpt_responses.py input.jsonl output.jsonl --min-length 100
  
  # Simple mode (creates input_filtered.jsonl)
  python filter_gpt_responses.py input.jsonl --simple --min-length 30
        """)
    
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('output_file', nargs='?', help='Output JSONL file path (optional in simple mode)')
    parser.add_argument('--min-length', type=int, default=50, 
                        help='Minimum length for GPT responses (default: 50)')
    parser.add_argument('--simple', action='store_true',
                        help='Simple mode: auto-generate output filename with _filtered suffix')
    
    args = parser.parse_args()
    
    if args.simple:
        simple_filter(args.input_file, args.min_length)
    else:
        if not args.output_file:
            parser.error("output_file is required unless --simple mode is used")
        filter_conversations_by_gpt_length(args.input_file, args.output_file, args.min_length)

if __name__ == "__main__":
    main()