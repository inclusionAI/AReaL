import json
import re
from exploration_or_cases import analyze_one_step_problem
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback

# Lock for thread-safe file writing
write_lock = Lock()

def extract_think_content_and_remainder(text: str) -> tuple:
    """
    Extract content between <think> and </think> tags, and preserve content after </think>.
    
    Args:
        text: String that may contain <think>...</think> tags
        
    Returns:
        Tuple of (think_content, remainder_content)
        - think_content: Content between the tags, or empty string if not found
        - remainder_content: Content after </think> tag, or empty string if not found
    """
    match = re.search(r'<think>(.*?)</think>(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    
    # If no match, check if there's content between tags without remainder
    match_no_remainder = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match_no_remainder:
        return match_no_remainder.group(1).strip(), ""
    
    return "", ""

def process_single_line(line_num: int, line: str, summary_file: str, max_lines: int) -> dict:
    """
    Process a single line from the JSONL file.
    
    Args:
        line_num: Line number being processed
        line: JSON line content
        summary_file: Path to summary file for logging
        max_lines: Maximum number of lines (for display)
        
    Returns:
        Dict with status: 'success', 'skipped', or 'error' and optional message
    """
    try:
        # Parse JSON line
        data = json.loads(line.strip())
        
        # Extract CoT content
        cot_content = data.get('CoT', '')
        
        if not cot_content:
            return {
                'status': 'skipped',
                'message': f"Line {line_num}: No CoT field found, skipping\n"
            }
        
        # Extract content between <think> and </think>, and preserve remainder
        think_content, remainder_content = extract_think_content_and_remainder(cot_content)
        
        if not think_content:
            return {
                'status': 'skipped',
                'message': f"Line {line_num}: No <think> tags found in CoT, skipping\n"
            }
        
        # Process the think content
        print(f"\n{'='*80}")
        print(f"Processing line {line_num}/{max_lines} (Thread)")
        print(f"{'='*80}\n")
        
        # Call the analyze function with remainder content
        result = analyze_one_step_problem(think_content, remainder_content)
        
        return {
            'status': 'success',
            'message': f"Line {line_num}: Successfully processed\n",
            'result': result,
            'remainder': remainder_content
        }
        
    except json.JSONDecodeError as e:
        return {
            'status': 'error',
            'message': f"Line {line_num}: JSON decode error - {str(e)}\n"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Line {line_num}: Error processing - {str(e)}\n{traceback.format_exc()}\n"
        }

def process_jsonl_file(jsonl_path: str, max_lines: int = 1000, output_dir: str = "open-math-reasoning-result", num_threads: int = 4):
    """
    Process a JSONL file and extract think content from CoT field using multiple threads.
    
    Args:
        jsonl_path: Path to the JSONL file
        max_lines: Maximum number of lines to process (default: 1000)
        output_dir: Directory to save processed results
        num_threads: Number of threads to use for parallel processing (default: 4)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp for this processing run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"{output_dir}/processing_summary_{timestamp}.txt"
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Read all lines first
    lines_to_process = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > max_lines:
                break
            lines_to_process.append((line_num, line))
    
    total_lines = len(lines_to_process)
    
    with open(summary_file, 'w', encoding='utf-8') as summary:
        summary.write(f"Processing started at: {datetime.now()}\n")
        summary.write(f"Input file: {jsonl_path}\n")
        summary.write(f"Max lines to process: {max_lines}\n")
        summary.write(f"Number of threads: {num_threads}\n")
        summary.write(f"Total lines to process: {total_lines}\n")
        summary.write("="*80 + "\n\n")
    
    # Process lines in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_line = {
            executor.submit(process_single_line, line_num, line, summary_file, max_lines): line_num
            for line_num, line in lines_to_process
        }
        
        # Process completed tasks
        for future in as_completed(future_to_line):
            line_num = future_to_line[future]
            try:
                result = future.result()
                
                # Thread-safe writing to summary file
                with write_lock:
                    with open(summary_file, 'a', encoding='utf-8') as summary:
                        summary.write(result['message'])
                
                # Update counters
                with write_lock:
                    if result['status'] == 'success':
                        processed_count += 1
                    elif result['status'] == 'skipped':
                        skipped_count += 1
                    elif result['status'] == 'error':
                        error_count += 1
                        print(result['message'])
                        
            except Exception as e:
                error_msg = f"Line {line_num}: Unexpected error in thread - {str(e)}\n{traceback.format_exc()}\n"
                print(error_msg)
                with write_lock:
                    with open(summary_file, 'a', encoding='utf-8') as summary:
                        summary.write(error_msg)
                    error_count += 1
    
    # Write summary statistics
    with open(summary_file, 'a', encoding='utf-8') as summary:
        summary.write("\n" + "="*80 + "\n")
        summary.write("Processing Summary:\n")
        summary.write(f"Total lines processed: {total_lines}\n")
        summary.write(f"Successfully processed: {processed_count}\n")
        summary.write(f"Skipped: {skipped_count}\n")
        summary.write(f"Errors: {error_count}\n")
        summary.write(f"Processing completed at: {datetime.now()}\n")
    
    print(f"\n{'='*80}")
    print("Processing Complete!")
    print(f"Total lines: {total_lines}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # TODO: Set your JSONL file path here
    jsonl_file_path = "/Users/zzy/Downloads/out_open_math_reasoning_small.jsonl"
    
    # Process the first 1000 lines with 4 threads
    process_jsonl_file(
        jsonl_path=jsonl_file_path,
        max_lines=60,
        output_dir="open-math-reasoning-result",
        num_threads=10  # Adjust the number of threads as needed
    )
