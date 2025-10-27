import json
import re
import openai
from typing import List, Tuple
import time
from dotenv import load_dotenv
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")
silicon_api_key = os.environ.get("SILICON_API_KEY")
silicon_base_url = os.environ.get("SILICON_BASE_URL")

client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

def extract_parallel_processing_sections(content: str) -> List[Tuple[str, int, int]]:
    """
    Extract all <parallel_processing> sections from the content.
    Returns a list of tuples: (section_content, start_pos, end_pos)
    """
    sections = []
    pattern = r"<parallel_processing>(.*?)</parallel_processing>"
    
    for match in re.finditer(pattern, content, re.DOTALL):
        section_content = match.group(1).strip()
        start_pos = match.start()
        end_pos = match.end()
        sections.append((section_content, start_pos, end_pos))
    
    return sections

def call_llm_api(parallel_content: str, api_type: str = "openai") -> str:
    """
    Call LLM API to summarize the parallel processing content.
    """
    prompt = f"""
Please analyze the following parallel processing section and provide a concise summary of the main conclusions and results achieved by the parallel threads.

Focus on:
1. What key insights or results were obtained
2. How the different threads contributed to solving the problem
3. The main conclusion that can be drawn from this parallel processing group

Parallel processing content:
{parallel_content}

Please provide a clear and concise summary in 2-3 sentences.

IMPORTANT: Do not say what these threads are doing, just summarize the results and conclusions.
GOOD: The root of x^2 = 4 is x = 2 or x = -2.
BAD: The parallel processing group calculated the root of x^2 = 4, which is x = 2 or x = -2.
"""
    try:
        response = client.chat.completions.create(
            model= "deepseek-v3-0324",  # Replace with your desired model
            temperature=0,
            seed = 42,
            messages=[
            {"role": "system", "content": """You are a helpful assistant that summarizes parallel processing results clearly and concisely."""},
            {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content.strip()
        return response_text 
            
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return generate_fallback_summary(parallel_content)

def generate_fallback_summary(parallel_content: str) -> str:
    """
    Generate a simple fallback summary if LLM API fails.
    """
    # Extract thread results
    thread_results = re.findall(r"<thread_result id='\d+'>(.*?)</thread_result>", parallel_content, re.DOTALL)
    
    if thread_results:
        # Take first few results and create a basic summary
        results_text = " ".join([result.strip() for result in thread_results[:3]])
        return f"The parallel processing group analyzed multiple aspects and concluded: {results_text[:150]}..."
    else:
        return "The parallel processing group completed its analysis of the problem components."

def process_main_thread_with_summaries(main_thread_content: str, api_type: str = "openai") -> str:
    """
    Process main_thread content by adding summaries after each parallel_processing section.
    """
    # Find all parallel processing sections
    sections = extract_parallel_processing_sections(main_thread_content)
    
    if not sections:
        return main_thread_content
    
    # Process the content by replacing each complete parallel_processing block
    modified_content = main_thread_content
    
    # Process sections in reverse order to maintain correct positions
    for section_content, start_pos, end_pos in reversed(sections):
        # Get LLM summary
        summary = call_llm_api(section_content, api_type)
        
        # Create the complete replacement: original block + parallel_result
        original_block = main_thread_content[start_pos:end_pos]
        replacement_block = original_block + f"\n<parallel_result>\n{summary}\n</parallel_result>"
        
        # Replace the original block with the new block
        modified_content = (
            modified_content[:start_pos] + 
            replacement_block + 
            modified_content[end_pos:]
        )
        
        # Add a small delay to avoid rate limiting within a single problem
        time.sleep(0.2)
    
    return modified_content

def process_single_line(line_data: Tuple[int, str]) -> Tuple[int, dict]:
    """
    Process a single line of JSONL data.
    Returns (line_num, processed_data) for maintaining order.
    """
    line_num, line = line_data
    try:
        # Parse JSON line
        data = json.loads(line.strip())
        
        # Check if main_thread key exists
        if 'main_thread' in data:
            # Process the main_thread content
            original_content = data['main_thread']
            processed_content = process_main_thread_with_summaries(original_content, api_type="openai")
            
            # Only update if content changed
            if processed_content != original_content:
                data['main_thread'] = processed_content
                print(f"✓ Line {line_num}: Added summaries for parallel processing sections")
                return line_num, {"data": data, "processed": True}
            else:
                print(f"○ Line {line_num}: No parallel processing sections found")
                return line_num, {"data": data, "processed": False}
        else:
            print(f"○ Line {line_num}: No main_thread key found")
            return line_num, {"data": data, "processed": False}
            
    except json.JSONDecodeError as e:
        print(f"✗ Line {line_num}: JSON decode error - {e}")
        return line_num, {"data": {"error": f"JSON decode error: {e}", "original_line": line}, "processed": False}
    except Exception as e:
        print(f"✗ Line {line_num}: Processing error - {e}")
        return line_num, {"data": {"error": f"Processing error: {e}", "original_line": line}, "processed": False}

def process_jsonl_file_parallel(input_file: str, output_file: str, batch_size: int = 10, delay_seconds: int = 10):
    """
    Process a JSONL file with parallel processing in batches.
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
        batch_size: Number of lines to process in parallel per batch
        delay_seconds: Seconds to wait between batches
    """
    # Read all lines first
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    total_lines = len(lines)
    processed_count = 0
    error_count = 0
    
    print(f"Total lines to process: {total_lines}")
    print(f"Batch size: {batch_size}, Delay between batches: {delay_seconds} seconds")
    
    # Split into batches
    batches = []
    for i in range(0, total_lines, batch_size):
        batch_lines = []
        for j in range(i, min(i + batch_size, total_lines)):
            batch_lines.append((j + 1, lines[j]))  # (line_number, line_content)
        batches.append(batch_lines)
    
    print(f"Split into {len(batches)} batches")
    
    # Process batches
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for batch_num, batch_lines in enumerate(batches, 1):
            print(f"\n=== Processing Batch {batch_num}/{len(batches)} ===")
            print(f"Lines in this batch: {[line_data[0] for line_data in batch_lines]}")
            
            # Process current batch in parallel
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                
                # Submit all tasks in the batch
                for line_data in batch_lines:
                    future = executor.submit(process_single_line, line_data)
                    futures[future] = line_data[0]  # Store line number
                
                print(f"Submitted {len(batch_lines)} tasks for batch {batch_num}")
                
                # Collect results for this batch
                batch_results = []
                for future in as_completed(futures):
                    line_num = futures[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        print(f"✗ Line {line_num}: Exception in processing - {e}")
                        error_count += 1
                        # Create error result
                        error_result = (line_num, {
                            "data": {"error": f"Exception: {e}", "original_line": ""},
                            "processed": False
                        })
                        batch_results.append(error_result)
                
                # Sort batch results by line number to maintain order
                batch_results.sort(key=lambda x: x[0])
                
                # Write results to file and update counters
                for line_num, result_info in batch_results:
                    data = result_info["data"]
                    was_processed = result_info["processed"]
                    
                    if "error" in data:
                        error_count += 1
                        # Write original line if there's an error
                        if "original_line" in data:
                            outfile.write(data["original_line"])
                        else:
                            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    else:
                        if was_processed:
                            processed_count += 1
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            print(f"Batch {batch_num} completed")
            
            # Wait between batches (except for the last batch)
            if batch_num < len(batches):
                print(f"Waiting {delay_seconds} seconds before next batch...")
                for remaining in range(delay_seconds, 0, -1):
                    print(f"Next batch starts in {remaining} seconds...", end='\r')
                    time.sleep(1)
                print("\n")
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total lines processed: {total_lines}")
    print(f"Lines with parallel processing summaries added: {processed_count}")
    print(f"Lines with errors: {error_count}")
    print(f"Results saved to: {output_file}")

def process_jsonl_file_custom_batch(input_file: str, output_file: str, batch_size: int = 10, delay_seconds: int = 10, max_workers: int = None):
    """
    Process JSONL file with custom batch settings.
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
        batch_size: Number of lines to process in parallel per batch
        delay_seconds: Seconds to wait between batches
        max_workers: Maximum number of workers (defaults to batch_size)
    """
    if max_workers is None:
        max_workers = min(batch_size, 20)  # Cap at 20 to avoid overwhelming the API
    
    return process_jsonl_file_parallel(input_file, output_file, batch_size, delay_seconds)

def test_with_example():
    """Test the functionality with a sample."""
    example_main_thread = """Question: Evaluate the sum $$\\sum_{n=1}^{\\infty}{\\frac{3^n+2^n}{6^n}}.$$
Assistant: 
<reasoning_process>
<think: type = ''>
I need to evaluate this infinite sum. Let me start by analyzing the structure.
</think: type = ''>
<parallel_processing>
<launch_threads>
<thread id='0'>
<task>
Series Decomposition
</task>
<objective>
Break down the sum into simpler components.
</objective>
</thread>
<thread id='1'>
<task>
Geometric Series Recognition
</task>
<objective>
Identify geometric series patterns.
</objective>
</thread>
</launch_threads>
<step_resolution>
<thread_result id='0'>
The sum can be split into: sum of 3^n/6^n + sum of 2^n/6^n = sum of (1/2)^n + sum of (1/3)^n.
</thread_result>
<thread_result id='1'>
Both series are geometric with ratios 1/2 and 1/3 respectively, starting from n=1.
</thread_result>
</step_resolution>
</parallel_processing>
<think: type = ''>
Now I can calculate each geometric series separately.
</think: type = ''>
</reasoning_process>"""

    print("Original content:")
    print(example_main_thread)
    print("\n" + "="*50 + "\n")
    
    # Process with API call
    processed = process_main_thread_with_summaries(example_main_thread, api_type="openai")
    print("Processed content:")
    print(processed)

# Configuration function for API setup
def setup_api(api_type: str, api_key: str):
    """Setup the LLM API with the provided key."""
    if api_type == "openai":
        openai.api_key = api_key
    elif api_type == "anthropic":
        # Set up Anthropic client
        pass
    else:
        print(f"Warning: Unknown API type '{api_type}'. Using fallback summary.")

# Example usage
if __name__ == "__main__":
    print("Available processing modes:")
    print("1. Test with example")
    print("2. Process file with default settings (batch_size=10, delay=10s)")
    print("3. Process file with custom settings")
    print("4. Process file sequentially (original method)")
    
    choice = input("Enter choice (1/2/3/4) or press Enter for option 2: ").strip()
    
    if choice == "1":
        test_with_example()
    elif choice == "3":
        input_file = input("Input file path (default: /home/zhangzy/parallel-thinking/data_processing/test_add_conclusion.jsonl): ").strip()
        if not input_file:
            input_file = "/home/zhangzy/parallel-thinking/data_processing/test_add_conclusion.jsonl"
        
        output_file = input("Output file path (default: output_with_summaries_parallel.jsonl): ").strip()
        if not output_file:
            output_file = "output_with_summaries_parallel.jsonl"
        
        batch_size = int(input("Batch size (default: 10): ").strip() or "10")
        delay_seconds = int(input("Delay seconds between batches (default: 10): ").strip() or "10")
        
        print(f"Processing {input_file} -> {output_file}")
        print(f"Batch size: {batch_size}, Delay: {delay_seconds} seconds")
        process_jsonl_file_parallel(input_file, output_file, batch_size, delay_seconds)
    elif choice == "4":
        # Original sequential processing
        input_file = "/home/zhangzy/parallel-thinking/data_processing/test_add_conclusion.jsonl"
        output_file = "output_with_summaries_sequential.jsonl"
        process_jsonl_file_sequential(input_file, output_file, api_type="openai")
    else:
        # Default parallel processing
        input_file = "/home/zhangzy/parallel-thinking/data_processing/new_760936.jsonl"
        output_file = "output_with_summaries_parallel_7611332.jsonl"
        print(f"Processing {input_file} -> {output_file}")
        print("Using default settings: batch_size=10, delay=10 seconds")
        process_jsonl_file_parallel(input_file, output_file, batch_size=15, delay_seconds=10)

def process_jsonl_file_sequential(input_file: str, output_file: str, api_type: str = "openai"):
    """
    Original sequential processing function (kept for comparison).
    """
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Check if main_thread key exists
                if 'main_thread' in data:
                    # Process the main_thread content
                    original_content = data['main_thread']
                    processed_content = process_main_thread_with_summaries(original_content, api_type)
                    
                    # Only update if content changed
                    if processed_content != original_content:
                        data['main_thread'] = processed_content
                        processed_count += 1
                        print(f"Processed line {line_num}: Added summaries for parallel processing sections")
                    else:
                        print(f"Line {line_num}: No parallel processing sections found")
                
                # Write the processed line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                error_count += 1
                # Write the original line if JSON parsing fails
                outfile.write(line)
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
                # Write the original line if processing fails
                outfile.write(line)
    
    print(f"\nProcessing complete!")
    print(f"Lines processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output saved to: {output_file}")