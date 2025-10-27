import json
import re
import os
from typing import List, Dict, Tuple
import requests
from datetime import datetime
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# MatrixLLM API config (can override via env). Defaults are from the user's snippet.
MATRIXLLM_API_KEY = ""
MATRIXLLM_API_URL = "https://matrixllm.alipay.com/v1/chat/completions"
MATRIXLLM_MODEL = "deepseek-v3-1-250821"

seed = 42

def call_model(messages: list, model: str = None, **kwargs) -> str:
    """Call the chat completion endpoint using requests.

    Args:
        messages: OpenAI-style messages list: [{"role": "system|user|assistant", "content": str}, ...]
        model: Model name to use. Defaults to MATRIXLLM_MODEL.
        **kwargs: Extra payload fields (temperature, max_tokens, seed, etc.).

    Returns:
        The assistant message content string.
    """
    url = MATRIXLLM_API_URL
    api_key = MATRIXLLM_API_KEY
    model = model or MATRIXLLM_MODEL

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "stream": False,
        "model": model,
        "messages": messages,
        **kwargs,
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=120)
    except Exception as e:
        raise RuntimeError(f"Request error: {e}")

    if response.status_code == 200:
        try:
            result = response.json()
            # Expecting OpenAI-like schema
            answer_text = result["choices"][0]["message"]["content"]
            return answer_text
        except Exception as e:
            raise RuntimeError(f"Invalid response JSON shape: {e}; raw: {response.text[:500]}")
    else:
        raise RuntimeError(f"API request failed: {response.status_code} - {response.text[:500]}")

def extract_parallel_processing_sections(text: str) -> List[Dict]:
    """
    Extract all parallel processing sections from the main thread
    """
    pattern = r'<parallel_processing>(.*?)</parallel_processing>'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    sections = []
    for match in matches:
        sections.append({
            'content': match.group(1).strip(),
            'start_pos': match.start(),
            'end_pos': match.end(),
            'full_match': match.group(0)
        })
    
    return sections

def generate_planning_prompt(parallel_content: str, problem_context: str = "", section_index: int = 0, total_sections: int = 1) -> str:
    """
    Create a prompt for the LLM to generate planning content
    """
    prompt = f"""You are an expert problem solver. Given a parallel processing section where multiple threads work on different aspects of a problem, generate a clear planning process that should come before this parallel execution.

This is parallel processing section #{section_index + 1} out of {total_sections} sections in the reasoning process.

CRITICAL: This planning must be DIFFERENT from other planning sections. Focus on the SPECIFIC threads and tasks in THIS section only.

The planning should analyze what needs to be done based on the parallel threads in THIS specific section.

IMPORTANT: Do NOT present it as separate threads or tasks, but as a coherent planning section that logically precedes the parallel processing. This planning should be a whole paragraph, rather than a list of tasks.

IMPORTANT: Your generated planning should be as concise as possible but still cover the necessary details for the parallel processing section presented.

IMPORTANT: Do NOT mention parallel process, just naturally integrate the planning into the context of the problem. For example, you can say "we need to consider the following aspects..." or "Okay, we shall start by ..." rather than "In this parallel processing section, we will...".

IMPORTANT: Make this planning UNIQUE by focusing on the specific thread tasks and objectives shown below.

Problem context: {problem_context}

Parallel processing section #{section_index + 1} of {total_sections}:
{parallel_content}

Generate a planning section that would logically precede THIS SPECIFIC parallel processing step. Focus on what makes this section different from others.

Planning:"""
    
    return prompt

def generate_conclusion_prompt(parallel_content: str, section_index: int = 0, total_sections: int = 1) -> str:
    """
    Create a prompt for the LLM to generate conclusion content
    """
    prompt = f"""Please analyze the following parallel processing section and provide a concise summary of the main conclusions and results achieved by the parallel threads.

This is section #{section_index + 1} out of {total_sections} sections. Make sure your summary is UNIQUE and specific to THIS section's threads.

Focus on:
1. What key insights or results were obtained from THIS specific section
2. How the different threads in THIS section contributed to solving the problem
3. The main conclusion that can be drawn from THIS particular parallel processing group

IMPORTANT: Make this summary DIFFERENT from other sections by focusing on the specific results shown in the thread_result tags below.

Parallel processing content for section #{section_index + 1}:
{parallel_content}

Please provide a clear and concise summary in 2-3 sentences that is UNIQUE to this section.

IMPORTANT: Do not say what these threads are doing, just summarize the results and conclusions.
GOOD: The root of x^2 = 4 is x = 2 or x = -2.
BAD: The parallel processing group calculated the root of x^2 = 4, which is x = 2 or x = -2.

Summary:"""
    
    return prompt

def call_llm_api(prompt: str, content_type: str = "planning", section_index: int = 0, line_num: int = 0) -> str:
    """
    Call LLM API to generate planning or conclusion content
    """
    try:
        # Use env/default model name provided by MatrixLLM config
        model = MATRIXLLM_MODEL

        if content_type == "planning":
            system_message = (
                "You are an expert at generating strategic planning content for problem-solving processes. "
                f"You are currently working on planning section #{section_index + 1}. Make sure your planning is UNIQUE and specific to this particular parallel processing step and different from other sections."
            )
            temperature = 0.5  # Increased for more variety
            seed_value = seed + (line_num * 100) + (section_index * 10)  # More unique seed
        else:  # conclusion
            system_message = (
                "You are a helpful assistant that summarizes parallel processing results clearly and concisely. "
                f"This is conclusion #{section_index + 1}, make it unique."
            )
            temperature = 0.3  # Increased from 0 for variety
            seed_value = seed + (line_num * 100) + (section_index * 10) + 5  # Different seed pattern

        # Build messages and call via requests
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        content = call_model(
            messages=messages,
            model=model,
            temperature=temperature,
            seed=seed_value,
            max_tokens=800,
        ).strip()

        # Clean up any potential XML tags in the response
        if content_type == "planning":
            content = re.sub(r"</?planning>", "", content)
        else:
            content = re.sub(r"</?parallel_result>", "", content)

        content = content.strip()
        return content

    except Exception as e:
        print(f"Error calling LLM API for {content_type}: {e}")

        # Fallback content with unique identifiers
        if content_type == "planning":
            return (
                f"Planning for section #{section_index + 1} (Line {line_num}): Before proceeding with this specific parallel processing step, "
                "let me analyze the unique problem structure and plan the approach for coordinating these particular reasoning threads."
            )
        else:
            return (
                f"Section #{section_index + 1} completed its analysis of the problem components with specific insights."
            )

def process_main_thread_content(main_thread: str, original_problem: str = "", line_num: int = 0) -> str:
    """
    Process main thread by adding both planning and conclusions to parallel processing sections
    """
    # Extract all parallel processing sections
    parallel_sections = extract_parallel_processing_sections(main_thread)
    
    if not parallel_sections:
        return main_thread
    
    total_sections = len(parallel_sections)
    modified_content = main_thread
    
    print(f"  Found {total_sections} parallel processing sections in line {line_num}")
    
    # Process sections in reverse order to maintain correct positions
    for i, section in enumerate(reversed(parallel_sections)):
        section_index = len(parallel_sections) - 1 - i  # Original section index
        
        print(f"    Processing section {section_index + 1}/{total_sections}")
        
        # Generate planning content with section-specific context
        planning_prompt = generate_planning_prompt(
            section['content'], 
            original_problem, 
            section_index, 
            total_sections
        )
        planning_content = call_llm_api(
            planning_prompt, 
            "planning", 
            section_index, 
            line_num
        )
        
        # Generate conclusion content with section-specific context
        conclusion_prompt = generate_conclusion_prompt(
            section['content'], 
            section_index, 
            total_sections
        )
        conclusion_content = call_llm_api(
            conclusion_prompt, 
            "conclusion", 
            section_index, 
            line_num
        )
        
        # Find the actual positions in the modified content
        pattern = re.escape(section['full_match'])
        match = re.search(pattern, modified_content)
        
        if match:
            actual_start = match.start()
            actual_end = match.end()
            
            # Create the replacement block with planning before and conclusion after
            planning_section = f"\n<planning>\n{planning_content}\n</planning>\n"
            conclusion_section = f"\n<parallel_result>\n{conclusion_content}\n</parallel_result>"
            
            replacement_block = planning_section + section['full_match'] + conclusion_section
            
            # Replace in the modified content
            modified_content = (
                modified_content[:actual_start] + 
                replacement_block + 
                modified_content[actual_end:]
            )
        
        # Small delay to avoid rate limiting and ensure different timestamps
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
            original_content = data['main_thread']
            original_problem = data.get('original_problem', '')
            
            # Check if there are parallel processing sections
            parallel_sections = extract_parallel_processing_sections(original_content)
            
            if parallel_sections:
                # Process the main_thread content with line number for unique seeds
                processed_content = process_main_thread_content(
                    original_content, 
                    original_problem, 
                    line_num
                )
                
                if processed_content != original_content:
                    data['main_thread'] = processed_content
                    print(f"✓ Line {line_num}: Added planning and conclusions for {len(parallel_sections)} parallel processing sections")
                    return line_num, {"data": data, "processed": True}
                else:
                    print(f"○ Line {line_num}: Processing failed, content unchanged")
                    return line_num, {"data": data, "processed": False}
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

def process_jsonl_file_batch(input_file: str, output_file: str, batch_size: int = 10, delay_seconds: int = 10):
    """
    Process a JSONL file with batch processing support.
    
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
            with ThreadPoolExecutor(max_workers=min(batch_size, 20)) as executor:
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
    print(f"Lines with planning and conclusions added: {processed_count}")
    print(f"Lines with errors: {error_count}")
    print(f"Results saved to: {output_file}")

def process_jsonl_file_sequential(input_file: str, output_file: str):
    """
    Process JSONL file sequentially (single-threaded)
    """
    print(f"Processing {input_file} sequentially...")
    
    processed_count = 0
    skipped_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            
            try:
                data = json.loads(line.strip())
                main_thread = data.get('main_thread', '')
                original_problem = data.get('original_problem', '')
                
                if not main_thread:
                    print(f"Line {line_num}: No main_thread found, skipping")
                    outfile.write(line)
                    skipped_count += 1
                    continue
                
                # Extract parallel processing sections
                parallel_sections = extract_parallel_processing_sections(main_thread)
                
                if not parallel_sections:
                    print(f"Line {line_num}: No parallel processing sections found, skipping")
                    outfile.write(line)
                    skipped_count += 1
                    continue
                
                print(f"Line {line_num}: Found {len(parallel_sections)} parallel processing sections")
                
                # Process the content
                processed_content = process_main_thread_content(main_thread, original_problem)
                
                if processed_content != main_thread:
                    data['main_thread'] = processed_content
                    processed_count += 1
                    print(f"  ✅ Successfully processed line {line_num}")
                else:
                    print(f"  ⚠️ No changes made to line {line_num}")
                
                # Write modified data
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error - {e}")
                outfile.write(line)  # Write original line
                skipped_count += 1
                
            except Exception as e:
                print(f"Line {line_num}: Processing error - {e}")
                outfile.write(line)  # Write original line
                skipped_count += 1
    
    print(f"\n📊 Processing Summary:")
    print(f"Total entries: {total_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped/errors: {skipped_count}")
    print(f"Output saved to: {output_file}")

def test_with_example():
    """Test the functionality with a sample."""
    example_main_thread = """Question: Factor the polynomial \\(x^4 - 2x^3 + 2x - 1\\).
Assistant: 
<reasoning_process>
<think: type = ''>
I need to factor this polynomial. Let me start by checking for rational roots.
</think: type = ''>
<parallel_processing>
<launch_threads>
<thread id='0'>
<task>
Reduced Polynomial Analysis
</task>
<objective>
analyzes the result of the synthetic division to get the reduced polynomial \\((x^3 - x^2 - x + 1)\\).
</objective>
</thread>
<thread id='1'>
<task>
Alternative Approach Consideration
</task>
<objective>
considers an alternative approach of factoring the quartic polynomial into two quadratics.
</objective>
</thread>
</launch_threads>
<step_resolution>
<thread_result id='0'>
The reduced polynomial is \\((x^3 - x^2 - x + 1)\\).
</thread_result>
<thread_result id='1'>
The model explores factoring the quartic polynomial into two quadratic polynomials as an alternative approach.
</thread_result>
</step_resolution>
</parallel_processing>
<think: type = ''>
Now I can continue with the factoring process.
</think: type = ''>
</reasoning_process>"""

    print("Original content:")
    print(example_main_thread)
    print("\n" + "="*50 + "\n")
    
    # Process with API call
    processed = process_main_thread_content(example_main_thread, "Factor the polynomial \\(x^4 - 2x^3 + 2x - 1\\).")
    print("Processed content:")
    print(processed)

def main():
    """
    Main function with multiple processing options
    """
    print("Available processing modes:")
    print("1. Test with example")
    print("2. Process file sequentially")
    print("3. Process file with batch processing")
    print("4. Process file with custom batch settings")
    
    choice = input("Enter choice (1/2/3/4) or press Enter for option 3: ").strip()
    
    if choice == "1":
        test_with_example()
    elif choice == "2":
        input_file = input("Input file path: ").strip()
        output_file = input("Output file path: ").strip()
        process_jsonl_file_sequential(input_file, output_file)
    elif choice == "4":
        input_file = input("Input file path: ").strip()
        output_file = input("Output file path: ").strip()
        batch_size = int(input("Batch size (default: 10): ").strip() or "10")
        delay_seconds = int(input("Delay seconds between batches (default: 10): ").strip() or "10")
        
        print(f"Processing {input_file} -> {output_file}")
        print(f"Batch size: {batch_size}, Delay: {delay_seconds} seconds")
        process_jsonl_file_batch(input_file, output_file, batch_size, delay_seconds)
    else:
        # Default batch processing
        input_file = "/home/admin/langzhang.zzy/inclusionAI/AReaL/parallel-thinking-main/data_processing/xml_output_new_no_first_step_parallel_large_conclusion_copy/converted_data_new_cleaned_tag.jsonl"
        output_file = "/home/admin/langzhang.zzy/inclusionAI/AReaL/parallel-thinking-main/data_processing/xml_output_new_no_first_step_parallel_large_conclusion_copy/converted_data_new_cleaned_with_parallel_tags_with_planning_conclusion.jsonl"
        print(f"Processing {input_file} -> {output_file}")
        print("Using default settings: batch_size=10, delay=10 seconds")
        process_jsonl_file_batch(input_file, output_file, batch_size=10, delay_seconds=10)

if __name__ == "__main__":
    main()