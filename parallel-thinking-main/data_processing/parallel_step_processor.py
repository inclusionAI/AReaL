import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
# Removed dependency on utils.py functions - we now directly identify parallel groups
import time

# Load environment variables
load_dotenv()
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")
seed = 42

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
model = "deepseek-v3"

def extract_steps_from_txt(txt_content):
    """
    Extract steps from txt file content.
    Expected format:
    Step i
    [Task content]
    The model [objective content]
    [Step content]
    
    Returns:
        tuple: (step_tasks, step_objectives, step_contents)
    """
    # Split content by "Step" pattern
    step_pattern = re.compile(r'Step (\d+)', re.MULTILINE)
    step_matches = list(step_pattern.finditer(txt_content))
    
    step_tasks = []
    step_objectives = []
    step_contents = []
    
    for i, match in enumerate(step_matches):
        start_pos = match.end()
        end_pos = step_matches[i + 1].start() if i + 1 < len(step_matches) else len(txt_content)
        
        step_text = txt_content[start_pos:end_pos].strip()
        lines = step_text.split('\n')
        
        # Extract task (first non-empty line after Step i)
        task = ""
        objective = ""
        content_start_idx = 0
        
        for j, line in enumerate(lines):
            line = line.strip()
            if line and not task:
                task = line.replace(": ", "").strip()
                content_start_idx = j + 1
                break
        
        # Extract objective (line starting with "The model")
        for j in range(content_start_idx, len(lines)):
            line = lines[j].strip()
            if line.startswith("The model"):
                objective = line.replace("The model", "").strip()
                content_start_idx = j + 1
                break
        
        # Extract remaining content
        remaining_lines = lines[content_start_idx:]
        content = '\n'.join(remaining_lines).strip()
        
        step_tasks.append(task)
        step_objectives.append(objective)
        step_contents.append(content)
    
    return step_tasks, step_objectives, step_contents

def identify_parallel_groups_with_llm(step_tasks, step_objectives, step_contents):
    """
    Use LLM to directly identify which steps can be processed in parallel.
    
    Returns:
        list: List of parallel groups (each group is a list of step indices)
    """
    # Prepare the full content for LLM analysis
    full_content = ""
    for i, (task, objective, content) in enumerate(zip(step_tasks, step_objectives, step_contents)):
        full_content += f"Step {i+1}:\n"
        full_content += f"Task: {task}\n"
        full_content += f"Objective: {objective}\n"
        full_content += f"Content: {content}\n\n"
    
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=seed,
        messages=[
            {
                "role": "system", 
                "content": """You are a helpful assistant for analyzing mathematical problem-solving steps for parallel processing.
                
                The user will give you the complete content of all steps in a mathematical solution. You need to analyze which steps can be executed in parallel (simultaneously) without depending on each other's results.
                
                Please identify groups of steps that can be processed in parallel. Return your answer in the following format:
                
                Group 1: [step_numbers]
                Group 2: [step_numbers]
                Group 3: [step_numbers]
                ...
                
                For example:
                Group 1: [1]
                Group 2: [2, 3, 4]
                Group 3: [5]
                Group 4: [6, 7]
                
                Rules:
                1. Steps in the same group can be executed simultaneously
                2. Steps in different groups must be executed sequentially (later groups depend on earlier groups)
                3. If a step doesn't depend on any previous step results, it must be grouped with other independent steps
                4. Single steps should still be in their own group (like Group 1: [1])
                5. First and last steps should be in their own group, they cannot be grouped with other steps
                6. Be confident and aggressive! For steps that are not sure, you can still group them with other steps
                7. As long as the steps are not dependent on each other, you can group them together.
                8. More parallel groups are better. Put as many steps as possible in parallel.
                9. Try your best to group steps together. Do not be too conservative. Add more groups if possible.
                Only return the groups in the specified format, no additional explanation.
                10. There should never be two consecutive parallel groups, if there is, move the first step of the second group as an independent group.
                """
            },
            {
                "role": "user", 
                "content": f"Please analyze these steps and identify parallel groups:\n\n{full_content}"
            }
        ]
    )
    
    llm_output = response.choices[0].message.content
    print(f"Parallel groups analysis: {llm_output}")
    
    # Parse the LLM output to extract parallel groups
    parallel_groups = parse_parallel_groups_output(llm_output)
    
    return parallel_groups

def parse_parallel_groups_output(llm_output):
    """
    Parse the LLM output to extract parallel groups.
    
    Expected format:
    Group 1: [1]
    Group 2: [2, 3, 4]
    Group 3: [5]
    
    Returns:
        list: List of parallel groups (each group is a list of step indices)
    """
    parallel_groups = []
    
    # Pattern to match "Group X: [step_numbers]"
    pattern = r'Group \d+:\s*\[([^\]]+)\]'
    matches = re.findall(pattern, llm_output)
    
    for match in matches:
        # Extract step numbers from the match
        step_numbers_str = match.strip()
        if step_numbers_str:
            try:
                # Parse comma-separated numbers
                step_numbers = [int(num.strip()) for num in step_numbers_str.split(',')]
                parallel_groups.append(step_numbers)
            except ValueError as e:
                print(f"Warning: Could not parse step numbers '{step_numbers_str}': {e}")
                continue
    
    # If no groups were parsed, create sequential groups (fallback)
    if not parallel_groups:
        print("Warning: Could not parse parallel groups from LLM output. Creating sequential groups as fallback.")
        # Create individual groups for each step
        total_steps = llm_output.count('Step ')
        if total_steps == 0:
            # Try to infer from content
            step_count = len(re.findall(r'Step \d+:', llm_output))
            total_steps = step_count if step_count > 0 else 5  # Default fallback
        
        for i in range(1, total_steps + 1):
            parallel_groups.append([i])
    
    print(f"Parsed parallel groups: {parallel_groups}")
    return parallel_groups

def get_step_results(step_tasks, step_objectives, step_contents):
    """
    Get results for each step using LLM API.
    
    Returns:
        list: Results for each step
    """
    step_results = []
    
    for i, (task, objective, content) in enumerate(zip(step_tasks, step_objectives, step_contents)):
        print(f"Processing step {i+1} result...")
        
        result = client.chat.completions.create(
            model=model,
            temperature=0,
            seed=seed,
            messages=[
                {
                    "role": "system", 
                    "content": """You are a helpful assistant for math problems thinking process analysis. 
                    The user will give you the target of the thinking step and the content of this step. 
                    You need to analyze the content of this step and give me the result of this step. 
                    The result should be a single line of text, do not include any other information. 
                    However, if some new variables are defined or some new theorems are recalled, 
                    you should also include them in the result"""
                },
                {
                    "role": "user", 
                    "content": f"Target: {task} - {objective}. \nPlease give me the result of this step: {content}."
                }
            ]
        )
        step_results.append(result.choices[0].message.content)
        
    return step_results

def generate_xml_files(problem_index, step_tasks, step_objectives, step_contents, step_results, parallel_groups, output_dir="./xml_output/"):
    """
    Generate main_thread_{problem_index}.xml and thread_1_{problem_index}.xml files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create thread ID to step mapping
    thread_id_to_step = {}
    thread_id_counter = 0
    
    main_file_path = os.path.join(output_dir, f"main_thread_{problem_index}.xml")
    thread_file_path = os.path.join(output_dir, f"thread_1_{problem_index}.xml")
    
    # Generate main thread XML
    with open(main_file_path, "w", encoding='utf-8') as output_file:
        # Process each parallel group
        for i in range(len(parallel_groups)):
            if len(parallel_groups[i]) > 1:
                # Parallel processing group
                output_file.write(f"<launch_threads>\n")
                for j in range(len(parallel_groups[i])):
                    step_index = parallel_groups[i][j] - 1
                    thread_id_to_step[thread_id_counter] = step_index

                    output_file.write(f"<thread id='{thread_id_counter}'>\n")
                    output_file.write(f"<task>\n")
                    output_file.write(f"{step_tasks[step_index]}\n")
                    output_file.write(f"</task>\n")
                    output_file.write(f"<objective>\n")
                    output_file.write(f"{step_objectives[step_index]}\n")
                    output_file.write(f"</objective>\n")
                    output_file.write(f"</thread>\n")
                    
                    thread_id_counter += 1
                    
                output_file.write(f"</launch_threads>\n")
                output_file.write(f"<step_resolution>\n")
                
                # Get the thread IDs for this parallel group
                group_thread_ids = list(range(thread_id_counter - len(parallel_groups[i]), thread_id_counter))
                
                for thread_id in group_thread_ids:
                    step_index = thread_id_to_step[thread_id]
                    output_file.write(f"<thread_result id='{thread_id}'>\n")
                    output_file.write(f"{step_results[step_index]}\n")
                    output_file.write(f"</thread_result>\n")
                    
                output_file.write(f"</step_resolution>\n")
            else:
                # For non-parallel steps, just write the content directly
                step_index = parallel_groups[i][0] - 1
                output_file.write(step_contents[step_index])
                output_file.write("\n")
        
        # Handle the last step if it exists and wasn't processed
        if len(step_contents) > 0:
            last_step_processed = False
            for group in parallel_groups:
                if len(step_contents) in group:
                    last_step_processed = True
                    break
            
            if not last_step_processed:
                output_file.write(step_contents[-1])
                output_file.write("\n")
    
    # Generate thread XML
    with open(thread_file_path, "w", encoding='utf-8') as f:
        for thread_id, step_index in thread_id_to_step.items():
            f.write(f"<thread id='{thread_id}'>\n")
            f.write(f"<task>\n")
            f.write(f"{step_tasks[step_index]}\n")
            f.write(f"</task>\n")
            f.write(f"<objective>\n")
            f.write(f"{step_objectives[step_index]}\n")
            f.write(f"</objective>\n")
            f.write(f"</thread>\n")
            f.write(f"<thread_processing id='{thread_id}'>\n")
            f.write(f"{step_contents[step_index]}")   
            f.write(f"</thread_processing>\n")
            f.write(f"<thread_result id='{thread_id}'>\n")
            f.write(f"{step_results[step_index]}")
            f.write(f"</thread_result>\n")
        
    return main_file_path, thread_file_path

def process_single_txt_file(txt_file_path, problem_index, output_dir="./xml_output/"):
    """
    Process a single txt file and generate XML files.
    
    Args:
        txt_file_path (str): Path to the txt file
        problem_index (int): Problem index for naming output files
        output_dir (str): Directory to save XML files
        
    Returns:
        dict: Result dictionary
    """
    try:
        print(f"Processing {txt_file_path} (problem {problem_index})")
        
        # Read txt file
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            txt_content = f.read()
        
        # Extract steps
        step_tasks, step_objectives, step_contents = extract_steps_from_txt(txt_content)
        print(f"Extracted {len(step_tasks)} steps")
        
        # Identify parallel groups
        parallel_groups = identify_parallel_groups_with_llm(step_tasks, step_objectives, step_contents)
        print(f"Identified parallel groups: {parallel_groups}")
        
        # Get step results
        step_results = get_step_results(step_tasks, step_objectives, step_contents)
        print(f"Generated {len(step_results)} step results")
        
        # Generate XML files
        main_xml_path, thread_xml_path = generate_xml_files(
            problem_index, step_tasks, step_objectives, step_contents, step_results, parallel_groups, output_dir
        )
        
        return {
            "problem_index": problem_index,
            "txt_file_path": txt_file_path,
            "status": "success",
            "main_xml_path": main_xml_path,
            "thread_xml_path": thread_xml_path,
            "num_steps": len(step_tasks),
            "parallel_groups": parallel_groups
        }
        
    except Exception as e:
        print(f"Error processing {txt_file_path}: {e}")
        return {
            "problem_index": problem_index,
            "txt_file_path": txt_file_path,
            "status": "error",
            "error": str(e)
        }

def find_txt_files(input_dir="./"):
    """
    Find all txt files in the input directory that contain step content.
    
    Returns:
        list: List of tuples (txt_file_path, problem_index)
    """
    txt_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            
            # Try to extract problem index from filename
            # Support various naming patterns
            problem_index = None
            
            # Pattern 1: temp_direct_{index}.txt
            if filename.startswith("temp_direct_"):
                try:
                    index_str = filename.replace("temp_direct_", "").replace(".txt", "")
                    problem_index = int(index_str)
                except ValueError:
                    continue
            
            # Pattern 2: temp_{index}.txt
            elif filename.startswith("temp_") and not filename.startswith("temp_direct_"):
                try:
                    index_str = filename.replace("temp_", "").replace(".txt", "")
                    problem_index = int(index_str)
                except ValueError:
                    continue
            
            # Pattern 3: problem_{index}.txt
            elif filename.startswith("problem_"):
                try:
                    index_str = filename.replace("problem_", "").replace(".txt", "")
                    problem_index = int(index_str)
                except ValueError:
                    continue
            
            # Pattern 4: Use filename without extension as index if it's a number
            else:
                try:
                    name_without_ext = filename.replace(".txt", "")
                    problem_index = int(name_without_ext)
                except ValueError:
                    # Use hash of filename as index if no number can be extracted
                    problem_index = hash(filename) % 10000
            
            # Verify the file contains step content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "Step " in content and len(content.strip()) > 0:
                        txt_files.append((file_path, problem_index))
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
    
    txt_files.sort(key=lambda x: x[1])  # Sort by problem index
    print(f"Found {len(txt_files)} txt files with step content")
    return txt_files

def main_sequential_processing(input_dir="./", output_dir="./xml_output/"):
    """
    Process all txt files sequentially.
    """
    txt_files = find_txt_files(input_dir)
    
    if not txt_files:
        print("No txt files with step content found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    results_file = "parallel_step_processing_results.jsonl"
    
    with open(results_file, "w", encoding='utf-8') as output_file:
        for txt_file_path, problem_index in txt_files:
            result = process_single_txt_file(txt_file_path, problem_index, output_dir)
            results.append(result)
            
            # Write result to JSONL file
            json.dump(result, output_file, ensure_ascii=False)
            output_file.write('\n')
            
            print(f"Completed {len(results)}/{len(txt_files)} files")
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    print(f"\n=== SUMMARY ===")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"XML files saved to: {output_dir}")
    print(f"Results log: {results_file}")

def main_parallel_processing(input_dir="./", output_dir="./xml_output/", max_workers=5):
    """
    Process all txt files in parallel.
    
    Args:
        input_dir (str): Directory containing txt files
        output_dir (str): Directory to save XML files
        max_workers (int): Maximum number of parallel workers
    """
    txt_files = find_txt_files(input_dir)
    
    if not txt_files:
        print("No txt files with step content found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(txt_files)} files with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {}
        for txt_file_path, problem_index in txt_files:
            future = executor.submit(process_single_txt_file, txt_file_path, problem_index, output_dir)
            futures[future] = (txt_file_path, problem_index)
        
        print("All tasks submitted! Collecting results...")
        
        results = []
        results_file = "parallel_step_processing_results_parallel.jsonl"
        
        with open(results_file, "w", encoding='utf-8') as output_file:
            for future in as_completed(futures):
                txt_file_path, problem_index = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    json.dump(result, output_file, ensure_ascii=False)
                    output_file.write('\n')
                    print(f"✓ Completed {txt_file_path} ({len(results)}/{len(txt_files)})")
                except Exception as e:
                    error_result = {
                        "problem_index": problem_index,
                        "txt_file_path": txt_file_path,
                        "status": "error",
                        "error": str(e)
                    }
                    results.append(error_result)
                    json.dump(error_result, output_file, ensure_ascii=False)
                    output_file.write('\n')
                    print(f"✗ Failed {txt_file_path}: {e}")
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    print(f"\n=== SUMMARY ===")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"XML files saved to: {output_dir}")
    print(f"Results log: {results_file}")

def main_batched_processing(input_dir="./", output_dir="./xml_output/", batch_size=10, delay_seconds=30):
    """
    Process txt files in batches with delays between batches.
    
    Args:
        input_dir (str): Directory containing txt files
        output_dir (str): Directory to save XML files
        batch_size (int): Number of files per batch
        delay_seconds (int): Seconds to wait between batches
    """
    txt_files = find_txt_files(input_dir)
    
    if not txt_files:
        print("No txt files with step content found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into batches
    batches = []
    for i in range(0, len(txt_files), batch_size):
        batch = txt_files[i:i + batch_size]
        batches.append(batch)
    
    print(f"Processing {len(txt_files)} files in {len(batches)} batches")
    print(f"Batch size: {batch_size}, Delay: {delay_seconds} seconds")
    
    all_results = []
    results_file = f"parallel_step_processing_batched_size{batch_size}_delay{delay_seconds}s.jsonl"
    
    with open(results_file, "w", encoding='utf-8') as output_file:
        for batch_num, batch_files in enumerate(batches, 1):
            print(f"\n=== Batch {batch_num}/{len(batches)}: {[f[1] for f in batch_files]} ===")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(batch_size, 10)) as executor:
                futures = {}
                for txt_file_path, problem_index in batch_files:
                    future = executor.submit(process_single_txt_file, txt_file_path, problem_index, output_dir)
                    futures[future] = (txt_file_path, problem_index)
                
                # Collect results for this batch
                batch_results = []
                for future in as_completed(futures):
                    txt_file_path, problem_index = futures[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        json.dump(result, output_file, ensure_ascii=False)
                        output_file.write('\n')
                        print(f"✓ {txt_file_path}")
                    except Exception as e:
                        error_result = {
                            "problem_index": problem_index,
                            "txt_file_path": txt_file_path,
                            "status": "error",
                            "error": str(e)
                        }
                        batch_results.append(error_result)
                        json.dump(error_result, output_file, ensure_ascii=False)
                        output_file.write('\n')
                        print(f"✗ {txt_file_path}: {e}")
                
                all_results.extend(batch_results)
            
            # Delay between batches (except for the last batch)
            if batch_num < len(batches):
                print(f"Waiting {delay_seconds} seconds before next batch...")
                time.sleep(delay_seconds)
    
    # Summary
    successful = sum(1 for r in all_results if r["status"] == "success")
    failed = len(all_results) - successful
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total processed: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"XML files saved to: {output_dir}")
    print(f"Results log: {results_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process txt files and generate XML files with parallel processing")
    parser.add_argument("--mode", choices=["sequential", "parallel", "batched"], default="parallel",
                       help="Processing mode (default: parallel)")
    parser.add_argument("--input-dir", default="./test_temp", help="Input directory containing txt files (default: ./)")
    parser.add_argument("--output-dir", default="./xml_output_test/", help="Output directory for XML files (default: ./xml_output/)")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of parallel workers (default: 5)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for batched processing (default: 10)")
    parser.add_argument("--delay-seconds", type=int, default=30, help="Delay between batches in seconds (default: 30)")
    
    args = parser.parse_args()
    
    if args.mode == "sequential":
        main_sequential_processing(args.input_dir, args.output_dir)
    elif args.mode == "parallel":
        main_parallel_processing(args.input_dir, args.output_dir, args.max_workers)
    elif args.mode == "batched":
        main_batched_processing(args.input_dir, args.output_dir, args.batch_size, args.delay_seconds)