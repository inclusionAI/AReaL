from api_call import call_model_claude
import json
import re
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


PARALLEL_GROUP_SYSTEM_PROMPT = r"""
You are a helpful assistant that analyzes mathematical problem-solving steps to identify which steps can be executed in parallel.

You will be given a sequence of steps from a mathematical problem solution. Your task is to identify groups of steps that are independent and can be executed in parallel.

Steps can be executed in parallel if:
1. They don't depend on each other's results
2. They explore different approaches or cases to the same problem
3. They verify different aspects independently

Steps CANNOT be executed in parallel if:
1. One step uses the result from another step
2. They are in a sequential chain of reasoning
3. One step builds directly upon another

You must output a list of parallel groups, where each group contains step numbers that can be executed in parallel. 

For example: [1, [2, 3], 4] means Step 1 runs first, then Steps 2 and 3 can run in parallel, followed by Step 4.

Be strict: Only group steps as parallel if you are certain they are independent based on the criteria above. The first step is for problem understanding and cannot be parallel with any other step.

Example of steps that can be runned in parallel: 

1. Continuous steps starting with "Alternatively", "On the other hand", "In another case"

2. Continuous steps that analyze different variables or parameters independently

3. Continuous steps that starts with "Wait", "But", "However", that conduct verification independently

IMPORTANT: Steps that are far away in the sequence should NOT be grouped together as parallel, even if they seem independent.

"""


def parse_steps_from_file(file_path):
    """Parse steps from a merged steps file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract only the content before the separator
    separator = "="*80 + "\nContent after </think> tag:\n" + "="*80
    original_content = content
    if separator in content:
        content = content.split(separator)[0]
    if original_content == content:
        print("No separator found; using full content")
    # Split by step separators
    steps = []
    step_pattern = r'Step (\d+)\n={80}\n(.*?)(?=Step \d+\n={80}|\Z)'
    matches = re.findall(step_pattern, content, re.DOTALL)
    
    for step_num, step_content in matches:
        steps.append({
            'step_number': int(step_num),
            'content': step_content.strip()
        })
    
    return steps


def format_steps_for_api(steps):
    """Format steps into a string for the API call."""
    formatted = []
    for step in steps:
        formatted.append(f"Step {step['step_number']}:\n{step['content']}\n")
    
    return "\n".join(formatted)


def identify_parallel_groups(file_path, model="gemini-2.5-flash"):
    """
    Identify parallel groups of steps from a merged steps file.
    
    Args:
        file_path: Path to the merged steps file
        model: Model to use for analysis
        
    Returns:
        Dictionary containing parallel groups and dependencies
    """
    # Parse steps from file
    steps = parse_steps_from_file(file_path)
    
    if not steps:
        print("No steps found in file")
        return None
    
    print(f"Found {len(steps)} steps")
    
    # Format steps for API
    user_prompt = format_steps_for_api(steps)
    
    # Call the model
    print("Calling model to identify parallel groups...")
    result = call_model_claude(
        user_prompt=user_prompt,
        system_prompt=PARALLEL_GROUP_SYSTEM_PROMPT,
        model=model,
        temperature=0.3,  # Lower temperature for more consistent output
        response_format={"type": "json_object"}  # Request JSON output
    )
    
    if not result:
        print("Failed to get response from model")
        return None
    
    # Extract the response
    response_text = result['choices'][0]['message']['content']
    
    
    return response_text
    

def print_parallel_groups(analysis):
    """Pretty print the parallel groups analysis."""
    if not analysis:
        print("No analysis to display")
        return
    
    print("\n" + "="*80)
    print("PARALLEL GROUPS ANALYSIS")
    print("="*80)
    
    if 'parallel_groups' in analysis:
        print(f"\nFound {len(analysis['parallel_groups'])} parallel groups:\n")
        
        for group in analysis['parallel_groups']:
            print(f"Group {group['group_id']}:")
            print(f"  Steps: {group['steps']}")
            print(f"  Description: {group['description']}")
            print()
    
    if 'sequential_dependencies' in analysis:
        print("Sequential Dependencies:")
        print("-" * 80)
        
        for dep in analysis['sequential_dependencies']:
            print(f"  Group {dep['before_group']} must complete before Group {dep['after_group']}")
            print(f"  Reason: {dep['reason']}")
            print()


def save_analysis(analysis, output_path):
    """Save the parallel groups analysis to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis saved to: {output_path}")


def find_txt_files(directory):
    """Find all .txt files in the given directory."""
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"Directory does not exist: {directory}")
        return []
    
    txt_files = list(directory_path.glob("*.txt"))
    return [str(f) for f in txt_files]


def process_single_file(file_path, lock=None):
    """Process a single file and save its analysis."""
    try:
        if lock:
            with lock:
                print(f"Processing: {file_path}")
        else:
            print(f"Processing: {file_path}")
        
        # Identify parallel groups
        analysis = identify_parallel_groups(file_path)
        
        if analysis:
            # Save to JSON file
            output_path = file_path.replace('.txt', '_parallel_analysis.json')
            save_analysis(analysis, output_path)
            return {"file": file_path, "status": "success", "output": output_path}
        else:
            return {"file": file_path, "status": "failed", "error": "Failed to analyze"}
    
    except Exception as e:
        return {"file": file_path, "status": "error", "error": str(e)}


def process_directory_parallel(directory, max_workers=20, model="gemini-2.5-flash"):
    """
    Process all txt files in a directory using multithreading.
    
    Args:
        directory: Path to directory containing txt files
        max_workers: Maximum number of parallel threads
        model: Model to use for analysis
        
    Returns:
        List of results for each processed file
    """
    # Find all txt files
    txt_files = find_txt_files(directory)
    
    if not txt_files:
        print(f"No .txt files found in directory: {directory}")
        return []
    
    print(f"Found {len(txt_files)} .txt files to process")
    print(f"Using {max_workers} parallel threads")
    print("="*80)
    
    # Create a lock for thread-safe printing
    print_lock = Lock()
    
    results = []
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, file_path, print_lock): file_path 
            for file_path in txt_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                with print_lock:
                    if result['status'] == 'success':
                        print(f"✓ Completed: {os.path.basename(file_path)}")
                    else:
                        print(f"✗ Failed: {os.path.basename(file_path)} - {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                with print_lock:
                    print(f"✗ Exception for {os.path.basename(file_path)}: {str(e)}")
                results.append({"file": file_path, "status": "exception", "error": str(e)})
    
    return results


def print_summary(results):
    """Print a summary of processing results."""
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = total - successful
    
    print(f"Total files: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for r in results:
            if r['status'] != 'success':
                print(f"  - {os.path.basename(r['file'])}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        target = sys.argv[1]
        max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        
        # Check if target is a directory or file
        if os.path.isdir(target):
            print(f"Processing directory: {target}")
            results = process_directory_parallel(target, max_workers=max_workers)
            print_summary(results)
        else:
            # Single file processing
            print(f"Processing single file: {target}")
            result = process_single_file(target)
            if result['status'] == 'success':
                print(f"✓ Successfully processed: {target}")
            else:
                print(f"✗ Failed to process: {target}")
                print(f"  Error: {result.get('error', 'Unknown error')}")
    else:
        # Default: process the result_merged directory
        default_dir = "/Users/zzy/Desktop/data_curation_new/new_result_merged"
        print(f"No directory specified. Using default: {default_dir}")
        print(f"Usage: python3 {sys.argv[0]} <directory_path> [max_workers]")
        print("="*80)
        
        if os.path.isdir(default_dir):
            results = process_directory_parallel(default_dir, max_workers=20)
            print_summary(results)
        else:
            print(f"Default directory does not exist: {default_dir}")
