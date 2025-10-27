from Batch_jsonl import jsonLTOdict
from agent_new import main
import json
import xml.etree.ElementTree as ET
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_xml_content(filepath):
    """
    Load XML file content as a string.
    
    Args:
        filepath (str): Path to the XML file
        
    Returns:
        str: XML content as string, or empty string if file doesn't exist
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except (FileNotFoundError, IOError):
        return ""

def process_problem_return_result(problem_data, problem_index):
    """
    Process a single problem and return result instead of writing directly.
    """
    problem = problem_data.get('Problem', problem_data.get('problem', ''))
    cot = problem_data.get('CoT', '')
    print(f"Processing problem {problem_index}")
    
    # Call main function which generates XML files
    main(problem, cot, problem_index)
    
    # Load the generated XML files
    main_thread_xml = load_xml_content(f"temp_main_{problem_index}.xml")
    thread_1_xml = load_xml_content(f"temp_thread1_{problem_index}.xml")
    
    # Create result dictionary
    result = {
        "problem_index": problem_index,
        "original_problem": problem,
        "original_cot": cot,
        "main_thread": main_thread_xml,
        "thread_1": thread_1_xml
    }
    
    # Clean up XML files after processing
    for xml_file in [f"temp_main_{problem_index}.xml", f"temp_thread1_{problem_index}.xml"]:
        if os.path.exists(xml_file):
            os.remove(xml_file)
    
    return result

def main_batch_1001_to_1500_with_delay():
    """
    Process problems 1001-1500 in batches of 50 with 1-minute delays.
    """
    # Load data from JSONL file
    dict_list = jsonLTOdict(length_limit=1500)  # Load up to 1500 problems
    print(f"Loaded {len(dict_list)} problems from JSONL")
    
    # Extract problems 1001-1500 (indices 1000-1499)
    if len(dict_list) < 1500:
        print(f"Warning: Only {len(dict_list)} problems available, need 1500")
        return
    
    target_problems = dict_list[1000:1500]  # Problems 1001-1500
    print(f"Processing problems 1001-1500 ({len(target_problems)} problems)")
    
    # Split into batches of 50
    batch_size = 50
    batches = []
    for i in range(0, len(target_problems), batch_size):
        batch = target_problems[i:i + batch_size]
        batch_start_index = 1001 + i  # Start from problem 1001
        batches.append((batch, batch_start_index))
    
    print(f"Split into {len(batches)} batches of up to {batch_size} problems each")
    
    all_results = []
    
    with open("output_1001_to_1500_batched.jsonl", "w", encoding='utf-8') as output_file:
        for batch_num, (batch_problems, batch_start_index) in enumerate(batches, 1):
            print(f"\n=== Processing Batch {batch_num}/{len(batches)} ===")
            
            # Calculate problem indices for this batch
            batch_indices = list(range(batch_start_index, batch_start_index + len(batch_problems)))
            print(f"Problems in this batch: {batch_indices[0]}-{batch_indices[-1]}")
            
            # Process current batch in parallel
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = {}
                for i, problem_data in enumerate(batch_problems):
                    problem_index = batch_start_index + i
                    future = executor.submit(process_problem_return_result, problem_data, problem_index)
                    futures[future] = problem_index
                
                print(f"Submitted {len(batch_problems)} tasks for batch {batch_num}")
                
                # Collect results for this batch
                batch_results = []
                for future in as_completed(futures):
                    problem_index = futures[future]
                    try:
                        result = future.result()
                        batch_results.append((problem_index, result))
                        print(f"✓ Completed problem {problem_index}")
                    except Exception as e:
                        print(f"✗ Problem {problem_index} failed: {e}")
                        # Still write error result
                        error_result = {
                            "problem_index": problem_index,
                            "original_problem": "",
                            "original_cot": "",
                            "main_thread": "",
                            "thread_1": "",
                            "error": str(e)
                        }
                        batch_results.append((problem_index, error_result))
                
                # Sort batch results by problem_index and write to file
                batch_results.sort(key=lambda x: x[0])
                for problem_index, result in batch_results:
                    json.dump(result, output_file, ensure_ascii=False)
                    output_file.write('\n')
                    all_results.append(result)
            
            print(f"Batch {batch_num} completed with {len(batch_results)} results")
            
            # Wait 1 minute before next batch (except for the last batch)
            if batch_num < len(batches):
                print("Waiting 1 minute before next batch to reduce server load...")
                for remaining in range(60, 0, -10):  # Count down from 60 seconds
                    print(f"Next batch starts in {remaining} seconds...", end='\r')
                    time.sleep(10)
                print("\nStarting next batch...")
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total problems processed: {len(all_results)}")
    print(f"Total batches: {len(batches)}")
    print(f"Results saved to: output_1001_to_1500_batched.jsonl")
    
    # Count successful vs failed
    successful = sum(1 for r in all_results if "error" not in r)
    failed = len(all_results) - successful
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

def main_batch_custom_range(start_index, end_index, batch_size=50, delay_minutes=1):
    """
    Process problems in a custom range with custom batch size and delay.
    
    Args:
        start_index (int): Starting problem index (1-based)
        end_index (int): Ending problem index (1-based)
        batch_size (int): Number of problems per batch
        delay_minutes (int): Minutes to wait between batches
    """
    # Load data from JSONL file
    dict_list = jsonLTOdict(length_limit=max(end_index, 1500))
    print(f"Loaded {len(dict_list)} problems from JSONL")
    
    # Extract target problems (convert to 0-based indexing)
    if len(dict_list) < end_index:
        print(f"Warning: Only {len(dict_list)} problems available, need {end_index}")
        return
    
    target_problems = dict_list[start_index-1:end_index]
    print(f"Processing problems {start_index}-{end_index} ({len(target_problems)} problems)")
    
    # Split into batches
    batches = []
    for i in range(0, len(target_problems), batch_size):
        batch = target_problems[i:i + batch_size]
        batch_start_index = start_index + i
        batches.append((batch, batch_start_index))
    
    print(f"Split into {len(batches)} batches of up to {batch_size} problems each")
    
    output_filename = f"output_{start_index}_to_{end_index}_batch{batch_size}_delay{delay_minutes}min.jsonl"
    
    with open(output_filename, "w", encoding='utf-8') as output_file:
        for batch_num, (batch_problems, batch_start_index) in enumerate(batches, 1):
            print(f"\n=== Batch {batch_num}/{len(batches)}: {batch_start_index}-{batch_start_index + len(batch_problems) - 1} ===")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(batch_size, 50)) as executor:
                futures = {}
                for i, problem_data in enumerate(batch_problems):
                    problem_index = batch_start_index + i
                    future = executor.submit(process_problem_return_result, problem_data, problem_index)
                    futures[future] = problem_index
                
                # Collect results
                batch_results = []
                for future in as_completed(futures):
                    problem_index = futures[future]
                    try:
                        result = future.result()
                        batch_results.append((problem_index, result))
                        print(f"✓ Problem {problem_index}")
                    except Exception as e:
                        print(f"✗ Problem {problem_index}: {e}")
                        error_result = {
                            "problem_index": problem_index,
                            "error": str(e)
                        }
                        batch_results.append((problem_index, error_result))
                
                # Write results
                batch_results.sort(key=lambda x: x[0])
                for _, result in batch_results:
                    json.dump(result, output_file, ensure_ascii=False)
                    output_file.write('\n')
            
            print(f"Batch {batch_num} completed")
            
            # Wait between batches
            if batch_num < len(batches):
                delay_seconds = delay_minutes * 1
                print(f"Waiting {delay_minutes} minute(s)...")
                for remaining in range(delay_seconds, 0, -10):
                    mins, secs = divmod(remaining, 60)
                    print(f"Next batch in {mins:02d}:{secs:02d}", end='\r')
                    time.sleep(10)
                print("\n")
    
    print(f"All batches completed! Results in {output_filename}")

if __name__ == "__main__":
    # Choose which function to run
    print("Available options:")
    print("1. Process problems 1001-1500 (50 per batch, 1-minute delay)")
    print("2. Custom range and batch settings")
    
    choice = input("Enter choice (1/2) or press Enter for option 1: ").strip()
    
    if choice == "2":
        start_index = int(input("Start index (default 1001): ") or "1001")
        end_index = int(input("End index (default 1500): ") or "1500")
        batch_size = int(input("Batch size (default 50): ") or "50")
        delay_minutes = int(input("Delay minutes (default 1): ") or "1")
        main_batch_custom_range(start_index, end_index, batch_size, delay_minutes)
    else:
        main_batch_1001_to_1500_with_delay()