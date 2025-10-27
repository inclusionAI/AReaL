import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from recovery_agent import save_xml_from_temp_recovery
import time

def find_available_temp_files():
    """
    Find all available temp files and return their indices.
    Supports temp_direct_{index}.txt format.
    
    Returns:
        list: Available problem indices
    """
    available_indices = []
    for filename in os.listdir("."):
        if filename.startswith("temp_direct_") and filename.endswith(".txt"):
            try:
                # Extract index from temp_direct_{index}.txt
                index_str = filename.replace("temp_direct_", "").replace(".txt", "")
                problem_index = int(index_str)
                
                available_indices.append(problem_index)
            except ValueError:
                continue
    
    available_indices.sort()
    print(f"Found {len(available_indices)} temp_direct_{{index}}.txt files: {available_indices}")
    return available_indices

def process_single_temp_file(problem_index, output_dir="./xml_output_higher_parallel/"):
    """
    Process a single temp file and generate XML files.
    
    Args:
        problem_index (int): The problem index
        output_dir (str): Directory to save XML files
        
    Returns:
        dict: Result dictionary
    """
    try:
        print(f"Processing problem {problem_index}")
        
        # Use the save_xml_from_temp_recovery function
        save_xml_from_temp_recovery(problem_index, output_dir)
        
        return {
            "problem_index": problem_index,
            "status": "success",
            "main_xml_path": f"{output_dir}/temp_main_{problem_index}.xml",
            "thread_xml_path": f"{output_dir}/temp_thread1_{problem_index}.xml"
        }
        
    except Exception as e:
        print(f"Error processing problem {problem_index}: {e}")
        return {
            "problem_index": problem_index,
            "status": "error",
            "error": str(e)
        }

def main_batch_xml_generation():
    """
    Main batch processing function that generates XML files from all available temp files.
    """
    # Find all available temp files
    available_indices = find_available_temp_files()
    
    if not available_indices:
        print("No temp files found!")
        return
    
    output_dir = "./xml_output_new_no_first_step_parallel/"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(available_indices)} problems...")
    
    results = []
    with open("xml_generation_results.jsonl", "w", encoding='utf-8') as output_file:
        for problem_index in available_indices:
            result = process_single_temp_file(problem_index, output_dir)
            results.append(result)
            
            # Write result to JSONL file
            json.dump(result, output_file, ensure_ascii=False)
            output_file.write('\n')
            
            print(f"Completed {len(results)}/{len(available_indices)} problems")
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    print(f"\n=== SUMMARY ===")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"XML files saved to: {output_dir}")
    print(f"Results log: xml_generation_results.jsonl")

def main_parallel_xml_generation(max_workers=10):
    """
    Generate XML files from temp files using parallel processing.
    
    Args:
        max_workers (int): Maximum number of parallel workers
    """
    available_indices = find_available_temp_files()
    
    if not available_indices:
        print("No temp files found!")
        return
    
    output_dir = "./xml_output_no_first_parallel_large_conclusion/"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(available_indices)} problems with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {}
        for problem_index in available_indices:
            future = executor.submit(process_single_temp_file, problem_index, output_dir)
            futures[future] = problem_index
        
        print("All tasks submitted! Collecting results...")
        
        results = []
        with open("xml_generation_results_parallel.jsonl", "w", encoding='utf-8') as output_file:
            for future in as_completed(futures):
                problem_index = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    json.dump(result, output_file, ensure_ascii=False)
                    output_file.write('\n')
                    print(f"✓ Completed problem {problem_index} ({len(results)}/{len(available_indices)})")
                except Exception as e:
                    error_result = {
                        "problem_index": problem_index,
                        "status": "error",
                        "error": str(e)
                    }
                    results.append(error_result)
                    json.dump(error_result, output_file, ensure_ascii=False)
                    output_file.write('\n')
                    print(f"✗ Failed problem {problem_index}: {e}")
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    print(f"\n=== SUMMARY ===")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"XML files saved to: {output_dir}")
    print(f"Results log: xml_generation_results_parallel.jsonl")

def main_batched_xml_generation(batch_size=10, delay_minutes=1):
    """
    Generate XML files in batches with delays to avoid overloading.
    
    Args:
        batch_size (int): Number of problems per batch
        delay_minutes (int): Minutes to wait between batches
    """
    available_indices = find_available_temp_files()
    
    if not available_indices:
        print("No temp files found!")
        return
    
    output_dir = "./xml_output_new_no_first_step_parallel/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into batches
    batches = []
    for i in range(0, len(available_indices), batch_size):
        batch = available_indices[i:i + batch_size]
        batches.append(batch)
    
    print(f"Processing {len(available_indices)} problems in {len(batches)} batches")
    print(f"Batch size: {batch_size}, Delay: {delay_minutes} minutes")
    
    all_results = []
    output_filename = f"xml_generation_batched_size{batch_size}_delay{delay_minutes}min.jsonl"
    
    with open(output_filename, "w", encoding='utf-8') as output_file:
        for batch_num, batch_indices in enumerate(batches, 1):
            print(f"\n=== Batch {batch_num}/{len(batches)}: {batch_indices} ===")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(batch_size, 10)) as executor:
                futures = {}
                for problem_index in batch_indices:
                    future = executor.submit(process_single_temp_file, problem_index, output_dir)
                    futures[future] = problem_index
                
                # Collect results for this batch
                batch_results = []
                for future in as_completed(futures):
                    problem_index = futures[future]
                    try:
                        result = future.result()
                        batch_results.append((problem_index, result))
                        print(f"✓ Problem {problem_index}")
                    except Exception as e:
                        error_result = {
                            "problem_index": problem_index,
                            "status": "error",
                            "error": str(e)
                        }
                        batch_results.append((problem_index, error_result))
                        print(f"✗ Problem {problem_index}: {e}")
                
                # Sort and write results
                batch_results.sort(key=lambda x: x[0])
                for _, result in batch_results:
                    json.dump(result, output_file, ensure_ascii=False)
                    output_file.write('\n')
                    all_results.append(result)
            
            print(f"Batch {batch_num} completed with {len(batch_results)} results")
            
            # Wait between batches (except for the last batch)
            if batch_num < len(batches):
                delay_seconds = delay_minutes * 5
                print(f"Waiting {delay_minutes} minutes before next batch...")
                for remaining in range(delay_seconds, 0, -30):
                    mins, secs = divmod(remaining, 60)
                    print(f"Next batch in {mins:02d}:{secs:02d}", end='\r')
                    time.sleep(30)
                print("\n")
    
    # Summary
    successful = sum(1 for r in all_results if r["status"] == "success")
    failed = len(all_results) - successful
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total processed: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"XML files saved to: {output_dir}")
    print(f"Results log: {output_filename}")

def main_test_xml_generation(max_problems=5):
    """
    Test XML generation with a limited number of problems.
    
    Args:
        max_problems (int): Maximum number of problems to test
    """
    available_indices = find_available_temp_files()
    
    if not available_indices:
        print("No temp files found!")
        return
    
    # Limit to first few problems for testing
    test_indices = available_indices[:max_problems]
    print(f"Testing XML generation with {len(test_indices)} problems: {test_indices}")
    
    output_dir = "./xml_output_test/"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for problem_index in test_indices:
        result = process_single_temp_file(problem_index, output_dir)
        results.append(result)
        print(f"Completed {len(results)}/{len(test_indices)}")
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    print(f"\n=== TEST SUMMARY ===")
    print(f"Total tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Test XML files saved to: {output_dir}")

if __name__ == "__main__":
    print("XML Generation from Temp Files")
    print("Available options:")
    print("1. Sequential processing")
    print("2. Parallel processing")
    print("3. Batched processing with delays")
    print("4. Test with first 5 problems")
    
    choice = input("Enter choice (1/2/3/4) or press Enter for option 3: ").strip()
    
    if choice == "1":
        main_batch_xml_generation()
    elif choice == "2":
        max_workers = int(input("Max workers (default 10): ") or "10")
        main_parallel_xml_generation(max_workers)
    elif choice == "4":
        max_problems = int(input("Number of test problems (default 5): ") or "5")
        main_test_xml_generation(max_problems)
    else:
        batch_size = int(input("Batch size (default 10): ") or "10")
        delay_minutes = int(input("Delay minutes (default 1): ") or "1")
        main_batched_xml_generation(batch_size, delay_minutes)