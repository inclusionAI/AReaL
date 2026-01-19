from openai import OpenAI
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback

# Initialize client
client = OpenAI()

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_WORKERS = 10  # Number of concurrent threads
OUTPUT_DIR = "chatgpt_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thread-safe file writing
write_lock = Lock()

def write_result(filename, data):
    """Thread-safe result writing"""
    with write_lock:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def call_chatgpt_with_retry(problem, problem_id, max_retries=MAX_RETRIES):
    """Call ChatGPT Responses API with retry mechanism"""
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model="o3",
                input=problem,
                reasoning={
                    "effort": "low",
                    "summary": "detailed",
                },
            )

            thought_summary_chunks = []
            for item in response.output or []:
                if item.type == "reasoning":
                    parts = [
                        part.text
                        for part in (item.summary or [])
                        if getattr(part, "type", None) == "summary_text"
                    ]
                    if parts:
                        thought_summary_chunks.append(" ".join(parts))

            thought_summary = "\n".join(chunk.strip() for chunk in thought_summary_chunks if chunk)
            final_answer = response.output_text or ""

            return {
                "success": True,
                "thought_summary": thought_summary,
                "final_answer": final_answer,
                "problem_id": problem_id,
                "attempts": attempt + 1
            }

        except Exception as e:
            print(f"Error on attempt {attempt + 1}/{max_retries} for problem {problem_id}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "problem_id": problem_id,
                    "attempts": max_retries
                }

def process_problem(problem_data, file_source):
    """Process a single problem"""
    problem_id = problem_data.get('id', 'unknown')
    problem_text = problem_data.get('problem', '')
    
    print(f"Processing problem {problem_id} from {file_source}...")
    
    result = call_chatgpt_with_retry(problem_text, problem_id)
    
    # Add metadata
    result['source_file'] = file_source
    result['original_problem'] = problem_text
    result['expected_answer'] = problem_data.get('answer', None)
    
    # Write result to file
    output_file = os.path.join(OUTPUT_DIR, f"{file_source}_results.jsonl")
    write_result(output_file, result)
    
    status = "✓" if result['success'] else "✗"
    print(f"{status} Completed problem {problem_id} from {file_source} (attempts: {result.get('attempts', 0)})")
    
    return result

def load_problems_from_file(filename):
    """Load problems from JSONL file"""
    problems = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems

def main():
    print("=" * 80)
    print("AIME Problem Solver with ChatGPT Responses API")
    print("=" * 80)
    
    # Load problems from both files
    print("\nLoading problems...")
    problems = load_problems_from_file('s1-parallel/data_100.jsonl')
    # aime25_problems = load_problems_from_file('AIME25.jsonl')
    
    print(f"Loaded {len(problems)} problems from s1-parallel/data_100.jsonl")
    print(f"Total: {len(problems)} problems")
    
    # Prepare task list
    tasks = []
    for problem in problems:
        tasks.append(('data_100', problem))
    
    # Process problems with thread pool
    print(f"\nStarting processing with {MAX_WORKERS} workers...")
    print(f"Results will be saved to {OUTPUT_DIR}/ directory")
    print("-" * 80)
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_problem, problem, source): (source, problem)
            for source, problem in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            source, problem = future_to_task[future]
            try:
                result = future.result()
                if result['success']:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"✗ Exception processing problem from {source}: {str(e)}")
                failed += 1
    
    # Summary
    elapsed_time = time.time() - start_time
    print("-" * 80)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total problems processed: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per problem: {elapsed_time/len(tasks):.2f} seconds")
    result_files = {os.path.join(OUTPUT_DIR, f"{source}_results.jsonl") for source, _ in tasks}
    print(f"\nResults saved to:")
    for file_path in sorted(result_files):
        print(f"  - {file_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
