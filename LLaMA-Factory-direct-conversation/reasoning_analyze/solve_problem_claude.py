from openai import AzureOpenAI
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback

# Initialize client
# api_key = "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6IjEzOGQ4MDFlLTllN2ItNDVjNC05OTU1LWZiZTU0YzE4NDc0MCIsInNlY3JldCI6InVteTY2YVZNWDFuZE54ckxwejZ2VkpOTm5wcmRUOGJ1RXlzeklzblo0SUE9In0.Y1VXsU5CfxnMbxbMPYyZT4j4pVhYdIrG1QrChDclJTg"
# api_base = "https://llm-proxy.perflab.nvidia.com"
api_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
api_base = os.environ.get("AZURE_OPENAI_BASE", None)
assert api_key is not None, "AZURE_OPENAI_API_KEY not set"
assert api_base is not None, "AZURE_OPENAI_BASE not set"
api_version = "2025-02-01-preview"
deployment_name = os.environ.get("MODEL", "claude-sonnet-4-20250514") # Keeping this variable but not using it as model name per example

print(f"Evaluting problems with claude {deployment_name}...")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=api_base,
    api_key=api_key,
)

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_WORKERS = 10  # Number of concurrent threads
OUTPUT_DIR = "claude_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thread-safe file writing
write_lock = Lock()

def write_result(filename, data):
    """Thread-safe result writing"""
    with write_lock:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def call_claude_with_retry(problem, problem_id, max_retries=MAX_RETRIES):
    """Call Claude API with retry mechanism"""
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                model="claude-opus-4-5-20251101",
                messages=[{"role": "user", "content": [{"type": "text", "text": problem}]}],
                extra_body={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 8000 # Must be higher than 1024
                    }
                },
                max_tokens=15000 # Must be higher than the budget_tokens
            )

            # Extract reasoning and content
            # Check if model_extra exists and has reasoning_content, otherwise default to empty
            model_extra = getattr(chat_completion.choices[0].message, 'model_extra', {})
            reasoning_content = model_extra.get('reasoning_content', '') if model_extra else None
            assert reasoning_content is not None, "reasoning_content is not found in model_extra"
            
            # If reasoning_content is not directly there, try to see if it's in the extra_body logic or standard response
            # constructing the thread_summary from reasoning_content
            
            final_answer = chat_completion.choices[0].message.content or ""

            return {
                "success": True,
                "thought_summary": reasoning_content,
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
    
    result = call_claude_with_retry(problem_text, problem_id)
    
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
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found.")
        return []
        
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    problems.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid json line in {filename}")
    return problems

def main():
    print("=" * 80)
    print("AIME Problem Solver with Claude API (AzureOpenAI)")
    print("=" * 80)
    
    # Load problems
    print("\nLoading problems...")
    input_file = 's1-parallel/data_100.jsonl'
    problems = load_problems_from_file(input_file)
    
    print(f"Loaded {len(problems)} problems from {input_file}")
    
    if not problems:
        print("No problems found. Exiting.")
        return

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
    if len(tasks) > 0:
        print(f"Average time per problem: {elapsed_time/len(tasks):.2f} seconds")
    
    result_files = {os.path.join(OUTPUT_DIR, f"{source}_{deployment_name}_results.jsonl") for source, _ in tasks}
    print(f"\nResults saved to:")
    for file_path in sorted(result_files):
        print(f"  - {file_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
