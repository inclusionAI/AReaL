import os
import re
from api_call import call_model_claude
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# System prompt for generating objective and result
OBJECTIVE_RESULT_SYSTEM_PROMPT = """You are an expert at analyzing mathematical problem-solving steps. 
For each step provided, you need to:
1. Generate a concise OBJECTIVE that describes what this step aims to accomplish. Use one imperative sentence to describe the goal of the step.
2. Generate a concise RESULT that summarizes what was achieved or concluded in this step

Format your response as:
OBJECTIVE: [your objective here]
RESULT: [your result here]

Keep both objective and result concise (1-2 sentences each).

You should respond with **ONE** objective and **ONE** result.
"""


def parse_steps_from_file(file_path):
    """Parse a text file and extract individual steps."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by step headers (e.g., "Step 1", "Step 2", etc.)
    step_pattern = r'Step \d+\n={70,}\n'
    steps = re.split(step_pattern, content)
    
    # Remove empty first element if exists
    steps = [step.strip() for step in steps if step.strip()]
    
    # Get step numbers
    step_numbers = re.findall(r'Step (\d+)', content)
    
    return list(zip(step_numbers, steps))


def generate_objective_and_result(step_content):
    """Call Gemini model to generate objective and result for a step."""
    user_prompt = f"""Please analyze the following mathematical reasoning step and provide an objective and result:

{step_content}
"""
    
    try:
        response = call_model_claude(
            user_prompt=user_prompt,
            system_prompt=OBJECTIVE_RESULT_SYSTEM_PROMPT,
            model="gemini-2.5-flash",
            temperature=0.3,
            max_retries=6
        )
        
        if response and 'choices' in response:
            content = response['choices'][0]['message']['content']
            return content
        else:
            return "OBJECTIVE: Error generating objective\nRESULT: Error generating result"
    except Exception as e:
        print(f"Error calling model: {e}")
        return "OBJECTIVE: Error generating objective\nRESULT: Error generating result"


def process_file(input_file_path, output_dir):
    """Process a single file and save the enhanced version."""
    thread_id = threading.current_thread().name
    print(f"\n[{thread_id}] Processing file: {input_file_path}")
    
    # Parse steps from file
    steps = parse_steps_from_file(input_file_path)
    print(f"[{thread_id}] Found {len(steps)} steps")
    
    # Process each step
    enhanced_content = []
    for step_num, step_content in steps:
        print(f"\n[{thread_id}] Processing Step {step_num}...")
        
        # Generate objective and result
        obj_result = generate_objective_and_result(step_content)
        
        # Format the enhanced step
        enhanced_step = f"""Step {step_num}
{'='*80}
{obj_result}

{step_content}
"""
        enhanced_content.append(enhanced_step)
    
    # Create output file
    input_filename = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_dir, f"enhanced_{input_filename}")
    
    # Write enhanced content
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(enhanced_content))
    
    print(f"\n[{thread_id}] Saved enhanced file to: {output_file_path}")
    return output_file_path


def process_directory(input_dir, output_dir, max_workers=4):
    """Process all txt files in a directory using parallel threads."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all txt files
    input_path = Path(input_dir)
    txt_files = list(input_path.glob("*.txt"))
    
    print(f"Found {len(txt_files)} txt files to process")
    print(f"Using {max_workers} parallel threads")
    
    # Process files in parallel
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file, str(txt_file), output_dir): txt_file
            for txt_file in txt_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            txt_file = future_to_file[future]
            try:
                result = future.result()
                successful += 1
                print(f"\n✓ Successfully processed: {txt_file.name}")
            except Exception as e:
                failed += 1
                print(f"\n✗ Error processing {txt_file}: {e}")
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Enhanced files saved to: {output_dir}")


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "/Users/zzy/Desktop/data_curation_new/new_result_merged"
    OUTPUT_DIR = "/Users/zzy/Desktop/data_curation_new/new_result_with_objectives"
    MAX_WORKERS = 20  # Number of parallel threads
    
    # Process all files in the directory
    process_directory(INPUT_DIR, OUTPUT_DIR, max_workers=MAX_WORKERS)
