import os
import re
import numpy as np
import requests
import json
from extract_step import extract_steps_regex
from utils import extract_step_descriptions, parse_dependency_output, identify_parallel_steps, identify_parallel_steps_no_continuous_multistep

def recover_from_temp_files(problem_index):
    """
    Recover step content processing results from temporary files when API fails.
    
    Args:
        problem_index (int): The problem index used in the temp file names
        
    Returns:
        tuple: Same format as get_step_content_direct - 
               (step_descriptions_task, step_descriptions_objective, step_content, 
                step_result, dependency_output, adj_matrix, parallel_groups)
    """
    temp_file_path = f"temp_direct_{problem_index}.txt"
    
    if not os.path.exists(temp_file_path):
        raise FileNotFoundError(f"Temporary file {temp_file_path} not found")
    
    print(f"Recovering from {temp_file_path}")
    
    # Read the temp file content
    with open(temp_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract step contents using regex and clean them
    step_content_raw = extract_steps_regex(content)
    
    # Clean the step content to remove task and objective
    step_content = []
    for content_item in step_content_raw:
        lines = content_item.strip().split('\n')
        cleaned_lines = []
        
        # Skip the first line (task) and the second line if it starts with "The model" (objective)
        start_idx = 1  # Skip first line (task)
        if len(lines) > 1 and lines[1].strip().startswith("The model"):
            start_idx = 2  # Skip both task and objective lines
        
        # Take the remaining lines as pure content
        for line in lines[start_idx:]:
            cleaned_lines.append(line)
        
        cleaned_content = '\n'.join(cleaned_lines).strip()
        step_content.append(cleaned_content)
    
    # Extract step descriptions from the temp file
    step_descriptions = []
    step_descriptions_task = []
    step_descriptions_objective = []
    
    # Parse step headers and processing content from the temp file
    step_pattern = r'Step (\d+):\s*(.*?)(?=\nStep|\Z)'
    step_matches = re.findall(step_pattern, content, re.DOTALL)
    
    for step_num, step_content_full in step_matches:
        lines = step_content_full.strip().split('\n')
        
        if len(lines) >= 1:
            # First line is the task (after "Step X: ")
            first_line = lines[0].strip()
            task = first_line
            objective = ""

            # Check if the first line contains both task and objective (e.g., "Task: Objective")
            if ':' in first_line:
                parts = first_line.split(':', 1)
                # If the part after the colon seems to be a description, split them
                if 'the model' in parts[1].lower():
                    task = parts[0].strip()
                    objective = parts[1].strip()

            step_descriptions_task.append(task)
            
            # If objective wasn't on the first line, look for it on subsequent lines
            if not objective:
                for line in lines[1:]:  # Skip the first line (task)
                    line = line.strip()
                    if line.startswith("The model"):
                        # Remove "The model" from the beginning
                        objective = line.replace("The model", "").strip()
                        break
            
            step_descriptions_objective.append(objective)
            step_descriptions.append(f"{task}: {objective}" if objective else task)
    
    print(f"Recovered {len(step_content)} step contents")
    print(f"Recovered {len(step_descriptions)} step descriptions")
    
    # Validate that we have matching numbers
    if len(step_content) != len(step_descriptions):
        print(f"Warning: Step content count ({len(step_content)}) doesn't match descriptions count ({len(step_descriptions)})")
        # Try to match them as best as possible
        min_len = min(len(step_content), len(step_descriptions))
        step_content = step_content[:min_len]
        step_descriptions = step_descriptions[:min_len]
        step_descriptions_task = step_descriptions_task[:min_len]
        step_descriptions_objective = step_descriptions_objective[:min_len]
    
    # Create placeholder values for missing components
    step_result = [f"Result of step {i+1}" for i in range(len(step_content))]
    dependency_output = "NO_DEPENDENCIES"  # Placeholder - no dependencies
    
    # Create minimal adjacency matrix (no dependencies)
    n = len(step_content)
    adj_matrix = np.zeros((n, n), dtype=int)
    
    # Create parallel groups (all steps can run in parallel since no dependencies)
    parallel_groups = [list(range(1, n+1))]
    
    print("Recovery completed successfully")
    
    return step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups

api_key = ''

def call_model(messages, model="deepseek-v3-1-250821", **kwargs):
    url = 'https://matrixllm.alipay.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "stream": False,
        "model": model,
        "messages": messages,
        **kwargs,
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        if response.status_code == 200:
            result = response.json()
            answer_text = result['choices'][0]['message']['content']
            print(result)
            return answer_text
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        print(response.text if 'response' in locals() else "No response")
    return ""

def recover_from_temp_files_enhanced(problem_index):
    """
    Enhanced recovery that regenerates results and dependencies from temp file only.
    
    Args:
        problem_index (int): The problem index used in the temp file names
        
    Returns:
        tuple: Same format as get_step_content_direct
    """
    seed = 42
    model = "deepseek-v3-1-250821"
    
    # First, do basic recovery with cleaned content
    step_descriptions_task, step_descriptions_objective, step_content, _, _, _, _ = recover_from_temp_files(problem_index)
    
    print("Regenerating results and dependencies...")
    
    # Regenerate step results
    step_result = []
    try:
        for i, (task, objective, content) in enumerate(zip(step_descriptions_task, step_descriptions_objective, step_content)):
            desc = f"{task}: {objective}" if objective else task
            messages=[
                {"role": "system", "content": "You are a helpful assistant for math problems thinking process analysis. The user will give you the target of the thinking step and the content of this step. You need to analyze the content of this step and give me the result of this step. The result should be a single line of text, do not include any other information. However, if some new variables are defined or some new theorems are recalled, you should also include them in the result"},
                {"role": "user", "content": f"Target: {desc}. \n Please give me the result of this step: {content}."}
            ]
            result_text = call_model(messages, model=model, temperature=0, seed=seed)
            step_result.append(result_text)
            print(f"Regenerated result for step {i+1}")
    except Exception as e:
        print(f"Error regenerating results: {e}")
        print("Using placeholder results")
    
    # Regenerate dependency analysis
    dependency_output = "NO_DEPENDENCIES"
    adj_matrix = np.zeros((len(step_content), len(step_content)), dtype=int)
    parallel_groups = [list(range(1, len(step_content)+1))]
    try:
        # Reconstruct the step descriptions for dependency analysis
        dependency_analysis = ""
        for i, (task, objective, content) in enumerate(zip(step_descriptions_task, step_descriptions_objective, step_content), 1):
            desc = f"{task}: {objective}" if objective else task
            dependency_analysis += f"{i}. **{task}**: {objective}\n {content} \n"
        
        messages = [
            {"role": "system", "content": "The user want to do math problem in parallel. Please analyze the dependency of each step in the solution process given by the user. The output result should be a list of tuples like '(i, j)' where i and j are the indice of the step, each tuple should be a pair of step index and where the first step depends on the second step. As the user need to do math problem in parallel,  But you should also make sure all dependencies are included in the output. If there is really dependency, you should never remove it."},
            {"role": "user", "content": f"Please analyze the dependency of each step in the following thought process: {dependency_analysis}."}
    ]
        
        dependency_output = call_model(messages, model=model, temperature=0, seed=seed)
        adj_matrix = parse_dependency_output(dependency_output)
        parallel_groups = identify_parallel_steps_no_continuous_multistep(adj_matrix)
        print("Regenerated dependency analysis")
        
    except Exception as e:
        print(f"Error regenerating dependencies: {e}")
        print("Using default no-dependency configuration")
    
    return step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups

# ...existing code...
def batch_recover_from_temp_files(problem_indices):
    """
    Recover from multiple temp files at once.
    
    Args:
        problem_indices (list): List of problem indices to recover
        
    Returns:
        dict: Dictionary mapping problem_index to recovery results
    """
    results = {}
    
    for problem_index in problem_indices:
        try:
            result = recover_from_temp_files(problem_index)
            results[problem_index] = result
            print(f"Successfully recovered problem {problem_index}")
        except Exception as e:
            print(f"Failed to recover problem {problem_index}: {e}")
            results[problem_index] = None
    
    return results

# Example usage functions
def example_usage():
    """
    Example of how to use the recovery functions
    """
    # Basic recovery from temp file
    try:
        problem_index = 1500  # Change this to your problem index
        result = recover_from_temp_files_enhanced(problem_index)
        step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups = result
        
        print(f"Recovered {len(step_content)} steps")
        print(f"First step task: {step_descriptions_task[0] if step_descriptions_task else 'None'}")
        
    except FileNotFoundError as e:
        print(f"Recovery failed: {e}")
    
    # Enhanced recovery with problem info
    # problem = "Your original problem statement"
    # CoT_solution = "Your original CoT solution"
    # result = recover_from_temp_files_enhanced(problem_index, problem, CoT_solution)
    
    # Batch recovery
    # problem_indices = [1, 2, 3, 4, 5]
    # batch_results = batch_recover_from_temp_files(problem_indices)

# ...existing code...

if __name__ == "__main__":
    # Test with the existing temp_direct_1500.txt file
    example_usage()