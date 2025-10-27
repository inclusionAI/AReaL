import re
import numpy as np
def extract_step_descriptions(llm_output):
    """
    Extract descriptions from numbered steps in LLM output.
    
    Args:
        llm_output (str): The text output from the LLM containing numbered steps
        
    Returns:
        list: A list of step descriptions without the numbering and formatting
    """
    # Pattern to match numbered steps like "1. **Title**: Description"
    pattern = r'(\d+)\.\s+\*\*([^:]+)\*\*:\s+(.*?)(?=\n\d+\.|\n\n|$)'
    
    # Find all matches in the text
    matches = re.findall(pattern, llm_output, re.DOTALL)
    
    # Extract just the descriptions
    step_descriptions = []
    for num, title, description in matches:
        # Clean up any extra whitespace
        clean_description = f"{title}: {description.strip()}"
        step_descriptions.append(clean_description)
    
    return step_descriptions
def extract_code_blocks(text):
    """
    Extract content from code blocks in text.
    
    Args:
        text (str): Text containing Markdown code blocks (```...```)
        
    Returns:
        list: List of extracted code block contents
    """
    # Pattern to match code blocks with or without language specification
    # This handles both ```python and ``` format
    pattern = r'```(?:\w+)?\s*(.*?)```'
    
    # Find all matches using DOTALL to capture newlines
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Clean up whitespace
    code_blocks = [block.strip() for block in matches]
    
    return code_blocks

# Example usage:
# llm_response = "Here's some code: ```python\nprint('hello')\n``` and more code: ```\nx = 5\n```"
# code_blocks = extract_code_blocks(llm_response)
# print(code_blocks)  # ['print('hello')', 'x = 5']
def parse_dependency_output(dependency_output):
    """
    Parse dependency output from LLM and create an adjacency matrix for the DAG.
    
    Args:
        dependency_output (str): The LLM output containing dependency tuples
        
    Returns:
        numpy.ndarray: Adjacency matrix where matrix[i][j] = 1 means step j+1 depends on step i+1
    """
    # Extract the tuples using regex
    pattern = r'\((\d+),\s*(\d+)\)'
    tuples = re.findall(pattern, dependency_output)
    
    # Convert to integer pairs (dependent, prerequisite)
    dependencies = [(int(dep), int(src)) for dep, src in tuples]
    
    # Find the maximum step number to determine matrix size
    max_step = max(max(dep, src) for dep, src in dependencies)
    
    # Create adjacency matrix (initialized with zeros)
    adj_matrix = np.zeros((max_step, max_step), dtype=int)
    
    # Fill the adjacency matrix: if (dep, src) means "dep depends on src"
    # then there's an edge from src to dep in the DAG
    for dep, src in dependencies:
        # Adjust for 0-indexing
        adj_matrix[src-1][dep-1] = 1
    
    # add dependency for steps too far away, more than 10 steps away
    for i in range(max_step):
        for j in range(max_step):
            if i < j and abs(i - j) >= 4:
                # If step j depends on step i and they are more than 10 steps apart,
                # we add a direct dependency to ensure proper execution order
                adj_matrix[i, j] = 1
    
    
    return adj_matrix
# Find steps that can be executed in parallel (no dependencies between them)
def identify_parallel_steps_no_continuous_multistep(adj_matrix):
    """
    Identify parallel steps with only the restriction that multi-step groups 
    cannot be continuous. Groups steps that can be executed in parallel
    based on dependency constraints, but ensures multi-step groups are
    separated by single-step groups.
    
    Args:
        adj_matrix (numpy.ndarray): Adjacency matrix representing dependencies
        
    Returns:
        list: List of parallel groups, where each group contains steps that can run in parallel
    """
    n = adj_matrix.shape[0]
    parallel_groups = [[1]]
    visited = set()
    visited.add(1)
    # Process steps in topological order
    
    while len(visited) < n:
        next_available = []
        
        # Find all steps that can be executed (all prerequisites satisfied)
        for step in range(1, n+1):
            if step not in visited:
                prerequisites = [i+1 for i in range(n) if adj_matrix[i, step-1] == 1]
                if all(p in visited for p in prerequisites):
                    next_available.append(step)
        
        if not next_available:
            break  # No more steps can be processed
        
        # Sort the available steps for consistent output
        next_available.sort()
        
        # Check if previous group was multi-step
        previous_was_multistep = (len(parallel_groups) > 0 and 
                                 len(parallel_groups[-1]) > 1)
        
        if previous_was_multistep and len(next_available) > 1:
            # Previous group was multi-step and current would be multi-step
            # Need to separate them - take only the first step as single-step group
            single_step = next_available[0]
            parallel_groups.append([single_step])
            visited.add(single_step)
            parallel_groups.append(next_available[1:])
            for step in next_available[1:]:
                visited.add(step)
            # Remove the processed step from next_available for potential next iteration
            # The remaining steps will be processed in the next iteration
            
        else:
            # Safe to add all available steps as a group
            parallel_groups.append(next_available)
            visited.update(next_available)
    
    return parallel_groups
def identify_parallel_steps(adj_matrix):
    n = adj_matrix.shape[0]
    parallel_groups = []
    
    # Rule 1: The first step should be in the first parallel group independently
    parallel_groups.append([1])
    visited = {1}
    
    # Process remaining steps in topological order
    while len(visited) < n:
        next_available = []
        for step in range(1, n+1):
            if step not in visited:
                prerequisites = [i+1 for i in range(n) if adj_matrix[i, step-1] == 1]
                if all(p in visited for p in prerequisites):
                    next_available.append(step)
        
        if not next_available:
            break  # No more steps can be processed
        
        # Rule 3: Group continuous steps only
        continuous_groups = []
        if next_available:
            # Sort the available steps
            next_available.sort()
            
            # Group continuous steps together
            current_group = [next_available[0]]
            for i in range(1, len(next_available)):
                if next_available[i] == next_available[i-1] + 1:
                    current_group.append(next_available[i])
                else:
                    continuous_groups.append(current_group)
                    current_group = [next_available[i]]
            continuous_groups.append(current_group)
        
        # Process groups based on the new rule:
        # Ensure single-step groups between multi-step groups
        for i, group in enumerate(continuous_groups):
            if len(group) > 1:
                # Check if previous group was multi-step
                if len(parallel_groups) > 0 and len(parallel_groups[-1]) > 1:
                    # Previous was multi-step, need a single step between
                    # Take the first step as single, rest as multi-step
                    parallel_groups.append([group[0]])
                    visited.add(group[0])
                    if len(group) > 1:
                        parallel_groups.append(group[1:])
                        visited.update(group[1:])
                else:
                    # Safe to add as multi-step group
                    parallel_groups.append(group)
                    visited.update(group)
            else:
                # Single step group - always safe to add
                parallel_groups.append(group)
                visited.update(group)
    non_single_group_index = []
    for i in range (len(parallel_groups)):
        # Convert to 1-indexed
        if len(parallel_groups[i]) > 1:
            non_single_group_index.append(i)
    # sort the single group between non-single groups
    for i in range(len(non_single_group_index)):
        start = non_single_group_index[i]
        end = non_single_group_index[i+1] if i+1 < len(non_single_group_index) else len(parallel_groups)
        single_steps = []
        for j in range(start + 1, end):
            if len(parallel_groups[j]) == 1:
                single_steps.append(parallel_groups[j][0])
        
        # Sort the single steps and update parallel_groups
        single_steps.sort()
        single_index = 0
        for j in range(start + 1, end):
            if len(parallel_groups[j]) == 1:
                parallel_groups[j] = [single_steps[single_index]]
                single_index += 1
    return parallel_groups
def identify_parallel_steps_unrestricted(adj_matrix):
    """
    Identify parallel steps without the restriction that multi-step groups 
    cannot be continuous. Groups steps that can be executed in parallel
    based purely on dependency constraints.
    
    Args:
        adj_matrix (numpy.ndarray): Adjacency matrix representing dependencies
        
    Returns:
        list: List of parallel groups, where each group contains steps that can run in parallel
    """
    n = adj_matrix.shape[0]
    parallel_groups = []
    visited = set()
    
    # Process steps in topological order
    while len(visited) < n:
        next_available = []
        
        # Find all steps that can be executed (all prerequisites satisfied)
        for step in range(1, n+1):
            if step not in visited:
                prerequisites = [i+1 for i in range(n) if adj_matrix[i, step-1] == 1]
                if all(p in visited for p in prerequisites):
                    next_available.append(step)
        
        if not next_available:
            break  # No more steps can be processed
        
        # All available steps can be executed in parallel
        # Sort them for consistent output
        next_available.sort()
        parallel_groups.append(next_available)
        visited.update(next_available)
    
    return parallel_groups
if __name__ == "__main__":
    # Example usage
    dependency_output = "(3,1), (4,1), (5,2), (7,4), (7,3), (8,4), (9,1), (5,3), (8,7)"
    adj_matrix = parse_dependency_output(dependency_output)
    print("Adjacency Matrix:")
    print(adj_matrix)
    
    parallel_steps = identify_parallel_steps_no_continuous_multistep(adj_matrix)
    print("Parallel Steps Groups:")
    print(parallel_steps)