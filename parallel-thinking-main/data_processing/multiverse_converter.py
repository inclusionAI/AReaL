import json
import re
from typing import Dict, List, Tuple

def extract_parallel_groups(parallel_cot: str) -> Tuple[List[List[Dict]], str]:
    """Extract parallel processing groups and sequential content from Parallel_CoT"""
    parallel_groups = []
    sequential_parts = []
    
    # Find all top-level <Parallel> blocks and their positions
    parallel_pattern = r'<Parallel>(.*?)</Parallel>'
    parallel_matches = list(re.finditer(parallel_pattern, parallel_cot, re.DOTALL))
    
    current_pos = 0
    
    for match in parallel_matches:
        # Add sequential content before this parallel block
        sequential_content = parallel_cot[current_pos:match.start()].strip()
        if sequential_content:
            sequential_parts.append(sequential_content)
        
        # Process the parallel block
        parallel_content = match.group(1)
        group = extract_single_parallel_group(parallel_content)
        if group:
            parallel_groups.append(group)
        
        current_pos = match.end()
    
    # Add remaining sequential content after the last parallel block
    remaining_content = parallel_cot[current_pos:].strip()
    if remaining_content:
        sequential_parts.append(remaining_content)
    
    # Combine all sequential content
    main_content = '\n'.join(sequential_parts)
    
    return parallel_groups, main_content

def extract_single_parallel_group(parallel_block: str) -> List[Dict]:
    """Extract steps from a single parallel block"""
    group = []
    
    # Find Goal section with Outlines
    goal_pattern = r'<Goal>(.*?)</Goal>'
    goal_match = re.search(goal_pattern, parallel_block, re.DOTALL)
    
    if not goal_match:
        return group
    
    goal_content = goal_match.group(1)
    
    # Extract outlines
    outline_pattern = r'<Outline>(.*?)</Outline>'
    outlines = re.findall(outline_pattern, goal_content, re.DOTALL)
    
    # Extract paths - handle nested parallel groups by flattening them
    path_pattern = r'<Path>(.*?)</Path>'
    paths = re.findall(path_pattern, parallel_block, re.DOTALL)
    
    # Process each path to handle nested parallel groups
    processed_paths = []
    for path in paths:
        processed_path = process_nested_parallel(path)
        processed_paths.append(processed_path)
    
    # Match outlines with processed paths
    for i, (outline, path) in enumerate(zip(outlines, processed_paths)):
        # Clean the path content - remove numbered prefixes like "1:", "2:", etc.
        path_cleaned = clean_numbered_prefixes(path)
        
        # Clean outline - remove numbered prefixes
        outline_cleaned = clean_numbered_prefixes(outline)
        
        group.append({
            'objective': outline_cleaned.strip(),
            'processing': path_cleaned.strip()
        })
    
    return group

def process_nested_parallel(path_content: str) -> str:
    """Process nested parallel groups within a path by flattening them"""
    # Remove "Let's think in parallel" phrases before nested <Parallel> tags
    path_content = re.sub(r'Let\'s think in parallel\.\s*(?=<Parallel>)', '', path_content, flags=re.IGNORECASE)
    
    # Find nested parallel blocks
    nested_parallel_pattern = r'<Parallel>(.*?)</Parallel>'
    nested_matches = re.findall(nested_parallel_pattern, path_content, re.DOTALL)
    
    for nested_match in nested_matches:
        # Extract all Path content from nested parallel and flatten it
        nested_paths = re.findall(r'<Path>(.*?)</Path>', nested_match, re.DOTALL)
        flattened_content = '\n'.join(nested_paths)
        
        # Replace the entire nested parallel block with flattened content
        path_content = re.sub(
            r'<Parallel>.*?</Parallel>', 
            flattened_content, 
            path_content, 
            count=1, 
            flags=re.DOTALL
        )
    
    return path_content

def clean_numbered_prefixes(text: str) -> str:
    """Remove numbered prefixes like '1:', '2:', etc. from text"""
    # Remove numbered prefixes at the beginning of lines
    text = re.sub(r'^\d+:\s*', '', text.strip(), flags=re.MULTILINE)
    # Process each line individually
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(r'^\d+:\s*', '', line)
        cleaned_lines.append(cleaned_line)
    return '\n'.join(cleaned_lines)

def extract_conclusion_content(parallel_cot: str) -> str:
    """Extract all conclusion content from within Parallel blocks, removing tags"""
    # Only extract conclusions from within Parallel blocks
    parallel_pattern = r'<Parallel>(.*?)</Parallel>'
    parallel_blocks = re.findall(parallel_pattern, parallel_cot, re.DOTALL)
    
    conclusions = []
    for block in parallel_blocks:
        conclusion_pattern = r'<Conclusion>(.*?)</Conclusion>'
        block_conclusions = re.findall(conclusion_pattern, block, re.DOTALL)
        conclusions.extend([c.strip() for c in block_conclusions])
    
    return '\n'.join(conclusions)

def call_llm_for_task(objective: str) -> str:
    """Generate task name from objective"""
    # Simple heuristic to generate task names
    if "substitut" in objective.lower():
        return "Variable Substitution"
    elif "domain" in objective.lower():
        return "Domain Consideration" 
    elif "trial" in objective.lower() or "test" in objective.lower():
        return "Trial Solution Check"
    elif "alternative" in objective.lower():
        return "Alternative Solution Check"
    elif "verify" in objective.lower() or "check" in objective.lower():
        return "Solution Verification"
    elif "simplif" in objective.lower():
        return "Expression Simplification"
    elif "solve" in objective.lower():
        return "Equation Solving"
    elif "factor" in objective.lower():
        return "Factorization"
    elif "evaluat" in objective.lower():
        return "Expression Evaluation"
    elif "analyz" in objective.lower() or "analy" in objective.lower():
        return "Mathematical Analysis"
    elif "confirm" in objective.lower():
        return "Result Confirmation"
    elif "demonstrat" in objective.lower():
        return "Mathematical Demonstration"
    elif "converg" in objective.lower():
        return "Convergence Analysis"
    elif "summar" in objective.lower():
        return "Mathematical Summary"
    elif "discuss" in objective.lower():
        return "Mathematical Discussion"
    elif "consider" in objective.lower():
        return "Mathematical Consideration"
    elif "inquir" in objective.lower():
        return "Mathematical Inquiry"
    else:
        # Extract first few words as task name
        words = objective.split()[:3]
        return " ".join(words).title()

def call_llm_for_result(objective: str, processing: str) -> str:
    """Generate thread result from objective and processing"""
    # Use the last meaningful sentence from processing
    sentences = [s.strip() for s in processing.split('.') if s.strip()]
    
    # Look for sentences that seem like conclusions
    for sentence in reversed(sentences):
        if len(sentence) > 15 and any(word in sentence.lower() for word in ['therefore', 'thus', 'so', 'hence', 'consequently']):
            return sentence + "."
    
    # If no conclusion sentence found, use the last sentence
    if sentences:
        return sentences[-1] + "."
    
    # Fallback
    return f"Completed: {objective}"

def process_dataset(input_file: str, output_file: str):
    """Process the JSONL dataset"""
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                problem = data.get("Problem", "")
                parallel_cot = data.get("Parallel_CoT", "")
                
                # Extract parallel groups and main content
                parallel_groups, main_content = extract_parallel_groups(parallel_cot)
                
                # Extract conclusion content (only from within Parallel blocks)
                conclusion_content = extract_conclusion_content(parallel_cot)
                
                # Generate main_thread content
                main_thread = generate_main_thread(main_content, parallel_groups, conclusion_content)
                
                # Generate thread_1 content  
                thread_1 = generate_thread_1(parallel_groups)
                
                # Create output record
                output_record = {
                    "Problem": problem,
                    "main_thread": main_thread,
                    "thread_1": thread_1
                }
                
                # Write to output file
                f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                
                print(f"Processed line {line_num}")
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

def generate_main_thread(main_content: str, parallel_groups: List[List[Dict]], conclusion_content: str) -> str:
    """Generate main_thread content"""
    content = []
    
    # Add main content first (sequential reasoning)
    if main_content.strip():
        content.append(main_content.strip())
    
    thread_id_counter = 0
    
    for group in parallel_groups:
        if len(group) > 1:
            # Parallel processing group
            content.append("<launch_threads>")
            
            # Generate threads
            group_thread_ids = []
            for step in group:
                task = call_llm_for_task(step['objective'])
                
                content.append(f"<thread id='{thread_id_counter}'>")
                content.append("<task>")
                content.append(task)
                content.append("</task>")
                content.append("<objective>")
                content.append(step['objective'])
                content.append("</objective>")
                content.append("</thread>")
                
                group_thread_ids.append(thread_id_counter)
                thread_id_counter += 1
            
            content.append("</launch_threads>")
            content.append("<step_resolution>")
            
            # Generate thread results
            for i, thread_id in enumerate(group_thread_ids):
                result = call_llm_for_result(group[i]['objective'], group[i]['processing'])
                content.append(f"<thread_result id='{thread_id}'>")
                content.append(result)
                content.append("</thread_result>")
            
            content.append("</step_resolution>")
    
    # Add conclusion content if present (from within Parallel blocks)
    if conclusion_content.strip():
        content.append(conclusion_content.strip())
    
    return '\n'.join(content)

def generate_thread_1(parallel_groups: List[List[Dict]]) -> str:
    """Generate thread_1 content"""
    content = []
    
    thread_id_counter = 0
    
    for group in parallel_groups:
        # Include ALL groups in thread_1
        for step in group:
            task = call_llm_for_task(step['objective'])
            result = call_llm_for_result(step['objective'], step['processing'])
            
            content.append(f"<thread id='{thread_id_counter}'>")
            content.append("<task>")
            content.append(task)
            content.append("</task>")
            content.append("<objective>")
            content.append(step['objective'])
            content.append("</objective>")
            content.append("</thread>")
            content.append(f"<thread_processing id='{thread_id_counter}'>")
            content.append(step['processing'])
            content.append("</thread_processing>")
            content.append(f"<thread_result id='{thread_id_counter}'>")
            content.append(result)
            content.append("</thread_result>")
            
            thread_id_counter += 1
    
    return '\n'.join(content)

if __name__ == "__main__":
    # Example usage
    input_file = "multiverse.jsonl"  # Replace with your input file path
    output_file = "multiverse_processed.jsonl"  # Replace with your output file path
    
    process_dataset(input_file, output_file)
    print("Dataset processing completed!")