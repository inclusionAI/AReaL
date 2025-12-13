import json
import os
import re
from pathlib import Path

# Configuration
TXT_DIR = "/Users/zzy/Desktop/data_curation_new/new_result_with_objectives"
JSON_DIR = "/Users/zzy/Desktop/data_curation_new/new_result_merged"
OUTPUT_DIR = "/Users/zzy/Desktop/data_curation_new/new_result_parallel_formatted"

def parse_parallel_structure(json_string):
    """
    Parse the JSON string to extract parallel groups.
    Flattens nested brackets to only keep the innermost structure.
    """
    # Parse the escaped JSON string
    data = json.loads(json_string)
    
    def flatten_nested_lists(items):
        """Recursively flatten nested lists to keep only innermost brackets."""
        result = []
        for item in items:
            if isinstance(item, list):
                # If it's a list, check if it contains nested lists
                has_nested_list = any(isinstance(x, list) for x in item)
                if has_nested_list:
                    # Flatten this level - extract non-list items and flatten nested lists
                    for sub_item in item:
                        if isinstance(sub_item, list):
                            result.append(sub_item)
                        else:
                            result.append(sub_item)
                else:
                    # No nested lists, keep as is
                    result.append(item)
            else:
                result.append(item)
        return result
    
    flattened = flatten_nested_lists(data)
    return flattened

def extract_steps_from_txt(txt_path):
    """
    Extract steps from the txt file.
    Returns a dict mapping step number to its content, objective, and result.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    steps = {}
    
    # Split by step separator
    step_pattern = r'Step (\d+)\n={80}\n(.*?)(?=\nStep \d+\n={80}|\n={80}\nContent after </think> tag:|\Z)'
    matches = re.findall(step_pattern, content, re.DOTALL)
    
    for step_num, step_content in matches:
        step_num = int(step_num)
        
        # Extract objective - between OBJECTIVE: and RESULT:
        objective_match = re.search(r'OBJECTIVE:\s*(.*?)(?=\nRESULT:)', step_content, re.DOTALL)
        objective = objective_match.group(1).strip() if objective_match else ""
        
        # Extract result - between RESULT: and the actual content (after the blank line)
        result_match = re.search(r'RESULT:\s*(.*?)(?=\n\n)', step_content, re.DOTALL)
        result = result_match.group(1).strip() if result_match else ""
        
        # Extract the actual content - everything after RESULT: ... and blank line
        # The content starts after the result line and a blank line
        if objective and result:
            # Remove OBJECTIVE and RESULT sections to get the actual content
            content_match = re.search(r'RESULT:.*?\n\n(.*)', step_content, re.DOTALL)
            clean_content = content_match.group(1).strip() if content_match else ""
        else:
            # No objective/result, use the whole content
            clean_content = step_content.strip()
        
        steps[step_num] = {
            'content': clean_content,
            'objective': objective,
            'result': result
        }
    
    # Extract content after </think> tag
    after_think_match = re.search(r'Content after </think> tag:\n={80}\n(.*)', content, re.DOTALL)
    after_think_content = after_think_match.group(1).strip() if after_think_match else ""
    
    return steps, after_think_content

def format_parallel_group(step_numbers, steps_data):
    """
    Format a parallel group with the specified template.
    """
    output = "Let's think in parallel\n<Parallel>\n<Goal>\n"
    
    # Add outlines
    for i, step_num in enumerate(step_numbers, 1):
        if step_num in steps_data:
            objective = steps_data[step_num]['objective']
            output += f"<Outline>\n{i}. {objective}\n</Outline>\n"
    
    output += "</Goal>\n"
    
    # Add paths
    for i, step_num in enumerate(step_numbers, 1):
        if step_num in steps_data:
            content = steps_data[step_num]['content']
            output += f"<Path>\n{i}. {content}\n</Path>\n"
    
    output += "<Conclusion>\n[Leave this blank]\n</Conclusion>\n</Parallel>\n"
    
    return output

def process_file_pair(txt_file, json_file):
    """
    Process a pair of txt and json files.
    """
    # Extract steps from txt file
    steps_data, after_think_content = extract_steps_from_txt(txt_file)
    
    # Load and parse parallel structure from json file
    with open(json_file, 'r', encoding='utf-8') as f:
        json_string = json.load(f)
    
    parallel_structure = parse_parallel_structure(json_string)
    
    # Build the output
    output = ""
    
    for item in parallel_structure:
        if isinstance(item, list):
            # It's a parallel group
            output += format_parallel_group(item, steps_data)
            output += "\n"
        else:
            # It's a single step
            if item in steps_data:
                # output += f"Step {item}\n"
                # output += "=" * 80 + "\n"
                output += steps_data[item]['content'] + "\n\n"
    
    # Append the content after </think> tag
    output += "</think>\n"
    output += after_think_content
    
    return output

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all txt files in the txt directory
    txt_files = sorted([f for f in os.listdir(TXT_DIR) if f.endswith('.txt')])
    
    processed_count = 0
    
    for txt_file in txt_files:
        # Extract timestamp from filename
        # Format: merged_steps_YYYYMMDD_HHMMSS.txt
        match = re.search(r'merged_steps_(\d+_\d+)', txt_file)
        if not match:
            continue
        
        timestamp = match.group(1)
        
        # Find corresponding json file
        json_file = f"merged_steps_{timestamp}_parallel_analysis.json"
        json_path = os.path.join(JSON_DIR, json_file)
        txt_path = os.path.join(TXT_DIR, txt_file)
        
        if not os.path.exists(json_path):
            print(f"Warning: No matching json file for {txt_file}")
            continue
        
        try:
            print(f"Processing {txt_file}...")
            output_content = process_file_pair(txt_path, json_path)
            
            # Write output
            output_file = f"formatted_{timestamp}.txt"
            output_path = os.path.join(OUTPUT_DIR, output_file)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            processed_count += 1
            print(f"  ✓ Saved to {output_file}")
            
        except Exception as e:
            print(f"  ✗ Error processing {txt_file}: {str(e)}")
    
    print(f"\nProcessed {processed_count} file pairs successfully!")

if __name__ == "__main__":
    main()
