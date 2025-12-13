import os
import re
from pathlib import Path


def extract_steps_from_file(file_path):
    """
    Extract step contents and their labels from a file.
    Returns a list of tuples: [(step_number, content, label), ...]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the "Original Labeled Text:" section
    match = re.search(r'Original Labeled Text:\s*={70,}', content)
    if not match:
        return []
    
    # Get everything after "Original Labeled Text:"
    start_pos = match.end()
    labeled_section = content[start_pos:]
    
    # Split into steps
    # Pattern: "Step N" followed by content, then either another "Step N" or the Model Output section
    step_pattern = r'Step (\d+)\s*\n(.*?)(?=\nStep \d+|Model Output:|\Z)'
    steps = re.findall(step_pattern, labeled_section, re.DOTALL)
    
    step_data = []
    for step_num, step_content in steps:
        step_content = step_content.strip()
        step_data.append((int(step_num), step_content))
    
    # Now extract labels from the "Model Output:" section
    model_output_match = re.search(r'Model Output:\s*={70,}(.*?)(?:Merged and Re-labeled Text:|\Z)', 
                                   content, re.DOTALL)
    if not model_output_match:
        return []
    
    model_output = model_output_match.group(1)
    
    # Extract step labels
    label_pattern = r'Step (\d+): (New Subproblem|Continue Previous Subproblem.*?)(?=\nStep \d+:|\Z)'
    labels = re.findall(label_pattern, model_output, re.DOTALL)
    
    # Create a mapping of step number to label
    label_map = {}
    for step_num, label in labels:
        label_map[int(step_num)] = label.strip()
    
    # Combine step data with labels
    result = []
    for step_num, step_content in step_data:
        label = label_map.get(step_num, "")
        result.append((step_num, step_content, label))
    
    return result


def merge_steps(step_data):
    """
    Merge steps based on "New Subproblem" labels.
    If a step is "New Subproblem", start a new merged step.
    Otherwise, append to the last merged step with "\n" separator.
    
    Returns a list of merged step contents.
    """
    if not step_data:
        return []
    
    merged_steps = []
    current_step = None
    
    for step_num, content, label in step_data:
        if "New Subproblem" in label:
            # Start a new step
            if current_step is not None:
                merged_steps.append(current_step)
            current_step = content
        else:
            # Continue previous step
            if current_step is not None:
                current_step += "\n\n" + content
            else:
                # If the first step is not "New Subproblem", treat it as new
                current_step = content
    
    # Add the last step
    if current_step is not None:
        merged_steps.append(current_step)
    
    return merged_steps


def process_file(input_path, output_path):
    """
    Process a single file: extract, merge, and save.
    """
    step_data = extract_steps_from_file(input_path)
    if not step_data:
        print(f"Warning: No steps found in {input_path}")
        return
    
    merged_steps = merge_steps(step_data)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, step in enumerate(merged_steps, 1):
            f.write(f"Step {i}\n")
            f.write("=" * 80 + "\n")
            f.write(step + "\n\n")
    
    print(f"Processed: {input_path.name} -> {output_path.name}")
    print(f"  Original steps: {len(step_data)}, Merged steps: {len(merged_steps)}")


def main():
    # Define directories
    input_dir = Path("/Users/zzy/Desktop/data_curation_new/new_result")
    output_dir = Path("/Users/zzy/Desktop/data_curation_new/new_result_merged")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Process all .txt files in input directory
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} files to process\n")
    
    for input_file in txt_files:
        output_file = output_dir / input_file.name
        try:
            process_file(input_file, output_file)
        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")
    
    print(f"\nProcessing complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
