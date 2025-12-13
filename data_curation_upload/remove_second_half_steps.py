import os
import re
from pathlib import Path


def process_file(file_path):
    """
    Read a file, count the steps, and keep only the first half of the steps.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        True if file was processed successfully, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by step pattern - looking for "Step N" followed by separator line
        # Pattern: "Step" followed by number, newline, separator line (===)
        step_pattern = r'Step \d+\n={80,}\n'
        
        # Find all step positions
        steps = list(re.finditer(step_pattern, content))
        
        if not steps:
            print(f"No steps found in {file_path.name}")
            return False
        
        total_steps = len(steps)
        
        # Calculate how many steps to keep (first half)
        keep_steps = total_steps // 2
        
        if keep_steps == 0:
            print(f"Only {total_steps} step(s) in {file_path.name}, keeping at least 1")
            keep_steps = 1
        
        # Find the position where we should cut
        # If we need to keep n/2 steps, we cut after the n/2-th step
        if keep_steps < total_steps:
            # Get the start position of the (keep_steps + 1)-th step
            cut_position = steps[keep_steps].start()
            
            # Keep content up to that position
            new_content = content[:cut_position].rstrip()
            
            # Check if there's a "Content after </think> tag:" section to preserve
            think_tag_pattern = r'={80,}\nContent after </think> tag:\n={80,}\n'
            think_match = re.search(think_tag_pattern, content)
            
            if think_match:
                # Find everything from the "Content after </think> tag:" section onwards
                final_content = content[think_match.start():]
                new_content = new_content + "\n\n" + final_content
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"Processed {file_path.name}: {total_steps} steps -> kept {keep_steps} steps")
            return True
        else:
            print(f"File {file_path.name} has {total_steps} step(s), no change needed")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        return False


def main():
    """
    Process all .txt files in the result_merged directory.
    """
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Define the directory containing the files to process
    target_dir = script_dir / "new_result_merged"
    
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return
    
    # Get all .txt files in the directory
    txt_files = list(target_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {target_dir}")
        return
    
    print(f"Found {len(txt_files)} .txt files to process\n")
    
    # Process each file
    processed_count = 0
    for file_path in txt_files:
        if process_file(file_path):
            processed_count += 1
    
    print(f"\n{'='*80}")
    print(f"Summary: Processed {processed_count} out of {len(txt_files)} files")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
