from api_call import call_model_claude

from example import test_string
from utils import label_steps
import re
from datetime import datetime
import os
STEP_TYPE_ANALYSIS =  """
You are given a series of mathematical reasoning steps. Your task is to classify each step into one of the following categories:
1. Exploration Step: A step that introduces new ideas, approaches, or avenues of thought that the model is not sure whether they will lead to a solution, but are worth investigating. 
2. Derivation Step: A step that logically follows from previous steps, building upon established ideas to move closer to a solution.
Notice that if some steps follow model's own previous exploration steps, you should classify them as Exploration Steps as well, even if they are deriving from previous steps as they are still part of the exploration process.

Beside this you should also consider whether this step is considering different cases or probaiblities. If a step is analyzing different cases of the same subproblem, you should add the tag "(Case Analysis)" after the step type.
Your output should be in the following format:
Step i: [Exploration Step|Derivation Step] [Optional: (Case Analysis)]
"""
def analyze_one_step_problem(step_text: str, remainder_content: str = ""):
    """
    Analyze and merge steps in the problem text.
    
    Args:
        step_text: The main text content to analyze (from <think> tags)
        remainder_content: Optional content after </think> tag to preserve
        
    Returns:
        Merged and re-labeled text
    """
    labeled_text = label_steps(step_text)
    print("================================")
    print ("Labeled Text:\n", labeled_text)
    print("================================")
    response = call_model_claude(
        user_prompt=labeled_text,
        system_prompt=STEP_TYPE_ANALYSIS,
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=32768,
        max_retries=3,  # Retry up to 3 times
        retry_delay=2   # Wait 2 seconds between retries
    )
    
    # Check if API call failed
    if response is None:
        raise Exception("API call failed after all retry attempts")
    
    # Extract the model output text
    model_output = response['choices'][0]['message']['content']
    print("================================")
    print("Model Output:\n", model_output)
    print("================================")
    
    # Parse the model output to get step classifications
    step_classifications = {}
    for line in model_output.strip().split('\n'):
        match = re.match(r'Step (\d+): (.+)', line)
        if match:
            step_num = int(match.group(1))
            classification = match.group(2)
            step_classifications[step_num] = classification
    
    # Parse the original labeled text to extract steps
    # Split by double newline to get each step block
    step_blocks = labeled_text.split('\n\n')
    steps = {}
    
    for block in step_blocks:
        # Each block should start with "Step N"
        lines = block.split('\n', 1)
        if len(lines) >= 1:
            step_match = re.match(r'^Step (\d+)$', lines[0])
            if step_match:
                step_num = int(step_match.group(1))
                # Content is everything after "Step N"
                content = lines[1] if len(lines) > 1 else ""
                steps[step_num] = content.strip()
    
    # Merge steps according to classifications
    merged_steps = []
    current_merged_content = []
    
    for step_num in sorted(steps.keys()):
        classification = step_classifications.get(step_num, "")
        
        if "New Subproblem" in classification:
            # Save previous merged content if exists
            if current_merged_content:
                merged_steps.append('\n\n'.join(current_merged_content))
            # Start new subproblem with this step's content
            current_merged_content = [steps[step_num]]
        elif "Continue Previous Subproblem" in classification:
            # Add this step's content to the current subproblem
            current_merged_content.append(steps[step_num])
        else:
            # Unknown classification, treat as new subproblem
            if current_merged_content:
                merged_steps.append('\n\n'.join(current_merged_content))
            current_merged_content = [steps[step_num]]
    
    # Save last merged content
    if current_merged_content:
        merged_steps.append('\n\n'.join(current_merged_content))
    
    # Re-label the merged steps
    merged_labeled_text = label_steps('\n\n'.join(merged_steps))
    
    print("================================")
    print("Merged and Re-labeled Text:\n", merged_labeled_text)
    print("================================")
    
    # Save to result directory with timestamp
    os.makedirs('result_analysis', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"result_analysis/merged_steps_{timestamp}.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("Original Labeled Text:\n")
        f.write("="*80 + "\n")
        f.write(labeled_text)
        f.write("\n\n")
        f.write("Model Output:\n")
        f.write("="*80 + "\n")
        f.write(model_output)
        f.write("\n\n")
        f.write("Merged and Re-labeled Text:\n")
        f.write("="*80 + "\n")
        f.write(merged_labeled_text)
        
        # Append remainder content if it exists
        if remainder_content:
            f.write("\n\n")
            f.write("="*80 + "\n")
            f.write("Content after </think> tag:\n")
            f.write("="*80 + "\n")
            f.write(remainder_content)
    
    print(f"Results saved to {output_filename}")
    
    return merged_labeled_text

if __name__ == "__main__":
    response = analyze_one_step_problem(test_string)
    print(response)