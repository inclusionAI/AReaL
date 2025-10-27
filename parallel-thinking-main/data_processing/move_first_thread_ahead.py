import json
import re

def extract_thread_processing_content(thread_1_content, thread_id):
    """
    Extract the content from <thread_processing id='X'> in thread_1
    """
    thread_processing_pattern = rf"<thread_processing id='{thread_id}'>(.*?)</thread_processing>"
    match = re.search(thread_processing_pattern, thread_1_content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return ""

def process_main_thread(main_thread_content, thread_1_content):
    """
    Process the main_thread content according to the specified rules:
    1. Extract thread_processing id='0' content from thread_1 and move it after <reasoning_process>
    2. Remove thread id='0' from parallel_processing in main_thread
    3. If only one thread remains, remove parallel_processing and put processing content in <think> tags
    4. Renumber remaining threads globally to start from 0
    """
    
    # Check if the pattern exists
    if "<reasoning_process>\n<parallel_processing>" not in main_thread_content:
        return main_thread_content
    
    # Split content only at the first occurrence and ensure we only process once
    reasoning_start = main_thread_content.find("<reasoning_process>\n<parallel_processing>")
    if reasoning_start == -1:
        return main_thread_content
    
    before_reasoning = main_thread_content[:reasoning_start]
    after_reasoning_marker = main_thread_content[reasoning_start + len("<reasoning_process>\n<parallel_processing>"):]
    
    # Find the end of the first reasoning_process section
    reasoning_end = after_reasoning_marker.find("</reasoning_process>")
    if reasoning_end == -1:
        return main_thread_content
    
    reasoning_content = after_reasoning_marker[:reasoning_end]
    after_reasoning = after_reasoning_marker[reasoning_end:]
    
    # Extract thread id='0' content from parallel processing
    thread_0_pattern = r"<thread id='0'>(.*?)</thread>"
    thread_0_match = re.search(thread_0_pattern, reasoning_content, re.DOTALL)
    
    if not thread_0_match:
        return main_thread_content
    
    # Remove thread id='0' from the parallel processing part
    reasoning_no_thread0 = re.sub(thread_0_pattern, "", reasoning_content, flags=re.DOTALL)
    
    # Remove thread_result id='0' from step_resolution sections
    reasoning_no_thread0 = re.sub(r"<thread_result id='0'>.*?</thread_result>\s*", "", reasoning_no_thread0, flags=re.DOTALL)
    
    # Find all thread IDs in the content (after removing thread 0)
    all_thread_ids = set()
    for match in re.finditer(r"<thread id='(\d+)'", reasoning_no_thread0):
        all_thread_ids.add(int(match.group(1)))
    for match in re.finditer(r"<thread_result id='(\d+)'", reasoning_no_thread0):
        all_thread_ids.add(int(match.group(1)))
    
    # Create mapping from old IDs to new IDs
    old_to_new_mapping = {}
    sorted_ids = sorted(all_thread_ids)
    for new_id, old_id in enumerate(sorted_ids):
        old_to_new_mapping[old_id] = new_id
    
    # Build the new content
    new_content = before_reasoning + "<reasoning_process>\n"
    
    # Add thread_0_processing content from thread_1 in <think> tags
    thread_0_content = extract_thread_processing_content(thread_1_content, '0')
    if thread_0_content:
        new_content += f"<think: type = ''>\n{thread_0_content}\n</think: type = ''>\n"
    
    # Check if only one thread remains after removing thread 0
    remaining_thread_matches = re.findall(r"<thread id='(\d+)'>(.*?)</thread>", reasoning_no_thread0, re.DOTALL)
    
    if len(remaining_thread_matches) <= 1:
        # Only one thread or no threads left, remove parallel_processing
        if remaining_thread_matches:
            # Get the single remaining thread ID and extract its processing content
            remaining_thread_id = remaining_thread_matches[0][0]
            remaining_thread_processing = extract_thread_processing_content(thread_1_content, remaining_thread_id)
            
            if remaining_thread_processing:
                new_content += f"<think: type = ''>\n{remaining_thread_processing}\n</think: type = ''>\n"
        
        # Extract any <think> tags that were between parallel processing groups
        think_pattern = r"</parallel_processing>\s*(<think: type = ''>.*?</think: type = ''>)"
        think_matches = re.findall(think_pattern, reasoning_no_thread0, re.DOTALL)
        for think_content in think_matches:
            new_content += think_content + "\n"
    else:
        # Multiple threads remain, keep parallel_processing structure but renumber all threads globally
        
        # Apply global renumbering to all thread IDs and thread_result IDs
        def renumber_thread_id(match):
            old_id = int(match.group(1))
            if old_id in old_to_new_mapping:
                return f"<thread id='{old_to_new_mapping[old_id]}'>"
            return match.group(0)
        
        def renumber_thread_result_id(match):
            old_id = int(match.group(1))
            if old_id in old_to_new_mapping:
                return f"<thread_result id='{old_to_new_mapping[old_id]}'>"
            return match.group(0)
        
        # Apply renumbering to the entire content
        renumbered_content = re.sub(r"<thread id='(\d+)'>", renumber_thread_id, reasoning_no_thread0)
        renumbered_content = re.sub(r"<thread_result id='(\d+)'>", renumber_thread_result_id, renumbered_content)
        
        # Clean up extra whitespace and ensure proper structure
        # Fix missing <parallel_processing> tags and extra newlines
        renumbered_content = re.sub(r'\n\s*\n+\s*<launch_threads>', '\n<parallel_processing>\n<launch_threads>', renumbered_content)
        renumbered_content = re.sub(r'</step_resolution>\s*\n+\s*<think:', '</step_resolution>\n</parallel_processing>\n<think:', renumbered_content)
        renumbered_content = re.sub(r'</think: type = \'\'>\s*\n+\s*<launch_threads>', '</think: type = \'\'>\n<parallel_processing>\n<launch_threads>', renumbered_content)
        
        # Remove extra blank lines
        renumbered_content = re.sub(r'\n\s*\n+', '\n', renumbered_content)
        
        # Ensure proper structure at the beginning if it starts with <launch_threads>
        if renumbered_content.strip().startswith('<launch_threads>'):
            renumbered_content = '<parallel_processing>\n' + renumbered_content.strip()
        
        new_content += renumbered_content
    
    new_content += "</reasoning_process>" + after_reasoning
    
    return new_content

def process_jsonl_file(input_file, output_file):
    """
    Process a JSONL file and apply the transformation to each main_thread
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Check if both main_thread and thread_1 keys exist
                if 'main_thread' in data and 'thread_1' in data:
                    # Process the main_thread content
                    data['main_thread'] = process_main_thread(data['main_thread'], data['thread_1'])
                
                # Write the processed line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                # Write the original line if JSON parsing fails
                outfile.write(line)
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                # Write the original line if processing fails
                outfile.write(line)

def test_with_example():
    example_main_thread = """Question: Evaluate the sum $$\\sum_{n=1}^{\\infty}{\\frac{3^n+2^n}{6^n}}.$$
Assistant: 
<reasoning_process>
<parallel_processing>
<launch_threads>
<thread id='0'>
<task>
Problem Understanding
</task>
<objective>
identifies the task of evaluating an infinite sum involving terms with exponents.
</objective>
</thread>
<thread id='1'>
<task>
Geometric Series Formula Recall
</task>
<objective>
recalls the formula for the sum of an infinite geometric series.
</objective>
</thread>
</launch_threads>
<step_resolution>
<thread_result id='0'>
The task is to evaluate the infinite sum of (3^n + 2^n)/6^n from n=1 to infinity by splitting it into two separate sums: sum of 3^n/6^n and sum of 2^n/6^n.
</thread_result>
<thread_result id='1'>
The sum of an infinite geometric series starting at n=0 is a/(1 - r), where a is the first term and r is the common ratio. For series starting at n=1, subtract the n=0 term.
</thread_result>
</step_resolution>
</parallel_processing>
<think: type = ''>
..."""

    example_thread_1 = """<thread id='0'>
<task>
Problem Understanding
</task>
<objective>
identifies the task of evaluating an infinite sum involving terms with exponents.
</objective>
<thread_processing id='0'>
Okay, so I need to evaluate the infinite sum of (3^n + 2^n)/6^n from n=1 to infinity. Hmm, let me think. First, maybe I can split the numerator into two separate terms. So, the sum becomes the sum of 3^n/6^n plus the sum of 2^n/6^n. That way, I can handle each part individually.
</thread_processing>
<thread_result id='0'>
The task is to evaluate the infinite sum of (3^n + 2^n)/6^n from n=1 to infinity by splitting it into two separate sums: sum of 3^n/6^n and sum of 2^n/6^n.
</thread_result>
<thread id='1'>
<task>
Geometric Series Formula Recall
</task>
<objective>
recalls the formula for the sum of an infinite geometric series.
</objective>
<thread_processing id='1'>
Now, these are both geometric series. Remember, the formula for the sum of a geometric series starting at n=0 is a/(1 - r), where a is the first term and r is the common ratio. But since our sums start at n=1, we need to subtract the n=0 term.
</thread_processing>
<thread_result id='1'>
The sum of an infinite geometric series starting at n=0 is a/(1 - r), where a is the first term and r is the common ratio. For series starting at n=1, subtract the n=0 term.
</thread_result>"""
    
    # Process main thread
    processed_main_thread = process_main_thread(example_main_thread, example_thread_1)
    
    print("Processed main_thread:")
    print(processed_main_thread)

# Example usage
if __name__ == "__main__":
    test_with_example()
    
    # Uncomment to process actual files
    input_file = "new.jsonl"  # Replace with your input file path
    output_file = "new_760936.jsonl"  # Replace with your output file path
    process_jsonl_file(input_file, output_file)
    print(f"Processing complete. Output saved to {output_file}")