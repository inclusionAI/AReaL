import json
import re

def reformat_conversation(data):
    """
    Reformat conversation data into instruction-output format with chat template.
    """
    conversations = data.get('conversations', [])
    
    # Extract system, human, and gpt messages
    system_prompt = ""
    user_content = ""
    gpt_content = ""
    
    for conv in conversations:
        if conv.get('from') == 'system':
            system_prompt = conv.get('value', '')
        elif conv.get('from') == 'human':
            user_content = conv.get('value', '')
        elif conv.get('from') in ['gpt', 'assistant']:
            gpt_content = conv.get('value', '')
    
    # Look for <thread_processing id='i'> pattern in gpt content
    thread_pattern = r'<thread_processing id=\'(\d+)\'>'
    match = re.search(thread_pattern, gpt_content)
    
    if match:
        # Split at the <thread_processing id='i'> token
        split_pos = match.start()
        assistant_start = gpt_content[:split_pos + len(match.group(0))]
        target_output = gpt_content[split_pos + len(match.group(0)):]
    else:
        # No thread_processing token found, all gpt content is target output
        assistant_start = ""
        target_output = gpt_content
    
    # Format instruction using chat template
    instruction = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_start}"""
    
    # Format output
    output = f"""{target_output}"""
    
    return {
        "instruction": instruction,
        "output": output
    }

def process_jsonl_file(input_file, output_file):
    """
    Process JSONL file and reformat each line.
    """
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse JSON line
                data = json.loads(line)
                
                # Reformat the conversation
                reformatted = reformat_conversation(data)
                
                # Write to output file
                outfile.write(json.dumps(reformatted, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Successfully processed {processed_count} conversations")

def test_with_example():
    """
    Test the reformatting function with the provided example.
    """
    example_data = {
        "conversations": [
            {
                "from": "system", 
                "value": "You are a helpful assistant that solve math problems.  \n\nThe input is a math problem and reasoning process before this thread is launched. Also the task and objective of this thread will be provided in the end of the input. You ahould complete the task and objective of this thread, start your processing with <thread_processing id = 'i'> where i is the index of the thread. End your processing with '</thread_processing>'. After this, put the result of this step between '<thread_result id = 'i'>' and '</thread_result>'. DO NOT output the special tag '<think>'  DO NOT output the special tag '<think>', what you need to do is to finish the reasoning of <thread_processing id='i'> and output its result, you only need to solve this partial step not the full problem\n Stop reasoning when you reach the end of the thread processing and then output the result in the format of '<thread_result id = 'i'>result</thread_result>'.\n NEVER solve the whole problem, you MUST STOP after the objective of this step is reached. Also, think for a while before you output the result, put the reasoning process in <thread_processing id='i'> ... </thread_processing> tag, where 'i' is the id of this thread. Put the result of **THIS STEP** (not the whole problem) in the <thread_result id='i'> ... </thread_result> tag"
            },
            {
                "from": "human", 
                "value": "Problem: Find all positive integer solutions to the equation $a^{2}=b^{3}+1$.\n<think>"
            },
            {
                "from": "gpt", 
                "value": "Okay, so I need to find all positive integer solutions to the equation \\( a^2 = b^3 + 1 \\). Hmm, let's start by thinking about small values for \\( b \\) and see what happens. Maybe there's a pattern or a limited number of solutions.\n\nFirst, let's try \\( b = 1 \\). Then the equation becomes \\( a^2 = 1^3 + 1 = 2 \\). But \\( \\sqrt{2} \\) isn't an integer, so that's not a solution.\n\nTo begin solving the equation \\(a^{2}=b^{3}+1\\), we first need to test small positive integer values of \\(b\\) to identify potential solutions. Since the equation involves perfect squares and cubes, we should systematically evaluate \\(b = 1\\) through \\(b = 5\\) to determine if \\(a^2\\) results in a perfect square. Additionally, we should briefly consider \\(b = 0\\) to confirm it doesn't satisfy the positive integer constraint. This initial exploration will help us identify any obvious solutions and establish a baseline for further analysis.\n\n<parallel_processing>\n<thread id='5'>\n<task>\nConsidering b = 0\n</task>\n<objective>\nbriefly considers \\( b = 0 \\) but dismisses it as it does not meet the problem's positive integer requirement.\n</objective>\n</thread>\n<thread_processing id='5'>Wait, maybe \\( b = 0 \\)? But the problem says positive integers, so \\( b \\) has to be at least 1. So that's out.\n</thread_processing>\n<thread_result id='5'>\nb = 0 is dismissed since the problem requires positive integers.\n</thread_result><|im_end|>"
            }
        ]
    }
    
    result = reformat_conversation(example_data)
    print("=== TEST RESULT ===")
    print("INSTRUCTION:")
    print(result["instruction"])
    print("\nOUTPUT:")
    print(result["output"])
    print("=== END TEST ===")

# Usage
if __name__ == "__main__":
    # Test with the example first
    print("Testing with provided example...")
    test_with_example()
    
    # Process actual files
    input_file = "/home/zhangzy/parallel-thinking/data_processing/new_dataset/val_set.jsonl"  # Replace with your input file path
    output_file = "/home/zhangzy/parallel-thinking/data_processing/new_dataset/val_set_raw_instruct.jsonl"  # Replace with your output file path
    
    print(f"\nProcessing {input_file}...")
    process_jsonl_file(input_file, output_file)
    print(f"Processing complete. Output saved to {output_file}")