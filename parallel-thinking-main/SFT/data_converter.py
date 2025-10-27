import json
import os
import re

def clean_assistant_response(response):
    """
    Clean the assistant response by removing specified tags while preserving the content within them
    """
    # Define patterns to remove tags but keep content
    tag_patterns = [
        (r'<parallel_result>', r'</parallel_result>'),
        (r'<parallel_processing>', r'</parallel_processing>'), 
        (r'<planning>', r'</planning>'),
        (r'<answer>', r'</answer>'),
        (r'<think:\s*type\s*=\s*[\'"][^\'"]*[\'"]>', r'</think:\s*type\s*=\s*[\'"][^\'"]*[\'"]>'),
    ]
    
    cleaned_response = response
    
    # Remove each tag pair but keep content
    for start_tag, end_tag in tag_patterns:
        # Pattern to match start_tag + content + end_tag and replace with just content
        pattern = f'{start_tag}(.*?){end_tag}'
        cleaned_response = re.sub(pattern, r'\1', cleaned_response, flags=re.DOTALL)
    
    # Clean up extra whitespace and newlines
    cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)
    cleaned_response = cleaned_response.strip()
    
    return cleaned_response

def convert_jsonl_to_sharegpt_format(input_file, output_file):
    """
    Convert your JSONL format to ShareGPT format for LLaMA Factory
    """
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract question from original_problem or problem_index context
            question = data.get('original_problem', '')
            
            # Extract answer from main_thread
            answer = data.get('main_thread', '')
            
            if question and answer:
                # Convert to ShareGPT format
                conversation = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt", 
                            "value": answer
                        }
                    ]
                }
                converted_data.append(conversation)
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_data)} conversations")
    print(f"Output saved to: {output_file}")

def convert_with_assistant_filtering(input_file, output_file):
    """
    Convert JSONL to ShareGPT format with assistant response filtering, reasoning process validation,
    and system prompt addition
    """
    print("Converting data to ShareGPT format with assistant filtering and system prompt...")
    
    # Define the system prompt from prompt.py
    system_prompt = """You are a helpful assistant that solve math problems. 
The input is a math problem and you need to solve it step by step. If you think you need to launch a thread, you can do so by using the '<launch_threads>' tag. Each thread will have its own task and objective, put the whole thread_launching process in the following format:
```
<launch_threads>
<thread id='0'>
<task>
[Task Name]
</task>
<objective> 
[Objective of the thread]
</objective>
</thread>
<thread id='1'>
<task>
[Task Name]
</task> 
<objective> 
[Objective of the thread]
</objective>
</thread>
</launch_threads>
```
You should complete the whole reasoning process of the original problem, rather than just a partial step in main mode. If you are in the main mode, start the reasoning process with the special tag '<think>'
"""
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            question = data.get('original_problem', '')
            main_thread = data.get('main_thread', '')
            
            if question and main_thread:
                # Extract just the assistant's response part
                if "Assistant: " in main_thread:
                    
                    # Split and get the assistant's response
                    assistant_response = main_thread.split("Assistant: ", 1)[1].strip()
                    
                    # Clean the assistant response by removing specified tags
                    cleaned_response = clean_assistant_response(assistant_response)
                    
                    # Ensure we have the reasoning process
                    if "<think>" in cleaned_response:
                        conversation = {
                            "conversations": [
                                {
                                    "from": "system",
                                    "value": system_prompt
                                },
                                {
                                    "from": "human",
                                    "value": question
                                },
                                {
                                    "from": "gpt", 
                                    "value": cleaned_response
                                }
                            ]
                        }
                        converted_data.append(conversation)
                    else:
                        print(f"Warning: No reasoning process found in entry {data.get('problem_index', 'unknown')}")
                else:
                    print(f"Warning: No assistant response found in entry {data.get('problem_index', 'unknown')}")
    
    print(f"Prepared {len(converted_data)} training examples")
    
    # Save training data
    os.makedirs('data', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Create dataset info
    dataset_info = {
        "parallel_thinking": {
            "file_name": os.path.basename(output_file),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value", 
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system"
            }
        }
    }
    
    # Save dataset info in the same directory as the output file
    output_dir = os.path.dirname(output_file)
    dataset_info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset info saved to: {dataset_info_path}")

def extract_reasoning_context(main_thread_content, target_thread_id):
    """
    Extract the reasoning context from main_thread before a specific thread was launched
    """
    # Find all launch_threads blocks
    launch_pattern = r'<launch_threads>\n(.*?)\n</launch_threads>'
    launch_blocks = re.findall(launch_pattern, main_thread_content, re.DOTALL)
    
    context = ""
    
    # Find the launch block that contains the target thread
    for block in launch_blocks:
        if f"<thread id='{target_thread_id}'>" in block:
            # Get everything before this launch_threads block
            launch_start = main_thread_content.find(f'<launch_threads>{block}</launch_threads>')
            context = main_thread_content[:launch_start].strip()
            break
    
    return context

def convert_jsonl_to_sharegpt_format_main_dialogue(input_file, output_file):
    """
    Convert your JSONL format to ShareGPT format for LLaMA Factory
    """
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract question from original_problem or problem_index context
            question = data.get('Problem', '')
            
            # Extract answer from main_thread
            answer = data.get('CoT', '')
            
            if question and answer:
                # Convert to ShareGPT format
                conversation = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt", 
                            "value": answer
                        }
                    ]
                }
                converted_data.append(conversation)
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_data)} conversations")
    print(f"Output saved to: {output_file}")

def convert_to_thread_dialogue_format(input_file, output_file):
    """
    Convert JSONL to thread dialogue format for training individual thread responses
    Each thread_processing becomes a separate training example
    Input: problem + reasoning before thread launch
    Output: thread processing + result for that specific thread
    """
    print("Converting data to thread dialogue format with system prompt...")
    
    # Define the system prompt from prompt_thread.py
    system_prompt = """You are a helpful assistant that solve math problems.  

The input is a math problem and reasoning process before this thread is launched. Also the task and objective of this thread will be provided in the end of the input. You ahould complete the task and objective of this thread, start your processing with <thread_processing id = 'i'> where i is the index of the thread. End your processing with '</thread_processing>'. After this, put the result of this step between '<thread_result id = 'i'>' and '</thread_result>'. DO NOT output the special tag '<think>'  DO NOT output the special tag '<think>', what you need to do is to finish the reasoning of <thread_processing id='i'> and output its result, you only need to solve this partial step not the full problem\n Stop reasoning when you reach the end of the thread processing and then output the result in the format of '<thread_result id = 'i'>result</thread_result>'.\n NEVER solve the whole problem, you MUST STOP after the objective of this step is reached. Also, think for a while before you output the result, put the reasoning process in <thread_processing id='i'> ... </thread_processing> tag, where 'i' is the id of this thread. Put the result of **THIS STEP** (not the whole problem) in the <thread_result id='i'> ... </thread_result> tag"""
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            question = data.get('original_problem', '')
            main_thread = data.get('main_thread', '')
            thread_1 = data.get('thread_1', '')
            
            if question and main_thread and thread_1:
                # Extract thread processing blocks from thread_1
                thread_processing_pattern = r'<thread_processing id\s*=\s*\'(\d+)\'>\s*(.*?)\s*</thread_processing>'
                thread_processings = re.findall(thread_processing_pattern, thread_1, re.DOTALL)
                
                # Extract thread results
                thread_result_pattern = r'<thread_result id=\'(\d+)\'>\s*(.*?)\s*</thread_result>'
                thread_results = re.findall(thread_result_pattern, thread_1, re.DOTALL)
                
                # Extract thread definitions (id, task, objective) from thread_1
                thread_def_pattern = r'<thread id=\'(\d+)\'>\s*<task>\s*(.*?)\s*</task>\s*<objective>\s*(.*?)\s*</objective>\s*</thread>'
                thread_definitions = re.findall(thread_def_pattern, thread_1, re.DOTALL)
                
                # Create dictionaries for easy lookup
                results_dict = {thread_id: result.strip() for thread_id, result in thread_results}
                thread_def_dict = {thread_id: {'task': task.strip(), 'objective': objective.strip()} 
                                 for thread_id, task, objective in thread_definitions}
                
                for thread_id, processing_content in thread_processings:
                    # Get the reasoning context before this specific thread was launched
                    reasoning_before_thread = extract_reasoning_before_thread_launch(main_thread, thread_id)
                    
                    # Clean the reasoning before thread launch
                    cleaned_reasoning_before = clean_assistant_response(reasoning_before_thread)
                    
                    # Get thread definition (task and objective)
                    thread_def = thread_def_dict.get(thread_id, {'task': '', 'objective': ''})
                    
                    # Create input: problem + reasoning before thread + thread definition
                    input_prompt = f"""Problem: {question}
{cleaned_reasoning_before}
<thread id='{thread_id}'>
<task>
{thread_def['task']}
</task>
<objective>
{thread_def['objective']}
</objective>
</thread>"""
                    
                    # Get the corresponding result and clean it
                    result = results_dict.get(thread_id, "")
                    cleaned_result = clean_assistant_response(result)
                    
                    # Clean the processing content as well
                    cleaned_processing_content = clean_assistant_response(processing_content.strip())
                    
                    # Create the target output (only processing + result for this thread)
                    target_output = f"""<thread_processing id='{thread_id}'>
{cleaned_processing_content}
</thread_processing>
<thread_result id='{thread_id}'>
{cleaned_result}
</thread_result>"""
                    
                    conversation = {
                        "conversations": [
                            {
                                "from": "system",
                                "value": system_prompt
                            },
                            {
                                "from": "human",
                                "value": input_prompt
                            },
                            {
                                "from": "gpt",
                                "value": target_output
                            }
                        ]
                    }
                    converted_data.append(conversation)
    
    print(f"Prepared {len(converted_data)} thread dialogue examples")
    
    # Save training data
    os.makedirs('data', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Create dataset info for thread dialogue
    dataset_info = {
        "thread_dialogue": {
            "file_name": os.path.basename(output_file),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value", 
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system"
            }
        }
    }
    
    # Save dataset info in the same directory as the output file
    output_dir = os.path.dirname(output_file)
    dataset_info_path = os.path.join(output_dir, 'thread_dialogue_dataset_info.json')
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"Thread dialogue dataset info saved to: {dataset_info_path}")

def extract_reasoning_before_thread_launch(main_thread_content, target_thread_id):
    """
    Extract only the reasoning that happened before a specific thread was launched
    Returns the reasoning process up to (but not including) the launch_threads block
    that contains the target thread
    """
    # Find the launch_threads block that contains our target thread
    launch_pattern = r'<launch_threads>\n(.*?)\n</launch_threads>'
    launch_matches = re.finditer(launch_pattern, main_thread_content, re.DOTALL)
    
    for match in launch_matches:
        launch_content = match.group(1)
        if f"<thread id='{target_thread_id}'>" in launch_content:
            # Get everything before this launch_threads block
            reasoning_before = main_thread_content[:match.start()].strip()
            
            # Extract only the reasoning process part (remove system/user parts if present)
            if "Assistant: " in reasoning_before:
                reasoning_before = reasoning_before.split("Assistant: ", 1)[1]
            
            # Clean up and return only the reasoning content
            reasoning_before = reasoning_before.strip()
            
            # Clean the reasoning content by removing special tags
            cleaned_reasoning_before = clean_assistant_response(reasoning_before)
            
            return cleaned_reasoning_before
    
    # If no matching launch block found, return empty
    return ""
if __name__ == "__main__":
    # Example usage for all functions
    
    # Original converter (includes all data)
    # print("=== Original Converter (All Data) ===")
    # convert_jsonl_to_sharegpt_format(
    #     "../data_processing/output.jsonl", 
    #     "data/parallel_thinking_all.jsonl"
    # )
    
    # print("\n" + "="*50 + "\n")
    
    # New converter with assistant filtering (filtered data with reasoning process)
    print("=== Assistant Filtering Converter (Reasoning Process Only) ===")
    convert_with_assistant_filtering(
        "/home/admin/langzhang.zzy/inclusionAI/AReaL/parallel-thinking-main/data_processing/xml_output_new_no_first_step_parallel_large_conclusion_copy/converted_data_new_cleaned_with_parallel_tags_with_planning_conclusion.jsonl",
        "/home/admin/langzhang.zzy/inclusionAI/AReaL/parallel-thinking-main/data_processing/xml_output_new_no_first_step_parallel_large_conclusion_copy/converted_data_new_cleaned_with_parallel_tags_with_planning_conclusion_main.jsonl"
    )
    
    print("\n" + "="*50 + "\n")
    
    # # New thread dialogue converter
    print("=== Thread Dialogue Converter ===")
    convert_to_thread_dialogue_format(
        "/home/admin/langzhang.zzy/inclusionAI/AReaL/parallel-thinking-main/data_processing/xml_output_new_no_first_step_parallel_large_conclusion_copy/converted_data_new_cleaned_with_parallel_tags_with_planning_conclusion.jsonl",
        "/home/admin/langzhang.zzy/inclusionAI/AReaL/parallel-thinking-main/data_processing/xml_output_new_no_first_step_parallel_large_conclusion_copy/converted_data_new_cleaned_with_parallel_tags_with_planning_conclusion_thread.jsonl"
    )
    # convert_jsonl_to_sharegpt_format_main_dialogue("/home/zhangzy/parallel-thinking/data_processing/xml_output/cleaned_with_planning.jsonl", 
    #                                                "/home/zhangzy/parallel-thinking/data_processing/xml_output/cleaned_with_planning_thread.jsonl")
    