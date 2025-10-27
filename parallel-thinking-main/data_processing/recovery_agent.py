def clean_step_content(content, task, objective):
    """
    Clean the step content by removing task, objective, and colon lines.
    
    Args:
        content (str): The raw step content
        task (str): The task name to remove
        objective (str): The objective to remove
        
    Returns:
        str: Cleaned content
    """
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip empty lines at the beginning
        if not line_stripped and not cleaned_lines:
            continue
            
        # Skip lines that match the task
        if line_stripped == f": {task}":
            continue
            
        # Skip lines that are just the task name
        if line_stripped == task:
            continue
            
        # Skip lines that start with "The model" (objective lines)
        if line_stripped.startswith("The model"):
            continue
            
        # Skip lines that are just ":"
        if line_stripped == ":":
            continue
            
        # Keep everything else
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

# ...existing code...

def save_xml_from_temp_recovery(problem_index, output_dir="./"):
    """
    Save XML files using recovered data from temp files.
    Creates both main XML and thread XML files like agent_new.py
    Loads problem statement and final answer from JSONL data file.
    
    Args:
        problem_index (int): The problem index used in temp file names
        output_dir (str): Directory to save XML files
    """
    from recover_from_temp import recover_from_temp_files_enhanced
    import os
    import json
    
    try:
        # Load problem statement and final answer from JSONL file
        problem_statement = None
        final_answer = None
        
        try:
            # Try to load from JSONL file
            with open('/home/admin/langzhang.zzy/inclusionAI/AReaL/out_open_math_reasoning_small.jsonl', 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == problem_index - 1:  # Adjust for zero-based index
                        data = json.loads(line.strip())
                        problem_statement = data.get('Problem', f"[Problem {problem_index} - statement not found]")
                        
                        # Extract final answer from CoT
                        cot_content = data.get('CoT', '')
                        if '</think>' in cot_content:
                            final_answer = cot_content.split('</think>', 1)[1].strip()
                        else:
                            final_answer = cot_content
                        break
                else:
                    # If we didn't find the line (problem_index too high)
                    problem_statement = f"[Problem {problem_index} - index out of range]"
                    final_answer = "[Final answer not found]"
                    
        except FileNotFoundError:
            try:
                # Alternative: try different JSONL file names
                with open(f'problem_{problem_index}.jsonl', 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                    data = json.loads(line)
                    problem_statement = data.get('Problem', f"[Problem {problem_index} - statement not found]")
                    cot_content = data.get('CoT', '')
                    if 'CoT' in cot_content:
                        final_answer = cot_content.split('CoT', 1)[1].strip()
                    else:
                        final_answer = cot_content
            except:
                problem_statement = f"[Problem {problem_index} - JSONL file not found]"
                final_answer = "[Final answer not found]"
        except Exception as e:
            print(f"Error reading JSONL file: {e}")
            problem_statement = f"[Problem {problem_index} - error reading file]"
            final_answer = "[Final answer not found]"
        
        # Recover the data from temp files
        step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups = recover_from_temp_files_enhanced(problem_index)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        thread_id_to_step = {}
        thread_id_counter = 0
        
        # Save main XML file
        main_xml_path = os.path.join(output_dir, f"temp_main_{problem_index}.xml")
        with open(main_xml_path, "w", encoding='utf-8') as output_file:
            output_file.write("Question: ")
            output_file.write(problem_statement)
            output_file.write("\n")
            output_file.write("Assistant: \n<think>\n")
            
            # Process parallel groups
            for i in range(len(parallel_groups)):
                if len(parallel_groups[i]) > 1:
                    output_file.write(f"<launch_threads>\n")
                    for j in range(len(parallel_groups[i])):
                        step_index = parallel_groups[i][j] - 1
                        thread_id_to_step[thread_id_counter] = step_index
                        
                        output_file.write(f"<thread id='{thread_id_counter}'>\n")
                        output_file.write(f"<task>\n")
                        output_file.write(f"{step_descriptions_task[step_index]}\n")
                        output_file.write(f"</task>\n")
                        output_file.write(f"<objective>\n")
                        output_file.write(f"{step_descriptions_objective[step_index]}\n")
                        output_file.write(f"</objective>\n")
                        output_file.write(f"</thread>\n")
                        
                        thread_id_counter += 1
                        
                    output_file.write(f"</launch_threads>\n")
                    output_file.write(f"<step_resolution>\n")
                    
                    # Get the thread IDs for this parallel group
                    group_thread_ids = list(range(thread_id_counter - len(parallel_groups[i]), thread_id_counter))
                    
                    for thread_id in group_thread_ids:
                        step_index = thread_id_to_step[thread_id]
                        output_file.write(f"<thread_result id='{thread_id}'>\n")
                        output_file.write(f"{step_result[step_index]}\n")
                        output_file.write(f"</thread_result>\n")
                        
                    output_file.write(f"</step_resolution>\n")
                    # output_file.write(f"</parallel_processing>\n")
                else:
                    # For non-parallel steps, clean the content before writing
                    step_index = parallel_groups[i][0] - 1
                    cleaned_content = step_content[step_index]
                    # output_file.write("<think: type = ''>\n")
                    output_file.write(cleaned_content)
                    output_file.write("\n")
                    # output_file.write("</think: type = ''>\n")
            
            output_file.write("</think>\n")
            # output_file.write("<answer>\n")
            output_file.write(final_answer)
            # output_file.write("\n")
            # output_file.write("</answer>\n")
        
        # Save thread XML file
        thread_xml_path = os.path.join(output_dir, f"temp_thread1_{problem_index}.xml")
        with open(thread_xml_path, "w", encoding='utf-8') as f:
            f.write("Problem: ")    
            f.write(problem_statement)
            f.write("\n")
            f.write("Assistant: \n")
            
            for thread_id, step_index in thread_id_to_step.items():
                f.write(f"<thread id='{thread_id}'>\n")
                f.write(f"<task>\n")
                f.write(f"{step_descriptions_task[step_index]}\n")
                f.write(f"</task>\n")
                f.write(f"<objective>\n")
                f.write(f"{step_descriptions_objective[step_index]}\n")
                f.write(f"</objective>\n")
                f.write(f"</thread>\n")
                
                # Clean the content before writing to thread_processing
                cleaned_content = clean_step_content(
                    step_content[step_index], 
                    step_descriptions_task[step_index], 
                    step_descriptions_objective[step_index]
                )
                f.write(f"<thread_processing id = '{thread_id}'>\n")
                f.write(f"{cleaned_content}\n")   
                f.write(f"</thread_processing>\n")
                f.write(f"<thread_result id='{thread_id}'>\n")
                f.write(f"{step_result[step_index]}\n")
                f.write(f"</thread_result>\n")
        
        print(f"XML files saved successfully!")
        print(f"Main XML: {main_xml_path}")
        print(f"Thread XML: {thread_xml_path}")
        print(f"Problem statement: {problem_statement[:100]}...")
        print(f"Final answer: {final_answer[:100]}...")
        
    except Exception as e:
        print(f"Error saving XML for problem {problem_index}: {e}")
        import traceback
        traceback.print_exc()

# ...existing code...


def print_xml_from_temp_recovery(problem_index, problem_statement=None):
    """
    Print XML file content using recovered data from temp files.
    This mimics the XML generation logic from agent_new.py
    
    Args:
        problem_index (int): The problem index used in temp file names
        problem_statement (str, optional): The original problem statement
    """
    from recover_from_temp import recover_from_temp_files_enhanced
    
    try:
        # Recover the data from temp files
        step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups = recover_from_temp_files_enhanced(problem_index)
        
        print("="*80)
        print(f"XML CONTENT FOR PROBLEM {problem_index}")
        print("="*80)
        
        # Print main XML structure
        print("Question:", problem_statement if problem_statement else f"[Problem {problem_index}]")
        print("Assistant:")
        print("<think>")
        
        # Create thread mapping (similar to agent_new.py)
        thread_id_to_step = {}
        thread_id_counter = 0
        
        # Process parallel groups
        for i in range(len(parallel_groups)):
            if len(parallel_groups[i]) > 1:
                print("<parallel_processing>")
                print("<launch_threads>")
                
                # Create threads for this parallel group
                for j in range(len(parallel_groups[i])):
                    step_index = parallel_groups[i][j] - 1
                    thread_id_to_step[thread_id_counter] = step_index
                    
                    print(f"<thread id='{thread_id_counter}'>")
                    print("<task>")
                    print(f"{step_descriptions_task[step_index]}")
                    print("</task>")
                    print("<objective>")
                    print(f"{step_descriptions_objective[step_index]}")
                    print("</objective>")
                    print("</thread>")
                    
                    thread_id_counter += 1
                
                print("</launch_threads>")
                print("<step_resolution>")
                
                # Print thread results
                group_thread_ids = list(range(thread_id_counter - len(parallel_groups[i]), thread_id_counter))
                for thread_id in group_thread_ids:
                    step_index = thread_id_to_step[thread_id]
                    print(f"<thread_result id='{thread_id}'>")
                    print(f"{step_result[step_index]}")
                    print("</thread_result>")
                
                print("</step_resolution>")
                print("</parallel_processing>")
            else:
                # For non-parallel steps, clean the content before printing
                step_index = parallel_groups[i][0] - 1
                cleaned_content = clean_step_content(
                    step_content[step_index], 
                    step_descriptions_task[step_index], 
                    step_descriptions_objective[step_index]
                )
                print("<think: type = ''>")
                print(cleaned_content)
                print("</think: type = ''>")
        
        print("</think>")
        print("<answer>")
        print("[Final answer would go here]")
        print("</answer>")
        
        print("\n" + "="*80)
        print("THREAD DETAILS (temp_thread1_X.xml format)")
        print("="*80)
        
        # Print thread details format
        print("Problem:", problem_statement if problem_statement else f"[Problem {problem_index}]")
        print("Assistant:")
        
        for thread_id, step_index in thread_id_to_step.items():
            print(f"<thread id='{thread_id}'>")
            print("<task>")
            print(f"{step_descriptions_task[step_index]}")
            print("</task>")
            print("<objective>")
            print(f"{step_descriptions_objective[step_index]}")
            print("</objective>")
            print("</thread>")
            
            # Clean the content before printing
            cleaned_content = clean_step_content(
                step_content[step_index], 
                step_descriptions_task[step_index], 
                step_descriptions_objective[step_index]
            )
            print(f"<thread_processing id = '{thread_id}'>")
            print(f"{cleaned_content}")
            print("</thread_processing>")
            print(f"<thread_result id='{thread_id}'>")
            print(f"{step_result[step_index]}")
            print("</thread_result>")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total steps: {len(step_content)}")
        print(f"Total threads: {len(thread_id_to_step)}")
        print(f"Parallel groups: {len(parallel_groups)}")
        print(f"Dependencies: {dependency_output}")
        
    except Exception as e:
        print(f"Error printing XML for problem {problem_index}: {e}")
if __name__ == "__main__":
    # Print XML content to console
    # print_xml_from_temp_recovery(problem_index=1, problem_statement="Sample math problem")
    
    # Save XML files
    save_xml_from_temp_recovery(problem_index=1500)