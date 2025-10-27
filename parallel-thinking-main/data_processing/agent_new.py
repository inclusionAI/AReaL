# from problem import problem, CoT_solution, problem2, CoT_solution_2
from api import get_step_content, get_step_content_direct
from api import delete_result_from_objective
from transformers import AutoTokenizer
def main(problem, CoT_solution, problem_index):
    
    
    # Define token threshold - you can adjust this value as needed
    TOKEN_THRESHOLD = 7000  # Adjust this threshold based on your needs
    
    if '</think>' in CoT_solution:
        parts = CoT_solution.split('</think>', 1)  # Split only on first occurrence
        CoT_solution = parts[0].strip()  # Keep the part before </think>
        result = parts[1].strip()        # Store the part after </think>
    else:
        result = ""  # If no </think> found, result is empty
    
    # Count tokens in CoT solution
    try:
        # tokenizer = AutoTokenizer.from_pretrained("/storage/openpsi/users/zzy/model/parallel_1_5b/nvidia-parallel-thinking_1_5B_lr5/checkpoint-267/tokenizer.json")
        # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-0324")
        tokens = CoT_solution.split(' ')
        token_count = len(tokens)
        print(f"CoT token count: {token_count}")
        
        # Choose function based on token count
        if token_count > TOKEN_THRESHOLD:
            print(f"CoT is long ({token_count} tokens > {TOKEN_THRESHOLD}), using get_step_content for splitting")
            print ( f"problem_index: {problem_index}", f"length: {token_count}")
            # raise Exception("CoT is too long, using get_step_content for splitting")
            step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups = get_step_content(problem, CoT_solution, problem_index)
        else:
            print(f"CoT is short ({token_count} tokens <= {TOKEN_THRESHOLD}), using get_step_content_direct")
            print ( f"problem_index: {problem_index}", f"length: {token_count}")
            
            step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups = get_step_content_direct(problem, CoT_solution, problem_index)
    
    except Exception as e:
        print(f"Error counting tokens: {e}")
        print("Falling back to get_step_content_direct")
        # step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups = get_step_content_direct(problem, CoT_solution, problem_index)
        raise Exception("CoT is too long")
    # step_descriptions_objective = delete_result_from_objective(step_descriptions_objective, step_result)
    # For debugging only - you can comment these out in production
    for i in range (len(step_content)):
        # print(f"Step {i+1}: BEFORE cleaning:")
        # print(step_content[i])
        step_content[i] = step_content[i].strip().replace(f"{step_descriptions_task[i]} :", "")
        step_content[i] = step_content[i].strip().replace(f"{step_descriptions_task[i]}", "")
        step_content[i] = step_content[i].strip().replace(f"{step_descriptions_objective[i]} :", "")
        step_content[i] = step_content[i].strip().replace(f"{step_descriptions_objective[i]}", "")
        step_content[i] = step_content[i].strip().replace("The model", "")
        step_content[i] = step_content[i].strip().replace(f"{step_descriptions_objective[i]} :", "")
        step_content[i] = step_content[i].strip().replace(f"{step_descriptions_objective[i]}", "")
        if step_content[i][0] == ':':
            step_content[i][0] == ' '
        if step_content[i][1] == ':':
            step_content[i][1] == ' '
        if step_content[i][2] == ':':
            step_content[i][2] == ' '
        if step_content[i][1] == '"':
            step_content[i][1] == ' '
        if step_content[i][2] == '"':
            step_content[i][2] == ' '
        if step_content[i][3] == '"':
            step_content[i][3] == ' '
        if step_content[i][-1] == '"':
            step_content[i][-1] == ' '
        step_content[i] = step_content[i][3:]  # Remove the first two characters
        step_content[i] = step_content[i].strip().replace("<think>", "")
        step_content[i] = step_content[i].strip().replace("</think>", "")
    #     print(f"Step {i+1}:")
    #     print(step_content[i])
    # print(step_descriptions_task)
    # print(step_descriptions_objective)
    # print("CONTENT: ", step_content)
    # print(step_result)
    # print(dependency_output)
    # print(adj_matrix)
    # print(parallel_groups)
    thread_id_to_step = {}
    with open(f"temp_main_{problem_index}.xml", "w") as output_file:
        output_file.write("Question: ")
        output_file.write(problem)
        output_file.write("\n")
        output_file.write("Assistant: \n<reasoning_process>\n")
        
        # Create a global thread ID counter and mapping of thread ID to step index
        thread_id_counter = 0
        
        
        # First loop to create threads and record their IDs
        for i in range(len(parallel_groups)):
            if len(parallel_groups[i]) > 1:
                output_file.write(f"<parallel_processing>\n<launch_threads>\n")
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
                output_file.write(f"</parallel_processing>\n")
            else:
                # For non-parallel steps, just write the content
                output_file.write("<think: type = ''>\n")
                output_file.write(step_content[parallel_groups[i][0]-1])
                output_file.write("\n")
                output_file.write("</think: type = ''>\n")
        
        output_file.write("</reasoning_process>\n")
        output_file.write("<answer>\n")
        output_file.write(f"{result}")
        output_file.write("\n")
        output_file.write("</answer>\n")
    with open(f"temp_thread1_{problem_index}.xml", "w") as f:
        f.write("Problem: ")    
        f.write(problem)
        f.write("\n")
        f.write("Assistant: \n")
        # f.write("<reasoning_process>\n")
        for thread_id, step_index in thread_id_to_step.items():
            f.write(f"<thread id='{thread_id}'>\n")
            f.write(f"<task>\n")
            f.write(f"{step_descriptions_task[step_index]}\n")
            f.write(f"</task>\n")
            f.write(f"<objective>\n")
            f.write(f"{step_descriptions_objective[step_index]}\n")
            f.write(f"</objective>\n")
            f.write(f"</thread>\n")
            f.write(f"<thread_processing id = '{thread_id}'>\n")
            f.write(f"{step_content[step_index]}\n")   
            f.write(f"</thread_processing>\n")
            f.write(f"<thread_result id='{thread_id}'>\n")
            f.write(f"{step_result[step_index]}\n")
            f.write(f"</thread_result>\n")
        # f.write("</reasoning_process>\n")
    print("XML file generated successfully!")

def count_tokens(text: str, tokenizer_name_or_path: str = "/storage/openpsi/users/zzy/model/parallel_1_5b/nvidia-parallel-thinking_1_5B_lr5/checkpoint-267") -> int:
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return len(text.split())  # Fallback to word count

if __name__ == "__main__":
    from problem import problem, CoT_solution, problem6, CoT_solution_6
    problem_index = 3  # Change this to 2 for the second problem
    main(problem, CoT_solution, problem_index)