# from problem import problem, CoT_solution, problem2, CoT_solution_2
def main(problem, CoT_solution, problem_index):
    from api import get_step_content
    from api import delete_result_from_objective
    
    step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups = get_step_content(problem, CoT_solution, problem_index)
    # step_descriptions_objective = delete_result_from_objective(step_descriptions_objective, step_result)
    # For debugging only - you can comment these out in production
    length = len(step_descriptions_task)
    element_to_remove = length 
    # Remove specific element (e.g., element with value 3) from all sublists
    parallel_groups = [[item for item in sublist if item != element_to_remove] for sublist in parallel_groups]
    # Remove empty sublists if any
    parallel_groups = [sublist for sublist in parallel_groups if sublist]

# ...existing code...
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
        output_file.write("Assistant: \n <reasoning_process>\n")
        
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
                    output_file.write(f"<task>")
                    output_file.write(f"{step_descriptions_task[step_index]}")
                    output_file.write(f"</task>")
                    output_file.write(f"<objective>")
                    output_file.write(f"{step_descriptions_objective[step_index]}")
                    output_file.write(f"</objective>")
                    output_file.write(f"</thread>\n")
                    
                    thread_id_counter += 1
                    
                output_file.write(f"</launch_threads>\n")
                output_file.write(f"<step_resolution>\n")
                
                # Get the thread IDs for this parallel group
                group_thread_ids = list(range(thread_id_counter - len(parallel_groups[i]), thread_id_counter))
                
                for thread_id in group_thread_ids:
                    step_index = thread_id_to_step[thread_id]
                    output_file.write(f"<thread_result id='{thread_id}'>\n")
                    output_file.write(f"{step_result[step_index]}")
                    output_file.write(f"</thread_result>\n")
                    
                output_file.write(f"</step_resolution>\n")
                output_file.write(f"</parallel_processing>\n")
            else:
                # For non-parallel steps, just write the content
                output_file.write("<think: type = ''>\n")
                output_file.write(step_content[parallel_groups[i][0]-1])
                output_file.write("\n")
                output_file.write("</think: type = ''>\n")
        output_file.write("<think: type = ''>\n")
        output_file.write(step_content[length-1])
        output_file.write("\n")
        output_file.write("</think: type = ''>\n")
        output_file.write("</reasoning_process>\n")
    with open(f"temp_thread1_{problem_index}.xml", "w") as f:
        f.write("Problem: ")    
        f.write(problem)
        f.write("\n")
        f.write("Assistant: \n")
        f.write("<reasoning_process>\n")
        for thread_id, step_index in thread_id_to_step.items():
            f.write(f"<thread id='{thread_id}'>\n")
            f.write(f"<task>")
            f.write(f"{step_descriptions_task[step_index]}")
            f.write(f"</task>")
            f.write(f"<objective>")
            f.write(f"{step_descriptions_objective[step_index]}")
            f.write(f"</objective>")
            f.write(f"</thread>\n")
            f.write(f"<thread_processing id = '{thread_id}'>\n")
            f.write(f"{step_content[step_index]}")   
            f.write(f"</thread_processing>\n")
            f.write(f"<thread_result id='{thread_id}'>\n")
            f.write(f"{step_result[step_index]}")
            f.write(f"</thread_result>\n")
        f.write("</reasoning_process>\n")
        
    print("XML file generated successfully!")
if __name__ == "__main__":
    from problem import problem, CoT_solution, problem2, CoT_solution_2
    problem_index = 2  # Change this to 2 for the second problem
    main(problem2, CoT_solution_2, problem_index)