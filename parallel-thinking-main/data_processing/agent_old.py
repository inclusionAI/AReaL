from api import get_step_content
from problem import problem, CoT_solution, problem2, CoT_solution_2
step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups = get_step_content(problem2, CoT_solution_2)
print (step_descriptions_task)
print (step_descriptions_objective)
print ("CONTENT: ", step_content)
print (step_result)
print (dependency_output)
print (adj_matrix)
print (parallel_groups)

with open("problem1_main.xml", "w") as output_file:
    output_file.write ("Question: ")
    output_file.write (problem2)
    output_file.write ("\n")
    output_file.write ("Assistant: \n <reasoning_process>\n")
    for i in range(len(parallel_groups)):
        if len(parallel_groups[i]) > 1:
            output_file.write (f"<parallel_processing>\n<launch_threads>\n")
            for j in range(len(parallel_groups[i])):
                output_file.write (f"<thread id= '{j}'>\n")
                output_file.write (f"<task>")
                output_file.write (f"{step_descriptions_task[parallel_groups[i][j]-1]}")
                output_file.write (f"</task>")
                output_file.write (f"<objective>")
                output_file.write (f"{step_descriptions_objective[parallel_groups[i][j]-1]}")
                output_file.write (f"</objective>")
                output_file.write (f"</thread>\n")
            output_file.write (f"</launch_threads>\n")
            output_file.write (f"<step_resolution>\n")
            for j in range(len(parallel_groups[i])):
                output_file.write (f"<thread_result: id = '{j}'>\n")
                output_file.write (f"{step_result[parallel_groups[i][j]-1]}")
                output_file.write (f"</thread_result>\n")
            output_file.write (f"</step_resolution>\n")
        else:
            output_file.write (step_content[parallel_groups[i][0]-1][0])
    output_file.write ("</reasoning_process>\n")