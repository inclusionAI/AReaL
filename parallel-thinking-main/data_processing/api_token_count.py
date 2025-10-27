from dotenv import load_dotenv
import os
import re
import numpy as np
from openai import OpenAI
# This code loads the OpenAI API key and base URL from environment variables using the dotenv package.
# It ensures that sensitive information is not hardcoded in the script, enhancing security.
from extract_step import extract_steps_regex
from dotenv import load_dotenv
# from problem import problem, problem2, problem3, CoT_solution, CoT_solution_2, CoT_solution_3, problem4, CoT_solution_4, problem5, CoT_solution_5
import os
load_dotenv()
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")
silicon_api_key = os.environ.get("SILICON_API_KEY")
silicon_base_url = os.environ.get("SILICON_BASE_URL")
seed = 42
# print(openai_base_url)  
client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

# Global token tracking
total_token_usage = {
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_tokens': 0
}

def track_token_usage(response, call_description=""):
    """Track token usage from API response"""
    global total_token_usage
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        total_token_usage['prompt_tokens'] += usage.prompt_tokens
        total_token_usage['completion_tokens'] += usage.completion_tokens
        total_token_usage['total_tokens'] += usage.total_tokens
        
        print(f"[{call_description}] Tokens - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
        print(f"[Running Total] Prompt: {total_token_usage['prompt_tokens']}, Completion: {total_token_usage['completion_tokens']}, Total: {total_token_usage['total_tokens']}")

def get_step_content(problem, CoT_solution, problem_index, turn = 0):
    global total_token_usage
    print(f"\n=== Starting get_step_content (turn {turn}) ===")
    
    if turn > 3:
        print("Warning: Too many turns, please check the problem and CoT solution.")
        raise ValueError("Too many turns, please check the problem and CoT solution.")
    turn += 1
    
    from dotenv import load_dotenv
    import os
    from openai import OpenAI
    # This code loads the OpenAI API key and base URL from environment variables using the dotenv package.
    # It ensures that sensitive information is not hardcoded in the script, enhancing security.

    from dotenv import load_dotenv
    # from problem import problem, problem2, problem3, CoT_solution, CoT_solution_2, CoT_solution_3, problem4, CoT_solution_4, problem5, CoT_solution_5
    import os
    load_dotenv()
    openai_api_key = os.environ.get("INFINI_API_KEY")
    openai_base_url = os.environ.get("INFINI_BASE_URL")
    silicon_api_key = os.environ.get("SILICON_API_KEY")
    silicon_base_url = os.environ.get("SILICON_BASE_URL")
    seed = 42
    # print(openai_base_url)  
    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    import re
    import numpy as np

    from utils import extract_step_descriptions, parse_dependency_output, identify_parallel_steps, extract_code_blocks  
    # You can choose a model from the following list
    # Or you can log into your Infini-AI or SiliconFlow account, and find an available model you want to use.
    # model = "Qwen/QVQ-72B-Preview"
    # model="llama-3.3-70b-instruct"
    model="deepseek-chat"

    # from problem import problem, CoT_solution, problem2, CoT_solution_2
    from utils import extract_step_descriptions
    model="deepseek-chat"

    user_content = f"Problem: {problem}\nCoT solution: {CoT_solution}"
    
    # API Call 1: Initial step analysis
    print("\nAPI Call 1: Initial step analysis")
    response = client.chat.completions.create(
      model=model,
      temperature=0,
      seed = seed,
      messages=[
        {"role": "system", "content": """You are a helpful assistant for math problems thinking process analysis. The user will give you a math problem and its CoT solution by LLM. You need to analyze the CoT solution and divide the solution into several steps, each thought process (like calculation, recalling theorems, reflection, etc) should be regarded as a single step. There should never be substeps under each step. Make the steps has as small granularity as possible. For each step, you need to give a short description of what the model do this description should only contain the task and objective of this step, and should never contain the result of this step. 
         EXAMPLES:
         GOOD: "1. **Length calculation**: The model calculates the length of the line segment AB."
         BAD: "1. **Length calculation**: The model calculates the length of the line segment AB, which is 5 cm."
         GOOD: "2. **Area calculation**: The model calculates the area of the triangle ABC."
         BAD: "2. **Area calculation**: The model calculates the area of the triangle ABC, by using the formula A = 1/2 * base * height, which is 10 cm^2."
         GOOD: "3. **Final result**: The model gives the final result of the problem."
         BAD: "3. **Final result**: The model gives the final result of the problem, which is 15 cm^2."
         GOOD: "18. **Line DI Equation**: The model identifies the line DI as the angle bisector and calculates its equation "
         BAD: "18. **Line DI Equation**: The model identifies the line DI as the angle bisector \( y = x \)."    
         GOOD: "7. **Probability Formula Recall**: The model recalls the general probability formula."
         BAD: "7. **Probability Formula Recall**: The model recalls the general probability formula, which is P(A) = n(A)/n(S)."  
         RULES:
         1. Each step should be a single thought process, do not include substeps.
         2. Each step should have a short description of what the model do, this description should ONLY contain the task and objective of this step, and should NEVER contain ANY result of this step.
         """},
        {"role": "user", "content": user_content}
      ]
    )
    track_token_usage(response, "Initial step analysis")
    
    if not response.choices:
      print("Warning: API is not working properly")
    # print(response.choices[0].message.content)
    llm_output = response.choices[0].message.content
    # print (llm_output)
    step_descriptions = extract_step_descriptions(llm_output)
    n = len(step_descriptions)
    print(f"Total steps: {n}")
    index = int(n/2)
    print ("INDEX: ", index)
    if index < 0 or index >= n:
        print("Warning: The index is out of range, please check the number of steps.")
        return get_step_content(problem, CoT_solution, problem_index, turn)
    
    # API Call 2: Extract half step content
    print(f"\nAPI Call 2: Extract half step content (step {index})")
    half_step = extract_one_step_content(problem, step_descriptions[int(index)], CoT_solution)
    
    # print ("Half step: ", half_step)
    # Divide the CoT into 2 halves according to the half step
    # Extract the first 50 characters and the last 50 characters of the half step
    half_step_head = half_step[:50]
    # print ("Half step head: ", half_step_head)
    half_step_tail = half_step[-50:]
    # print ("Half step tail: ", half_step_tail)
    if len(CoT_solution.split(half_step_head)) == 1:
      print ("Warning: The CoT solution may not be divided correctly, please check the CoT solution.")
      return get_step_content(problem, CoT_solution, problem_index, turn)
    CoT_solution_first = CoT_solution.split(half_step_head)[0]
    with open(f"first_half_{problem_index}.txt", "w") as f:
        f.write(CoT_solution_first)
    if len(CoT_solution.split(half_step_tail)) == 1:
      print ("Warning: The CoT solution may not be divided correctly, please check the CoT solution.")
    CoT_solution_second = CoT_solution.split(half_step_tail)[-1]
    with open(f"second_half_{problem_index}.txt", "w") as f:
        f.write(CoT_solution_second)
    if len(CoT_solution_second.split("\n")) == 1:
      print ("Warning: The second half of the CoT solution may not be divided correctly, please check the CoT solution.")
    
    # if len(CoT_solution) == 1:
    #   print("Warning: Dividing failed, the CoT solution may not be divided into two halves.")
    # Divide LLM output into 2 halves according to the Index
    
    llm_output_0 = ""
    for i in range(1, index+1):
        step = f"Step {i}: \n"
        llm_output_0 =  llm_output_0 + step + step_descriptions[i-1] + "\n"
    llm_output_1 = ""
    for j in range(index + 1, n):
        step = f"Step {j+1}: \n"
        llm_output_1 = llm_output_1 + step + step_descriptions[j] + "\n"
    with open(f"llm_out_0_{problem_index}.txt", "w") as f:
        f.write(llm_output_0)
        # print("LLM output 0: ", llm_output_0)
    with open(f"llm_out_1_{problem_index}.txt", "w") as f:
        f.write(llm_output_1)
        # print("LLM output 1: ", llm_output_1)
    
    # API Call 3: Get content together for first half
    print("\nAPI Call 3: Get content together for first half")
    step_info_0 = get_content_together(llm_output_0, CoT_solution_first)
    
    # API Call 4: Get content together for second half
    print("\nAPI Call 4: Get content together for second half")
    step_info_1 = get_content_together(llm_output_1, CoT_solution_second)
    
    with open(f"temp_{problem_index}.txt", "w") as f:
        f.write(step_info_0)
        f.write("\n\n")
        f.write(f"Step {index + 1}: {step_descriptions[index]}")
        f.write("\n")
        f.write(half_step)
        f.write(step_info_1)
    file = open(f'temp_{problem_index}.txt', 'r')
    content = file.read()
    step_content = extract_steps_regex(content)
    step_result = []
    step_descriptions_task = []
    step_descriptions_objective = []
    for desc in step_descriptions:
        # Split at the first colon
        parts = desc.split(':', 1)
        if len(parts) == 2:
            task = parts[0].strip()
            objective = parts[1].strip().replace("The model", "")
            step_descriptions_task.append(task)
            step_descriptions_objective.append(objective)
            
        else:
            # Handle case where there might not be a colon
            step_descriptions_task.append(desc)
            step_descriptions_objective.append("")
    print ("NUMBER OF STEP")
    print (len(step_content))
    if (len(step_content) != len(step_descriptions) ):
        print("Warning: The length of step_content and step_descriptions must be the same. Try again.")
        return get_step_content(problem, CoT_solution, problem_index, turn)
    
    # API Calls 5-N: Extract results for each step
    print(f"\nAPI Calls 5-{4+len(step_descriptions)}: Extract results for each step")
    for i, desc in enumerate(step_descriptions, 0):
        print(f"\nAPI Call {5+i}: Extract result for step {i+1}")
        result = client.chat.completions.create(
          model=model,
          temperature=0,
            seed = seed,
          messages=[
            {"role": "system", "content": "You are a helpful assistant for math problems thinking process analysis. The user will give you the target of the thinking step and the content of this step. You need to analyze the content of this step and give me the result of this step. The result should be a single line of text, do not include any other information."},
            {"role": "user", "content": f" Target: {desc}. \n Please give me the result of this step: {step_content[i]}. "},
          ]
        )
        track_token_usage(result, f"Extract result for step {i+1}")
        step_result.append(result.choices[0].message.content)
        
    # API Call N+1: Dependency analysis
    print(f"\nAPI Call {5+len(step_descriptions)}: Dependency analysis")
    dependency_analysis = response.choices[0].message.content
    response = client.chat.completions.create(
      model=model,
      temperature=0,
        seed = seed,
      messages=[
        {"role": "system", "content": "The user want to do math problem in parallel. Please analyze the dependency of each step in the solution process given by the user. The output result should be a list of tuples like '(i, j)' where i and j are the indice of the step, each tuple should be a pair of step index and where the first step depends on the second step. As the user need to do math problem in parallel, please make the dependency as small as possible. But you should also make sure all dependencies are included in the output. If there is really dependency, you should never remove it."},
        {"role": "user", "content": f"Please analyze the dependency of each step in the following thought process: {dependency_analysis}. "}
      ]
    )
    track_token_usage(response, "Dependency analysis")

    # print(response.choices[0].message.content)
    dependency_output = response.choices[0].message.content
    adj_matrix = parse_dependency_output(dependency_output)

    # print("\nAdjacency Matrix for Dependency DAG:")
    # print(adj_matrix)

    # Optionally visualize or analyze the graph
    # print(f"\nNumber of steps: {adj_matrix.shape[0]}")
    # print(f"Total dependencies: {np.sum(adj_matrix)}")

    parallel_groups = identify_parallel_steps(adj_matrix)
    # print("\nParallel Execution Groups:")
    # for i, group in enumerate(parallel_groups, 1):
    #     print(f"Group {i}: Steps {group}")
    if (len(step_content) != len(step_descriptions_task) or len(step_content) != len(step_result)) :
        print("Warning: The length of step_content, step_descriptions_task, step_descriptions_objective, step_result, and dependency_output must be the same. Try again.")
        step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups = get_step_content(problem, CoT_solution, problem_index, turn)
        
        # raise ValueError("The length of step_content, step_descriptions_task, step_descriptions_objective, step_result, and dependency_output must be the same.")
    print("CORRECT OUTPUT LENGTH")
    print(f"\n=== get_step_content completed. Total API calls: {6+len(step_descriptions)} ===")
    return step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups

def delete_result_from_objective(step_descriptions_objective, step_result):
    """
    Remove the result from the objective descriptions.
    
    Args:
        step_descriptions_objective (list): List of step descriptions with objectives.
        
    Returns:
        list: Updated list with results removed from objectives.
    """
    new_objective = []
    print("\n=== Starting delete_result_from_objective ===")
    for i in range(len(step_descriptions_objective)):
        step_descriptions_objective_i = step_descriptions_objective[i]
        step_result_i = step_result[i]
        print(f"\nAPI Call: Delete result from objective {i+1}")
        response = client.chat.completions.create(
            model="deepseek-v3",
            temperature=0,
            seed = seed,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for math problems thinking process analysis. The user will give you the objective of a step and the result of this step. You need to remove the result from the objective. If there is no result in the objective, just return the objective as it is."},
                {"role": "user", "content": f"Objective: {step_descriptions_objective_i}. Result: {step_result_i}. Please remove the result from the objective."}
            ]
        )
        track_token_usage(response, f"Delete result from objective {i+1}")
        new_objective.append(response.choices[0].message.content.strip())
    print(f"\n=== delete_result_from_objective completed. Total API calls: {len(step_descriptions_objective)} ===")
    return step_descriptions_objective

def get_content_together(steps, CoT):
  # print ("STEP:", steps)
  # print ("CoT:", CoT)
  print(f"API Call: Get content together")
  content = client.chat.completions.create(
      model="deepseek-chat",
      temperature=0,
      seed = seed,
      max_tokens = 8192,
      messages=[
        {"role": "system", "content": """You are a helpful assistant for math problems thinking process analysis. The user will give you a CoT solution by LLM of a math problem. You will also be given the steps of the CoT solution, each step has a name and description. Now you need find the content of each step in the CoT solution, what you need to do is just to **copy** the content of each step from the CoT solution. The content of each step must exist in the CoT solution. 
         IMPORTANT RULES:
         1. The content of each step must exist in the CoT solution.
         2. The content of each step should never overlap with other steps. For any snippet of the CoT solution, it should only be included in one step.
         3. You need to include the full content of each step starting from the first thought of the step to the final conclusion of the step. NO WORD SHOULD BE LEFT OUT.
         4. The content contains the thinking process of the step, including the calculation, reasoning, and any other thought process, it should be much longer than the description of the step.
         5. You should strictly follow the steps given by the user, do not skip any step, do not add any extra steps, also do not change the number before each step.
         6. You should ALWAYS write the content in the following format:
         Step <number>: <Task> \n <Objective> \n <Content of the step>
         Notice that <number>, <Task>, <Objective> has all been given in the steps, you should just **copy** them and then fill in the <Content of the step>. DO NOT put <number>, <Task>, <Objective> in your content, replace them with the actual content of the step.
         7. VERY IMPORTANT: The number of steps you give should be the same as the number of steps I give you, do not add or remove any steps.
         """},
        {"role": "user", "content": f": Steps: {steps}. CoT solution: {CoT}. "}
      ]
    )
  track_token_usage(content, "Get content together")
  print ("CONTENT: ", content.choices[0].message.content.strip())
  return content.choices[0].message.content.strip()

from utils import extract_code_blocks
def extract_one_step_content(problem, step, CoT_solution):
    import re
    model = "deepseek-chat"
    # from problem import problem, CoT_solution, problem2, CoT_solution_2
    from utils import extract_step_descriptions
    print(f"API Call: Extract one step content")
    loacl_response = client.chat.completions.create(
          model=model,
          temperature=0,
            seed = seed,
          messages=[
            {"role": "system", "content": """You are a helpful assistant for math problems thinking process analysis. The user will give you a math problem and its CoT solution by LLM. And there will be a summary of **ONE** step in the CoT solution. Now just find the content of this step in the CoT solution. Just give user the full content of this step starting from the first thought of the step to the final conclusion of the step, do not give me any other information. The content of this step must exist in the CoT solution. What you need to do is just to **copy** the content of this step from the CoT solution. 
            You need to include the full content of this step starting from the first thought of the step to the final conclusion of the step. Put your result in a code block that starts with ``` and ends with ```.
             Do not add any extra information, just copy the content of this step from the CoT
             IMPORTANT RULES:
             1. The content of this step must exist in the CoT solution.
             2. Do NOT include any other information, just copy the content of this step from the CoT solution.
             3. What you need to do is just to **copy** the content of this step from the CoT solution, do not add any extra information.
             4. Every token you give me should be the content of this step, do not add any extra information.
             EXAMPLES:
             GOOD: "```When we add 1 with 2, we get 3.```"
             BAD: "```Sum calculation: The model calculates the sum of 1 and 2.\n When we add 1 with 2, we get 3.```"
            """},
            {"role": "user", "content": f""" Please give me the content of  "{step}" in the following thought process: "{CoT_solution}". 
             """},
          ]
        )
    track_token_usage(loacl_response, "Extract one step content")
    result = loacl_response.choices[0].message.content
    # Extract the code block content
    ret = re.findall(r'```(.*?)```', result, re.DOTALL)
  
    if len(ret) == 0:
        print("Warning: No code blocks found in LLM response. Using raw response.")
        print(f"Raw response: {result}")
        return result.strip()
    print(ret[0])
    return ret[0]

def get_step_content_direct(problem, CoT_solution, problem_index, turn = 0):
    """
    Extract step content from CoT solution without splitting it into halves.
    This is a simplified version of get_step_content that processes the entire CoT at once.
    """
    global total_token_usage
    print(f"\n=== Starting get_step_content_direct (turn {turn}) ===")
    
    if turn > 3:
        print("Warning: Too many turns, please check the problem and CoT solution.")
        raise ValueError("Too many turns, please check the problem and CoT solution.")
    turn += 1
    
    from utils import extract_step_descriptions, parse_dependency_output, identify_parallel_steps, extract_code_blocks  
    
    model="deepseek-chat"

    user_content = f"Problem: {problem}\nCoT solution: {CoT_solution}"
    
    # API Call 1: Initial step analysis
    print("\nAPI Call 1: Initial step analysis")
    response = client.chat.completions.create(
      model=model,
      temperature=0,
      seed = seed,
      messages=[
        {"role": "system", "content": """You are a helpful assistant for math problems thinking process analysis. The user will give you a math problem and its CoT solution by LLM. You need to analyze the CoT solution and divide the solution into several steps, each thought process (like calculation, recalling theorems, reflection, etc) should be regarded as a single step. There should never be substeps under each step. Make the steps has as small granularity as possible. For each step, you need to give a short description of what the model do this description should only contain the task and objective of this step, and should never contain the result of this step. 
         EXAMPLES:
         GOOD: "1. **Length calculation**: The model calculates the length of the line segment AB."
         BAD: "1. **Length calculation**: The model calculates the length of the line segment AB, which is 5 cm."
         GOOD: "2. **Area calculation**: The model calculates the area of the triangle ABC."
         BAD: "2. **Area calculation**: The model calculates the area of the triangle ABC, by using the formula A = 1/2 * base * height, which is 10 cm^2."
         GOOD: "3. **Final result**: The model gives the final result of the problem."
         BAD: "3. **Final result**: The model gives the final result of the problem, which is 15 cm^2."
         GOOD: "18. **Line DI Equation**: The model identifies the line DI as the angle bisector and calculates its equation "
         BAD: "18. **Line DI Equation**: The model identifies the line DI as the angle bisector \( y = x \)."    
         GOOD: "7. **Probability Formula Recall**: The model recalls the general probability formula."
         BAD: "7. **Probability Formula Recall**: The model recalls the general probability formula, which is P(A) = n(A)/n(S)."  
         RULES:
         1. Each step should be a single thought process, do not include substeps.
         2. Each step should have a short description of what the model do, this description should ONLY contain the task and objective of this step, and should NEVER contain ANY result of this step.
         """},
        {"role": "user", "content": user_content}
      ]
    )
    track_token_usage(response, "Initial step analysis")
    
    if not response.choices:
        print("Warning: API is not working properly")
        return get_step_content_direct(problem, CoT_solution, problem_index, turn)
    
    llm_output = response.choices[0].message.content
    step_descriptions = extract_step_descriptions(llm_output)
    n = len(step_descriptions)
    print(f"Total steps: {n}")
    
    # Format all steps for get_content_together
    formatted_steps = ""
    for i, desc in enumerate(step_descriptions, 1):
        step = f"Step {i}: \n"
        formatted_steps = formatted_steps + step + desc + "\n"
    
    # API Call 2: Use get_content_together to extract all step contents at once
    print("\nAPI Call 2: Get content together for all steps")
    step_info = get_content_together(formatted_steps, CoT_solution)
    
    # Save to temporary file and extract using regex
    with open(f"temp_direct_{problem_index}.txt", "w") as f:
        f.write(step_info)
    
    file = open(f'temp_direct_{problem_index}.txt', 'r')
    content = file.read()
    step_content = extract_steps_regex(content)
    
    # Parse step descriptions into tasks and objectives
    step_descriptions_task = []
    step_descriptions_objective = []
    for desc in step_descriptions:
        parts = desc.split(':', 1)
        if len(parts) == 2:
            task = parts[0].strip()
            objective = parts[1].strip().replace("The model", "")
            step_descriptions_task.append(task)
            step_descriptions_objective.append(objective)
        else:
            step_descriptions_task.append(desc)
            step_descriptions_objective.append("")
    
    print(f"NUMBER OF STEPS: {len(step_content)}")
    
    # Validate that all arrays have the same length
    if len(step_content) != len(step_descriptions):
        print("Warning: The length of step_content and step_descriptions must be the same. Try again.")
        return get_step_content_direct(problem, CoT_solution, problem_index, turn)
    
    # API Calls 3-N: Extract results for each step
    print(f"\nAPI Calls 3-{2+len(step_descriptions)}: Extract results for each step")
    step_result = []
    for i, desc in enumerate(step_descriptions):
        print(f"\nAPI Call {3+i}: Extract result for step {i+1}")
        result = client.chat.completions.create(
            model=model,
            temperature=0,
            seed=seed,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for math problems thinking process analysis. The user will give you the target of the thinking step and the content of this step. You need to analyze the content of this step and give me the result of this step. The result should be a single line of text, do not include any other information."},
                {"role": "user", "content": f"Target: {desc}. \n Please give me the result of this step: {step_content[i]}."}
            ]
        )
        track_token_usage(result, f"Extract result for step {i+1}")
        step_result.append(result.choices[0].message.content)
    
    # API Call N+1: Dependency analysis
    print(f"\nAPI Call {3+len(step_descriptions)}: Dependency analysis")
    dependency_analysis = response.choices[0].message.content
    dependency_response = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=seed,
        messages=[
            {"role": "system", "content": "The user want to do math problem in parallel. Please analyze the dependency of each step in the solution process given by the user. The output result should be a list of tuples like '(i, j)' where i and j are the indice of the step, each tuple should be a pair of step index and where the first step depends on the second step. As the user need to do math problem in parallel, please make the dependency as small as possible. But you should also make sure all dependencies are included in the output. If there is really dependency, you should never remove it."},
            {"role": "user", "content": f"Please analyze the dependency of each step in the following thought process: {dependency_analysis}."}
        ]
    )
    track_token_usage(dependency_response, "Dependency analysis")
    
    dependency_output = dependency_response.choices[0].message.content
    adj_matrix = parse_dependency_output(dependency_output)
    parallel_groups = identify_parallel_steps(adj_matrix)
    
    # Final validation
    if (len(step_content) != len(step_descriptions_task) or len(step_content) != len(step_result)):
        print("Warning: The length of step_content, step_descriptions_task, step_descriptions_objective, step_result, and dependency_output must be the same. Try again.")
        return get_step_content_direct(problem, CoT_solution, problem_index, turn)
    
    print("CORRECT OUTPUT LENGTH")
    print(f"\n=== get_step_content_direct completed. Total API calls: {3+len(step_descriptions)} ===")
    return step_descriptions_task, step_descriptions_objective, step_content, step_result, dependency_output, adj_matrix, parallel_groups

def print_token_summary():
    """Print a summary of total token usage"""
    print(f"\n{'='*60}")
    print("TOKEN USAGE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Prompt Tokens: {total_token_usage['prompt_tokens']:,}")
    print(f"Total Completion Tokens: {total_token_usage['completion_tokens']:,}")
    print(f"Total Tokens: {total_token_usage['total_tokens']:,}")
    print(f"{'='*60}")

def reset_token_counter():
    """Reset the global token counter"""
    global total_token_usage
    total_token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }