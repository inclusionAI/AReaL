from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_local_model_with_special_tokens():
    # Use the fixed model path
    model_path = "/nvme0n1/zzy_model/fresh_mixed_7151717/deepseek-parallel-thinking"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Define system prompt and user message
    system_prompt = """You are a helpful assistant that solve math problems.  

The input is a math problem and reasoning process before this thread is launched. Also the task and objective of this thread will be provided in the end of the input. You ahould complete the task and objective of this thread, start your processing with <thread_processing id = 'i'> where i is the index of the thread. End your processing with '</thread_processing>'. After this, put the result of this step between '<thread_result id = 'i'>' and '</thread_result>'. DO NOT output the special tag '<think>'  DO NOT output the special tag '<think>', what you need to do is to finish the reasoning of <thread_processing id='i'> and output its result, you only need to solve this partial step not the full problem\n Stop reasoning when you reach the end of the thread processing and then output the result in the format of '<thread_result id = 'i'>result</thread_result>'.\n NEVER solve the whole problem, you MUST STOP after the objective of this step is reached. Also, think for a while before you output the result, put the reasoning process in <thread_processing id='i'> ... </thread_processing> tag, where 'i' is the id of this thread. Put the result of **THIS STEP** (not the whole problem) in the <thread_result id='i'> ... </thread_result> tag"""
    
    user_question = r"""Problem: Find the minimum value of the expression \n\\[ P = 9x^2 + 2y^2 + 5z^2 - 6xy + 12xz - 6y + 12z + 2022. \\]\n<think>\n\nOkay, so I have this problem where I need to find the minimum value of the expression P = 9x² + 2y² + 5z² - 6xy + 12xz - 6y + 12z + 2022. Hmm, quadratic in three variables. Since it's asking for the minimum, maybe I can complete the squares for each variable or use partial derivatives. Let me think.\n\nTo begin, we need to evaluate the expression's structure to determine the most effective method for finding its minimum. The presence of both quadratic and linear terms suggests two viable approaches: completing the square to rewrite the expression in a more tractable form or analyzing critical points via partial derivatives. The first approach will help us identify whether the expression can be simplified into a sum of squares, potentially revealing its minimum value, while the second will allow us to systematically examine the behavior of the function by solving for points where all partial derivatives vanish. Given the mixed terms and coefficients, both methods must be carefully considered to avoid overlooking any peculiarities in the expression's behavior, such as unboundedness or degenerate critical points.\n\n<launch_threads>\n<thread id='0'>\n<task>\nInitial Approach Selection\n</task>\n<objective>\n considers completing the square or using partial derivatives to find the minimum.\n</objective>\n</thread>\n<thread id='1'>\n<task>\nCritical Point Analysis\n</task>\n<objective>\n switches to using partial derivatives to find critical points.\n</objective>\n</thread>\n</launch_threads>\n<step_resolution>\n<thread_result id='0'>\ncompleting the square is selected as the initial approach.\n</thread_result>\n<thread_result id='1'>\nThe model finds that the expression can be made arbitrarily small due to the negative coefficient in the squared term, indicating no finite minimum exists.\n</thread_result>\n</step_resolution>\n\nSection #1 determined that completing the square was initially viable, but critical point analysis revealed the expression has no finite minimum due to an unbounded negative squared term. The parallel threads converged to show the problem lacks a solution through standard optimization approaches.\n\nTo approach this optimization problem systematically, we first need to analyze the structure of the quadratic expression \\( P \\) from two complementary perspectives. On one hand, we should examine how the variables interact by grouping terms involving \\( x \\), which will reveal potential patterns for completing the square—this is crucial for identifying any inherent symmetries or simplifications. Simultaneously, we must investigate the critical points by computing the partial derivatives with respect to each variable, as these will help us locate potential minima by solving the resulting system of equations. Both approaches are essential: the grouping strategy may uncover a simplified form of \\( P \\), while the partial derivatives will provide exact conditions for optimality. By pursuing these two lines of analysis in tandem, we can efficiently narrow down the solution space.\n\n<launch_threads>\n<thread id='2'>\n<task>\nExpression Grouping\n</task>\n<objective>\n groups terms involving \\( x \\) to prepare for completing the square.\n</objective>\n</thread>\n<thread id='3'>\n<task>\nPartial Derivative Calculation\n</task>\n<objective>\n computes partial derivatives with respect to \\( x \\), \\( y \\), and \\( z \\).\n</objective>\n</thread>\n</launch_threads>\n<step_resolution>\n<thread_result id='2'>\n3x(3x - 2y + 4z)\n</thread_result>\n<thread_result id='3'>\n3x - y +2z =0, 4y -6x -6 =0, 10z +12x +12 =0\n</thread_result>\n</step_resolution>\n\nThread 2 successfully grouped terms involving \\( x \\) into the factored form \\( 3x(3x - 2y + 4z) \\). Thread 3 derived the partial derivative equations \\( 3x - y + 2z = 0 \\), \\( 4y - 6x - 6 = 0 \\), and \\( 10z + 12x + 12 = 0 \\), establishing critical relationships for solving the system. These results provide both a simplified expression and necessary conditions for further analysis.\n\nTo proceed, we need to analyze the expression from two complementary perspectives: restructuring it to reveal hidden patterns and deriving conditions for optimality. First, we should reorganize the quadratic terms involving \\( x \\) to identify a perfect square, which will simplify the expression and potentially reveal its minimum structure. Simultaneously, we must establish the critical points by setting the partial derivatives with respect to each variable to zero, as these equations will define the system that must be solved to locate the extremum. Both approaches will provide essential insights—the first by transforming the expression into a more manageable form, and the second by mathematically confirming where the minimum occurs. This dual analysis ensures we capture both the algebraic simplification and the calculus-based optimization required for the solution.\n\n<launch_threads>\n<thread id='4'>\n<task>\nCompleting the Square for \\( x \\)\n</task>\n<objective>\n attempts to complete the square for the \\( x \\)-terms.\n</objective>\n</thread>\n<thread id='5'>\n<task>\nSystem of Equations Formation\n</task>\n<objective>\n sets the partial derivatives to zero to form a system of equations.\n</objective>\n</thread>\n</launch_threads>\n<step_resolution>\n<thread_result id='4'>\n9[ [x - (y -2z)/3]² - (y² -4yz +4z²) ] + 2y² +5z² -6y +12z +2022\n</thread_result>\n<thread_result id='5'>\n1) 3x - y + 2z = 0, 2) -6x + 4y - 6 = 0, 3) 12x + 10z + 12 = 0\n</thread_result>\n</step_resolution>\n\nThread 4 successfully completed the square for the \\( x \\)-terms, resulting in the expression \\( 9\\left[ \\left(x - \\frac{y - 2z}{3}\\right)^2 - \\frac{y^2 - 4yz + 4z^2}{9} \\right] + 2y^2 + 5z^2 - 6y + 12z + 2022 \\). Thread 5 derived a system of three linear equations: \\( 3x - y + 2z = 0 \\), \\( -6x + 4y - 6 = 0 \\), and \\( 12x + 10z + 12 = 0 \\). These results provide the foundation for solving the optimization problem.\n\nTo proceed, we need to address two key aspects simultaneously: first, we must carefully substitute the completed square form back into the original expression and simplify it to reveal any remaining patterns or simplifications, ensuring all terms are correctly accounted for. Second, we need to solve the system of equations derived from setting the partial derivatives to zero, as this will give us the critical points where the expression might attain its minimum. Both steps are crucial—the simplification will confirm the structure of the expression, while solving the system will pinpoint the exact values of \\(x\\), \\(y\\), and \\(z\\) that potentially minimize \\(P\\). We must ensure accuracy in algebraic manipulation during substitution and methodical solving of the equations to avoid errors in either process.\n\n<thread id='7'>\n<task>\nEquation Solving\n</task>\n<objective>\nsolves the system of equations to find critical points.\n</objective>\n</thread>\n<thread_processing id='7'>\n"""
    
    # Create messages in chat format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    # Manually create the chat template format with custom assistant start
    formatted_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_question}<|im_end|>
<|im_start|>assistant
"""
    
    print("Formatted prompt:")
    print(formatted_prompt)
    print("\n" + "="*50 + "\n")
    
    # Generate response
    with torch.no_grad():
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=1.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (excluding the input)
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
    print("GENERATED RESPONSE:")
    print(response)

if __name__ == "__main__":
    test_local_model_with_special_tokens()