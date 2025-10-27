from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_local_model_with_special_tokens():
    # Use the fixed model path
    model_path = "/nvme0n1/zzy_model/fresh_7172108/deepseek-parallel-thinking/"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Define system prompt and user message
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
    
    user_question = r"""Problem: Alice and Bob play the following game. A stack of $n$ tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either $1$ token or $4$ tokens from the stack. Whoever removes the last token wins. Find the number of positive integers $n$ less than or equal to $2024$ for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play."""
    
    # Create messages in chat format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    # Manually create the chat template format with custom assistant start
    formatted_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_question}<think>\n<|im_end|>
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
            temperature=0.2,
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