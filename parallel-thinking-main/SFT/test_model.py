import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

# Find the latest checkpoint
base_dir = "/nvme0n1/zzy_model/fresh_mixed/deepseek-parallel-thinking"
checkpoints = []

if os.path.exists(base_dir):
    for item in os.listdir(base_dir):
        if item.startswith("checkpoint-"):
            match = re.search(r'checkpoint-(\d+)', item)
            if match:
                checkpoints.append((int(match.group(1)), item))

if checkpoints:
    latest_checkpoint_num, latest_checkpoint = max(checkpoints)
    model_path = os.path.join(base_dir, latest_checkpoint)
    print(f"Using latest checkpoint: {model_path}")
else:
    print("No checkpoints found!")
    exit(1)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print("Model loaded successfully!")
    
    # Define your prompt and first token
    test_question = """Teacher $D$ has 60 red glass beads and 50 black glass beads. A magical machine, when used once, will turn 4 red glass beads into 1 black glass bead, or turn 5 black glass beads into 2 red glass beads. After Teacher $D$ used the machine 30 times, all the red glass beads were gone. At this point, there are $\qquad$ black glass beads."""

    
    first_token = "<reasoning_process>"  # Your desired first token
    
    # Method 1: Concatenate prompt + first token as string
    prompt_with_first_token = test_question + first_token
    inputs = tokenizer(prompt_with_first_token, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nQuestion:", test_question)
    print("First token provided:", first_token)
    print("Full response:", response[len(test_question):].strip())
    
except Exception as e:
    print(f"Error: {e}")