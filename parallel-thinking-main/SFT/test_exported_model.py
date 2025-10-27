import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path

# Use absolute path instead of relative path
model_path = "/nvme0n1/zzy_model/saves_20250623_095112/deepseek-parallel-thinking"

# Check if the model exists
if not os.path.exists(model_path):
    print(f"Model not found at: {model_path}")
    
    # Try alternative paths
    alternative_paths = [
        "exported_model/deepseek-parallel-thinking-merged",
        "./exported_model/deepseek-parallel-thinking-merged",
        "/home/zhangzy/parallel-thinking/exported_model/deepseek-parallel-thinking-merged",
        "/home/zhangzy/parallel-thinking/SFT/exported_model/deepseek-parallel-thinking-merged"
    ]
    
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            model_path = os.path.abspath(alt_path)
            print(f"Found model at: {model_path}")
            break
    else:
        print("Model not found in any expected location. Please check the export completed successfully.")
        exit(1)

print(f"Loading model from: {model_path}")

try:
    # Load tokenizer and model with absolute path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True,
        local_files_only=True  # Force local loading
    )
    
    print("Model loaded successfully!")
    
    # Test the model
    test_question = "What is parallel thinking?"
    prompt = f"User: {test_question}\n\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            temperature=0.1, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print("Question:", test_question)
    print("Response:", response.split("Assistant:")[-1].strip())
    print("="*50)
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nDebugging information:")
    print(f"Model path: {model_path}")
    print(f"Path exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        print("Files in model directory:")
        for file in os.listdir(model_path):
            print(f"  - {file}")