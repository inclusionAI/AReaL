import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Model path - this is a merged model, no adapter needed
model_path = "/nvme0n1/zzy_model/saves_20250622_135434/deepseek-parallel-thinking"

print(f"Loading merged model from: {model_path}")

# Check if this is a DeepSpeed checkpoint directory
if os.path.exists(os.path.join(model_path, "pytorch_model.bin.index.json")):
    print("Detected sharded model checkpoint")
elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
    print("Detected single model checkpoint")
elif os.path.exists(os.path.join(model_path, "model.safetensors")):
    print("Detected safetensors format")

# Load tokenizer first
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

print("Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True  # Help with memory efficiency
    )
    print("Model loaded successfully!")
    
    # Unwrap from DeepSpeed/DDP wrappers if present
    if hasattr(model, 'module'):
        print("Unwrapping model from .module wrapper...")
        model = model.module
    
    # Check for DeepSpeed engine wrapper
    if hasattr(model, '_modules') and 'module' in model._modules:
        print("Unwrapping from DeepSpeed engine...")
        model = model._modules['module']
    
    # Check model type
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative loading method...")
    
    # Try loading without local_files_only in case of format issues
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("Model loaded with alternative method!")
        
        # Apply unwrapping again
        if hasattr(model, 'module'):
            print("Unwrapping model from .module wrapper...")
            model = model.module
            
    except Exception as e2:
        print(f"Alternative loading also failed: {e2}")
        exit(1)

print("Merged model loaded successfully!")

# Test with a sample question
test_question = "A company was contracted to construct three buildings, with the second building being two times as tall as the first building. The third building had to be three times as tall as the combined height of the first and second buildings. If the first building was 600 feet, calculate the total height of the three buildings together."

# Format prompt for DeepSeek/Qwen format
prompt = f"User: {test_question}\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")

# Move inputs to same device as model
if hasattr(model, 'device'):
    inputs = inputs.to(model.device)
else:
    # If model is distributed across devices, try to get the first device
    first_param_device = next(model.parameters()).device
    inputs = inputs.to(first_param_device)

print(f"Inputs device: {inputs['input_ids'].device}")
print(f"Model device: {next(model.parameters()).device}")

# Generate response
print("Generating response...")
try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*80)
    print("Test Question:", test_question)
    print("\nModel Response:")
    print(response.split("Assistant:")[-1].strip())
    print("="*80)
    
except Exception as e:
    print(f"Error during generation: {e}")
    print("This might be the CUDA device-side assert error.")
    print("Let's try with CPU fallback...")
    
    # Try CPU fallback
    try:
        model = model.to('cpu')
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n" + "="*80)
        print("Test Question (CPU):", test_question)
        print("\nModel Response:")
        print(response.split("Assistant:")[-1].strip())
        print("="*80)
        
    except Exception as cpu_error:
        print(f"CPU fallback also failed: {cpu_error}")