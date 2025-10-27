from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_local_model_with_special_tokens():
    # Use the fixed model path
    model_path = "/nvme0n1/zzy_model/fresh_mixed_7131740/deepseek-parallel-thinking/"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Check if special tokens are loaded
    print("Additional special tokens:", tokenizer.additional_special_tokens)
    print("Special token IDs:", {token: tokenizer.convert_tokens_to_ids(token) 
                               for token in tokenizer.additional_special_tokens})
    
    # Create direct prompt without chat template
    system_instruction = "You are a helpful assistant that uses parallel thinking to solve problems."
    user_question =r"""Find all positive integer solutions to the equation $a^{2}=b^{3}+1$.\n<think>\n""",


    # Format prompt to match training data format
    prompt = user_question 
    
    print("="*50)
    print("PROMPT:")
    print(repr(prompt))
    print("="*50)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Input token IDs:", inputs['input_ids'][0].tolist()[:20], "...")  # Show first 20 tokens
    
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and print
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_only = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=False)
    
    print("FULL RESPONSE:")
    print(full_response)
    print("\n" + "="*50)
    print("GENERATED PART ONLY:")
    print(repr(generated_only))
    print("="*50)
    
    # Check which special tokens appeared in the generation
    print("SPECIAL TOKEN ANALYSIS:")
    special_tokens_in_output = []
    if tokenizer.additional_special_tokens:
        for token in tokenizer.additional_special_tokens:
            if token in generated_only:
                special_tokens_in_output.append(token)
    
    print(f"Special tokens found in output: {special_tokens_in_output}")
    
    # Also check for reasoning structure
    reasoning_indicators = [
        "<reasoning_process>",
        "<think",
        "<parallel_processing>",
        "<launch_threads>",
        "<thread",
        "</reasoning_process>"
    ]
    
    found_indicators = [indicator for indicator in reasoning_indicators if indicator in generated_only]
    print(f"Reasoning structure indicators found: {found_indicators}")

def test_tokenization():
    """Test how special tokens are tokenized"""
    model_path = "/nvme0n1/zzy_model/fresh_mixed_780924/deepseek-parallel-thinking"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("\n" + "="*50)
    print("TOKENIZATION TEST")
    print("="*50)
    
    # Test key tokens
    test_tokens = [
        "<reasoning_process>",
        "<think: type = ''>",
        "</think>",
        "<parallel_processing>",
        "<launch_threads>",
        "<thread id='0'>",
        "</reasoning_process>"
    ]
    
    for token in test_tokens:
        tokenized = tokenizer.tokenize(token)
        token_ids = tokenizer.convert_tokens_to_ids(tokenized)
        print(f"'{token}':")
        print(f"  Tokens: {tokenized}")
        print(f"  IDs: {token_ids}")
        print()

if __name__ == "__main__":
    test_local_model_with_special_tokens()
    test_tokenization()