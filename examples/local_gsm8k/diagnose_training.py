#!/usr/bin/env python3
"""Diagnose training issues."""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")

# Check first sample
sample = dataset[0]
print("Sample question:")
print(sample["question"])
print("\nSample answer:")
print(sample["answer"])

# Encode
full_text = sample["question"] + sample["answer"] + tokenizer.eos_token
question_text = sample["question"]

full_tokens = tokenizer.encode(full_text)
question_tokens = tokenizer.encode(question_text)

print(f"\nQuestion tokens: {len(question_tokens)}")
print(f"Full tokens: {len(full_tokens)}")
print(f"Answer tokens: {len(full_tokens) - len(question_tokens)}")

# Build loss mask
loss_mask = [0] * len(question_tokens) + [1] * (len(full_tokens) - len(question_tokens))

print(f"Loss mask: {loss_mask[:50]}...")
print(f"Loss mask sum: {sum(loss_mask)}")

# Load trained model and check if it even loads
print("\nTrying to load trained model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "./outputs/gsm8k-fixed",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    print("Model loaded successfully!")
    
    # Test a simple generation
    inputs = tokenizer("What is 2+2?", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print(f"Generated: {tokenizer.decode(outputs[0])}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()

