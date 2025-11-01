#!/usr/bin/env python3
"""
Training script using HuggingFace Trainer for proper SFT.
This fixes the loss masking issues from the manual implementation.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Auto-detect device if not specified
    if device == "auto":
        # Check for MPS (macOS only) - safe check for Windows compatibility
        try:
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        except (AttributeError, RuntimeError):
            mps_available = False
        
        if mps_available:
            device = "mps"
            print("Auto-detected: Using MPS (Metal Performance Shaders) backend")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Auto-detected: Using CUDA backend")
        else:
            device = "cpu"
            print("Auto-detected: Using CPU backend")
    else:
        print(f"Using specified device: {device}")
    
    # Load model with appropriate dtype and device handling
    if device == "mps":
        # For MPS, use float32 (bfloat16 not supported)
        torch_dtype = torch.float32
        # Load on CPU first, then move to MPS
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model = model.to("mps")
    elif device == "cuda":
        # For CUDA, use bfloat16 for better performance
        torch_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
    else:
        # For CPU, use float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    
    return model, tokenizer


def prepare_dataset(tokenizer, max_length=512, max_samples=None):
    """Prepare GSM8K dataset for training."""
    
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited dataset to {len(dataset)} samples")
    
    def process_function(sample):
        """Format question and answer for training."""
        # Tokenize question and full text
        question_text = sample['question']
        full_text = sample['question'] + sample['answer']
        
        question_tokens = tokenizer.encode(question_text, add_special_tokens=False, max_length=max_length, truncation=True)
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False, max_length=max_length, truncation=True)
        
        # Add EOS token to full_tokens
        if len(full_tokens) < max_length:
            full_tokens.append(tokenizer.eos_token_id)
        elif full_tokens[-1] != tokenizer.eos_token_id:
            # Replace last token with EOS if we truncated
            full_tokens[-1] = tokenizer.eos_token_id
        
        # Create labels: -100 for question tokens, actual token IDs for answer
        # Make sure labels length matches full_tokens length
        labels = [-100] * len(question_tokens) + full_tokens[len(question_tokens):]
        
        # Ensure labels length matches input_ids
        if len(labels) != len(full_tokens):
            # This can happen if question was truncated
            labels = [-100] * min(len(question_tokens), len(full_tokens)) + full_tokens[len(question_tokens):]
            labels = labels[:len(full_tokens)]
        
        return {
            "input_ids": full_tokens,
            "labels": labels,
        }
    
    print(f"Processing {len(dataset)} samples...")
    dataset = dataset.map(
        process_function,
        remove_columns=["question", "answer"],
        desc="Processing dataset"
    )
    
    return dataset


def train(
    model_path: str,
    output_dir: str,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    max_length: int = 128,
    max_samples: int = 500,
    max_time: int = None,
    max_steps: int = None,
    save_steps: int = 100,
    use_wandb: bool = True,
    device: str = "auto",
):
    """Train model using HuggingFace Trainer."""
    
    # Determine actual device to use
    actual_device = device
    if device == "auto":
        # Check for MPS (macOS only) - safe check for Windows compatibility
        try:
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        except (AttributeError, RuntimeError):
            mps_available = False
        
        if mps_available:
            actual_device = "mps"
        elif torch.cuda.is_available():
            actual_device = "cuda"
        else:
            actual_device = "cpu"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, device=actual_device)
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer, max_length=max_length, max_samples=max_samples)
    
    # Data collator - handles padding properly
    # We'll let it handle the shifting of labels for next-token prediction
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Training arguments
    training_args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "logging_steps": 10,
        "save_steps": save_steps,
        "save_total_limit": 2,
        "bf16": actual_device == "cuda",  # Use bf16 on CUDA
        "fp16": False,  # Don't use fp16 (use bf16 on CUDA, float32 on MPS/CPU)
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "push_to_hub": False,
        "report_to": "wandb" if use_wandb else "none",
        "run_name": f"gsm8k-sft-{Path(output_dir).name}",
    }
    
    # Only add max_steps if it's not None
    if max_steps is not None:
        training_args_dict["max_steps"] = max_steps
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start timer if max_time is set
    if max_time:
        import time
        start_time = time.time()
        
        def should_stop_callback(state):
            if time.time() - start_time > max_time:
                print(f"\nMax time ({max_time}s) reached. Stopping training...")
                return True
            return False
    else:
        should_stop_callback = None
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Training completed!")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train model using HuggingFace Trainer")
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/gsm8k-hf-trainer",
        help="Output directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of training samples",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        help="Maximum training time in seconds",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train
    trainer = train(
        model_path=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_samples=args.max_samples,
        max_time=args.max_time,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        use_wandb=not args.no_wandb,
        device=args.device,
    )


if __name__ == "__main__":
    main()

