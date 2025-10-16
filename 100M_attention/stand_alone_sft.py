#!/usr/bin/env python3
"""
Standalone SFT training script with custom positional encoding.

This script demonstrates how to fine-tune a model with custom periodic positional
encoding without modifying LLaMA-Factory source code. It uses minimal dependencies
and provides full control over the training process.

Usage:
    python standalone_sft_custom_pos.py \\
        --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \\
        --dataset_path LLaMA-Factory/data/alpaca_en_demo.json \\
        --output_dir outputs/standalone_sft \\
        --num_train_epochs 3

Features:
- Custom periodic positional encoding (100M+ context)
- LoRA fine-tuning support
- Automatic logging and checkpointing
- Compatible with Hugging Face datasets
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# Import custom positional encoding
from custom_model_patcher import (
    inject_custom_positional_encoding_factory,
    verify_custom_encoding_applied,
    count_custom_encoding_parameters,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Classes
# ============================================================================

@dataclass
class AlpacaDataItem:
    """Data item for Alpaca-style instruction datasets."""
    instruction: str
    input: str = ""
    output: str = ""


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        prompt_template: str = "qwen",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        
        # Load data
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def _format_prompt_qwen(self, item: Dict) -> str:
        """Format prompt using Qwen chat template."""
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        
        if input_text:
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
        
        messages = [{"role": "user", "content": user_message}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for tokenizers without chat template
            prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    def _format_prompt_llama3(self, item: Dict) -> str:
        """Format prompt using LLaMA-3 chat template."""
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        
        if input_text:
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
        
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format prompt based on template
        if self.prompt_template == "qwen":
            prompt = self._format_prompt_qwen(item)
        elif self.prompt_template == "llama3":
            prompt = self._format_prompt_llama3(item)
        else:
            # Simple format
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        output = item.get('output', '')
        full_text = prompt + output
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # Create labels (mask prompt tokens)
        prompt_tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        prompt_length = len(prompt_tokens['input_ids'])
        labels = tokenized['input_ids'].copy()
        labels[:prompt_length] = [-100] * prompt_length  # Mask prompt tokens
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
        }


# ============================================================================
# Model Setup
# ============================================================================

def setup_model_and_tokenizer(
    model_name_or_path: str,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    custom_pos_encoding: bool = True,
    custom_pos_period: int = 128000,
    custom_pos_max_length: Optional[int] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
):
    """Setup model and tokenizer with custom positional encoding and LoRA."""
    
    logger.info("=" * 70)
    logger.info("Loading model and tokenizer")
    logger.info("=" * 70)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "right"  # For training
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )
    
    # Inject custom positional encoding
    if custom_pos_encoding:
        logger.info("=" * 70)
        logger.info("Injecting custom periodic positional encoding")
        logger.info("=" * 70)
        
        try:
            custom_module, replaced_name = inject_custom_positional_encoding_factory(
                model=model,
                max_length=custom_pos_max_length,
                dropout=0.0,
                learned_scaling=True,
                period=custom_pos_period,
            )
            
            if verify_custom_encoding_applied(model):
                custom_params = count_custom_encoding_parameters(model)
                logger.info(f"✓ Custom positional encoding successfully applied!")
                logger.info(f"  Replaced: {replaced_name}")
                logger.info(f"  Period: {custom_pos_period}")
                logger.info(f"  Additional trainable params: {custom_params:,}")
            
        except Exception as e:
            logger.error(f"Failed to inject custom positional encoding: {e}")
            raise
        
        logger.info("=" * 70)
    
    # Prepare model for k-bit training if using quantization
    if load_in_8bit or load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    if use_lora:
        logger.info("=" * 70)
        logger.info("Applying LoRA")
        logger.info("=" * 70)
        
        # Get target modules (adjust based on your model)
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_type = getattr(config, 'model_type', '')
        
        if 'qwen' in model_type.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif 'llama' in model_type.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            # Generic fallback
            target_modules = ["q_proj", "v_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        logger.info("=" * 70)
    
    return model, tokenizer


# ============================================================================
# Training
# ============================================================================

def train(
    model,
    tokenizer,
    train_dataset,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    logging_steps: int = 10,
    save_steps: int = 100,
    save_total_limit: int = 3,
    fp16: bool = False,
    bf16: bool = True,
):
    """Run training with Hugging Face Trainer."""
    
    logger.info("=" * 70)
    logger.info("Starting training")
    logger.info("=" * 70)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        fp16=fp16,
        bf16=bf16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        report_to="tensorboard",
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info(f"Training for {num_train_epochs} epochs")
    logger.info(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("=" * 70)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name or path"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="LLaMA-Factory/data/alpaca_en_demo.json",
        help="Path to training dataset (JSON format)"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="qwen",
        choices=["qwen", "llama3", "simple"],
        help="Prompt template to use"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    # Custom positional encoding arguments
    parser.add_argument(
        "--use_custom_pos_encoding",
        action="store_true",
        default=True,
        help="Use custom periodic positional encoding"
    )
    parser.add_argument(
        "--custom_pos_period",
        type=int,
        default=128000,
        help="Period for periodic positional encoding"
    )
    parser.add_argument(
        "--custom_pos_max_length",
        type=int,
        default=None,
        help="Max length for custom positional encoding"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Quantization arguments
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit")
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/standalone_sft",
        help="Output directory for checkpoints"
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max number of checkpoints to keep")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16 training")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        custom_pos_encoding=args.use_custom_pos_encoding,
        custom_pos_period=args.custom_pos_period,
        custom_pos_max_length=args.custom_pos_max_length,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Load dataset
    train_dataset = InstructionDataset(
        data_path=args.dataset_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        prompt_template=args.prompt_template,
    )
    
    # Train
    train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    
    logger.info(f"\n✓ Training complete! Model saved to: {args.output_dir}")
    logger.info(f"\nTo use the model:")
    logger.info(f"  python inference_partial.py \\")
    logger.info(f"    --model-name-or-path {args.output_dir} \\")
    logger.info(f"    --prompt 'Your prompt here' \\")
    logger.info(f"    --use-chat-template")


if __name__ == "__main__":
    main()
