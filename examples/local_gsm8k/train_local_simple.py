#!/usr/bin/env python3
"""
Simplified local training script for GSM8K fine-tuning on Mac M2.
This script bypasses the complex distributed infrastructure and runs locally.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Import AReaL utilities for dataset and rewards
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from areal.dataset.gsm8k import get_gsm8k_sft_dataset
    from areal.reward.math_parser import process_results
    import areal.utils.logging as logging
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback: define the functions locally if areal is not installed
    from datasets import load_dataset
    import logging as std_logging
    logger = std_logging.getLogger(__name__)
    
    def get_gsm8k_sft_dataset(path, split, tokenizer, max_length=None):
        """Load GSM8K dataset for SFT training."""
        dataset = load_dataset(path=path, name="main", split=split)
        
        def process(sample):
            seq_token = tokenizer.encode(
                sample["question"] + sample["answer"] + tokenizer.eos_token
            )
            prompt_token = tokenizer.encode(sample["question"])
            loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
            return {"input_ids": seq_token, "loss_mask": loss_mask}
        
        dataset = dataset.map(process).remove_columns(["question", "answer"])
        
        if max_length is not None:
            dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
        
        return dataset
    
    def process_results(completions, answer):
        """Simple math parser - check if answer matches."""
        # This is a simplified version
        import re
        for completion in completions:
            # Look for \boxed{} pattern
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', completion)
            if boxed_match:
                predicted_answer = boxed_match.group(1).strip()
                expected_answer = answer.strip()
                # Try to extract numeric answer
                try:
                    # Extract numbers and compare
                    predicted_num = re.findall(r'\d+\.?\d*', predicted_answer)
                    expected_num = re.findall(r'\d+\.?\d*', expected_answer)
                    if predicted_num and expected_num:
                        return [abs(float(predicted_num[0]) - float(expected_num[0])) < 0.01]
                except:
                    pass
        return [False]


class SimpleTrainer:
    """Simplified trainer for local training without distributed setup."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        lr: float = 5e-5,
        batch_size: int = 8,
        max_epochs: int = 3,
        max_length: int = 512,
        gradient_accumulation_steps: int = 4,
        save_steps: int = 50,
        eval_steps: int = 100,
        use_wandb: bool = True,
        project_name: str = "areal-gsm8k",
        device: str = "auto",
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        
        # Setup device
        if device == "auto":
            # Check for MPS (macOS only) - safe check for Windows compatibility
            try:
                mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            except (AttributeError, RuntimeError):
                mps_available = False
            
            if mps_available:
                self.device = torch.device("mps")
                logger.info("Using MPS (Metal Performance Shaders) backend")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA backend")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU backend")
        else:
            self.device = torch.device(device)
        
        if logger:
            logger.info(f"Device: {self.device}")
        else:
            print(f"Device: {self.device}")
        
        # Load model and tokenizer
        if logger:
            logger.info(f"Loading model from {model_path}")
        else:
            print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use float16 for Mac MPS compatibility, bfloat16 for CUDA
        # Note: bfloat16 is not supported on MPS, must use float16
        torch_dtype = torch.float16 if self.device.type == "mps" else torch.bfloat16
        
        # Always load to CPU first to avoid MPS memory issues, then move to device
        try:
            if self.device.type == "mps":
                # For MPS, load to CPU first, then move
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                )
                self.model = self.model.to(self.device)
            else:
                # For CUDA, use device_map for better memory management
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                )
        except Exception as e:
            if logger:
                logger.error(f"Failed to load model: {e}")
            else:
                print(f"Failed to load model: {e}")
            # Fallback: always load to CPU then move
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if logger:
                logger.info("Enabled gradient checkpointing")
            else:
                print("Enabled gradient checkpointing")
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        
        # Initialize W&B
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=project_name,
                config={
                    "model": model_path,
                    "lr": lr,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "max_length": max_length,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "device": str(self.device),
                },
            )
    
    def train(self, max_steps: int = None, max_time_seconds: int = None):
        """Train the model on GSM8K dataset."""
        
        # Load dataset
        if logger:
            logger.info("Loading GSM8K dataset...")
        else:
            print("Loading GSM8K dataset...")
        train_dataset = get_gsm8k_sft_dataset(
            path="openai/gsm8k",
            split="train",
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        
        # Limit dataset for quick test run (first 500 samples)
        train_dataset = train_dataset.select(range(min(500, len(train_dataset))))
        if logger:
            logger.info(f"Training on {len(train_dataset)} samples")
        else:
            print(f"Training on {len(train_dataset)} samples")
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * self.max_epochs // self.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        
        # Training loop
        global_step = 0
        start_time = time.time()
        
        _log = logger.info if logger else print
        _log(f"Starting training for {self.max_epochs} epochs")
        _log(f"Total steps: {total_steps}")
        _log(f"Steps per epoch: {len(train_loader)}")
        
        for epoch in range(self.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Check time limit
                if max_time_seconds and (time.time() - start_time) > max_time_seconds:
                    _log(f"Reached time limit of {max_time_seconds}s. Stopping training.")
                    self.save_checkpoint(epoch, step, "time_limit")
                    return
                
                # Check step limit
                if max_steps and global_step >= max_steps:
                    _log(f"Reached step limit of {max_steps}. Stopping training.")
                    self.save_checkpoint(epoch, step, "step_limit")
                    return
                
                # Forward pass
                # Get logits
                outputs = self.model(**{k: v for k, v in batch.items() if k != "loss_mask"})
                logits = outputs.logits
                
                # Get labels and loss mask
                labels = batch["labels"]
                loss_mask = batch["loss_mask"]
                
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_loss_mask = loss_mask[..., 1:].contiguous()  # Shift mask too
                
                # Compute loss manually with masking
                # Reshape to [batch*seq, vocab]
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                shift_loss_mask = shift_loss_mask.view(-1)
                
                # Get log probs
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                
                # Get negative log likelihood for target tokens
                per_token_loss = -log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                
                # Apply loss mask
                masked_loss = per_token_loss * shift_loss_mask
                
                # Average only over non-masked tokens
                loss = masked_loss.sum() / shift_loss_mask.sum().clamp(min=1e-8)
                
                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    _log(f"Warning: NaN/inf loss at step {global_step}. Skipping update.")
                    self.optimizer.zero_grad()
                    continue
                
                # Clear device cache periodically to prevent memory buildup
                if self.device.type == "mps" and global_step % 10 == 0:
                    try:
                        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    except (AttributeError, RuntimeError):
                        pass  # MPS not available on this platform
                elif self.device.type == "cuda" and global_step % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Scale loss by gradient accumulation steps
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients to prevent NaN
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Check for NaN gradients
                    if torch.isnan(torch.tensor(grad_norm)):
                        _log(f"Warning: NaN gradients at step {global_step}. Skipping update.")
                        self.optimizer.zero_grad()
                        global_step += 1
                        continue
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Logging
                    if global_step % 10 == 0:
                        avg_loss = loss.item() * self.gradient_accumulation_steps
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "grad_norm": f"{grad_norm:.2f}"})
                        
                        if WANDB_AVAILABLE:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/grad_norm": grad_norm,
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                                "global_step": global_step,
                                "epoch": epoch + 1,
                                "steps_per_epoch": len(train_loader),
                            })
                    
                    # Save checkpoint
                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(epoch, step, f"step_{global_step}")
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            _log(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        # Final save
        self.save_checkpoint(epoch, len(train_loader), "final")
        _log("Training completed!")
        
        if WANDB_AVAILABLE:
            wandb.finish()
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        loss_mask = [torch.tensor(item["loss_mask"]) for item in batch]
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []  # Standard attention mask (1 for real tokens, 0 for padding)
        padded_loss_mask = []  # Separate loss mask (1 for tokens to compute loss on)
        
        for ids, mask in zip(input_ids, loss_mask):
            pad_len = max_len - len(ids)
            padded_ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
            # Create standard attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.cat([
                torch.ones(len(ids), dtype=torch.long), 
                torch.zeros(pad_len, dtype=torch.long)
            ])
            # Loss mask is already created in dataset (0 for prompt, 1 for answer)
            loss_mask_padded = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(attention_mask)
            padded_loss_mask.append(loss_mask_padded)
        
        return {
            "input_ids": torch.stack(padded_input_ids).to(self.device),
            "attention_mask": torch.stack(padded_attention_mask).to(self.device),
            "labels": torch.stack(padded_input_ids).to(self.device),  # Same as input_ids for LM
            "loss_mask": torch.stack(padded_loss_mask).to(self.device).float(),  # Float for multiplication
        }
    
    def save_checkpoint(self, epoch, step, suffix):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{suffix}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        _log = logger.info if logger else print
        _log(f"Saved checkpoint to {checkpoint_dir}")
        
        # Also save to main directory
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)


def main():
    parser = argparse.ArgumentParser(description="Simple local training for GSM8K")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/gsm8k-local",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--max-epochs", type=int, default=3, help="Maximum epochs"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Maximum training steps"
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=1800,
        help="Maximum training time in seconds (30 min default)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument(
        "--eval-steps", type=int, default=100, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=True, help="Use Weights & Biases"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="areal-gsm8k-mac",
        help="W&B project name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    trainer = SimpleTrainer(
        model_path=args.model,
        output_dir=args.output_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        use_wandb=args.wandb,
        project_name=args.wandb_project,
        device=args.device,
    )
    
    trainer.train(max_steps=args.max_steps, max_time_seconds=args.max_time)


if __name__ == "__main__":
    main()

