import os
import torch
import wandb
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import GRPOConfig
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
from areal.dataset.__init__ import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.reward.__init__ import get_custom_reward_fn
from areal.utils.device import log_gpu_stats
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from .areal.workflow.vision_rlvr import VisionRLVRWorkflow
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
from realhf.base import seeding, stats_tracker


def create_test_config():
    """Create test configuration with hardcoded parameters"""
    config = GRPOConfig()
    
    # Basic settings
    config.seed = 42
    config.total_train_epochs = 1
    config.async_training = False
    
    # Model and tokenizer paths (modify these to actual paths)
    config.tokenizer_path = "/path/to/your/tokenizer"
    
    # Training dataset configuration
    config.train_dataset.path = "/path/to/your/train/dataset"
    config.train_dataset.type = "your_dataset_type"
    config.train_dataset.batch_size = 2  # Small batch for testing
    config.train_dataset.shuffle = False  # Disable shuffle for reproducibility
    config.train_dataset.num_workers = 0  # Single process
    config.train_dataset.drop_last = False
    config.train_dataset.reward_fn = "your_reward_function"
    
    # Actor configuration
    config.actor.recompute_logprob = True
    config.actor.use_decoupled_loss = False
    config.actor.kl_ctl = 0.1
    
    # Reference model configuration (disabled for testing)
    config.ref = None
    
    # Generation configuration
    config.gconfig.stop_token_ids = []
    
    
    # Rollout configuration
    config.rollout = {}  # Configure according to RemoteSGLangEngine requirements
    
    return config


def main():
    """Main function for single-step inference testing"""
    # Use hardcoded configuration instead of command line arguments
    config = create_test_config()
    
    
    # Single process settings
    rank = 0
    world_size = 1
    
    print("Loading processor and tokenizer...")
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
    
    # Set random seed
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    print(f"Random seed set to {config.seed}")
    
    # Create training dataset
    print("Loading training dataset...")
    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        split="train",
        type=config.train_dataset.type,
        processor=processor,
    )
    
    # Create dataloader
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    
    print(f"Dataset loaded with {len(train_dataset)} samples")
    print(f"Dataloader created with {len(train_dataloader)} batches")
    
    # Create finetune specification
    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    
    # Initialize inference engine
    print("Initializing rollout engine...")
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)
    
    # Initialize training engine
    print("Initializing actor...")
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    
    # No reference model for testing
    ref = None
    
    # Weight update metadata (single process version)
    print("Setting up weight update metadata...")
    weight_update_meta = WeightUpdateMeta.from_disk(config.saver)
    
    # Configure stop tokens
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    print(f"Stop token IDs: {config.gconfig.stop_token_ids}")
    
    # Get reward function
    print("Loading reward function...")
    reward_fn = get_custom_reward_fn(
        path=config.train_dataset.reward_fn,
    )
    
    # Create workflow
    print("Creating workflow...")
    workflow = VisionRLVRWorkflow(
        reward_fn=reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        processor=processor,
        enable_thinking=False,
        dump_dir="./test_generated",
    )
    
    print("Starting single-step inference test...")
    
    # Perform only one training step
    global_step = 0
    epoch = 0
    step = 0
    
    # Get one batch of data
    data_generator = iter(train_dataloader)
    try:
        data = next(data_generator)
        print(f"Successfully loaded data batch with {len(data)} samples")
    except StopIteration:
        print("ERROR: Dataloader is empty, cannot perform test")
        return
    
    # Execute rollout
    print("Executing rollout...")
    batch = rollout.rollout_batch(data, workflow=workflow)
    print("Rollout completed successfully")
    
    # Move to device
    batch = batch.to(actor.device)
    print(f"Batch moved to device: {actor.device}")
    
    # Synchronize CUDA
    torch.cuda.synchronize()
    
    # Recompute log probabilities if needed
    if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
        print("Recomputing log probabilities...")
        logp = actor.compute_logp(batch)
        batch["prox_logp"] = logp
        print("Log probabilities recomputed")
    
    # Compute advantages
    print("Computing advantages...")
    actor.compute_advantages(batch)
    print("Advantages computed")
    
    # Perform PPO update
    stats = actor.ppo_update(batch)
        
    # Log to wandb
    if stats and len(stats) > 0:
        final_reward = stats[0].get("grpo_actor/final_reward/avg", 0.0)
        task_reward = stats[0].get("grpo_actor/task_reward/avg", 0.0)
        print(f"Final reward: {final_reward}")
        print(f"Task reward: {task_reward}")
        
    actor.step_lr_scheduler()
    print("PPO update completed")
    
    
    # Cleanup
    print("Cleaning up...")
    rollout.destroy()
    actor.destroy()
    
    print("Single-step inference test completed successfully!")


if __name__ == "__main__":
    main()