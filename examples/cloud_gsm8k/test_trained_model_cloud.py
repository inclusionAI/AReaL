#!/usr/bin/env python3
"""
Test script for trained GRPO model on GSM8K dataset (Cloud version).
This script loads a trained model checkpoint and evaluates it on the GSM8K test set.
"""

import os
import sys
import time
import warnings
from datetime import datetime, timedelta

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", message=".*Gloo.*Rank.*connected.*")

# Suppress Gloo messages
os.environ["GLOG_minloglevel"] = "2"

import torch.distributed as dist

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.printing import tabulate_stats
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from areal.reward.math_parser import process_results

    return int(process_results(answer, completions)[0])


def main(args):
    # Extract custom training parameters from YAML before config validation
    # These parameters are not part of GRPOConfig, so we read them directly from YAML
    import argparse
    from omegaconf import OmegaConf
    from pathlib import Path
    
    # Parse args to find config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the main configuration file", required=True)
    # Skip script path if present
    if args and args[0].endswith(".py"):
        args = args[1:]
    parsed_args, _ = parser.parse_known_args(args)
    
    # Load YAML directly to extract custom fields
    config_file = Path(parsed_args.config)
    if not config_file.is_absolute():
        # Make it absolute relative to current working directory
        config_file = Path.cwd() / config_file
    raw_yaml = OmegaConf.load(config_file)
    
    # Remove custom fields from config to avoid validation errors
    # Use OmegaConf/Hydra delete syntax (~key) to remove keys
    custom_keys = ["max_train_samples", "training_mode", "circuit_breaker_enabled", "circuit_breaker_threshold"]
    override_args = []
    for key in custom_keys:
        if key in raw_yaml:
            # Use ~ prefix to delete the key in OmegaConf/Hydra
            override_args.append(f"~{key}")
    
    # Add overrides to args to remove custom keys
    args_with_overrides = args + override_args
    
    # Now load config normally - the overrides will remove the custom keys
    config, _ = load_expr_config(args_with_overrides, GRPOConfig)
    config: GRPOConfig

    # Set up logging to network volume (persists after pod stops)
    log_dir = os.path.join("/workspace", "outputs", "grpo", "test_logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_model_{ts}.log")
    
    def _log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
    
    _log(f"\n{'='*80}")
    _log(f"Testing model from config: {config.experiment_name}/{config.trial_name}")
    _log(f"Log file: {log_path}")
    _log(f"{'='*80}\n")

    # Initialize distributed process group if not already initialized
    # Use a unique port to avoid conflicts with training processes
    if not dist.is_initialized():
        import socket
        import random
        
        # Find a free port in a high range to avoid conflicts
        def find_free_port():
            # Use a high port range (30000-60000) to avoid conflicts
            for _ in range(10):  # Try up to 10 random ports
                port = random.randint(30000, 60000)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('', port))
                        return port
                    except OSError:
                        continue
            # Fallback: let OS choose
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                return s.getsockname()[1]
        
        # Try to initialize with a unique port
        max_retries = 5
        initialized = False
        for attempt in range(max_retries):
            try:
                free_port = find_free_port()
                master_addr = os.environ.get("MASTER_ADDR", "localhost")
                init_method = f"tcp://{master_addr}:{free_port}"
                
                rank = int(os.environ.get("RANK", "0"))
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
                
                # Add a small delay to ensure port is fully released
                if attempt > 0:
                    time.sleep(1)
                
                dist.init_process_group(
                    backend="gloo",
                    init_method=init_method,
                    rank=rank,
                    world_size=world_size,
                    timeout=timedelta(seconds=30)
                )
                _log(f"Initialized process group on port {free_port}")
                initialized = True
                break
            except (dist.DistNetworkError, OSError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s, 8s
                    _log(f"Port conflict (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    _log(f"Failed to initialize process group after {max_retries} attempts: {e}")
                    _log("Continuing without distributed stats aggregation...")
                    # Continue without distributed - stats won't be aggregated but test can proceed
                    break
    else:
        _log("Process group already initialized")
        initialized = True
    
    # Create a group for stats all-reduce (only if dist is initialized)
    if dist.is_initialized():
        try:
            group = dist.new_group()
        except Exception as e:
            _log(f"Warning: Could not create stats group: {e}, continuing without group")
            group = None
    else:
        group = None

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"eval{rank}")

    # Create dataset and dataloaders
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    
    # Optional: Limit test samples via environment variable
    max_test_samples = int(os.getenv("MAX_TEST_SAMPLES", "0"))
    if max_test_samples > 0 and len(valid_dataset) > max_test_samples:
        _log(f"[EVAL] Limiting test set from {len(valid_dataset)} to {max_test_samples} samples")
        valid_dataset = valid_dataset.select(range(max_test_samples))
    
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=rank,
        world_size=world_size,
        dataset_config=config.valid_dataset,
    )

    # Initialize inference engine
    config.rollout.max_head_offpolicyness = int(1e12)
    eval_rollout = RemoteSGLangEngine(config.rollout)
    eval_rollout.initialize()

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    _log(f"Evaluating on {len(valid_dataset)} test samples...")
    _log(f"Using model from: {config.rollout.experiment_name}/{config.rollout.trial_name}")
    _log(f"Max new tokens: {config.gconfig.max_new_tokens}")
    _log(f"Generated responses will be saved to: {workflow.dump_dir}")
    _log(f"Note: Individual responses are saved to dump directory. Check files there for full question/response pairs.")

    # Run evaluation.
    cnt = 0
    for data in valid_dataloader:
        for item in data:
            eval_rollout.submit(item, workflow)
            cnt += 1
    
    _log(f"Submitted {cnt} evaluation tasks. Waiting for completion...")
    eval_rollout.wait(cnt, timeout=None)

    eval_rollout_stats = stats_tracker.export_all(reduce_group=group)
    
    results_text = "\n" + "="*80 + "\n"
    results_text += "EVALUATION RESULTS\n"
    results_text += "="*80 + "\n"
    results_text += tabulate_stats(eval_rollout_stats) + "\n"
    _log(results_text)
    
    # Extract accuracy from reward stats
    accuracy = None
    reward_key = None
    
    # Check different possible reward keys
    for key in ["eval-rollout/task_reward", "eval-rollout/reward", "eval-rollout/final_reward"]:
        if key in eval_rollout_stats:
            reward_stats = eval_rollout_stats[key]
            if isinstance(reward_stats, dict):
                if "avg" in reward_stats:
                    accuracy = reward_stats["avg"] * 100
                    reward_key = key
                    break
            elif isinstance(reward_stats, (int, float)):
                accuracy = reward_stats * 100 if reward_stats <= 1.0 else reward_stats
                reward_key = key
                break
    
    if accuracy is not None:
        accuracy_text = f"\n{'='*80}\n"
        accuracy_text += f"ACCURACY: {accuracy:.2f}%\n"
        accuracy_text += f"{'='*80}\n"
        _log(accuracy_text)
    else:
        warning_text = f"\n{'='*80}\n"
        warning_text += "WARNING: Could not extract accuracy from stats.\n"
        warning_text += f"Available keys: {list(eval_rollout_stats.keys())}\n"
        warning_text += f"{'='*80}\n"
        _log(warning_text)
    
    _log(f"\n{'='*80}")
    _log(f"Log saved to: {log_path}")
    _log(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    if accuracy is not None:
        print(f"ACCURACY: {accuracy:.2f}%")
    print(f"Log saved to: {log_path}")
    print(f"{'='*80}\n")
    
    eval_rollout.destroy()
    
    # Only destroy process group if we initialized it
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            _log(f"Warning: Could not destroy process group: {e}")
    
    # Exit cleanly
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])

