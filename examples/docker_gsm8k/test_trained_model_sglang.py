#!/usr/bin/env python3
"""
Test script for trained GRPO model on GSM8K dataset.

This script loads a trained model checkpoint and evaluates it on the GSM8K test set.
"""

import os
import sys
import warnings

# Ensure AReaL is in the Python path
# This is needed when running the script directly (not as a module)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also set PYTHONPATH environment variable for subprocesses
os.environ.setdefault("PYTHONPATH", project_root)

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
    # Parse command-line arguments for model path and max samples
    import argparse
    from pathlib import Path
    from omegaconf import OmegaConf
    
    # Support multiple ways to pass these parameters:
    # 1. Environment variables (works with launcher): EVAL_MODEL_PATH, EVAL_MAX_SAMPLES
    # 2. Command-line args (works when run directly): --model-path, --max-samples
    # Note: Hydra-style overrides (eval_model_path=...) don't work because Hydra validates
    # against the config structure, and these keys aren't part of GRPOConfig.
    model_path = os.getenv("EVAL_MODEL_PATH", None)
    max_samples = os.getenv("EVAL_MAX_SAMPLES", None)
    max_samples = int(max_samples) if max_samples else None
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the main configuration file", required=True)
    parser.add_argument("--model-path", help="Path to trained model checkpoint (overrides config)", default=model_path)
    parser.add_argument("--max-samples", type=int, help="Maximum number of test samples (overrides MAX_TEST_SAMPLES env var)", default=max_samples)
    # Parse known args to extract custom args, then pass remaining to load_expr_config
    # When running through launcher, Hydra may have already consumed some args
    try:
        parsed_args, remaining_args = parser.parse_known_args(args)
    except SystemExit:
        # If argparse fails (e.g., when Hydra has consumed args), use environment variables
        parsed_args = argparse.Namespace()
        parsed_args.config = args[args.index("--config") + 1] if "--config" in args else None
        parsed_args.model_path = model_path
        parsed_args.max_samples = max_samples
        # Remove --config and its value from remaining_args
        filtered_args = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg == "--config":
                skip_next = True
                continue
            if not arg.startswith("--model-path") and not arg.startswith("--max-samples"):
                filtered_args.append(arg)
        remaining_args = filtered_args
    
    # Load YAML directly to extract and remove custom fields that aren't in GRPOConfig
    config_file = Path(parsed_args.config)
    if not config_file.is_absolute():
        config_file = Path.cwd() / config_file
    raw_yaml = OmegaConf.load(config_file)
    
    # Remove custom fields that aren't part of GRPOConfig
    # These are training-specific fields that shouldn't be in the config for evaluation
    custom_keys = ["max_train_samples", "training_mode", "circuit_breaker_enabled", "circuit_breaker_threshold"]
    override_args = []
    for key in custom_keys:
        if key in raw_yaml:
            # Use ~ prefix to delete the key in OmegaConf/Hydra
            override_args.append(f"~{key}")
    
    # Add overrides to args to remove custom keys
    args_with_overrides = remaining_args + ["--config", str(parsed_args.config)] + override_args
    
    config, _ = load_expr_config(args_with_overrides, GRPOConfig)
    config: GRPOConfig

    # Note: We do NOT override experiment/trial from checkpoint path for server discovery
    # The server is launched with the config's experiment/trial name, so we need to use
    # the same name to find it. The checkpoint path is only used for loading model weights
    # (which happens in the SGLang server, not in this evaluation script).
    if parsed_args.model_path:
        print(f"[EVAL] Checkpoint path provided: {parsed_args.model_path}")
        print(f"[EVAL] Note: Server discovery uses config experiment/trial: {config.rollout.experiment_name}/{config.rollout.trial_name}")
        print(f"[EVAL] The checkpoint path is used by the SGLang server to load model weights.")

    # Set up distributed environment variables if not already set
    # This is needed for single-process evaluation
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("gloo")
    # Create a group for stats all-reduce.
    group = dist.new_group()

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"eval{rank}")

    # Create dataset and dataloaders
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    
    # Optional: Limit test samples via command-line argument, environment variable, or config
    # Priority: --max-samples > MAX_TEST_SAMPLES env var > 0 (no limit)
    if parsed_args.max_samples:
        max_test_samples = parsed_args.max_samples
    else:
        max_test_samples = int(os.getenv("MAX_TEST_SAMPLES", "0"))
    
    if max_test_samples > 0 and len(valid_dataset) > max_test_samples:
        print(f"[EVAL] Limiting test set from {len(valid_dataset)} to {max_test_samples} samples")
        valid_dataset = valid_dataset.select(range(max_test_samples))
    
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=rank,
        world_size=world_size,
        dataset_config=config.valid_dataset,
    )

    # Initialize inference engine
    config.rollout.max_head_offpolicyness = int(1e12)
    # Increase timeouts for evaluation
    config.rollout.setup_timeout = 600  # 10 minutes for server startup
    config.rollout.request_timeout = 3600  # 1 hour for requests
    eval_rollout = RemoteSGLangEngine(config.rollout)
    print("[EVAL] Initializing SGLang engine and connecting to server...")
    eval_rollout.initialize()
    print("[EVAL] SGLang engine initialized successfully!")

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

    print(f"Evaluating on {len(valid_dataset)} test samples...")
    print(f"Using model from: {config.rollout.experiment_name}/{config.rollout.trial_name}")

    # Run evaluation.
    cnt = 0
    for data in valid_dataloader:
        for item in data:
            eval_rollout.submit(item, workflow)
            cnt += 1
    
    print(f"Submitted {cnt} evaluation tasks. Waiting for completion...")
    eval_rollout.wait(cnt, timeout=None)

    eval_rollout_stats = stats_tracker.export_all(reduce_group=group)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(tabulate_stats(eval_rollout_stats))
    
    # Extract accuracy from reward stats
    # Try multiple possible keys for reward
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
                # Direct value
                accuracy = reward_stats * 100 if reward_stats <= 1.0 else reward_stats
                reward_key = key
                break
    
    if accuracy is not None:
        print(f"\n{'='*80}")
        print(f"ACCURACY: {accuracy:.2f}%")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("WARNING: Could not extract accuracy from stats.")
        print(f"Available keys: {list(eval_rollout_stats.keys())}")
        print(f"{'='*80}\n")
    
    eval_rollout.destroy()
    dist.destroy_process_group()
    
    # Exit cleanly to avoid JobException
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])

