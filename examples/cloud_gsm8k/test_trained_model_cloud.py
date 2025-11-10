#!/usr/bin/env python3
"""
Test script for trained GRPO model on GSM8K dataset (Cloud version).
This script loads a trained model checkpoint and evaluates it on the GSM8K test set.
"""

import os
import sys
import warnings

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

    return int(process_results(completions, answer)[0])


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

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
    
    # Optional: Limit test samples via environment variable
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
    
    # Exit cleanly
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])

