"""
Consolidated GRPO training script for GSM8K.

This script handles all training configurations (fast, 1hour, 3hour, full)
by reading settings from the YAML config file and command-line overrides.

Usage:
    python -m areal.launcher.local examples/cloud_gsm8k/gsm8k_grpo_train.py \
        --config examples/cloud_gsm8k/gsm8k_grpo_fast.yaml

Config parameters:
    - max_train_samples: Limit dataset size (None for full dataset)
    - training_mode: Display name for training mode (e.g., "FAST", "1-HOUR", "3-HOUR", "FULL")
    - circuit_breaker_enabled: Enable circuit breaker for zero-reward detection (default: True)
    - circuit_breaker_threshold: Number of consecutive zero rewards before stopping (default: 50)
"""
import os
import sys
import logging
import getpass
import glob
import re
import subprocess
import time
from copy import deepcopy

import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import cycle_dataloader
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from areal.reward.math_parser import process_results

    return int(process_results(answer, completions)[0]) # TODO: this is a binary reward, might want to change to fraction reward


def main(args):
    # Store args and script_dir for later use in testing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_args = args.copy() if isinstance(args, list) else args
    
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
    config_file_str = str(config_file)  # Store for later use
    raw_yaml = OmegaConf.load(config_file)
    
    # Extract custom training parameters
    max_train_samples = raw_yaml.get("max_train_samples", None)
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    training_mode = raw_yaml.get("training_mode", "TRAINING")
    circuit_breaker_enabled = raw_yaml.get("circuit_breaker_enabled", True)
    circuit_breaker_threshold = int(raw_yaml.get("circuit_breaker_threshold", 50))
    
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

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Create dataset and dataloaders
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )

    # Apply dataset size limit if specified
    original_size = len(train_dataset)
    if max_train_samples is not None and len(train_dataset) > max_train_samples:
        print(f"[{training_mode}] Limiting dataset from {len(train_dataset)} to {max_train_samples} samples")
        train_dataset = train_dataset.select(range(max_train_samples))
    elif max_train_samples is None:
        print(f"[{training_mode}] Using full dataset: {len(train_dataset)} samples")

    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.valid_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    # Use disk-based weight updates for single-GPU setups to avoid NCCL duplicate GPU error
    # NCCL requires separate GPUs for each rank, but single GPU setups share the same GPU
    # between training and inference processes
    if config.cluster.n_gpus_per_node == 1 and allocation_mode.gen.world_size == 1:
        weight_update_meta = WeightUpdateMeta.from_disk(
            config.experiment_name,
            config.trial_name,
            config.cluster.fileroot,
        )
    else:
        weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    # Always set dump_dir for consistency (harmless if not needed)
    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )

    # Eval workflow with temperature=0.6 for better evaluation
    eval_workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    # Print training summary
    print(f"\n{'='*80}")
    print(f"[{training_mode} MODE]")
    print(f"  Dataset size: {len(train_dataset)} samples" + 
          (f" (limited from {original_size})" if max_train_samples and original_size > max_train_samples else ""))
    print(f"  Batch size: {config.train_dataset.batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Total steps: {max_steps}")
    print(f"  Estimated time: ~{max_steps} minutes (~{max_steps/60:.1f} hours) at ~1 step/min")
    if circuit_breaker_enabled:
        print(f"  Circuit breaker: Enabled (threshold: {circuit_breaker_threshold} consecutive zero rewards)")
    print(f"{'='*80}\n")

    # Circuit breaker: Track consecutive zero rewards
    zero_reward_streak = 0

    data_generator = cycle_dataloader(train_dataloader)
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = actor.prepare_batch(
                    train_dataloader,
                    granularity=actor.config.group_size,
                    workflow=workflow,
                    should_accept=lambda sample: True,
                )
            else:
                batch = actor.rollout_batch(
                    next(data_generator),
                    granularity=actor.config.group_size,
                    workflow=workflow,
                    should_accept=lambda sample: True,
                )

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
                base_model_path=config.actor.path,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                if actor.is_data_parallel_head():
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats[0].update(
            stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        )
        stats_logger.commit(epoch, step, global_step, stats)

        # Circuit breaker: Check task reward and stop if zero for too long
        if circuit_breaker_enabled:
            task_reward_avg = stats[0].get("grpo_actor/task_reward/avg", None)
            if task_reward_avg is not None:
                if task_reward_avg == 0.0:
                    zero_reward_streak += 1
                    if zero_reward_streak >= circuit_breaker_threshold:
                        error_msg = (
                            f"\n{'='*80}\n"
                            f"CIRCUIT BREAKER TRIGGERED!\n"
                            f"{'='*80}\n"
                            f"Task reward has been zero for {zero_reward_streak} consecutive steps.\n"
                            f"Current step: {global_step} (Epoch {epoch}, Step {step})\n"
                            f"Stopping training to prevent model corruption.\n"
                            f"\nPossible causes:\n"
                            f"  - SGLang server crashed or disconnected\n"
                            f"  - Inference server not responding\n"
                            f"  - Network connectivity issues\n"
                            f"\nNext steps:\n"
                            f"  1. Check SGLang server logs for errors\n"
                            f"  2. Verify SGLang server is running\n"
                            f"  3. Resume training from checkpoint before step {global_step - zero_reward_streak + 1}\n"
                            f"  4. Use: python examples/cloud_gsm8k/resume_training.py \\\n"
                            f"         --experiment-name {config.experiment_name} \\\n"
                            f"         --trial-name {config.trial_name} \\\n"
                            f"         --before-step {global_step - zero_reward_streak + 1}\n"
                            f"{'='*80}\n"
                        )
                        if actor.is_data_parallel_head():
                            print(error_msg)
                            logging.error(error_msg)
                        # Save a final checkpoint before stopping
                        saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)
                        recover_handler.dump(
                            actor,
                            step_info,
                            saver,
                            evaluator,
                            stats_logger,
                            train_dataloader,
                            tokenizer=tokenizer,
                        )
                        # Clean shutdown
                        stats_logger.close()
                        eval_rollout.destroy()
                        rollout.destroy()
                        if ref is not None:
                            ref.destroy()
                        actor.destroy()
                        sys.exit(1)
                else:
                    # Reset streak if reward is non-zero
                    if zero_reward_streak > 0:
                        if actor.is_data_parallel_head():
                            print(f"✅ Task reward recovered: {task_reward_avg:.4f} (was zero for {zero_reward_streak} steps)")
                    zero_reward_streak = 0
            elif actor.is_data_parallel_head() and global_step % 10 == 0:
                # Log warning if task_reward is missing from stats
                print(f"⚠️  Warning: task_reward/avg not found in stats at step {global_step}")

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()
    
    # Run test on first 50 samples after training completes
    # Test both baseline model and trained model
    if actor.is_data_parallel_head():
        print(f"\n{'='*80}")
        print("Training completed! Running tests on first 50 samples...")
        print(f"Testing both BASELINE and TRAINED models for comparison")
        print(f"{'='*80}\n")
        
        # Get baseline model path (original model before training)
        baseline_model_path = config.actor.path
        print(f"Baseline model: {baseline_model_path}")
        
        # Find the latest checkpoint
        checkpoint_dir = os.path.join(
            config.cluster.fileroot,
            "checkpoints",
            getpass.getuser(),
            config.experiment_name,
            config.trial_name,
            "default"
        )
        
        # Get max_new_tokens from config, but increase for testing to allow longer reasoning
        # Training uses shorter sequences for efficiency, but testing needs full answers
        training_max_tokens = config.gconfig.max_new_tokens
        test_max_tokens = max(training_max_tokens, 512)  # Use at least 512 for testing
        max_new_tokens = str(test_max_tokens)
        print(f"Using max_new_tokens: {max_new_tokens} for testing (training used {training_max_tokens})")
        
        # Get the latest checkpoint (by modification time)
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
            if checkpoints:
                # Sort by modification time, get the latest
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
                
                print(f"Trained model checkpoint: {latest_checkpoint}")
                
                # Determine if this is a reasoning model
                is_reasoning = "reasoning" in config.experiment_name.lower() or "reasoning" in config.train_dataset.path.lower()
                
                # Save logs to network volume (persists after pod stops)
                test_log_dir = os.path.join(config.cluster.fileroot, "test_logs")
                
                # Use configurable batch size (default: 16 for faster testing)
                test_batch_size = os.environ.get("TEST_BATCH_SIZE", "16")
                
                if is_reasoning:
                    # Use reasoning test script
                    test_script = os.path.join(script_dir, "test_reasoning_model_cloud.py")
                    
                    # Test baseline model first
                    print(f"\n{'='*80}")
                    print("Testing BASELINE model...")
                    print(f"{'='*80}\n")
                    baseline_cmd = [
                        sys.executable,
                        test_script,
                        "--model-path", baseline_model_path,
                        "--max-samples", "50",
                        "--max-new-tokens", max_new_tokens,  # Use config value
                        "--log-dir", test_log_dir,
                        "--model-name", "BASELINE",
                        "--batch-size", test_batch_size,
                    ]
                    
                    # Test trained model
                    print(f"\n{'='*80}")
                    print("Testing TRAINED model...")
                    print(f"{'='*80}\n")
                    trained_cmd = [
                        sys.executable,
                        test_script,
                        "--model-path", latest_checkpoint,
                        "--max-samples", "50",
                        "--max-new-tokens", max_new_tokens,  # Use config value
                        "--log-dir", test_log_dir,
                        "--model-name", "TRAINED",
                        "--batch-size", test_batch_size,
                    ]
                else:
                    # Use regular test script (supports both --model-path and --config)
                    test_script = os.path.join(script_dir, "test_trained_model_cloud.py")
                    
                    # Test baseline model first
                    print(f"\n{'='*80}")
                    print("Testing BASELINE model...")
                    print(f"{'='*80}\n")
                    baseline_cmd = [
                        sys.executable,
                        test_script,
                        "--model-path", baseline_model_path,
                        "--max-samples", "50",
                        "--max-new-tokens", str(max_new_tokens),
                        "--log-dir", test_log_dir,
                        "--model-name", "BASELINE",
                        "--batch-size", test_batch_size,
                    ]
                    
                    # Test trained model
                    print(f"\n{'='*80}")
                    print("Testing TRAINED model...")
                    print(f"{'='*80}\n")
                    trained_cmd = [
                        sys.executable,
                        test_script,
                        "--model-path", latest_checkpoint,
                        "--max-samples", "50",
                        "--max-new-tokens", str(max_new_tokens),
                        "--log-dir", test_log_dir,
                        "--model-name", "TRAINED",
                        "--batch-size", test_batch_size,
                    ]
                
                try:
                    baseline_accuracy = None
                    trained_accuracy = None
                    
                    # Test baseline model
                    if baseline_cmd is not None:
                        print(f"\n{'='*80}")
                        print("Testing BASELINE model...")
                        print(f"{'='*80}\n")
                        # Don't capture output so it prints directly to terminal
                        baseline_result = subprocess.run(baseline_cmd, check=False, capture_output=False)
                        
                        # Extract accuracy from log file (test script prints to both stdout and log file)
                        # Wait for log file to be written (30 seconds to ensure file is fully written)
                        time.sleep(30)
                        
                        if baseline_result.returncode == 0:
                            # Try to find the log file and extract accuracy from it
                            log_pattern = os.path.join(test_log_dir, f"test_model_baseline_*.log")
                            log_files = glob.glob(log_pattern)
                            if log_files:
                                # Get the most recent log file
                                latest_log = max(log_files, key=os.path.getmtime)
                                try:
                                    with open(latest_log, 'r', encoding='utf-8') as f:
                                        log_content = f.read()
                                        # Try multiple patterns to match accuracy
                                        patterns = [
                                            r'BASELINE\s+MODEL\s+ACCURACY:\s+(\d+\.\d+)%',
                                            r'FINAL\s+ACCURACY:\s+(\d+\.\d+)%',
                                            r'(?:[A-Z]+\s+)?(?:MODEL\s+)?ACCURACY:\s+(\d+\.\d+)%',
                                        ]
                                        for pattern in patterns:
                                            match = re.search(pattern, log_content, re.IGNORECASE)
                                            if match:
                                                baseline_accuracy = float(match.group(1))
                                                break
                                except Exception as e:
                                    print(f"Warning: Could not read baseline log file: {e}")
                            print(f"\n✅ Baseline test completed successfully!")
                        else:
                            print(f"\n⚠️  Baseline test exited with code {baseline_result.returncode}")
                    
                    # Test trained model
                    print(f"\n{'='*80}")
                    print("Testing TRAINED model...")
                    print(f"{'='*80}\n")
                    # Don't capture output so it prints directly to terminal
                    trained_result = subprocess.run(trained_cmd, check=False, capture_output=False)
                    
                    # Extract accuracy from log file (test script prints to both stdout and log file)
                    # Wait for log file to be written (30 seconds to ensure file is fully written)
                    time.sleep(30)
                    
                    if trained_result.returncode == 0:
                        # Try to find the log file and extract accuracy from it
                        log_pattern = os.path.join(test_log_dir, f"test_model_trained_*.log")
                        log_files = glob.glob(log_pattern)
                        if log_files:
                            # Get the most recent log file
                            latest_log = max(log_files, key=os.path.getmtime)
                            try:
                                with open(latest_log, 'r', encoding='utf-8') as f:
                                    log_content = f.read()
                                    # Try multiple patterns to match accuracy
                                    patterns = [
                                        r'TRAINED\s+MODEL\s+ACCURACY:\s+(\d+\.\d+)%',
                                        r'FINAL\s+ACCURACY:\s+(\d+\.\d+)%',
                                        r'(?:[A-Z]+\s+)?(?:MODEL\s+)?ACCURACY:\s+(\d+\.\d+)%',
                                    ]
                                    for pattern in patterns:
                                        match = re.search(pattern, log_content, re.IGNORECASE)
                                        if match:
                                            trained_accuracy = float(match.group(1))
                                            break
                            except Exception as e:
                                print(f"Warning: Could not read trained log file: {e}")
                        print(f"\n✅ Trained test completed successfully!")
                    else:
                        print(f"\n⚠️  Trained model test exited with code {trained_result.returncode}")
                    
                    # Print comparison summary
                    print(f"\n{'='*80}")
                    print("TEST RESULTS SUMMARY")
                    print(f"{'='*80}")
                    if baseline_accuracy is not None:
                        print(f"Baseline Model Accuracy: {baseline_accuracy:.2f}%")
                    else:
                        print(f"Baseline Model Accuracy: N/A (test may have failed)")
                    
                    if trained_accuracy is not None:
                        print(f"Trained Model Accuracy:  {trained_accuracy:.2f}%")
                    else:
                        print(f"Trained Model Accuracy:  N/A (test may have failed)")
                    
                    if baseline_accuracy is not None and trained_accuracy is not None:
                        improvement = trained_accuracy - baseline_accuracy
                        improvement_pct = (improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
                        print(f"\nImprovement: {improvement:+.2f}% ({improvement_pct:+.1f}% relative)")
                        if improvement > 0:
                            print("✅ Training improved model performance!")
                        elif improvement < 0:
                            print("⚠️  Training degraded model performance")
                        else:
                            print("➡️  Training did not change model performance")
                    
                    print(f"{'='*80}\n")
                    
                    if baseline_result.returncode == 0 and trained_result.returncode == 0:
                        print(f"✅ All tests completed successfully!")
                    else:
                        print(f"⚠️  Some tests may have failed. Check logs above for details.")
                    print(f"{'='*80}\n")
                    
                    # Auto-upload logs if configured
                    upload_method = os.environ.get("AUTO_UPLOAD_LOGS_METHOD")
                    if upload_method:
                        # Normalize method to lowercase (handle EMAIL -> email, etc.)
                        upload_method = upload_method.lower()
                        print(f"\n{'='*80}")
                        print(f"📤 Auto-uploading test logs via {upload_method}...")
                        print(f"{'='*80}\n")
                        try:
                            upload_script = os.path.join(script_dir, "upload_logs.py")
                            upload_cmd = [
                                sys.executable,
                                upload_script,
                                "--log-dir", test_log_dir,
                                "--method", upload_method,
                                "--latest-only",  # Only upload latest baseline and trained logs
                            ]
                            
                            # Add method-specific arguments from environment
                            if upload_method == "email":
                                if os.environ.get("AUTO_UPLOAD_EMAIL_TO"):
                                    upload_cmd.extend(["--email-to", os.environ.get("AUTO_UPLOAD_EMAIL_TO")])
                            elif upload_method == "gdrive":
                                if os.environ.get("AUTO_UPLOAD_GDRIVE_FOLDER_ID"):
                                    upload_cmd.extend(["--gdrive-folder-id", os.environ.get("AUTO_UPLOAD_GDRIVE_FOLDER_ID")])
                            elif upload_method == "s3":
                                if os.environ.get("AUTO_UPLOAD_S3_BUCKET"):
                                    upload_cmd.extend(["--s3-bucket", os.environ.get("AUTO_UPLOAD_S3_BUCKET")])
                                if os.environ.get("AUTO_UPLOAD_S3_PREFIX"):
                                    upload_cmd.extend(["--s3-prefix", os.environ.get("AUTO_UPLOAD_S3_PREFIX")])
                            elif upload_method == "hf":
                                if os.environ.get("AUTO_UPLOAD_HF_REPO_ID"):
                                    upload_cmd.extend(["--hf-repo-id", os.environ.get("AUTO_UPLOAD_HF_REPO_ID")])
                            elif upload_method == "wandb":
                                if os.environ.get("AUTO_UPLOAD_WANDB_PROJECT"):
                                    upload_cmd.extend(["--wandb-project", os.environ.get("AUTO_UPLOAD_WANDB_PROJECT")])
                                if os.environ.get("AUTO_UPLOAD_WANDB_RUN_NAME"):
                                    upload_cmd.extend(["--wandb-run-name", os.environ.get("AUTO_UPLOAD_WANDB_RUN_NAME")])
                            elif upload_method == "webhook":
                                if os.environ.get("AUTO_UPLOAD_WEBHOOK_URL"):
                                    upload_cmd.extend(["--webhook-url", os.environ.get("AUTO_UPLOAD_WEBHOOK_URL")])
                            
                            upload_result = subprocess.run(upload_cmd, check=False, capture_output=True)
                            if upload_result.returncode == 0:
                                print(f"✅ Logs uploaded successfully via {upload_method}!")
                            else:
                                print(f"⚠️  Log upload failed: {upload_result.stderr.decode() if upload_result.stderr else 'Unknown error'}")
                        except Exception as upload_error:
                            print(f"⚠️  Failed to upload logs: {upload_error}")
                            print(f"   You can manually upload logs using:")
                            print(f"   python {upload_script} --log-dir {test_log_dir} --method {upload_method} ...")
                        print(f"{'='*80}\n")
                except Exception as e:
                    print(f"\n{'='*80}")
                    print(f"⚠️  Failed to run test: {e}")
                    print(f"   You can run the test manually with:")
                    print(f"   Baseline: python {test_script} --model-path {baseline_model_path} --max-samples 50 --max-new-tokens {max_new_tokens} --model-name BASELINE")
                    print(f"   Trained:  python {test_script} --model-path {latest_checkpoint} --max-samples 50 --max-new-tokens {max_new_tokens} --model-name TRAINED")
                    print(f"{'='*80}\n")
            else:
                print(f"⚠️  No checkpoints found in {checkpoint_dir}")
        else:
            print(f"⚠️  Checkpoint directory not found: {checkpoint_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])

