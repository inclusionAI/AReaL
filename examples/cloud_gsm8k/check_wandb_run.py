#!/usr/bin/env python3
"""
Check WandB run status and diagnose issues.
Usage:
    python examples/cloud_gsm8k/check_wandb_run.py --trial-name trial_20251115_233602
    python examples/cloud_gsm8k/check_wandb_run.py --trial-name trial_20251115_233602 --api-key YOUR_KEY
"""

import argparse
import os
import sys
from typing import Optional

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Install with: pip install wandb")
    sys.exit(1)


def analyze_run(trial_name: str, api_key: Optional[str] = None, project: str = "gsm8k-grpo-local", entity: Optional[str] = None):
    """Analyze a WandB run by trial name."""
    
    # Set API key if provided
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Try to find the run
    # The run name format is: gsm8k-grpo-cloud-1hour_{trial_name}_train
    # But we can also search by trial name in the group or name field
    
    print(f"\n{'='*80}")
    print(f"Analyzing WandB Run: {trial_name}")
    print(f"{'='*80}\n")
    
    # Search for runs in the project
    try:
        if entity:
            runs = api.runs(f"{entity}/{project}")
        else:
            # Try to auto-detect entity from current user
            runs = api.runs(project)
    except Exception as e:
        print(f"ERROR: Could not access project '{project}': {e}")
        print("\nTrying to find entity automatically...")
        # Try common entity names
        for possible_entity in ["tong-zhao-georgia-institute-of-technology", None]:
            try:
                if possible_entity:
                    runs = api.runs(f"{possible_entity}/{project}")
                    entity = possible_entity
                    print(f"Found entity: {entity}")
                    break
                else:
                    runs = api.runs(project)
                    break
            except:
                continue
        else:
            print("ERROR: Could not find project. Please specify --entity")
            return
    
    # Find runs matching the trial name
    matching_runs = []
    for run in runs:
        # Check if trial name appears in run name, group, or id
        if trial_name in run.name or trial_name in str(run.group) or trial_name in run.id:
            matching_runs.append(run)
    
    if not matching_runs:
        print(f"❌ No runs found matching trial name: {trial_name}")
        print(f"\nAvailable runs in project '{project}':")
        for run in list(runs)[:10]:  # Show first 10
            print(f"  - {run.name} (group: {run.group}, id: {run.id})")
        return
    
    # Analyze each matching run
    for run in matching_runs:
        print(f"\n{'='*80}")
        print(f"Run: {run.name}")
        print(f"ID: {run.id}")
        print(f"Group: {run.group}")
        print(f"State: {run.state}")
        print(f"{'='*80}\n")
        
        # Check run state
        if run.state == "finished":
            print("✅ Run completed successfully")
        elif run.state == "running":
            print("⚠️  Run is still running (or terminated externally without proper shutdown)")
        elif run.state == "crashed":
            print("❌ Run crashed")
        elif run.state == "failed":
            print("❌ Run failed")
        else:
            print(f"⚠️  Run state: {run.state}")
        
        # Check timing information
        print(f"\n⏰ Timing Information:")
        if hasattr(run, 'created_at'):
            print(f"  Created: {run.created_at}")
        if hasattr(run, 'updated_at'):
            print(f"  Last updated: {run.updated_at}")
        if hasattr(run, 'runtime'):
            print(f"  Runtime: {run.runtime} seconds")
        
        # Check if run appears to have stopped updating
        import time
        if hasattr(run, 'updated_at'):
            try:
                from datetime import datetime
                if isinstance(run.updated_at, str):
                    last_update = datetime.fromisoformat(run.updated_at.replace('Z', '+00:00'))
                else:
                    last_update = run.updated_at
                now = datetime.now(last_update.tzinfo) if hasattr(last_update, 'tzinfo') and last_update.tzinfo else datetime.now()
                time_since_update = (now - last_update).total_seconds()
                print(f"  Time since last update: {time_since_update/60:.1f} minutes ({time_since_update/3600:.2f} hours)")
                if time_since_update > 3600:  # More than 1 hour
                    print("  ⚠️  WARNING: Run hasn't updated in over 1 hour - likely terminated externally")
            except Exception as e:
                print(f"  ⚠️  Could not calculate time since update: {e}")
        
        # Get summary metrics
        print("\n📊 Summary Metrics:")
        summary = run.summary
        if summary:
            # Key metrics to check
            key_metrics = [
                "grpo_actor/task_reward/avg",
                "grpo_actor/loss",
                "grpo_actor/entropy",
                "grpo_actor/grad_norm",
                "rollout/reward",
                "rollout/length",
            ]
            
            for metric in key_metrics:
                if metric in summary:
                    value = summary[metric]
                    print(f"  {metric}: {value}")
            
            # Check for zero task reward
            if "grpo_actor/task_reward/avg" in summary:
                task_reward = summary["grpo_actor/task_reward/avg"]
                if task_reward == 0.0:
                    print("\n⚠️  WARNING: Task reward is zero! This suggests:")
                    print("    - SGLang server may have crashed/disconnected")
                    print("    - All rollouts failed")
                    print("    - Model may have degraded")
        
        # Get history to check for patterns
        print("\n📈 Training History (last 20 steps):")
        try:
            # Use pandas=False to avoid pandas dependency issues
            history = run.history(pandas=False)
            if len(history) > 0:
                # Get last 20 steps
                last_steps = history[-20:] if len(history) > 20 else history
                
                # Extract metrics from history (list of dicts)
                task_rewards = []
                losses = []
                steps = []
                
                for row in last_steps:
                    if "grpo_actor/task_reward/avg" in row:
                        task_rewards.append(row["grpo_actor/task_reward/avg"])
                    if "grpo_actor/loss" in row:
                        losses.append(row["grpo_actor/loss"])
                    if "_step" in row:
                        steps.append(row["_step"])
                
                # Check task reward trend
                if task_rewards:
                    print(f"  Task reward range: {min(task_rewards):.4f} - {max(task_rewards):.4f}")
                    print(f"  Last task reward: {task_rewards[-1]:.4f}")
                    
                    # Check for zero reward streak
                    zero_streak = 0
                    for val in reversed(task_rewards):
                        if val == 0.0:
                            zero_streak += 1
                        else:
                            break
                    if zero_streak > 0:
                        print(f"  ⚠️  WARNING: {zero_streak} consecutive zero-reward steps at the end!")
                
                # Check loss
                if losses:
                    print(f"  Loss range: {min(losses):.4f} - {max(losses):.4f}")
                    print(f"  Last loss: {losses[-1]:.4f}")
                
                # Show step count
                if steps:
                    print(f"  Total steps: {max(steps):.0f}")
                    print(f"  Last step: {steps[-1]:.0f}")
                    
                # Show recent task reward values
                if task_rewards and len(task_rewards) >= 5:
                    print(f"\n  Recent task rewards (last 5 steps):")
                    for i, val in enumerate(task_rewards[-5:], start=len(task_rewards)-4):
                        step_num = steps[i-1] if i-1 < len(steps) else "?"
                        print(f"    Step {step_num}: {val:.4f}")
        except Exception as e:
            print(f"  ⚠️  Could not fetch history: {e}")
        
        # Check for errors in logs
        print("\n🔍 Checking for errors...")
        try:
            # Try to get run logs
            files = run.files()
            log_files = [f for f in files if "log" in f.name.lower() or "error" in f.name.lower()]
            if log_files:
                print(f"  Found {len(log_files)} log files")
                # Try to read first log file
                try:
                    first_log = log_files[0]
                    log_content = first_log.download(replace=True)
                    # Check for common error patterns
                    error_patterns = [
                        "CUDA out of memory",
                        "OOM",
                        "out of memory",
                        "Connection refused",
                        "Connection timeout",
                        "Circuit breaker",
                        "SGLang",
                        "crash",
                        "error",
                        "Error",
                        "ERROR",
                    ]
                    found_errors = []
                    with open(log_content.name, 'r') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines[-100:], start=len(lines)-100):  # Last 100 lines
                            for pattern in error_patterns:
                                if pattern in line:
                                    found_errors.append((i, pattern, line.strip()[:100]))
                                    break
                    
                    if found_errors:
                        print(f"  ⚠️  Found {len(found_errors)} potential errors in logs:")
                        for line_num, pattern, line in found_errors[-10:]:  # Last 10 errors
                            print(f"    Line {line_num}: [{pattern}] {line}")
                    else:
                        print("  ✅ No obvious errors found in logs")
                except Exception as e:
                    print(f"  ⚠️  Could not read log file: {e}")
            else:
                print("  ℹ️  No log files found")
        except Exception as e:
            print(f"  ⚠️  Could not check logs: {e}")
        
        # Print run URL
        print(f"\n🔗 WandB Run URL: {run.url}")
        
        # Check config
        print("\n⚙️  Configuration:")
        if run.config:
            important_configs = [
                "experiment_name",
                "trial_name",
                "total_train_epochs",
                "actor.gradient_checkpointing",
                "sglang.mem_fraction_static",
                "train_dataset.batch_size",
            ]
            for key in important_configs:
                if key in run.config:
                    print(f"  {key}: {run.config[key]}")


def main():
    parser = argparse.ArgumentParser(description="Check WandB run status and diagnose issues")
    parser.add_argument(
        "--trial-name",
        type=str,
        required=True,
        help="Trial name (e.g., trial_20251115_233602)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="WandB API key (or set WANDB_API_KEY env var)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="gsm8k-grpo-local",
        help="WandB project name (default: gsm8k-grpo-local)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="WandB entity/username (optional, will try to auto-detect)",
    )
    
    args = parser.parse_args()
    
    # Get API key from env if not provided
    api_key = args.api_key or os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("ERROR: WandB API key not provided.")
        print("Either set WANDB_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    analyze_run(args.trial_name, api_key, args.project, args.entity)


if __name__ == "__main__":
    main()

