#!/usr/bin/env python3
"""
List available checkpoints for a given experiment and trial.
Can filter by maximum global step to find checkpoints before a crash.
"""

import argparse
import os
import re
import getpass
from pathlib import Path
from typing import List, Tuple, Optional


def parse_checkpoint_dir(dirname: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse checkpoint directory name: epoch{epoch}epochstep{step}globalstep{global_step}
    Returns (epoch, step, global_step) or None if format doesn't match.
    """
    pattern = r"epoch(\d+)epochstep(\d+)globalstep(\d+)"
    match = re.match(pattern, dirname)
    if match:
        epoch, step, global_step = map(int, match.groups())
        return (epoch, step, global_step)
    return None


def find_checkpoints(
    fileroot: str,
    experiment_name: str,
    trial_name: str,
    max_global_step: Optional[int] = None,
    name: str = "default",
) -> List[Tuple[int, int, int, str]]:
    """
    Find all available checkpoints.
    
    Returns:
        List of (epoch, step, global_step, checkpoint_path) tuples, sorted by global_step.
    """
    checkpoint_root = os.path.join(
        fileroot,
        "checkpoints",
        getpass.getuser(),
        experiment_name,
        trial_name,
        name,
    )
    
    if not os.path.exists(checkpoint_root):
        return []
    
    checkpoints = []
    for dirname in os.listdir(checkpoint_root):
        parsed = parse_checkpoint_dir(dirname)
        if parsed is None:
            continue
        
        epoch, step, global_step = parsed
        
        # Filter by max_global_step if specified
        if max_global_step is not None and global_step > max_global_step:
            continue
        
        checkpoint_path = os.path.join(checkpoint_root, dirname)
        
        # Verify checkpoint is complete (has model files)
        if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")) or \
           os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
            checkpoints.append((epoch, step, global_step, checkpoint_path))
    
    # Sort by global_step
    checkpoints.sort(key=lambda x: x[2])
    return checkpoints


def find_recover_checkpoint(
    fileroot: str,
    experiment_name: str,
    trial_name: str,
) -> Optional[str]:
    """Find the recover checkpoint directory."""
    import getpass
    
    recover_path = os.path.join(
        fileroot,
        "checkpoints",
        getpass.getuser(),
        experiment_name,
        trial_name,
        "recover_checkpoint",
    )
    
    if os.path.exists(recover_path):
        return recover_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="List available checkpoints for AReaL training"
    )
    parser.add_argument(
        "--fileroot",
        type=str,
        default="/workspace/outputs/grpo",
        help="Root directory for checkpoints (default: /workspace/outputs/grpo)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment name (e.g., gsm8k-grpo-cloud-h200)",
    )
    parser.add_argument(
        "--trial-name",
        type=str,
        required=True,
        help="Trial name (e.g., trial_20251112_203112)",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only show checkpoints with global_step <= this value (useful for finding checkpoints before a crash)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Only show the latest checkpoint",
    )
    parser.add_argument(
        "--recover-info",
        action="store_true",
        help="Show recover checkpoint info if available",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Checkpoint Listing for: {args.experiment_name}/{args.trial_name}")
    print(f"{'='*80}\n")
    
    # Find checkpoints
    checkpoints = find_checkpoints(
        args.fileroot,
        args.experiment_name,
        args.trial_name,
        max_global_step=args.max_step,
    )
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        if args.max_step:
            print(f"   (Filtered to global_step <= {args.max_step})")
        print(f"\nCheckpoint root: {os.path.join(args.fileroot, 'checkpoints', os.getenv('USER', 'root'), args.experiment_name, args.trial_name)}")
        return
    
    # Show recover checkpoint if requested
    if args.recover_info:
        recover_path = find_recover_checkpoint(
            args.fileroot,
            args.experiment_name,
            args.trial_name,
        )
        if recover_path:
            print(f"✅ Recover checkpoint found: {recover_path}\n")
        else:
            print("❌ No recover checkpoint found\n")
    
    # Display checkpoints
    if args.latest:
        latest = checkpoints[-1]
        epoch, step, global_step, path = latest
        print(f"Latest checkpoint:")
        print(f"  Global Step: {global_step}")
        print(f"  Epoch: {epoch}, Step: {step}")
        print(f"  Path: {path}")
    else:
        print(f"Found {len(checkpoints)} checkpoint(s):\n")
        print(f"{'Global Step':<12} {'Epoch':<8} {'Step':<8} {'Path'}")
        print("-" * 80)
        for epoch, step, global_step, path in checkpoints:
            print(f"{global_step:<12} {epoch:<8} {step:<8} {path}")
        
        if checkpoints:
            latest = checkpoints[-1]
            epoch, step, global_step, path = latest
            print(f"\n✅ Latest checkpoint: Step {global_step} (Epoch {epoch}, Step {step})")
            print(f"   Path: {path}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

