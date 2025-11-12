#!/usr/bin/env python3
"""
Resume AReaL training from a checkpoint.

This script:
1. Lists available checkpoints
2. Finds the latest checkpoint before a specified step (or uses latest available)
3. Sets up recovery configuration
4. Provides the command to resume training
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Add parent directory to path to import list_checkpoints
sys.path.insert(0, os.path.dirname(__file__))
try:
    from list_checkpoints import find_checkpoints, find_recover_checkpoint, parse_checkpoint_dir
except ImportError:
    # Fallback: define functions inline if import fails
    import re
    import getpass
    
    def parse_checkpoint_dir(dirname: str) -> Optional[Tuple[int, int, int]]:
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
            
            if max_global_step is not None and global_step > max_global_step:
                continue
            
            checkpoint_path = os.path.join(checkpoint_root, dirname)
            
            if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")) or \
               os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
                checkpoints.append((epoch, step, global_step, checkpoint_path))
        
        checkpoints.sort(key=lambda x: x[2])
        return checkpoints
    
    def find_recover_checkpoint(
        fileroot: str,
        experiment_name: str,
        trial_name: str,
    ) -> Optional[str]:
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


def find_best_checkpoint(
    fileroot: str,
    experiment_name: str,
    trial_name: str,
    before_step: Optional[int] = None,
) -> Optional[Tuple[int, int, int, str]]:
    """
    Find the best checkpoint to resume from.
    
    If before_step is specified, finds the latest checkpoint with global_step < before_step.
    Otherwise, finds the latest checkpoint overall.
    
    Returns:
        (epoch, step, global_step, checkpoint_path) or None
    """
    checkpoints = find_checkpoints(
        fileroot,
        experiment_name,
        trial_name,
        max_global_step=before_step - 1 if before_step else None,
    )
    
    if not checkpoints:
        return None
    
    # Return the latest checkpoint
    return checkpoints[-1]


def create_recovery_config(
    checkpoint_path: str,
    output_config_path: str,
    original_config_path: str,
):
    """Create a recovery config file that points to the checkpoint."""
    # Read original config
    import yaml
    from omegaconf import OmegaConf
    
    original_config = OmegaConf.load(original_config_path)
    
    # Update recover config
    original_config.recover.mode = "manual"
    original_config.recover.checkpoint_path = checkpoint_path
    
    # Save updated config
    with open(output_config_path, "w") as f:
        OmegaConf.save(original_config, f)
    
    print(f"✅ Created recovery config: {output_config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Resume AReaL training from a checkpoint"
    )
    parser.add_argument(
        "--fileroot",
        type=str,
        default="/workspace/outputs/grpo",
        help="Root directory for checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "--trial-name",
        type=str,
        required=True,
        help="Trial name",
    )
    parser.add_argument(
        "--before-step",
        type=int,
        default=None,
        help="Find latest checkpoint before this step (e.g., 188 to resume before crash)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/cloud_gsm8k/gsm8k_grpo_cloud.yaml",
        help="Original config file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done, don't create files",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Resume Training: {args.experiment_name}/{args.trial_name}")
    print(f"{'='*80}\n")
    
    # Find best checkpoint
    if args.before_step:
        print(f"🔍 Looking for latest checkpoint before step {args.before_step}...")
    else:
        print(f"🔍 Looking for latest checkpoint...")
    
    checkpoint_info = find_best_checkpoint(
        args.fileroot,
        args.experiment_name,
        args.trial_name,
        before_step=args.before_step,
    )
    
    if not checkpoint_info:
        print("❌ No suitable checkpoint found!")
        if args.before_step:
            print(f"   (No checkpoints with global_step < {args.before_step})")
        return 1
    
    epoch, step, global_step, checkpoint_path = checkpoint_info
    
    print(f"\n✅ Found checkpoint:")
    print(f"   Global Step: {global_step}")
    print(f"   Epoch: {epoch}, Step: {step}")
    print(f"   Path: {checkpoint_path}")
    
    # Check for recover checkpoint
    recover_path = find_recover_checkpoint(
        args.fileroot,
        args.experiment_name,
        args.trial_name,
    )
    
    if recover_path and os.path.exists(os.path.join(recover_path, "step_info.json")):
        print(f"\n✅ Recover checkpoint found: {recover_path}")
        with open(os.path.join(recover_path, "step_info.json")) as f:
            recover_info = json.load(f)
            print(f"   Recover info: Step {recover_info.get('global_step')}")
    else:
        print(f"\n⚠️  No recover checkpoint found - will need to use manual recovery")
    
    # Generate resume command
    print(f"\n{'='*80}")
    print("Resume Command:")
    print(f"{'='*80}\n")
    
    # Method 1: Use recover mode (if recover checkpoint exists)
    if recover_path and os.path.exists(os.path.join(recover_path, "step_info.json")):
        print("# Method 1: Use automatic recovery (recommended)")
        print("# This uses the recover_checkpoint directory automatically")
        print(f"python -m areal.launcher.local {args.config.replace('.yaml', '.py')} \\")
        print(f"    --config {args.config} \\")
        print(f"    experiment_name={args.experiment_name} \\")
        print(f"    trial_name={args.trial_name} \\")
        print(f"    recover.mode=auto")
        print()
    
    # Method 2: Manual recovery - copy checkpoint to recover_checkpoint
    print("# Method 2: Manual recovery (copy checkpoint to recover_checkpoint)")
    print("# Step 1: Copy the checkpoint to recover_checkpoint directory")
    import getpass
    recover_checkpoint_dir = os.path.join(
        args.fileroot,
        "checkpoints",
        getpass.getuser(),
        args.experiment_name,
        args.trial_name,
        "recover_checkpoint",
    )
    print(f"mkdir -p {recover_checkpoint_dir}")
    print(f"# Then copy checkpoint files to {recover_checkpoint_dir}")
    print(f"# Or use the script below to do it automatically")
    print()
    
    # Method 3: Direct checkpoint path (if supported)
    print("# Method 3: Direct checkpoint recovery")
    print("# Note: This requires modifying the config to point to the checkpoint")
    print(f"# Checkpoint path: {checkpoint_path}")
    print()
    
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

