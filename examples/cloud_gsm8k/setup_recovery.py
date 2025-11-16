#!/usr/bin/env python3
"""
Set up recovery from a specific checkpoint by copying it to recover_checkpoint directory.
"""

import argparse
import os
import shutil
import sys
import getpass
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))
from list_checkpoints import find_checkpoints


def find_best_checkpoint(
    fileroot: str,
    experiment_name: str,
    trial_name: str,
    before_step: int = None,
) -> tuple:
    """Find the best checkpoint to resume from."""
    checkpoints = find_checkpoints(
        fileroot,
        experiment_name,
        trial_name,
        max_global_step=before_step - 1 if before_step else None,
    )
    
    if not checkpoints:
        return None
    
    return checkpoints[-1]


def setup_recovery(
    fileroot: str,
    experiment_name: str,
    trial_name: str,
    before_step: int = None,
    checkpoint_path: str = None,
    dry_run: bool = False,
):
    """Set up recovery by copying checkpoint to recover_checkpoint directory."""
    
    # Find checkpoint if not provided
    if checkpoint_path is None:
        if before_step:
            print(f"🔍 Finding latest checkpoint before step {before_step}...")
        else:
            print(f"🔍 Finding latest checkpoint...")
        
        checkpoint_info = find_best_checkpoint(
            fileroot,
            experiment_name,
            trial_name,
            before_step=before_step,
        )
        
        if not checkpoint_info:
            print("❌ No suitable checkpoint found!")
            return 1
        
        epoch, step, global_step, checkpoint_path = checkpoint_info
        print(f"✅ Found checkpoint: Step {global_step} (Epoch {epoch}, Step {step})")
    else:
        # Parse checkpoint path to get step info
        checkpoint_dir = os.path.basename(checkpoint_path)
        import re
        pattern = r"epoch(\d+)epochstep(\d+)globalstep(\d+)"
        match = re.match(pattern, checkpoint_dir)
        if match:
            epoch, step, global_step = map(int, match.groups())
            print(f"✅ Using checkpoint: Step {global_step} (Epoch {epoch}, Step {step})")
        else:
            print(f"⚠️  Warning: Could not parse step info from checkpoint path")
            global_step = None
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
        return 1
    
    # Set up recover checkpoint directory
    recover_checkpoint_dir = os.path.join(
        fileroot,
        "checkpoints",
        getpass.getuser(),
        experiment_name,
        trial_name,
        "recover_checkpoint",
    )
    
    print(f"\n📋 Recovery Setup:")
    print(f"   Source: {checkpoint_path}")
    print(f"   Target: {recover_checkpoint_dir}")
    
    if dry_run:
        print(f"\n[DRY RUN] Would copy checkpoint to recover_checkpoint directory")
        return 0
    
    # Create recover checkpoint directory
    os.makedirs(recover_checkpoint_dir, exist_ok=True)
    
    # Copy checkpoint files
    print(f"\n📦 Copying checkpoint files...")
    try:
        # Copy all files from checkpoint directory
        for item in os.listdir(checkpoint_path):
            src = os.path.join(checkpoint_path, item)
            dst = os.path.join(recover_checkpoint_dir, item)
            
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        print(f"✅ Checkpoint copied successfully!")
        print(f"\n📝 Next steps:")
        print(f"   1. Resume training with:")
        print(f"      python -m areal.launcher.local examples/cloud_gsm8k/gsm8k_grpo_train.py \\")
        print(f"          --config examples/cloud_gsm8k/gsm8k_grpo_cloud.yaml \\")
        print(f"          experiment_name={experiment_name} \\")
        print(f"          trial_name={trial_name} \\")
        print(f"          recover.mode=auto")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error copying checkpoint: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Set up recovery from a checkpoint"
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
        help="Find latest checkpoint before this step",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Direct path to checkpoint (overrides --before-step)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done",
    )
    
    args = parser.parse_args()
    
    return setup_recovery(
        args.fileroot,
        args.experiment_name,
        args.trial_name,
        before_step=args.before_step,
        checkpoint_path=args.checkpoint_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())

