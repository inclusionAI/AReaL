#!/usr/bin/env python3
"""
Quick validation script to test reasoning training config can be loaded
and basic components are wired correctly.
This does NOT run actual training, just validates the setup.
"""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from omegaconf import OmegaConf
from areal.dataset import get_custom_dataset
from areal.api.cli_args import DatasetConfig
from areal.utils.hf_utils import load_hf_tokenizer

def validate_reasoning_config():
    """Validate that reasoning config can be loaded and used."""
    print("=" * 60)
    print("Reasoning Training Config Validation")
    print("=" * 60)
    
    config_path = os.path.join(script_dir, "gsm8k_grpo_reasoning_fast.yaml")
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    print(f"\n1. Loading config from: {config_path}")
    config = OmegaConf.load(config_path)
    print(f"   ✓ Config loaded")
    
    # Validate key settings
    print(f"\n2. Validating key settings...")
    checks = [
        ("experiment_name", config.experiment_name == "gsm8k-grpo-reasoning-fast"),
        ("training_mode", config.training_mode == "REASONING-FAST"),
        ("max_new_tokens", config.gconfig.max_new_tokens == 1024),
        ("max_length", config.train_dataset.max_length == 2048),
        ("reasoning_path", "reasoning" in config.train_dataset.path.lower()),
    ]
    
    all_ok = True
    for name, check in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {name}: {check}")
        if not check:
            all_ok = False
    
    # Test dataset loading with config
    print(f"\n3. Testing dataset loading with config...")
    try:
        tokenizer = load_hf_tokenizer(config.actor.path)
        dataset_config = DatasetConfig(**config.train_dataset)
        
        # Load a small sample
        dataset = get_custom_dataset(
            split="train",
            dataset_config=dataset_config,
            tokenizer=tokenizer
        )
        
        print(f"   ✓ Dataset loaded: {len(dataset)} samples")
        
        # Check first sample format
        first_sample = dataset[0]
        if "messages" in first_sample:
            messages = first_sample["messages"]
            if len(messages) >= 2:
                if messages[0]["role"] == "system" and "reasoning" in messages[0]["content"].lower():
                    print(f"   ✓ System prompt contains reasoning instruction")
                if messages[1]["role"] == "user":
                    print(f"   ✓ User message present")
        
        return all_ok
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run validation."""
    success = validate_reasoning_config()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Validation passed! Reasoning training config is ready.")
        print("\nTo start training, run:")
        print("  python3 -m areal.launcher.local examples/docker_gsm8k/gsm8k_grpo_train.py \\")
        print("    --config examples/docker_gsm8k/gsm8k_grpo_reasoning_fast.yaml \\")
        print("    experiment_name=gsm8k-grpo-reasoning-fast trial_name=trial0")
        return 0
    else:
        print("✗ Validation failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

