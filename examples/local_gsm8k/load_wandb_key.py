#!/usr/bin/env python3
"""
Helper script to load W&B API key from local file and set it as an environment variable.
This ensures the key is not hardcoded in scripts or tracked by git.
"""

import os

def load_wandb_api_key():
    """Load W&B API key from wandb/api_key.txt file and set as environment variable."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(script_dir, "wandb", "api_key.txt")
    
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            key = f.read().strip()
        os.environ["WANDB_API_KEY"] = key
        return key
    else:
        # Fallback to environment variable if file doesn't exist
        existing_key = os.environ.get("WANDB_API_KEY")
        if existing_key:
            return existing_key
        raise FileNotFoundError(
            f"W&B API key file not found at {key_file}. "
            "Either create the file or set WANDB_API_KEY environment variable."
        )

if __name__ == "__main__":
    # When run directly, just load and print confirmation
    try:
        key = load_wandb_api_key()
        print("✓ W&B API key loaded successfully")
        print(f"  Key length: {len(key)} characters")
        print(f"  First 8 chars: {key[:8]}...")
    except Exception as e:
        print(f"✗ Error loading W&B API key: {e}")

