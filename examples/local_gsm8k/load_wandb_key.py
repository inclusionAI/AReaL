#!/usr/bin/env python3
"""
Helper script to load W&B API key from local file and set it as an environment variable.
This ensures the key is not hardcoded in scripts or tracked by git.
"""

import os

def load_wandb_api_key():
    """
    Load W&B API key from wandb/.wandb_api_key file in repository root and set as environment variable.
    
    This uses the same location as the Docker training scripts for consistency.
    The key file is located at: <repo_root>/wandb/.wandb_api_key
    """
    # Get repository root (two levels up from examples/local_gsm8k/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    key_file = os.path.join(project_root, "wandb", ".wandb_api_key")
    
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            key = f.read().strip()
        # Trim whitespace (leading/trailing) and newlines
        key = key.strip().replace('\r', '').replace('\n', '')
        os.environ["WANDB_API_KEY"] = key
        return key
    else:
        # Fallback to environment variable if file doesn't exist
        existing_key = os.environ.get("WANDB_API_KEY")
        if existing_key:
            # Trim whitespace from existing env var too
            existing_key = existing_key.strip().replace('\r', '').replace('\n', '')
            os.environ["WANDB_API_KEY"] = existing_key
            return existing_key
        raise FileNotFoundError(
            f"W&B API key file not found at {key_file}. "
            "Either create the file or set WANDB_API_KEY environment variable."
        )

if __name__ == "__main__":
    # When run directly, just load and print confirmation
    try:
        key = load_wandb_api_key()
        print("[OK] W&B API key loaded successfully")
        print(f"  Key length: {len(key)} characters")
        print(f"  First 8 chars: {key[:8]}...")
    except Exception as e:
        print(f"[ERROR] Error loading W&B API key: {e}")

