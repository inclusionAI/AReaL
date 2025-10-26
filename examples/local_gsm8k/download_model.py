#!/usr/bin/env python3
"""
Download model manually to speed up training setup.
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login


def download_model(model_id: str, local_dir: str = None):
    """Download model from HuggingFace."""
    
    if local_dir is None:
        model_name = model_id.split("/")[-1]
        local_dir = f"./models/{model_name}"
    
    local_path = Path(local_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_id} to {local_dir}...")
    print("This may take a while (model is ~3GB)...")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
        )
        print(f"\n✅ Model downloaded successfully!")
        print(f"Location: {local_path.absolute()}")
        print(f"\nYou can now use it with:")
        print(f"  python examples/local_gsm8k/train_local_simple.py \\")
        print(f"    --model {local_path.absolute()}")
        return local_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model ID from HuggingFace",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory to save model (default: ./models/MODEL_NAME)",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Login to HuggingFace (required for private repos)",
    )
    
    args = parser.parse_args()
    
    # Login if requested
    if args.login:
        print("Please login to HuggingFace:")
        login()
    
    # Download model
    download_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()

