from huggingface_hub import upload_folder
import os

# --- Configuration ---
# 1. Path to your local model files
local_dir = "/Path/to/your/local/model/folder"

# 2. Your Hugging Face username (REPLACE THIS!)
hf_username = "your-hf-username"

# 3. The name for your repo on Hugging Face
repo_name = "your-repo-name"
# --- End Configuration ---

# --- Upload ---
repo_id = f"{hf_username}/{repo_name}"
print(f"Uploading files from '{local_dir}' to '{repo_id}'...")

try:
    upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        # This will create the repo if it doesn't exist (requires write token)
        # Optional: Add a commit message
        # commit_message="Upload initial model checkpoint"
    )
    print("\n✅ Upload Complete!")
    print(f"View repository at: https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"\n❌ Upload Failed:")
    print(e)