# Manual Model Download Guide

The automatic model download can be slow. Here are faster alternatives:

## Option 1: Use Python Script (Recommended)

```bash
source venv/bin/activate

# Download the model
python examples/local_gsm8k/download_model.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# Now use the local path
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800
```

## Option 2: Use HuggingFace CLI

```bash
source venv/bin/activate

# Install CLI if not already installed
pip install -U huggingface_hub[hf_transfer]

# Download model
huggingface-cli download \
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir-use-symlinks False

# Use it
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800
```

## Option 3: Use Git (Fastest for Large Models)

Git is often faster than direct download for large files:

```bash
# Install Git LFS if not already installed
# On Mac: brew install git-lfs
git lfs install

# Clone the model repository
cd models
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
cd ..

# Now use the local path
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800
```

## Option 4: Use HuggingFace Hub Repository

You can also download from the browser:

1. Go to: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
2. Click "Files" tab
3. Download all files manually
4. Create folder: `./models/DeepSeek-R1-Distill-Qwen-1.5B/`
5. Place all files there
6. Use the path in training script

## Finding Your HuggingFace Cache

The model is actually cached by transformers. You can check:

```bash
# Find where models are cached
python -c "from transformers import AutoTokenizer; import os; print(os.path.expanduser('~/.cache/huggingface/'))"
```

Models are typically in:
- `~/.cache/huggingface/hub/` (Linux/Mac)
- `C:\Users\<USER>\.cache\huggingface\hub\` (Windows)

## Using Cached Model

If the model is already downloaded (from a previous run), you can:

```bash
# Check if model is cached
ls ~/.cache/huggingface/hub/ | grep DeepSeek

# Use the cached model by model ID
python examples/local_gsm8k/train_local_simple.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800
```

It will use the cached version if available.

## Speeding Up HuggingFace Downloads

Enable faster transfers:

```bash
export HF_ENABLE_HUB_TRANSFER=1
pip install "huggingface_hub[hf_transfer]" --upgrade

# Now downloads use faster protocol
python examples/local_gsm8k/download_model.py
```

## Recommended Workflow

**Best approach for Mac:**

```bash
# 1. Enable fast transfer
export HF_ENABLE_HUB_TRANSFER=1
pip install "huggingface_hub[hf_transfer]" --upgrade

# 2. Download model with progress bar
python examples/local_gsm8k/download_model.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# 3. Train using local model
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800
```

This avoids re-downloading during training!

## Model Size

- **DeepSeek-R1-Distill-Qwen-1.5B**: ~3GB
- Download time: 5-10 minutes on good connection
- After download: Training starts immediately

## Alternative: Use Smaller Model for Testing

If you want to test faster, try a smaller model:

```bash
python examples/local_gsm8k/train_local_simple.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max-time 1800
```

This model is ~1GB and downloads much faster!

