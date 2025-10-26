# Fast Model Download Guide

## Quick Option: Just Run This

```bash
source venv/bin/activate

# Download model (takes 5-10 minutes)
python examples/local_gsm8k/download_model.py

# Then train
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800
```

## Alternative: Use Your Browser

If you prefer to download from browser:

### Step 1: Download Files

Go to: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

Click the "Files" tab and download ALL files (especially the large .safetensors files).

### Step 2: Create Directory

```bash
mkdir -p ./models/DeepSeek-R1-Distill-Qwen-1.5B
```

### Step 3: Move Files

Place all downloaded files into `./models/DeepSeek-R1-Distill-Qwen-1.5B/`

### Step 4: Train

```bash
python examples/local_gsm8k/train_local_simple.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800
```

## Files You Need to Download

Minimum required files:
- `config.json`
- `tokenizer_config.json`
- `vocab.json` (or `merges.txt` for BPE)
- `model.safetensors` or `pytorch_model.bin`
- `generation_config.json`

Just download everything in the Files tab.

## Use Cache Location

Or find where it's already downloaded:

```bash
# Find cache
python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B').init_kwargs)"

# Copy to local directory
cp -r ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*/ ./models/DeepSeek-R1-Distill-Qwen-1.5B/
```

## Simplest Approach

Actually, the **easiest** is to just start training - if the model is downloading slowly, let it run in the background and take a coffee break! ☕

The script will cache it for future use.

```bash
# This will download automatically (first time only)
python examples/local_gsm8k/train_local_simple.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-time 1800

# Next time it will use the cache
```

