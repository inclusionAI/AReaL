# Parallel Thinking Model Fine-tuning with LLaMA Factory

This directory contains all the code needed to fine-tune a Llama-3-8B-Instruct model using LLaMA Factory to learn parallel thinking reasoning patterns from your JSONL data.

## Files Overview

- `train_parallel_thinking.py` - Main training script that handles the complete pipeline
- `data_converter.py` - Converts your JSONL format to ShareGPT format for LLaMA Factory
- `train_config.yaml` - Training configuration with all hyperparameters
- `test_model.py` - Script to test the fine-tuned model
- `requirements.txt` - Required dependencies

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the complete training pipeline:**
```bash
python train_parallel_thinking.py
```

This will:
- Convert your JSONL data to the correct format
- Setup LLaMA Factory if not already installed
- Start the fine-tuning process
- Create a test script for evaluation

3. **Test your fine-tuned model:**
```bash
python test_model.py
```

## Data Format

The script converts your JSONL data from:
```json
{"original_problem": "question text", "main_thread": "detailed reasoning"}
```

To ShareGPT format:
```json
{
  "conversations": [
    {"from": "human", "value": "Question: question text"},
    {"from": "gpt", "value": "detailed reasoning"}
  ]
}
```

## Training Configuration

Key parameters in `train_config.yaml`:
- **Model**: Llama-3-8B-Instruct
- **Method**: LoRA fine-tuning (rank=8, alpha=16)
- **Batch size**: 2 per device with 4 gradient accumulation steps
- **Learning rate**: 5e-5 with cosine scheduler
- **Epochs**: 3
- **Max sequence length**: 2048 tokens

## Output

The fine-tuned model will be saved to:
- `LLaMA-Factory/saves/llama3-parallel-thinking/` (LoRA adapter)

## Usage Notes

- Adjust `CUDA_VISIBLE_DEVICES` in the script based on your GPU setup
- Modify batch size and other parameters based on your GPU memory
- The training uses 4-bit quantization to reduce memory usage
- Model supports both merged inference and adapter-based inference

## Monitoring

Training logs will show:
- Loss progression
- Learning rate schedule
- Memory usage
- Training speed

You can monitor GPU usage with `nvidia-smi` during training.
