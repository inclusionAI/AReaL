# Windows 11 / NVIDIA GPU Compatibility

This document explains the changes made to ensure your training code works on Windows 11 with an NVIDIA 4080s GPU.

## Changes Made

### 1. Safe MPS Detection (`train_local_simple.py` & `train_hf_trainer.py`)
- **Issue**: Code checked `torch.backends.mps.is_available()` which doesn't exist on Windows and would crash
- **Fix**: Added safe MPS detection with try/except and `hasattr()` checks
- **Result**: Code now gracefully handles Windows/Linux systems where MPS is unavailable

### 2. MPS Cache Clearing (`train_local_simple.py`)
- **Issue**: `torch.mps.empty_cache()` would crash on Windows
- **Fix**: Added conditional check and safe error handling
- **Bonus**: Added CUDA cache clearing for Windows GPU setups

### 3. Device Auto-Detection (`train_hf_trainer.py`)
- **Issue**: Code was hardcoded to use CPU, ignoring available GPUs
- **Fix**: Implemented proper device auto-detection (CUDA > MPS > CPU priority)
- **Result**: Will now automatically use your NVIDIA 4080s GPU on Windows

### 4. Proper CUDA dtype Handling
- **Issue**: Mixed float16/bfloat16 logic wasn't CUDA-optimized
- **Fix**: Uses `bfloat16` on CUDA for better performance and numerical stability
- **Result**: Better training performance on NVIDIA GPUs

## Testing Your Setup

Run the compatibility test script first:

```powershell
python examples/local_gsm8k/test_windows_compatibility.py
```

This will verify:
- ✅ CUDA availability and GPU detection
- ✅ Device selection logic
- ✅ Basic tensor operations
- ✅ Cache management

## Running Training

### Option 1: Using train_local_simple.py (Manual Training Loop)

```powershell
python examples/local_gsm8k/train_local_simple.py ^
    --model Qwen/Qwen2.5-0.5B-Instruct ^
    --output-dir ./outputs/gsm8k-windows ^
    --batch-size 4 ^
    --gradient-accumulation-steps 8 ^
    --max-epochs 3 ^
    --max-length 512 ^
    --device cuda ^
    --wandb
```

### Option 2: Using train_hf_trainer.py (HuggingFace Trainer - Recommended)

```powershell
python examples/local_gsm8k/train_hf_trainer.py ^
    --model Qwen/Qwen2.5-0.5B-Instruct ^
    --output-dir ./outputs/gsm8k-windows ^
    --batch-size 2 ^
    --gradient-accumulation-steps 16 ^
    --num-epochs 3 ^
    --learning-rate 5e-5 ^
    --max-length 512 ^
    --max-samples 500 ^
    --device cuda ^
    --wandb
```

## Key Differences from macOS

1. **Device**: Uses `cuda` instead of `mps`
2. **dtype**: Uses `bfloat16` instead of `float16` (better for NVIDIA GPUs)
3. **Cache**: Uses `torch.cuda.empty_cache()` instead of `torch.mps.empty_cache()`
4. **Performance**: Should be faster than macOS MPS due to better CUDA optimization

## Expected Performance (NVIDIA RTX 4080s)

With a 16GB RTX 4080s, you should be able to:
- Use batch size of 2-4 (vs 1 on Mac)
- Train faster due to CUDA optimization
- Use longer sequences (512 tokens should work fine)

## Troubleshooting

### CUDA not detected?
1. Verify CUDA installation: `nvidia-smi`
2. Install PyTorch with CUDA: https://pytorch.org/get-started/locally/
3. Select: Windows, CUDA, and your CUDA version

### Out of Memory?
- Reduce `--batch-size` to 1 or 2
- Increase `--gradient-accumulation-steps` to maintain effective batch size
- Reduce `--max-length` to 256 or 128

### Import Errors?
- Ensure you're in the project root directory
- Install dependencies: `pip install -r examples/local_gsm8k/requirements.txt`

## Notes

- Shell scripts (`.sh` files) won't run directly on Windows PowerShell
- Use Python commands directly instead
- Path handling is cross-platform compatible (uses `pathlib`)
- All device checks are now safe for Windows


