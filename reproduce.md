# Reproduce

Hang is not reproducible.

## Changes

`scripts/configs/math_grpo.yaml`:

```
stats_logger.wandb.mode: offline -> disabled
```

## Environments

- GPU: 8x H800
- CPU: 32 cores
- Memory: 64 GB
- Storage: 1TB local storage

## Experiment

Run with

```bash
python -m areal.launcher.local scripts/math_grpo.py --config scripts/configs/math_grpo.yaml trial_name=0806_debug
```

## Result

CUDA OOM may occur in the training process after several epochs:

```
[2025-08-06 15:24:55] Failed to update parameter online: CUDA out of memory. Tried to allocate 28.00 MiB. GPU 0 has a total capacity of 79.11 GiB of which 5.19 MiB is free. Process 289367 has 79.10 GiB memory in use. Of the allocated memory 76.91 GiB is allocated by PyTorch, with 61.70 MiB allocated in private pools (e.g., CUDA Graphs), and 494.83 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables). The full weights of the ModelRunner are partially updated. Please discard the whole weights.
```
