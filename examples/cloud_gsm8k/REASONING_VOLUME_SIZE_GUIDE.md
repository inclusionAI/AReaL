# Reasoning Model Volume Size Guide for RunPod

## Overview

This guide provides recommended network volume sizes for reasoning model training on RunPod. Reasoning models generate longer outputs and may require more storage than regular GRPO training.

## Storage Components

### What Gets Saved to Volume:

1. **Model Checkpoints** (largest component)
   - Per epoch checkpoints
   - Actor model weights (~1GB per checkpoint for Qwen2.5-0.5B)
   - Reference model weights (if saved separately)
   - Optimizer states

2. **Training Logs**
   - Training statistics
   - Rollout traces (if enabled)
   - System logs

3. **Test Logs**
   - Full validation test results
   - Per-sample outputs (longer for reasoning models)
   - Accuracy metrics

4. **Generated Samples**
   - Rollout outputs during training
   - Longer reasoning chains (up to 1024 tokens vs 512 for regular)

5. **Dataset Cache** (if cached locally)
   - HuggingFace dataset cache
   - Tokenized data cache

## Recommended Volume Sizes by Training Config

### Reasoning Fast (200 samples, 1 epoch)
- **Recommended**: **30GB**
- **Minimum**: 20GB
- **Breakdown**:
  - Checkpoints: ~1-2GB (1 epoch)
  - Logs: ~1-2GB
  - Generated samples: ~2-3GB
  - Safety margin: ~5GB

### Reasoning 1-Hour (500 samples, 2 epochs)
- **Recommended**: **40GB**
- **Minimum**: 30GB
- **Breakdown**:
  - Checkpoints: ~2-3GB (2 epochs)
  - Logs: ~2-3GB
  - Generated samples: ~3-5GB
  - Safety margin: ~5GB

### Reasoning 3-Hour (1000 samples, 3 epochs)
- **Recommended**: **50GB**
- **Minimum**: 40GB
- **Breakdown**:
  - Checkpoints: ~3-4GB (3 epochs)
  - Logs: ~3-4GB
  - Generated samples: ~5-8GB
  - Safety margin: ~5GB

### Reasoning 2000 Samples 4 GPUs (2000 samples, 3 epochs)
- **Recommended**: **60GB**
- **Minimum**: 50GB
- **Breakdown**:
  - Checkpoints: ~4-6GB (3 epochs, multi-GPU may save more)
  - Logs: ~4-5GB (more extensive with multi-GPU)
  - Generated samples: ~8-12GB (longer reasoning chains)
  - Safety margin: ~10GB

## General Recommendations

### For Single Training Run
- **Fast/1-hour configs**: 30-40GB
- **3-hour config**: 50GB
- **Multi-GPU configs**: 60GB

### For Multiple Training Runs
If you plan to run multiple experiments without cleaning up:

- **2-3 runs**: 100GB
- **5+ runs**: 150-200GB

### For Production/Long-term Use
- **Recommended**: 100-150GB
- Allows multiple experiments
- Room for model downloads
- Dataset caching
- Backup checkpoints

## Why Reasoning Models Need More Space

1. **Longer Outputs**: Reasoning models generate up to 1024 tokens (vs 512 for regular)
   - Generated samples are ~2x larger
   - Test logs contain more text per sample

2. **More Detailed Logs**: Reasoning model logs include:
   - Full reasoning chains
   - Step-by-step outputs
   - XML format parsing results

3. **Multiple Checkpoints**: Training typically saves per-epoch checkpoints
   - 3 epochs = 3 checkpoints
   - Each checkpoint ~1GB for Qwen2.5-0.5B

## Space Calculation Formula

```
Total Space = (Checkpoints × Epochs × 1GB) + 
              (Logs × 2GB) + 
              (Generated Samples × Samples × 0.01GB) + 
              (Safety Margin × 10GB)
```

**Example for 2000 samples, 3 epochs:**
```
= (1GB × 3) + (2GB × 2) + (0.01GB × 2000) + 10GB
= 3GB + 4GB + 20GB + 10GB
= ~37GB (minimum)
+ Safety margin = ~50-60GB recommended
```

## Monitoring Volume Usage

### Check Volume Usage
```bash
# Inside pod
df -h /workspace/outputs

# Check specific directories
du -sh /workspace/outputs/grpo/checkpoints/*
du -sh /workspace/outputs/grpo/logs/*
du -sh /workspace/outputs/grpo/test_logs/*
```

### Clean Up Old Checkpoints
```bash
# Remove old checkpoints (keep only latest)
# WARNING: Only do this if you're sure you don't need them!
rm -rf /workspace/outputs/grpo/checkpoints/old_experiment_name/
```

## Upgrading Volume Size

If you run out of space:

1. **RunPod Dashboard** → Volumes
2. **Select your volume** → Edit/Resize
3. **Increase size** (RunPod allows resizing)
4. **Wait for resize** to complete
5. **Continue training**

**Note**: RunPod volumes can be resized, but it's better to start with adequate size to avoid interruptions.

## Best Practices

1. ✅ **Start with 50GB** for most use cases (covers most configs)
2. ✅ **Use 60GB** for multi-GPU training
3. ✅ **Use 100GB+** if running multiple experiments
4. ✅ **Monitor usage** during training
5. ✅ **Clean up old experiments** periodically
6. ✅ **Download important checkpoints** to local backup
7. ✅ **Use cloud storage** (S3/GCS) for long-term archival

## Cost Considerations

RunPod volume pricing (approximate):
- **50GB**: ~$0.50/month
- **100GB**: ~$1.00/month
- **200GB**: ~$2.00/month

**Recommendation**: The cost difference between 50GB and 100GB is minimal (~$0.50/month), so it's worth getting 100GB for flexibility.

## Summary Table

| Training Config | Samples | Epochs | Recommended Size | Minimum Size |
|----------------|---------|--------|------------------|--------------|
| Reasoning Fast | 200 | 1 | **30GB** | 20GB |
| Reasoning 1-Hour | 500 | 2 | **40GB** | 30GB |
| Reasoning 3-Hour | 1000 | 3 | **50GB** | 40GB |
| Reasoning 4 GPUs | 2000 | 3 | **60GB** | 50GB |
| Multiple Runs | - | - | **100GB+** | 80GB |

**Most Common Recommendation**: **50-60GB** covers all single-run scenarios with safety margin.

