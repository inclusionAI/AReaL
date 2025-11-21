# Test Logs Guide - Accessing Results After Pod Stops

## Overview

Test logs are automatically saved to the **network volume** so you can access them **without starting the pod again**!

## Where Test Logs Are Saved

All test logs are saved to:
```
/workspace/outputs/grpo/test_logs/
```

This directory is on the **network volume** (`areal-outputs`), which means:
- ✅ Logs persist after the pod stops
- ✅ You can access them from RunPod dashboard
- ✅ You can download them without starting a new pod
- ✅ No need to keep the pod running just to read results

## Log File Names

### Reasoning Model Tests
- Format: `test_reasoning_YYYYMMDD_HHMMSS.log`
- Example: `test_reasoning_20251119_143022.log`

### Regular Model Tests
- Format: `test_model_YYYYMMDD_HHMMSS.log`
- Example: `test_model_20251119_143022.log`

## Accessing Logs

### Method 1: RunPod Dashboard (Easiest)

1. **Go to RunPod Dashboard**: https://www.runpod.io/console/volumes
2. **Find your volume**: `areal-outputs` (or whatever you named it)
3. **Click on the volume** to view contents
4. **Navigate to**: `grpo/test_logs/`
5. **Download** the log files you want

### Method 2: Start a Temporary Pod (If Needed)

If you need to access logs via command line:

1. **Deploy a small pod** (RTX 4090 or smaller)
2. **Mount the same volume**: `/workspace/outputs` → `areal-outputs`
3. **Access logs**:
   ```bash
   # Inside pod
   ls -lh /workspace/outputs/grpo/test_logs/
   cat /workspace/outputs/grpo/test_logs/test_reasoning_20251119_143022.log
   ```

### Method 3: Download via API (Advanced)

If RunPod provides API access, you can download logs programmatically.

## What's in the Logs

### Reasoning Model Test Logs

Each log file contains:
- Model path and configuration
- Test dataset size
- Per-sample results (question, generated answer, correct answer, correctness)
- Final accuracy percentage
- Detailed output for first few samples and incorrect answers

Example log structure:
```
============================================================
Testing REASONING model: /workspace/outputs/grpo/checkpoints/...
Log file: /workspace/outputs/grpo/test_logs/test_reasoning_20251119_143022.log
============================================================

Using device: cuda
Model loaded and set to eval mode
Testing on FULL dataset: 1319 samples

--- Sample 1 ---
Question: Janet's ducks lay 16 eggs per day...
Generated: <reasoning>...</reasoning><answer>16</answer>
Correct Answer: 16
Result: CORRECT

Progress: 10/1319 | Correct: 8/10 | Accuracy: 80.00%
...

============================================================
FINAL ACCURACY: 45.23% (596/1319)
Log saved to: /workspace/outputs/grpo/test_logs/test_reasoning_20251119_143022.log
============================================================
```

### Regular Model Test Logs

Each log file contains:
- Model configuration (experiment_name, trial_name)
- Test dataset size
- Evaluation statistics (reward stats, accuracy)
- Final accuracy percentage

Example log structure:
```
================================================================================
Testing model from config: gsm8k-grpo-cloud-1hour/trial0
Log file: /workspace/outputs/grpo/test_logs/test_model_20251119_143022.log
================================================================================

Evaluating on 1319 test samples...
Using model from: gsm8k-grpo-cloud-1hour/trial0

EVALUATION RESULTS
================================================================================
eval-rollout/task_reward:
  avg: 0.4523
  min: 0.0
  max: 1.0
  ...

================================================================================
ACCURACY: 45.23%
================================================================================
```

## Automatic Logging

Test logs are **automatically created** when:
1. Training completes successfully
2. The training script automatically runs the test script
3. Test results are saved to the network volume

You don't need to do anything - logs are saved automatically!

## Manual Testing

If you want to run tests manually (e.g., after pod restart):

### Reasoning Model
```bash
python examples/cloud_gsm8k/test_reasoning_model_cloud.py \
    --model-path /workspace/outputs/grpo/checkpoints/... \
    --all \
    --log-dir /workspace/outputs/grpo/test_logs
```

### Regular Model
```bash
python examples/cloud_gsm8k/test_trained_model_cloud.py \
    --config examples/cloud_gsm8k/gsm8k_grpo_1hour.yaml \
    --all
```

Note: Regular model tests automatically save to `/workspace/outputs/grpo/test_logs/` - no need to specify `--log-dir`.

## Best Practices

1. ✅ **Check logs after training** - They're automatically saved
2. ✅ **Download important logs** - Save them locally as backup
3. ✅ **Use timestamps** - Log files include timestamps for easy identification
4. ✅ **Compare logs** - Track accuracy improvements across training runs
5. ✅ **No need to keep pod running** - Logs persist in network volume

## Troubleshooting

### Logs Not Found

**Problem**: Can't find logs in `/workspace/outputs/grpo/test_logs/`

**Solutions**:
1. Check if test script ran successfully (look for "Test completed successfully!" message)
2. Verify network volume is mounted at `/workspace/outputs`
3. Check if test script has write permissions to the volume
4. Look for logs in the training script output (stdout/stderr)

### Logs Are Empty

**Problem**: Log file exists but is empty or incomplete

**Solutions**:
1. Test script may have crashed - check training script output
2. Disk space issue - check volume capacity
3. Permission issue - verify write permissions

### Want Different Log Location

**Problem**: Want to save logs to a different location

**Solutions**:
1. **Reasoning model**: Use `--log-dir` argument:
   ```bash
   python test_reasoning_model_cloud.py --log-dir /custom/path ...
   ```
2. **Regular model**: Modify the `log_dir` variable in `test_trained_model_cloud.py`

## Summary

- ✅ Test logs are **automatically saved** to network volume
- ✅ Location: `/workspace/outputs/grpo/test_logs/`
- ✅ Accessible **without starting pod** via RunPod dashboard
- ✅ Persist after pod stops
- ✅ Include timestamps for easy identification
- ✅ Contain full test results and accuracy metrics

No need to keep the pod running just to read test results! 🎉

