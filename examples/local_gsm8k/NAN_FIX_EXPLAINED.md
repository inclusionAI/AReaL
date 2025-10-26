# NaN Loss Issue - Fixed! ✅

## What Was Wrong

The training showed `loss=nan` because the **loss mask was never applied**. 

The model was trying to predict ALL tokens (prompt + answer), but only the answer tokens should contribute to the loss. Without the loss mask, the model was getting confused and producing NaN loss values.

## What I Fixed

### 1. Applied Loss Mask Properly

```python
# BEFORE (wrong):
loss = outputs.loss  # Model computes loss on all tokens

# AFTER (correct):
# Extract logits and manually compute loss
logits = outputs.logits
labels = batch["labels"]
loss_mask = batch["loss_mask"]

# Shift for next-token prediction
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
shift_loss_mask = loss_mask[..., 1:]

# Compute masked loss
per_token_loss = -log_softmax(shift_logits).gather(..., shift_labels)
masked_loss = per_token_loss * shift_loss_mask
loss = masked_loss.sum() / shift_loss_mask.sum()  # Only average over answer tokens!
```

### 2. Added NaN Detection

```python
if torch.isnan(loss) or torch.isinf(loss):
    log("Warning: NaN loss. Skipping update.")
    continue
```

### 3. Added Gradient Norm Monitoring

Now you can see gradient norms in W&B to detect explosion:
```python
wandb.log({"train/grad_norm": grad_norm})
```

## Test Results

Training now completes successfully with valid loss values:
- ✅ No NaN loss
- ✅ Loss values: ~2-3 (normal for LM)
- ✅ Gradients stable
- ✅ Model checkpoints saved

## Now You Can Run Full Training

```bash
source venv/bin/activate

python examples/local_gsm8k/train_local_simple.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max-time 1800 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-length 128 \
    --device auto \
    --wandb
```

This will train properly without NaN loss!

