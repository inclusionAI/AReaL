# Final Training Results Analysis

## Summary

| Model | Training Loss | Test Accuracy | Notes |
|-------|---------------|---------------|-------|
| **Base Model** | N/A (pretrained) | **40% (8/20)** | No finetuning |
| **Trained Model** | **0.37** (from 0.49) | **5% (1/20)** | ❌ Regression |

## Training Statistics

- **Dataset**: 1,500 GSM8K samples
- **Training Time**: 31 minutes
- **Final Loss**: 0.37 (improved from 0.49)
- **Epochs Completed**: 3
- **Steps**: ~423 total

## What Went Wrong

### The Problem: Format Mismatch

**Training Format (GSM8K):**
```
Question: How many eggs does Janet sell?
Answer: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs.
#### 9.
```

**Test Prompt:**
```
Question: How many eggs does Janet sell?
Please provide your final answer within \boxed{}.
```

**Issue**: Model learned GSM8K format (`<<...>>`), but test asks for `\boxed{}` format!

### Why Accuracy Dropped

1. Model learned GSM8K answer format during training
2. Test uses different prompt format (`\boxed{}`)
3. Model doesn't know how to respond to new format
4. Generates GSM8K format which doesn't match expected output

## Loss Function Explanation

### How Loss Masking Works

```python
# Training data structure:
input_ids = [q1, q2, q3, ..., a1, a2, a3, ..., EOS]
labels    = [-100, -100, ..., a1, a2, a3, ..., EOS]

# CrossEntropyLoss with ignore_index=-100
loss = mean( 
    -log(probability_of_correct_token) 
    for each position where label != -100
)

# Only answer tokens contribute to loss!
# Question tokens (label=-100) are ignored
```

### Loss Function Details

1. **Input**: Model gets full sequence `[question + answer]`
2. **Predictions**: Model outputs logits for each token position
3. **Labels**: `-100` for questions, actual token IDs for answers
4. **Loss**: Computed only on answer positions
5. **Backprop**: Gradients flow only for answer predictions

### Why It Decreased (0.49 → 0.37)

- Model learned to predict answer tokens correctly
- Training loss improved (lower is better)
- Model can now generate answer format it saw during training
- **BUT** our test uses a different format!

## Answer Extraction Process

### From Ground Truth
```python
# GSM8K answer format:
answer = "Janet sells ... <<computation>> ... #### 18."

# Extraction:
gt_answer = extract_last_number_after("####")  # = 18
```

### From Model Output
```python
# Model might generate:
completion = "Step 1: Janet sells ... \nStep 2: ... #### 18."

# OR (if format mismatch):
completion = "Janet sells ... <<16-3-4=9>> ..."  # No ####!

# Extraction tries:
pred_answer = extract_last_number_after("####")  # Fails if no ####
```

### Comparison
```python
# Compare as floats
is_correct = (abs(pred_answer - gt_answer) < 0.01)
```

## Key Insights

### What Worked ✅
1. **Training pipeline**: HuggingFace Trainer works correctly
2. **Loss masking**: `-100` labels properly ignored
3. **Loss improvement**: 0.49 → 0.37 shows learning
4. **Model generates**: Coherent step-by-step reasoning
5. **No NaN**: Stable training with gradient checkpointing

### What Didn't Work ❌
1. **Format mismatch**: Training/test format incompatible
2. **Limited data**: 1,500 samples not enough
3. **Short training**: 3 epochs too few
4. **Small model**: 0.5B parameters insufficient for complex math

## How to Fix

### Quick Fix: Use GSM8K Format in Testing
```python
# Change test script to match training format
prompt = f"{question}\n{answer}"  # Remove \boxed{} requirement
```

### Proper Fix: Align All Formats
1. Use consistent format across training and testing
2. Train on full 7,473 samples
3. More epochs (10+)
4. Larger model (1.5B+)
5. Better prompt engineering

## Training Pipeline Summary

```
┌─────────────────────────────────────────┐
│ 1. Load Dataset (GSM8K 1,500 samples)  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 2. Process: Format as question+answer  │
│    - Tokenize question and full text    │
│    - Create labels: -100 for question  │
│      tokens, real IDs for answer tokens │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 3. Forward Pass:                        │
│    - Model sees: [q1,q2,...,a1,a2,...] │
│    - Model outputs: logits for each     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 4. Loss Computation:                    │
│    - Only compute loss on answer tokens │
│    - Ignore -100 labels                 │
│    - Loss = negative log likelihood     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 5. Backward Pass:                        │
│    - Gradients only for answer positions │
│    - Update model weights               │
└─────────────────────────────────────────┘
```

## Conclusion

The training **successfully improved the loss** (0.49 → 0.37) and the model learned to generate GSM8K-format answers. However, the test results show a regression because of a **format mismatch** between training and testing.

The pipeline works correctly, but for production use, you need to:
1. Ensure consistent formats between training and evaluation
2. Use more training data
3. Train for longer
4. Consider a larger model

---

**Training completed**: 2024-10-26 14:54  
**W&B Run**: https://wandb.ai/tong-zhao-georgia-institute-of-technology/huggingface/runs/s8nhvvnx

