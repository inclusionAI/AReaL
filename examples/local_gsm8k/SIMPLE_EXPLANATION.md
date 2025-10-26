# Simple Explanation: Training Pipeline and Answer Extraction

## Training Pipeline Overview

### 1. The Loss Function (Causal Language Modeling)

Think of it like a **"fill in the blank"** game where:
- You show the model: `"Question: X. Answer: Y"`
- You only want it to learn to predict the **answer part** (Y)
- You don't care if it predicts the question (X) correctly

**How we do this:**
```python
# Full sequence
text = "Question: 2+2=? Answer: 2+2=4"

# Labels for loss
labels = [-100, -100, -100, 4]  # Ignore question, predict answer "4"
```

The `-100` means "ignore this in loss calculation". So:
- Model sees: `"Question: 2+2=? Answer: ..."`
- We only compute loss on the answer part
- Model learns: "Given this question, predict this answer"

### 2. Training Process

```
┌─────────────────────────────────┐
│  1. Input: Question + Answer    │
│     "Q: 2+2=? A: 2+2=4"         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  2. Label question tokens as    │
│     -100 (ignore in loss)       │
│     [-100, -100, -100, 4]       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  3. Model predicts next token   │
│     for each position            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  4. Loss = -log(prob of correct)│
│     Only computed on answer part│
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  5. Update model weights        │
│     Loss: 0.49 → 0.37 ✓         │
└─────────────────────────────────┘
```

### 3. Answer Extraction and Comparison

When testing, we need to extract the numerical answer from both:
- Ground truth answer
- Model's generated answer

**From Ground Truth:**
```python
answer = "Janet sells ... <<computation>> ... #### 18."
# Extract last number after "####"
gt_answer = 18.0
```

**From Model Output:**
```python
completion = "Janet sells ... #### 18."
# Extract last number after "####"
pred_answer = 18.0
```

**Compare:**
```python
is_correct = (abs(pred_answer - gt_answer) < 0.01)
# 18.0 == 18.0 → True ✓
```

## Why Accuracy Dropped

### The Format Mismatch

**Training showed model:**
```
Q: How many eggs?
A: Janet sells 16-3-4=<<16-3-4=9>>9 eggs. #### 9.
```

**Test asks model:**
```
Q: How many eggs?
Please provide your final answer within \boxed{}.
```

**Model responds with:**
```
Step 1: Janet sells...
Step 2: ... =<<16-3-4=9>>9 eggs
#### 9
```

**But our test script looks for:**
```
... \boxed{9}
```

**Result**: Format mismatch! Model learned GSM8K format, but test expects different format.

## Key Concept: Loss Masking

The magic is in the `-100` labels:

```python
# Input sequence: "What is 2+2? The answer is 4."
#                q1  q2 q3 q4? a1 a2 a3 a4  4 .

# Labels for loss:
labels =  [-100,-100,-100,-100,-100,-100,-100,-100,-100, 4, -100]
#          ^^^^^^ question part ^^^^^^^^^  ^^^^^^^^^  pred final
#          Ignore in loss              |answer tokens| 
```

Only the `4` label contributes to loss. The model learns:
- Given question tokens, predict answer tokens
- Don't worry about reconstructing the question

## Why Training "Worked" (Loss Improved)

1. **Loss went down**: 0.49 → 0.37 ✓
2. **Model learned**: To predict GSM8K format answers
3. **Generated text**: Coherent step-by-step reasoning
4. **BUT**: Wrong format for our test!

## Conclusion

The training pipeline works correctly:
- ✅ Loss masking works (`-100` labels)
- ✅ Only answer tokens trained
- ✅ Loss decreased (model learning)
- ❌ Format mismatch caused test failure

**The issue isn't the training algorithm—it's the format mismatch between training and testing!**

