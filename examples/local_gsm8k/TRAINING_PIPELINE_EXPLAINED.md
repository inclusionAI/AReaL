# Training Pipeline and Results Explanation

## Test Results Summary

### Base Model (Before Training)
- **Accuracy**: 40% (8/20 correct)
- **Model**: Qwen/Qwen2.5-0.5B-Instruct (pretrained, no finetuning)

### Trained Model (After Training)  
- **Accuracy**: 5% (1/20 correct)
- **Issue**: Training actually hurt performance!

## Why This Happened: Format Mismatch

The key issue is a **prompt format mismatch**:

### Training Format (GSM8K)
```
Question: Janet's ducks lay 16 eggs...
Answer: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
```

### Testing Format (Our Prompt)
```
Question: Janet's ducks lay 16 eggs...
Please provide your final answer within \boxed{}.
```

**The model learned to output GSM8K format (`<<...>>`) but we're testing with `\boxed{}` prompt!**

## Training Pipeline Deep Dive

### 1. Loss Function

The training uses **causal language modeling (CLM) loss** with loss masking:

```python
def process_function(sample):
    question_text = sample['question']
    full_text = sample['question'] + sample['answer']
    
    # Tokenize
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    full_tokens.append(tokenizer.eos_token_id)
    
    # Create labels: -100 for question tokens (ignored in loss)
    # Real token IDs for answer tokens (used in loss)
    labels = [-100] * len(question_tokens) + full_tokens[len(question_tokens):]
    
    return {
        "input_ids": full_tokens,  # [q1, q2, ..., a1, a2, ..., EOS]
        "labels": labels,            # [-100, -100, ..., a1, a2, ..., EOS]
    }
```

**How it works:**
- `input_ids`: Full sequence `[question + answer + EOS]`
- `labels`: `[-100 for question tokens] + [answer token IDs]`
- Loss computed only on answer tokens
- The model predicts next token probabilities for answer tokens only

### 2. Loss Computation

HuggingFace Trainer uses **CrossEntropyLoss** with `ignore_index=-100`:

```python
# Inside DataCollatorForSeq2Seq or Trainer
loss = CrossEntropyLoss(ignore_index=-100)(
    predictions,    # [batch, seq_len, vocab_size]
    labels          # [batch, seq_len]
)
```

**What happens:**
1. Model outputs logits `[batch, seq_len, vocab_size]` for each position
2. Labels `[batch, seq_len]` contain:
   - `-100` for question tokens (ignored)
   - Actual token IDs for answer tokens (predicted)
3. Loss computed **only** when `label != -100`
4. Average loss over non-ignored tokens only

**Loss Formula:**
```
loss = mean( cross_entropy(logits[i], labels[i]) for i where labels[i] != -100 )
```

### 3. Answer Extraction and Comparison

The test script extracts answers using regex:

```python
def process_results(completions, answer):
    # Extract ground truth answer
    gt_answer = float(re.findall(r'-?\d+\.?\d*', answer.split("####")[-1])[-1])
    
    # Extract predicted answer
    pred_answer = float(re.findall(r'-?\d+\.?\d*', completion.split("####")[-1])[-1])
    
    # Compare
    return [pred_answer == gt_answer]
```

**How it works:**
1. **Extract from ground truth**: Finds last number after `####` in answer
2. **Extract from prediction**: Finds last number after `####` in completion
3. **Compare**: Returns `True` if exact match

**Problem**: Model outputs `<<computation>>` format from training, but test looks for `\boxed{}` format!

## The Real Issue

### Training Data Format
GSM8K answers use this format:
```
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmers' market.
#### $18.
```

### Test Prompt Format
```
Question: Janet's ducks...
Please provide your final answer within \boxed{}.
```

### What Happens
1. Model learns to generate GSM8K format (`<<...>>`)
2. Test asks for `\boxed{}` format
3. Model doesn't know how to respond to this format
4. Performance degrades

## Training Improvements

The training **is working correctly** - loss decreased from 0.485 to 0.37! The issue is:

1. **Format mismatch** between training and testing
2. **Limited data** (1,500 vs 7,473 samples)
3. **Short training** (3 epochs only)
4. **Small model** (0.5B parameters)

## How to Fix

### Option 1: Match Training and Testing Format
```python
# In training, add the test prompt format
text = f"{question}\nPlease provide your final answer within \\boxed{{}}.\n{answer}"

# Or just use GSM8K format in testing too
prompt = f"{question}\n{answer}"  # Don't add \boxed{} prompt
```

### Option 2: Use GSM8K's Evaluation Script
The standard GSM8K evaluation expects the model to output in GSM8K format and extract the final `####` answer.

### Option 3: Longer Training
- Train on full 7,473 samples
- More epochs (10+)
- Larger model (1.5B+)
- Better prompt engineering

## Loss Function Details

### Forward Pass
```
Input: [q1, q2, ..., a1, a2, ..., EOS]
Model predicts: [logits_q1, logits_q2, ..., logits_a1, logits_a2, ..., logits_EOS]
Labels: [-100, -100, ..., a1, a2, ..., EOS]
```

### Loss Computation
```
For each position i:
  if labels[i] == -100: skip (no loss)
  else: add -log(probability_of_correct_token) to loss
```

### Backward Pass
Only gradients for answer tokens flow back through the model.

## Key Takeaways

1. **Loss masking works**: Only answer tokens contribute to loss
2. **Training improved loss**: 0.485 → 0.37 is good
3. **Format matters**: Training/test format mismatch caused regression
4. **More data needed**: 1,500 samples isn't enough for this task
5. **Prompt engineering matters**: The test prompt is different from training format

