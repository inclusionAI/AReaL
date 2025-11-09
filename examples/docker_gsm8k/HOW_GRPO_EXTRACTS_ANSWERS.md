# How GRPO Extracts Answers and Learns Format

## Answer Extraction Process

### Step 1: Model Generates Completion

The model generates a completion based on the prompt. For GSM8K, the prompt is:

```
"Janet has 16 eggs. She sells 3 and breaks 4. How many left?
Please put your final answer within \boxed{}."
```

The model might generate various formats:
- `"16 - 3 - 4 = 9. \boxed{9}"` ✅ (follows instruction)
- `"The answer is 9"` ✅ (extractor can handle this)
- `"I calculate: 16 - 3 = 13, 13 - 4 = 9. So 9 eggs remain."` ✅ (extractor finds last number)
- `"I don't know"` ❌ (no number found)

### Step 2: Answer Extraction (Format-Agnostic!)

The `extract_answer()` function in `areal/reward/math_parser.py` is **very flexible**. It tries multiple strategies:

```python
def extract_answer(pred_str, data_name, use_last_number=True):
    # Strategy 1: Look for \boxed{...}
    if "boxed" in pred_str:
        # Extract content from \boxed{answer}
        ...
    
    # Strategy 2: Look for "the answer is"
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    
    # Strategy 3: Look for "final answer is"
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    
    # Strategy 4: Fallback - extract last number
    else:
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]  # Take the last number
```

**Key Point**: The extractor doesn't require a specific format! It can extract answers from:
- `\boxed{9}`
- `The answer is 9`
- `Final answer: 9`
- `... 9` (just finds the last number)
- `9.` (with period)
- `Answer: 9 eggs` (extracts number)

### Step 3: Mathematical Comparison

The extracted answer is compared to the ground truth using `math_equal()`:

```python
def math_equal(prediction, reference, ...):
    # 1. Direct string match
    if str(prediction) == str(reference):
        return True
    
    # 2. Numerical comparison (handles 9.0 == 9)
    if is_digit(prediction) and is_digit(reference):
        return numeric_equal(float(prediction), float(reference))
    
    # 3. Symbolic comparison (handles "1/2" == "0.5")
    return symbolic_equal(prediction, reference)
```

This is **very robust** - it handles:
- `9` == `9.0` == `9.00`
- `1/2` == `0.5`
- `"9"` == `9`
- Fractions, decimals, etc.

## How the Model Learns the Format

### The Prompt Instructs the Model

Looking at `areal/dataset/gsm8k.py`:

```python
def get_gsm8k_rl_dataset(...):
    def process(sample):
        messages = [{
            "role": "user",
            "content": sample["question"] + "\nPlease put your final answer within \\boxed{}."
        }]
        return {"messages": messages}
```

**The prompt explicitly asks**: `"Please put your final answer within \boxed{}."`

### Learning Process

1. **Initial State (Base Model)**:
   - Model sees the instruction: `"Please put your final answer within \boxed{}."`
   - May or may not follow it initially
   - Generates various formats: `"The answer is 9"`, `"9"`, `"\boxed{9}"`, etc.

2. **Reward Signal**:
   - If answer is correct → reward = 1
   - If answer is wrong → reward = 0
   - **The extractor finds the answer regardless of format**

3. **GRPO Learning**:
   - Model generates 4 solutions per problem
   - Compares them relatively
   - **Learns to produce answers that:**
     - Are mathematically correct (primary goal)
     - Can be extracted by the parser (secondary, but helpful)
     - Follow the instruction format (if it helps extraction)

4. **Why It Works Without SFT**:
   - **Instruction following**: Pre-trained models (like Qwen) are already trained to follow instructions
   - **Flexible extractor**: Doesn't require perfect format
   - **RL learning**: Model learns through trial and error which formats work

## Example: Learning Progression

### Early Training (Base Model)

**Problem**: "Janet has 16 eggs. She sells 3 and breaks 4. How many left?"

**Solution 1**: `"16 - 3 - 4 = 9. The answer is 9."`
- Extractor finds: `9` ✅
- Ground truth: `9`
- Reward: `1` (correct!)

**Solution 2**: `"I think it's 8"`
- Extractor finds: `8` ✅
- Ground truth: `9`
- Reward: `0` (wrong)

**Solution 3**: `"Let me calculate... 16 minus 3 is 13, minus 4 is 9. So 9."`
- Extractor finds: `9` (last number) ✅
- Ground truth: `9`
- Reward: `1` (correct!)

### After Training

**Solution 1**: `"16 - 3 - 4 = 9. \boxed{9}"` ✅
- Follows instruction
- Extractor finds: `9`
- Reward: `1`

**Solution 2**: `"16 - 3 = 13, 13 - 4 = 9. \boxed{9}"` ✅
- Follows instruction
- Extractor finds: `9`
- Reward: `1`

The model learns to:
1. ✅ Produce correct answers (primary)
2. ✅ Follow the `\boxed{}` format (secondary, but reinforced)

## Why This Works Without SFT

### 1. **Instruction Following Capability**

Pre-trained models like Qwen are already trained on instruction-following data. When they see:
```
"Please put your final answer within \boxed{}."
```

They have a good chance of following it, even without fine-tuning.

### 2. **Flexible Answer Extraction**

The `extract_answer()` function is **format-agnostic**:
- Doesn't require `\boxed{}`
- Can extract from natural language
- Falls back to finding the last number

So even if the model doesn't follow the format perfectly, the extractor can still find the answer.

### 3. **RL Learning Signal**

Through GRPO:
- Correct answers → positive reward → model learns to produce them
- Wrong answers → negative reward → model learns to avoid them
- Over time, the model learns which formats are most reliable

### 4. **Multiple Samples Per Problem**

GRPO generates 4 solutions per problem:
- Even if some formats are unclear, at least one might be extractable
- Model learns which formats work best through relative comparison

## The Answer Extraction Code

Here's the actual extraction logic:

```python
# From areal/reward/math_parser.py

def extract_answer(pred_str, data_name, use_last_number=True):
    # Try to find \boxed{answer}
    if "boxed" in pred_str:
        # Extract from \boxed{9} or \boxed{answer}
        ...
    
    # Try to find "the answer is"
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    
    # Try to find "final answer is"
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    
    # Fallback: extract last number
    else:
        pattern = "-?\d*\.?\d+"
        numbers = re.findall(pattern, pred_str.replace(",", ""))
        if numbers:
            pred = numbers[-1]  # Last number in the text
```

**Key Insight**: The extractor is **robust and format-agnostic**. It doesn't require the model to use a specific format - it can extract answers from many different formats.

## Summary

### How GRPO Extracts Answers

1. **Flexible Extractor**: `extract_answer()` tries multiple strategies:
   - `\boxed{answer}` format
   - "the answer is" format
   - "final answer is" format
   - **Fallback**: Extract last number in text

2. **Robust Comparison**: `math_equal()` handles:
   - Numerical equality (9 == 9.0)
   - Symbolic equality (1/2 == 0.5)
   - String matching

### How Model Learns Format Without SFT

1. **Prompt Instruction**: Explicitly asks for `\boxed{}` format
2. **Pre-trained Capability**: Base models can follow instructions
3. **Flexible Extractor**: Works even if format isn't perfect
4. **RL Learning**: Model learns through rewards which formats work best
5. **Multiple Samples**: 4 solutions per problem increases chance of extractable answer

**The model doesn't need SFT because:**
- ✅ The extractor is flexible (doesn't require perfect format)
- ✅ Pre-trained models can follow instructions
- ✅ RL training teaches the model to produce extractable answers
- ✅ The format is learned implicitly through reward signals

The beauty of GRPO is that it's **format-agnostic** - it works with whatever format the model produces, as long as it contains the correct answer!

