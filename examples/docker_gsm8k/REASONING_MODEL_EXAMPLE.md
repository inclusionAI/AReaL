# Reasoning Model Training Example

This document shows a concrete example of what the reasoning model is encouraged to produce.

## Example Problem

**Question from GSM8K:**
```
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
```

**Ground Truth Answer:** `72`

---

## Original GRPO Model Output

### Input Prompt:
```
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Please put your final answer within \boxed{}.
```

### Typical Model Output (Short, Direct):
```
In April: 48 clips
In May: 48 / 2 = 24 clips
Total: 48 + 24 = 72 clips
\boxed{72}
```

**Characteristics:**
- ✅ Correct answer: `72`
- ✅ Uses `\boxed{}` format
- ⚠️ Minimal reasoning (just calculations)
- ⚠️ No structured thinking process
- ⚠️ May skip steps or be unclear

### Alternative Output (Even Shorter):
```
48 + 24 = 72
\boxed{72}
```

**Characteristics:**
- ✅ Correct answer
- ❌ No explanation of where "24" came from
- ❌ Not educational or explainable

---

## Reasoning Model Output (What We're Encouraging)

### Input Prompt:
**System Message:**
```
Respond in the following format:
<reasoning>
[Your step-by-step reasoning process here]
</reasoning>
<answer>
[Your final numerical answer here]
</answer>
```

**User Message:**
```
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
```

### Encouraged Model Output (Structured Reasoning):
```
<reasoning>
Let me break down this problem step by step.

Step 1: Identify what we know
- Natalia sold clips to 48 friends in April
- In May, she sold half as many clips as in April

Step 2: Calculate May's sales
- May sales = April sales / 2
- May sales = 48 / 2
- May sales = 24 clips

Step 3: Calculate total sales
- Total = April sales + May sales
- Total = 48 + 24
- Total = 72 clips

Step 4: Verify the answer
- April: 48 clips
- May: 24 clips (which is indeed half of 48)
- Total: 48 + 24 = 72 ✓
</reasoning>
<answer>
72
</answer>
```

**Characteristics:**
- ✅ Correct answer: `72`
- ✅ Uses XML format (`<reasoning>` and `<answer>`)
- ✅ **Structured step-by-step reasoning**
- ✅ **Clear problem-solving process**
- ✅ **Self-verification included**
- ✅ **Educational and explainable**

---

## Why This Matters

### 1. **Transparency**
The reasoning model shows **how** it arrived at the answer, not just **what** the answer is.

### 2. **Debugging**
If the answer is wrong, you can see exactly where the reasoning went wrong:
```
<reasoning>
Step 1: April = 48 clips
Step 2: May = 48 / 2 = 24 clips
Step 3: Total = 48 + 24 = 72 clips
</reasoning>
<answer>
72
</answer>
```
vs. a wrong answer with reasoning:
```
<reasoning>
Step 1: April = 48 clips
Step 2: May = 48 * 2 = 96 clips  ← ERROR HERE!
Step 3: Total = 48 + 96 = 144 clips
</reasoning>
<answer>
144
</answer>
```

### 3. **Educational Value**
Students can learn from the reasoning process, not just see the final answer.

### 4. **Trust & Verification**
Users can verify the model's thinking process before trusting the answer.

---

## Training Process

### How GRPO Encourages This Format

1. **System Prompt** instructs the model to use XML format
2. **Reward Function** extracts answer from `<answer>` tags (with fallback)
3. **Generation Parameters** allow longer outputs (1024 tokens vs 512)
4. **Training** rewards completions that:
   - Follow the XML format
   - Show clear reasoning steps
   - Produce correct answers

### Example Training Scenario

**Generation 1 (Early Training):**
```
The answer is 72.
```
- ❌ No XML format
- ❌ No reasoning
- ✅ Correct answer
- **Reward:** Lower (format not followed, but answer correct)

**Generation 2 (Mid Training):**
```
<reasoning>
48 + 24 = 72
</reasoning>
<answer>
72
</answer>
```
- ✅ XML format followed
- ⚠️ Minimal reasoning
- ✅ Correct answer
- **Reward:** Medium (format good, reasoning could be better)

**Generation 3 (Well Trained):**
```
<reasoning>
Step 1: April sales = 48 clips
Step 2: May sales = 48 / 2 = 24 clips
Step 3: Total = 48 + 24 = 72 clips
</reasoning>
<answer>
72
</answer>
```
- ✅ XML format followed
- ✅ Clear reasoning steps
- ✅ Correct answer
- **Reward:** High (all criteria met)

---

## Real-World Analogy

**Original GRPO:** Like a student who just writes the answer on a math test
```
Question: Solve 2x + 5 = 13
Answer: x = 4
```

**Reasoning Model:** Like a student who shows their work
```
Question: Solve 2x + 5 = 13

Work:
Step 1: Subtract 5 from both sides
  2x + 5 - 5 = 13 - 5
  2x = 8

Step 2: Divide both sides by 2
  2x / 2 = 8 / 2
  x = 4

Answer: x = 4
```

---

## Summary

The reasoning model is encouraged to produce:

1. **Structured Output**: XML format with `<reasoning>` and `<answer>` tags
2. **Step-by-Step Thinking**: Clear problem decomposition and solution steps
3. **Transparency**: Shows the "why" behind the answer, not just the "what"
4. **Educational Value**: Can be used for teaching and learning
5. **Verifiability**: Users can check the reasoning process

This makes the model more similar to:
- **DeepSeek-R1** (reasoning models)
- **GPT-4 with Chain-of-Thought**
- **Human problem-solving** (showing work)

Rather than just a "black box" that produces answers.

