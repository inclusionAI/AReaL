# How the Code Encourages Reasoning Format

This document explains the technical implementation details of how the reasoning model training encourages the XML reasoning format.

## Overview: The Complete Flow

```
1. Dataset Loader → Adds System Prompt
2. Chat Template → Formats Messages
3. Model Generation → Produces Output
4. Answer Extraction → Prioritizes XML Format
5. GRPO Training → Rewards Format Compliance
```

---

## Step 1: Dataset Loader - Injecting the System Prompt

### Location: `areal/dataset/gsm8k.py`

**Code:**
```python
def get_gsm8k_reasoning_rl_dataset(path, split, tokenizer, max_length=None):
    # Load the dataset
    dataset = load_dataset(path=actual_path, name="main", split=split)
    
    # System prompt instructing the model to use reasoning format
    SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
[Your step-by-step reasoning process here]
</reasoning>
<answer>
[Your final numerical answer here]
</answer>"""

    def process(sample):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},  # ← System prompt added here
            {"role": "user", "content": sample["question"]}
        ]
        return {"messages": messages}
    
    dataset = dataset.map(process)
    return dataset
```

**What This Does:**
- **Automatically detects** reasoning mode when path contains "reasoning" (e.g., `openai/gsm8k-reasoning`)
- **Injects system prompt** into every training sample
- **Instructs the model** to use XML format before it even generates

**Key Point:** The system prompt is part of the input, so the model sees it during every forward pass.

---

## Step 2: Chat Template - Formatting Messages

### Location: `areal/workflow/rlvr.py`

**Code:**
```python
def default_get_input_ids_fn(data, tokenizer, enable_thinking):
    input_ids = tokenizer.apply_chat_template(
        data,  # Contains: [{"role": "system", ...}, {"role": "user", ...}]
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return input_ids
```

**What This Does:**
- Takes the `messages` array (with system + user messages)
- Applies the model's chat template (e.g., Qwen's template)
- Formats it into the proper token sequence
- Adds generation prompt to indicate where the model should start generating

**Example Output (Tokenized):**
```
<|im_start|>system
Respond in the following format:
<reasoning>
[Your step-by-step reasoning process here]
</reasoning>
<answer>
[Your final numerical answer here]
</answer><|im_end|>
<|im_start|>user
Natalia sold clips to 48 of her friends...<|im_end|>
<|im_start|>assistant
[Model generates here]
```

**Key Point:** The system prompt is now part of the tokenized input that the model processes.

---

## Step 3: Model Generation - Producing Output

### Location: `areal/workflow/rlvr.py` (rollout)

**Code:**
```python
async def arun_episode(self, engine: InferenceEngine, data):
    # Get input_ids with system prompt included
    input_ids = self.get_input_ids_fn(
        self.data_extract_prompt_fn(data), 
        self.tokenizer, 
        self.enable_thinking
    )
    
    # Generate multiple samples (e.g., 4 per problem)
    n_samples = self.gconfig.n_samples
    req = ModelRequest(
        input_ids=input_ids,
        gconfig=self.gconfig.new(n_samples=1),
        ...
    )
    resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])
    
    # Decode completions
    for resp in resps:
        completions_str = self.tokenizer.decode(resp.output_tokens)
        # completions_str now contains model's output
```

**What This Does:**
- Model receives input with system prompt
- Generates `n_samples` (typically 4) different completions
- Each completion is decoded to text

**Example Completions (Early Training):**
```
Completion 1: "The answer is 72."
Completion 2: "48 + 24 = 72"
Completion 3: "<reasoning>Let me think... 48/2=24, 48+24=72</reasoning><answer>72</answer>"
Completion 4: "I need to calculate..."
```

**Key Point:** The model has seen the system prompt, but may not follow it perfectly initially.

---

## Step 4: Answer Extraction - Prioritizing XML Format

### Location: `areal/reward/math_parser.py`

**Code:**
```python
def extract_answer(pred_str, data_name, use_last_number=True):
    # Extract from XML reasoning format: <answer>...</answer>
    # This should be checked FIRST before other formats
    pred = None
    if "<answer>" in pred_str and "</answer>" in pred_str:
        answer_start = pred_str.find("<answer>")
        answer_end = pred_str.find("</answer>")
        if answer_start < answer_end:
            pred = pred_str[answer_start + len("<answer>"):answer_end].strip()
            # Continue with normal processing to clean the answer below
    
    # If XML format not found, try other extraction methods
    if pred is None:
        if "boxed" in pred_str:
            # Extract from \boxed{...}
        elif "final answer is" in pred_str:
            # Extract from "final answer is ..."
        else:
            # Fallback: extract last number
```

**What This Does:**
1. **First Priority:** Check for `<answer>...</answer>` tags
2. **Second Priority:** Check for `\boxed{}` format
3. **Third Priority:** Check for "final answer is" format
4. **Fallback:** Extract last number in text

**Why This Encourages Format:**
- If the model uses XML format, the answer is extracted **reliably**
- If the model doesn't use XML format, extraction may fail or be less reliable
- **Reliable extraction = Higher reward** (correct answer found)
- **Unreliable extraction = Lower reward** (answer not found or wrong)

**Example:**
```python
# Model output with XML format:
completion = "<reasoning>Step 1: 48/2=24, Step 2: 48+24=72</reasoning><answer>72</answer>"
extracted = extract_answer(completion, "gsm8k")  # Returns: "72" ✅

# Model output without XML format:
completion = "The answer is 72."
extracted = extract_answer(completion, "gsm8k")  # Returns: "72" (fallback) ⚠️

# Model output with wrong format:
completion = "I think it might be around 70 or so."
extracted = extract_answer(completion, "gsm8k")  # Returns: "70" (wrong!) ❌
```

**Key Point:** XML format provides the most reliable answer extraction, leading to higher rewards.

---

## Step 5: Reward Function - Computing Rewards

### Location: `examples/docker_gsm8k/gsm8k_grpo_train.py`

**Code:**
```python
def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from areal.reward.math_parser import process_results
    
    return int(process_results(completions, answer)[0])
```

**What `process_results` Does:**
```python
def process_results(completions, solution):
    # completions: List of model outputs (e.g., 4 completions)
    # solution: Ground truth answer (e.g., "72")
    
    results = []
    for completion in completions:
        extracted = extract_answer(completion, "gsm8k")
        # Compare extracted answer with ground truth
        is_correct = (extracted == solution)
        results.append(1 if is_correct else 0)
    
    return results  # e.g., [1, 1, 0, 0] for 2 correct, 2 wrong
```

**What This Does:**
- For each completion, extracts the answer
- Compares with ground truth
- Returns binary reward: `1` (correct) or `0` (incorrect)

**Example:**
```python
completions = [
    "<reasoning>48/2=24, 48+24=72</reasoning><answer>72</answer>",  # Format + Correct
    "The answer is 72.",  # No format, but correct
    "<reasoning>48*2=96</reasoning><answer>96</answer>",  # Format but wrong
    "I don't know."  # No format, no answer
]

ground_truth = "72"

rewards = process_results(completions, ground_truth)
# Returns: [1, 1, 0, 0]
```

**Key Point:** Completions that follow XML format are more likely to have correct answers extracted, leading to higher rewards.

---

## Step 6: GRPO Training - Rewarding Format Compliance

### Location: `areal/engine/ppo/actor.py` (GRPO algorithm)

**How GRPO Works:**
1. **Group Generation:** For each problem, generate `n_samples` (e.g., 4) completions
2. **Group Rewards:** Compute rewards for all completions in the group
3. **Group-Relative Normalization:** Normalize rewards within each group
4. **Policy Update:** Increase probability of tokens in high-reward completions

**Example Training Step:**
```python
# Problem: "Natalia sold 48 clips in April, half in May. Total?"

# Group of 4 completions:
completions = [
    "<reasoning>Step 1: May=48/2=24, Step 2: Total=48+24=72</reasoning><answer>72</answer>",  # Reward: 1
    "<reasoning>48+24=72</reasoning><answer>72</answer>",  # Reward: 1
    "The answer is 72.",  # Reward: 1 (but less reliable extraction)
    "I think it's 70."  # Reward: 0
]

# Group rewards: [1, 1, 1, 0]
# Group mean: 0.75
# Group std: 0.433

# Normalized advantages:
# Completion 1: (1 - 0.75) / 0.433 = +0.577  ← Boosted (has format + correct)
# Completion 2: (1 - 0.75) / 0.433 = +0.577  ← Boosted (has format + correct)
# Completion 3: (1 - 0.75) / 0.433 = +0.577  ← Boosted (correct, but no format)
# Completion 4: (0 - 0.75) / 0.433 = -1.732  ← Penalized (wrong)

# Policy update:
# - Increase probability of tokens in completions 1, 2, 3
# - Decrease probability of tokens in completion 4
```

**Why This Encourages Format:**
1. **Reliable Extraction:** XML format → More reliable answer extraction → Higher chance of correct reward
2. **Consistent Rewards:** Format-compliant completions get consistent rewards
3. **Policy Learning:** Model learns that XML format leads to higher rewards
4. **Over Time:** Model increasingly produces XML format to maximize rewards

**Key Point:** GRPO doesn't explicitly reward format, but format compliance leads to more reliable answer extraction, which leads to higher rewards, which the model learns to maximize.

---

## Step 7: Generation Parameters - Supporting Longer Reasoning

### Location: `examples/docker_gsm8k/gsm8k_grpo_reasoning_1hour.yaml`

**Code:**
```yaml
gconfig:
  max_new_tokens: 1024  # Increased from 512 to allow longer reasoning chains

train_dataset:
  max_length: 2048  # Increased from 512 to accommodate reasoning
```

**What This Does:**
- **Allows longer outputs:** Model can generate up to 1024 tokens (vs 512)
- **Accommodates reasoning:** System prompt + reasoning + answer fits in 2048 tokens
- **Supports detailed steps:** Model has space to show step-by-step thinking

**Why This Matters:**
- Without enough tokens, model might truncate reasoning
- With more tokens, model can produce full reasoning chains
- This enables the model to follow the format properly

**Key Point:** Generation parameters provide the capacity for the model to produce full reasoning chains.

---

## Summary: How It All Works Together

### The Complete Chain:

1. **Dataset Loader** → Injects system prompt into every sample
2. **Chat Template** → Formats system + user messages properly
3. **Model Generation** → Model sees prompt and generates output
4. **Answer Extraction** → XML format prioritized for reliable extraction
5. **Reward Function** → Correct answers (from reliable extraction) get reward = 1
6. **GRPO Training** → Model learns that XML format → reliable extraction → higher rewards
7. **Generation Parameters** → Provide capacity for full reasoning chains

### Why This Works:

- **Implicit Learning:** Model doesn't get explicit "format reward", but format compliance leads to reliable answer extraction, which leads to higher rewards
- **Self-Reinforcing:** As model learns format, it gets more reliable rewards, which reinforces format usage
- **Natural Evolution:** Over training, model increasingly produces XML format because it maximizes rewards

### The Key Insight:

The reasoning format is encouraged **indirectly** through:
1. **System prompt** (explicit instruction)
2. **Answer extraction** (format makes extraction more reliable)
3. **Reward signals** (reliable extraction → correct answers → higher rewards)
4. **GRPO learning** (model maximizes rewards by following format)

The model learns: **"XML format → Reliable extraction → Correct answers → Higher rewards"**

---

## Code Locations Summary

| Component | File | Key Function |
|-----------|------|--------------|
| Dataset Loader | `areal/dataset/gsm8k.py` | `get_gsm8k_reasoning_rl_dataset()` |
| Dataset Router | `areal/dataset/__init__.py` | Checks for "reasoning" in path |
| Chat Template | `areal/workflow/rlvr.py` | `default_get_input_ids_fn()` |
| Answer Extraction | `areal/reward/math_parser.py` | `extract_answer()` (XML priority) |
| Reward Function | `examples/docker_gsm8k/gsm8k_grpo_train.py` | `gsm8k_reward_fn()` |
| GRPO Training | `areal/engine/ppo/actor.py` | Group-relative policy optimization |
| Config | `examples/docker_gsm8k/gsm8k_grpo_reasoning_*.yaml` | Generation parameters |

---

## Example: Complete Flow for One Training Step

```python
# 1. Dataset Loader
sample = {
    "messages": [
        {"role": "system", "content": "Respond in the following format:\n<reasoning>...</reasoning>\n<answer>...</answer>"},
        {"role": "user", "content": "Natalia sold 48 clips in April, half in May. Total?"}
    ],
    "answer": "72"
}

# 2. Chat Template
input_ids = tokenizer.apply_chat_template(sample["messages"], ...)
# Result: Tokenized sequence with system prompt included

# 3. Model Generation (4 samples)
completions = [
    "<reasoning>Step 1: May=48/2=24, Step 2: Total=48+24=72</reasoning><answer>72</answer>",
    "<reasoning>48+24=72</reasoning><answer>72</answer>",
    "The answer is 72.",
    "I think it's 70."
]

# 4. Answer Extraction
extracted = [
    extract_answer(completions[0], "gsm8k"),  # "72" (from <answer> tag)
    extract_answer(completions[1], "gsm8k"),  # "72" (from <answer> tag)
    extract_answer(completions[2], "gsm8k"),  # "72" (fallback)
    extract_answer(completions[3], "gsm8k")   # "70" (wrong)
]

# 5. Reward Function
rewards = [1, 1, 1, 0]  # 3 correct, 1 wrong

# 6. GRPO Training
# - Normalize rewards within group
# - Compute advantages
# - Update policy to increase probability of tokens in high-reward completions
# - Model learns: XML format → reliable extraction → higher rewards
```

This is how the code encourages the reasoning format through the complete training pipeline!

