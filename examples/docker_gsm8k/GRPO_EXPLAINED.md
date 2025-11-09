# What is GRPO and Why It Helps with Math

## What is GRPO?

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning algorithm designed specifically for training language models on tasks with **sparse rewards** - like math problems where you only get a reward at the end (correct or incorrect).

### Key Innovation: No Value Function Needed

Traditional RL methods (like standard PPO) require:
- **Actor**: The model being trained
- **Critic**: A separate value function that estimates expected future rewards

GRPO eliminates the need for a critic by using **group-relative normalization** instead.

## How GRPO Works

### Step 1: Generate Multiple Solutions

For each math problem, GRPO generates **multiple candidate solutions** (e.g., 4 different attempts):

```
Problem: "Janet has 16 eggs. She sells 3 and breaks 4. How many left?"

Solution 1: "16 - 3 - 4 = 9. Answer: 9" ✅ (correct, reward = 1)
Solution 2: "16 - 3 = 13, then 13 - 4 = 9. Answer: 9" ✅ (correct, reward = 1)
Solution 3: "16 + 3 - 4 = 15. Answer: 15" ❌ (incorrect, reward = 0)
Solution 4: "I think it's 8" ❌ (incorrect, reward = 0)
```

### Step 2: Group-Relative Advantage Normalization

Instead of using absolute rewards, GRPO normalizes rewards **within each group** of solutions:

**Traditional PPO:**
- Uses absolute rewards: `advantage = reward - value_estimate`
- Requires training a separate value function (critic)

**GRPO:**
- Normalizes within the group: `advantage = (reward - group_mean) / group_std`
- No value function needed!

**Example:**
```
Group rewards: [1, 1, 0, 0]
Group mean: 0.5
Group std: 0.5

Normalized advantages:
- Solution 1: (1 - 0.5) / 0.5 = +1.0  (boosted)
- Solution 2: (1 - 0.5) / 0.5 = +1.0  (boosted)
- Solution 3: (0 - 0.5) / 0.5 = -1.0  (penalized)
- Solution 4: (0 - 0.5) / 0.5 = -1.0  (penalized)
```

### Step 3: Policy Update

The model learns to:
- **Increase probability** of tokens in correct solutions (positive advantage)
- **Decrease probability** of tokens in incorrect solutions (negative advantage)

## Why GRPO Helps with Math

### 1. **Sparse Rewards Problem**

Math problems have **binary rewards**:
- ✅ Correct answer → reward = 1
- ❌ Wrong answer → reward = 0

Traditional RL struggles because:
- Most attempts get reward = 0 (no learning signal)
- Need accurate value estimates (hard to train)

**GRPO Solution:**
- By comparing within a group, even if all solutions are wrong, the "less wrong" ones get higher relative advantage
- Creates learning signal even from failures!

### 2. **No Value Function Training**

Traditional PPO needs to train two models:
- Actor (policy)
- Critic (value function)

This is:
- **Expensive**: 2x compute and memory
- **Unstable**: Value function can be hard to train
- **Complex**: More hyperparameters to tune

**GRPO Solution:**
- Only trains the actor (policy)
- Simpler, faster, more stable

### 3. **Emphasizes Relative Quality**

GRPO focuses on **which solution is better** rather than absolute reward:

```
Example: All solutions are wrong, but one is closer

Solution A: "16 - 3 - 4 = 8" (wrong, but correct steps)
Solution B: "I don't know" (completely wrong)
Solution C: "42" (random guess)
Solution D: "banana" (nonsense)

Traditional RL: All get reward = 0, no learning
GRPO: Solution A gets highest relative advantage → model learns better reasoning!
```

### 4. **Works Well with Small Models**

For small models (like 0.5B), training a value function is especially hard:
- Limited capacity
- Value function competes with policy for model capacity

**GRPO Solution:**
- All model capacity goes to improving the policy
- Better use of limited parameters

## Mathematical Formulation

The GRPO objective:

$$
J_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}\left( r_{i,t}(\theta),\ 1-\epsilon,\ 1+\epsilon \right) \hat{A}_{i,t} \right) \right]
$$

Where:
- $r_{i,t}(\theta)$ = importance weight (new policy / old policy)
- $\hat{A}_{i,t}$ = **group-relative advantage** = $\frac{r_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$
- $G$ = group size (number of samples per prompt)

## In Your Training Configuration

Looking at your `gsm8k_grpo.yaml`:

```yaml
gconfig:
  n_samples: 4  # Generate 4 solutions per problem (group size)

actor:
  group_size: ${gconfig.n_samples}  # Group size = 4
  reward_norm:
    mean_level: group  # Normalize mean within groups
    std_level: group   # Normalize std within groups
    group_size: ${gconfig.n_samples}  # 4 samples per group
```

This means:
- For each math problem, generate 4 different solutions
- Compare rewards within those 4 solutions
- Learn from relative differences, not absolute rewards

## Why It's Better Than Supervised Fine-Tuning (SFT)

### SFT Limitations:
- Only learns from **correct** examples
- Can't learn from mistakes
- No exploration of alternative solutions

### GRPO Advantages:
- Learns from **both correct and incorrect** solutions
- Explores multiple solution paths
- Discovers better reasoning strategies through trial and error
- Can improve even when most attempts fail (via relative comparison)

## Real-World Example

**Before GRPO (Base Model):**
- Generates one solution
- If wrong, no learning signal
- Accuracy: ~18%

**After GRPO Training:**
- Generates 4 solutions per problem
- Compares them relatively
- Learns which reasoning patterns work better
- Even wrong solutions contribute to learning
- Accuracy: Can improve to 30-40%+ with proper training

## Key Takeaway

GRPO is **perfect for math** because:
1. ✅ Handles sparse rewards (only correct/incorrect at the end)
2. ✅ No need for expensive value function training
3. ✅ Learns from relative quality, not just absolute correctness
4. ✅ Works well with small models
5. ✅ Explores multiple solution strategies

The group-relative normalization is the key innovation that makes RL practical for math problems!

