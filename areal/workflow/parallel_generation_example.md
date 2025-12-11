# Parallel Generation Workflow Example

This example demonstrates how to use the `ParallelGenerationWorkflow` for structured parallel generation with Goals, Paths, and Conclusions.

## Overview

The `ParallelGenerationWorkflow` enables models to generate complex reasoning with:
1. **Goal Generation**: Model generates a goal with multiple numbered outlines
2. **Parallel Path Generation**: Each outline spawns a separate reasoning path
3. **Conclusion Generation**: Model synthesizes all paths into a final answer

## Generation Structure

```
<Goal>
  <Outline>1. First approach to solve the problem</Outline>
  <Outline>2. Second approach to solve the problem</Outline>
  <Outline>3. Third approach to solve the problem</Outline>
</Goal>
<Path>
1. [Detailed reasoning for first approach...]
</Path>
<Path>
2. [Detailed reasoning for second approach...]
</Path>
<Path>
3. [Detailed reasoning for third approach...]
</Path>
<Conclusion>
[Final synthesis and answer combining insights from all paths...]
</Conclusion>
```

## Usage Example

```python
from areal.api.cli_args import GenerationHyperparameters
from areal.workflow.parallel_generation import ParallelGenerationWorkflow

# Define a reward function
def math_reward_fn(prompt: str, completion: str, input_ids: list[int], 
                   output_ids: list[int], **kwargs) -> float:
    """Example reward function for math problems."""
    # Extract the answer from kwargs (e.g., ground truth)
    ground_truth = kwargs.get("answer", "")
    
    # Parse the completion to extract the model's answer
    # (This is simplified - you'd want more robust parsing)
    import re
    match = re.search(r"Final Answer:\s*(.+)", completion)
    if match:
        model_answer = match.group(1).strip()
        # Check if answers match
        if model_answer == ground_truth:
            return 1.0
    return 0.0

# Configure generation parameters
gconfig = GenerationHyperparameters(
    n_samples=1,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=8192,
)

# Create the workflow
workflow = ParallelGenerationWorkflow(
    reward_fn=math_reward_fn,
    gconfig=gconfig,
    tokenizer="path/to/tokenizer",
    enable_thinking=False,
    max_goal_tokens=2048,      # Max tokens for goal generation
    max_path_tokens=4096,      # Max tokens per path
    max_conclusion_tokens=2048, # Max tokens for conclusion
    rollout_stat_scope="rollout",
    dump_dir="./rollout_dumps",  # Optional: save generations to disk
)

# Use in training
# The workflow will be called by the inference engine during rollout
# See areal/examples for complete training setups
```

## Integration with AReaL Training

To use this workflow in a training loop:

```python
from areal.api.engine_api import InferenceEngine

# During rollout phase
engine = InferenceEngine(...)

# Prepare your data
data = {
    "messages": [
        {"role": "user", "content": "Solve this math problem: ..."}
    ],
    "answer": "42",  # Ground truth for reward computation
    "query_id": "problem_001",
}

# The workflow's arun_episode will be called automatically
# It returns a trajectory dict with:
# - input_ids: Full sequence (prompt + generation)
# - loss_mask: Mask for training (1 for generated tokens, 0 for prompt)
# - logprobs: Log probabilities for each token
# - versions: Model versions for each token
# - attention_mask: All ones
# - rewards: Scalar reward value

trajectory = await workflow.arun_episode(engine, data)
```

## Configuration Options

### Generation Limits
- `max_goal_tokens`: Maximum tokens for goal generation (default: 2048)
- `max_path_tokens`: Maximum tokens per individual path (default: 4096)
- `max_conclusion_tokens`: Maximum tokens for conclusion (default: 2048)

### Workflow Options
- `enable_thinking`: Enable thinking tokens in the prompt (default: False)
- `rollout_stat_scope`: Scope for statistics tracking (default: "rollout")
- `dump_dir`: Directory to save generation dumps (default: None)

### Custom Functions
- `get_input_ids_fn`: Custom function to convert data to input_ids
- `data_extract_prompt_fn`: Custom function to extract prompt from data

## Key Features

1. **Automatic Outline Detection**: The workflow automatically detects numbered outlines in the goal text and generates corresponding paths.

2. **Parallel Generation**: All paths are generated in parallel using `asyncio.gather()`, improving efficiency.

3. **Proper Logprob Tracking**: The workflow tracks logprobs and versions for all generated tokens, crucial for policy gradient training.

4. **Flexible Reward Functions**: Support both synchronous and asynchronous reward functions.

5. **Debugging Support**: Optional dump directory saves all generations to disk for inspection.

## Notes

- The workflow expects the model to generate structured output with `<Goal>`, `<Path>`, and `<Conclusion>` tags
- If no outlines are detected in the goal, it defaults to generating a single path with prefix "1"
- All marker tokens (`<Goal>`, `</Goal>`, `<Path>`, etc.) have logprob=0.0 and version=-1
- The reward is computed on the complete generation (goal + paths + conclusion)

## Comparison with RLVR Workflow

| Feature | RLVR Workflow | Parallel Generation Workflow |
|---------|---------------|------------------------------|
| Structure | Single linear generation | Goal → Paths → Conclusion |
| Sampling | Multiple samples per prompt | Single structured sample |
| Parallel Paths | No | Yes (multiple reasoning paths) |
| Use Case | Standard RL training | Complex reasoning tasks |

## Advanced: Custom Reward Functions

You can use async reward functions for more complex reward computation:

```python
async def async_math_reward_fn(prompt: str, completion: str, 
                                input_ids: list[int], output_ids: list[int],
                                **kwargs) -> float:
    """Async reward function that can do complex computation or API calls."""
    # Example: call an external API to verify the answer
    ground_truth = kwargs.get("answer", "")
    
    # Extract answer from completion
    import re
    match = re.search(r"Final Answer:\s*(.+)", completion)
    if not match:
        return 0.0
    
    model_answer = match.group(1).strip()
    
    # Could do: await api_call_to_verify(model_answer, ground_truth)
    # For now, simple string match
    await asyncio.sleep(0.01)  # Simulate async work
    
    return 1.0 if model_answer == ground_truth else 0.0

# Use async reward function
workflow = ParallelGenerationWorkflow(
    reward_fn=async_math_reward_fn,  # Async functions are automatically wrapped
    # ... other params
)
```

## Import Path Reference

```python
# For use in config files (import by string):
workflow_str = "areal.workflow.parallel_generation.ParallelGenerationWorkflow"

# Or direct import:
from areal.workflow.parallel_generation import ParallelGenerationWorkflow
```
