# RolloutWorkflow Reference

This document describes the `RolloutWorkflow` abstraction, the core interface for
implementing rollout generation in AReaL's reinforcement learning pipeline.

**Note**: This page targets developers seeking a deep understanding of the codebase. For
agentic RL training, use the high-level API described in the
[Agentic RL Guide](../tutorial/agentic_rl.md).

## Overview

A `RolloutWorkflow` defines how to generate training trajectories from input data. It
encapsulates the logic for:

- Tokenizing prompts and preparing model inputs
- Calling the inference engine to generate completions
- Computing rewards for generated outputs
- Packaging results into tensor dictionaries for training

## Interface

```python
from areal.api.workflow_api import RolloutWorkflow

class RolloutWorkflow(ABC):
    @abstractmethod
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        """Run a single episode of the workflow."""
        ...
```

### Parameters

| Parameter | Type              | Description                                     |
| --------- | ----------------- | ----------------------------------------------- |
| `engine`  | `InferenceEngine` | Inference engine for generating model responses |
| `data`    | `dict[str, Any]`  | A single sample from the dataloader             |

### Return Types

The `arun_episode` method supports three return types:

| Return Type                                 | Description                                                                                        |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `dict[str, torch.Tensor]`                   | Standard tensor format for training                                                                |
| `dict[str, InteractionWithTokenLogpReward]` | Token-level interactions (auto-converted to tensors); produced by the high-level `ArealOpenAI` API |
| `None`                                      | Rejected trajectory, excluded from training                                                        |

## Tensor Dictionary Format

When returning a tensor dictionary, the following fields are expected:

| Field            | Shape                   | Type    | Required | Description                         |
| ---------------- | ----------------------- | ------- | -------- | ----------------------------------- |
| `input_ids`      | `[batch_size, seq_len]` | int32   | Yes      | Token IDs (prompt + completion)     |
| `attention_mask` | `[batch_size, seq_len]` | bool    | Yes      | Valid token mask                    |
| `loss_mask`      | `[batch_size, seq_len]` | int32   | No       | Completion token mask (1 = train)   |
| `logprobs`       | `[batch_size, seq_len]` | float32 | No       | Log probabilities per token         |
| `rewards`        | `[batch_size]`          | float32 | No       | Per-sequence rewards                |
| `versions`       | `[batch_size, seq_len]` | int32   | No       | Weight version when token generated |

Example return value:

```python
return {
    "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32),
    "attention_mask": torch.ones(1, 5, dtype=torch.bool),
    "loss_mask": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
    "logprobs": torch.tensor([[0.0, 0.0, -0.5, -0.3, -0.2]], dtype=torch.float32),
    "rewards": torch.tensor([1.0], dtype=torch.float32),
    "versions": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32),
}
```

## Workflow Context

Inside `arun_episode`, access the execution context via the `workflow_context` module.
Each workflow instance has its own isolated context:

```python
from areal.infra import workflow_context

async def arun_episode(self, engine, data):
    # Get current execution context
    ctx = workflow_context.get()

    # Check if running in evaluation mode
    if ctx.is_eval:
        # Use different parameters for evaluation
        ...

    # Get task ID for logging
    task_id = ctx.task_id

    # Get stats scope based on mode ("rollout" or "eval-rollout")
    scope = workflow_context.stat_scope()
```

## Trajectory Dumping

When `InferenceEngineConfig.dump_to_file=True`, trajectories are automatically saved to
disk for debugging and analysis.

### Configuration

```yaml
rollout:
  dump_to_file: true
  fileroot: "/path/to/logs"
  tokenizer_path: "model/tokenizer" # Required for text decoding
```

### Output Location

Trajectories are saved to:

```
{fileroot}/{experiment_name}/{trial_name}/[rollout|eval-rollout]/{version}/{task_id}.jsonl
```

Example:

```
/tmp/areal/my_exp/trial1/rollout/5/42.jsonl
```

### Output Format

Each line in the JSONL file contains:

```json
{
  "task_id": 42,
  "sample_idx": 0,
  "seqlen": 256,
  "prompt_len": 128,
  "head_version": 5,
  "tail_version": 5,
  "reward": 1.0,
  "prompt": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
  "completion": "The answer is 4.<|im_end|>"
}
```

## Implementing Custom Workflows

To create a custom workflow:

1. **Subclass `RolloutWorkflow`**:

```python
from areal.api.workflow_api import RolloutWorkflow

class MyWorkflow(RolloutWorkflow):
    def __init__(self, tokenizer, gconfig, **kwargs):
        self.tokenizer = tokenizer
        self.gconfig = gconfig

    async def arun_episode(self, engine, data):
        # 1. Prepare input
        input_ids = self.tokenizer.encode(data["prompt"])

        # 2. Generate completion
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig,
            tokenizer=self.tokenizer,
        )
        resp = await engine.agenerate(req)

        # 3. Compute reward
        reward = self.compute_reward(resp, data)

        # 4. Return tensor dict (or None to reject)
        if reward < 0:
            return None

        return self.build_tensor_dict(resp, reward)
```

2. **Register with trainer**:

```python
trainer.train(
    workflow=MyWorkflow,
    workflow_kwargs={
        "tokenizer": tokenizer,
        "gconfig": config.gconfig,
    },
)
```

## Workflow Resolution

Workflows can be specified in multiple ways:

| Format         | Example                          | Description                |
| -------------- | -------------------------------- | -------------------------- |
| Instance       | `MyWorkflow(...)`                | Pre-instantiated workflow  |
| Class          | `MyWorkflow`                     | Class (requires kwargs)    |
| String path    | `"my_module.MyWorkflow"`         | Dynamic import             |
| Agent workflow | Any class with `async def run()` | Wrapped with proxy support |

The training system automatically resolves these to `RolloutWorkflow` instances.

## See Also

- [Agentic RL Tutorial](../tutorial/agentic_rl.md) - Training with agent frameworks
- [Adding Custom Workflows](../customization/agent.md) - Step-by-step guide
