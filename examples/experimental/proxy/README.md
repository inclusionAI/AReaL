# AReaL with Proxy Mode

## Quick Start

```bash
python3 examples/experimental/proxy/train.py \
    --config examples/experimental/proxy/config.yaml \
    scheduler.type=local \
```

This script will run the example agent in `areal/workflow/openai/math_agent.py`. You can
also modify `workflow` to run other example agents defined in the same file.

## Write Your Own Agent

1. Write a Python **`async` function** that inputs a dict as agent inputs. The function
   must directly use OpenAI Python SDK or implicitly use it through a high-level agent
   package (e.g., [OpenAI Agent](https://openai.github.io/openai-agents-python/),
   [CAMEL-AI](https://www.camel-ai.org/)).The function can either return a float as the
   final reward, or a dict where the reward of each interaction is keyed by the
   completion or response ID.

1. Wrap the function within AReaL's `AgentWorkflow`. This class is for pure typing and
   API regulation usage. You can add configurable options in the `__init__` method and
   expose them in the training scripts, in which way you can then configure these
   options in CLI args or yaml.

```python
async def my_agent(data: dict) -> float:
    ...

from areal.api.workflow_api import AgentWorkflow

class MyAgentWorkflow(AgentWorkflow):
    async def run(self, data: dict):
        return await my_agent(data)
```

3. Place your agent code in a path that can be imported in Python, and modify the
   `workflow` in the training script to that path.

See `areal/workflow/openai/` for concrete examples.
