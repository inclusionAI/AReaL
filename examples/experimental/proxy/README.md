# AReaL with Proxy Mode

## Quick Start

```bash
export PYTHONPATH=/path/to/AReaL:$PYTHONPATH

python3 -m areal.launcher.local AReaL/examples/experimental/proxy/gsm8k_grpo_proxy.py --config AReaL/examples/math/gsm8k_grpo.yaml \
+tool_call_parser="qwen25" \
+agent_module_path="examples.experimental.proxy.math_agent" \
actor.path=Qwen/Qwen2.5-1.5B \
experiment_name=proxy-agent \
trial_name=run1
```

This script will run the example agent in `math_agent.py`. You can also modify
agent_module_path to `math_with_python_tool.py` or `multi_agent_math.py` to run the
other two example agents.

## Write Your Own Agent

1. Write an Agent using a framework that you are familiar with, such as
   [OpenAI Agent](https://openai.github.io/openai-agents-python/)
1. Write an AReaL interface function named `run_agent_return_reward`, where the input
   data is a piece of data in the dataset, and the function needs to return a float
   representing the final reward:

```python
async def run_agent_return_reward(data: Any) -> float:

    def gsm8k_reward_fn(result, answer):
        from areal.reward.math_parser import process_results
        return float(process_results(result, answer)[0])

    result = await run_agent(data)
    reward = gsm8k_reward_fn(result.final_output, data["answer"])
    return reward
```

3. Place your agent code in a path that can be imported in Python, and modify the
   agent_module_path in the configuration file to that path:

```yaml
agent_module_path: "examples.any_agents.agent.math.math_with_python_tool"
```
