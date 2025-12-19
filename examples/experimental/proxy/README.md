# AReaL with Proxy Mode

## Quick Start

```bash
export PYTHONPATH=/path/to/AReaL:$PYTHONPATH

python3 -m areal.launcher.local AReaL/examples/experimental/proxy/train.py --config AReaL/examples/math/gsm8k_grpo.yaml \
+agent_module_path="examples.experimental.proxy.gsm8k_agent" \
actor.path=Qwen/Qwen2.5-1.5B \
experiment_name=proxy-agent \
trial_name=run1
```

This script will run the example agent in `gsm8k_agent.py`. You can also modify
agent_module_path to `gsm8k_multi_turn_agent`, `gsm8k_openai_agent`,
`math_with_python_tool` or `multi_agent_math` to run other example agents.

## Write Your Own Agent

1. Write an Agent that calls OpenAI-compatible APIs (e.g. chat completions, responses)
   using a framework that you are familiar with, such as
   [OpenAI Agent](https://openai.github.io/openai-agents-python/)
1. Write an AReaL interface function named `run_agent_return_reward`, where the input
   data is a piece of data in the dataset, and the function needs to return a float
   representing the final reward:

```python
async def run_agent_return_reward(data: Any) -> float:
    from areal.reward import get_math_verify_worker

    worker = get_math_verify_worker()

    def gsm8k_reward_fn(result, answer):
        try:
            worker = get_math_verify_worker()
            return worker.verify(str(result), str(answer))
        except Exception:
            return 0.0

    result = await run_agent(data)
    reward = gsm8k_reward_fn(result.final_output, data["answer"])
    return reward
```

3. Wraps `run_agent_return_reward` function into a `run_and_submit` function, you can
   use the `run_and_submit_rewards` function in `areal.utils.proxy_utils` to do this.

```python
async def run_and_submit(data: dict):
    await run_and_submit_rewards(func=run_agent_return_reward, data=data)
```

4. Place your agent code in a path that can be imported in Python, and modify the
   agent_module_path in the configuration file to that path:

```yaml
agent_module_path: "examples.experimental.proxy.gsm8k_agent"
```

## The configuration file for the examples

The `config.yaml` file is identical to `examples/multi-turn-math/gsm8k_grpo_mt.yaml` and
can be used by any examples here. You may modify the configuration file to fit your
needs, or override configuration values in the command line (see quick start above as an
example).
