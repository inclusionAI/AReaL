# Designating Server Addresses for Remote Inference Engines

By default, launchers in AReaL records server addresses whem launching them and then set
the environment variable `AREAL_LLM_SERVER_ADDRS` to pass them to the remote inference
engine in training scripts. However, this method does not cover cases that involve a
training model and one or more serving-only models, such as LLM-as-Judge and multi-agent
games. To support these cases, the remote inference enigne in AReaL support manually
designating server addresses by passing `addr` argument to
`RemoteSGLangEngine.initialize`.

## Manually Recording Server Addresses

For example:

```python
from areal.engine.sglang_remote import RemoteSGLangEngine

def main(args):
    ...
    # Manually recorded SGLang servers addresses.
    addresses = ["192.168.0.10", "192.168.0.11", "192.168.0.12", "192.168.0.13"]
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(addr=addresses, ft_spec=ft_spec)
```

## Retrieving Server Addresses by `wait_sglang_server_addrs`

For convenience, AReaL launchers could launch SGLang servers and automatically record
their addresses with key-value storage. AReaL provides a utility function named
`wait_sglang_server_addrs` to retrive these addresses easily in the training script.

For example, use the launcher with `LLM_SERVER_ONLY` allocation mode to launch inference
servers:

```bash
# SLURM launcher for example, you could use other launchers.
python3 -m areal.launcher.slurm my_script.py --config sglang_server.yaml allocation_mode=sglang.d4 experiment_name=sglang-server trial_name=xxx
```

Then in your training script, you could retrieve SGLang server addresses using
`areal.utils.launcher.wait_sglang_server_addrs` and corresponding experiment and trial
names.

```python
from areal.utils.launcher import wait_sglang_server_addrs
from areal.utils import names, name_resolve
from areal.api.alloc_mode import AllocationMode

def main(args):
    config, _ = load_expr_config(args, ...)
    ...
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    n_sglang_servers = allocation_mode.gen.data_parallel_size
    addresses = wait_sglang_server_addrs(
        config.experiment_name
        config.trial_name,
        n_sglang_servers,
    )
```

## An LLM Judge Example

Following code snippet shows an example that involves LLM-as-Judge in the training loop,
using multiple remote inference engines with different server addresses.

```python
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai import ArealOpenAI
from areal.api.alloc_mode import AllocationMode

rollout_engine = RemoteSGLangEngine(config.actor)
# Use addresses in `AREAL_LLM_SERVER_ADDRS`, set by launchers
rollout_engine.initialize(None, ft_spec)

# Collect server addresses for LLM judge manually
addresses = ["192.168.0.10", "192.168.0.11", "192.168.0.12", "192.168.0.13"]
# or, collect server addresses by `wait_sglang_server_addrs`
allocation_mode = AllocationMode.from_str(config.allocation_mode)
n_sglang_servers = allocation_mode.gen.data_parallel_size
addresses = wait_sglang_server_addrs(
    config.experiment_name
    config.trial_name,
    n_sglang_servers,
)

llm_judge_engine = RemoteSGLangEngine(config.llm_judge)
llm_judge_engine.initialize(addresses, ft_spec)

client = ArealOpenAI(engine=rollout_engine, tokenizer=tokenizer)
llm_judge_client = ArealOpenAI(engine=llm_judge_engine, tokenizer=tokenizer)

# collect a trajectory with client
async def collect_a_trajectory(client, data):
   ... // agent workflow
   return answer

# run llm as judge
async def run_llm_judge(ground_truth, answer):
    judge_prompt = ... // prompt for llm as judge
    judge_completion = await llm_judge_client.chat.completions.create(
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=1.0,
            max_completion_tokens=8192,
        )
    return judge_completion.choices[0].message.content
```
