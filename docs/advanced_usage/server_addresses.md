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

## Recording Server Addresses by Name Resolve

AReaL also provides a key-value store called name resolve, whose backend could be shared
storage, distributed KV store service such as `etcd` or key-value store in ray. You can
use name resolve to record inference server addresses and retrive them in the training
script easily.

When launching inference server:

```python
from areal.utils import name_resolve
server_addr = "192.168.0.11"
... # launch your inference server with server_addr
# after server ready
name = "xxx/xxx" # or anything else for a key
name_resolve.add_subentry(name, server_addr)
```

In the training script:

```python
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import name_resolve

def main(args):
    ...
    # Retrive SGLang servers addresses recorded by name_resolve.
    name = "xxx/xxx" # or anything else you have set when recording server addresses
    addresses = name_resolve.get_subtree(name)
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(addr=addresses, ft_spec=ft_spec)
```

## An LLM Judge Example

Following code snippet shows an example that involves LLM-as-Judge in the training loop,
using multiple remote inference engines with different server addresses.

```python
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai import ArealOpenAI

rollout_engine = RemoteSGLangEngine(config.actor)
# Use addresses in `AREAL_LLM_SERVER_ADDRS`, set by launchers
rollout_engine.initialize(None, ft_spec)

# Collect server addresses for LLM judge manually
addresses = ["192.168.0.10", "192.168.0.11", "192.168.0.12", "192.168.0.13"]
# or, collect server addresses by name resolve in AReaL
from areal.utils import name_resolve
name = "xxx/llm_judge" # or anything else you have set when recording server addresses
addresses = name_resolve.get_subtree(name)

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
