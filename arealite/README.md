# AReaLite (TBD)

A simplified and easy-to-read version of AReaL with a minimal set of APIs.

## How to Use (Dev WIP)

1. Launch a wrapped SGLang server:

```bash
python3 arealite/launch_llm_server.py
```

This command reads the configuration from `config/llm_server.yaml` and launches a local SGLang server. It registers the server address for later use by the trainer.

2. Run simple tests:

```bash
python3 arealite/tests/test_rollout.py
```

This command reads the configuration from `config/async_ppo.yaml`, which uses the same experiment and trial names as those used to launch LLM servers, so the test automatically finds the server address. You can then run the rollout loop with a pre-defined environment and agent to collect training data.

## Customization

All customizations follow similar procedures:

1. Inherit the base class (e.g., `Agent`) and write your customized object under the `impl/` folder (e.g., `impl/agent/my_agent.py`).

2. Modify the factory class in the API file (e.g., `api/agent_api.py`) to allow initialization of your customized class, e.g.:

```diff
@dataclass
class AgentFactory:
    args: TrainingArgs
    client_config: LLMClientConfig

    def make_agent(self, config: AgentConfig) -> Agent:
        if config.type == "math_code_single_step":
            from arealite.impl.agent.math_code_single_step_agent import (
                MathCodeSingleStepAgent,
            )

            return MathCodeSingleStepAgent(
                self.args,
                self.client_config,
                config,
            )
+       elif config.type == "my_agent":
+           from arealite.impl.agent.my_agent import MyAgent
+           return MyAgent(
+               self.args,
+               self.client_config,
+               config,
+           )
        else:
            raise NotImplementedError(f"Unknown agent type: {config.type}")
```

3. Modify the CLI args so you can customize your agent through the command line:

```python
@dataclass
class MyAgentConfig:
    my_param: str = ''
    my_int: int = 0
```

```diff
@dataclass
class AgentConfig:
    type: str = field(default="", metadata={"help": "Agent type"})
    math_code_single_step: Optional[MathCodeSingleStepAgentConfig] = None
+   my_agent: Optional[MyAgentConfig] = None
```

Then you can use your own agent with a command like:

```bash
python3 train.py trainer.ppo.collector.agent.type=my_agent \
    trainer.ppo.collector.agent.my_agent.my_param='hello'
```

## Unit Tests

Unit tests are placed under the `tests` folder, but currently they are essentially not *unit* tests but rather standalone scripts to ensure that individual components are runnable. 

They need further refactoring to use `pytest`.

## TODOs

- [ ] Finalize API design.
- [x] Implement standalone SGLang server (`impl/sglang_server.py`).
- [x] Implement SGLang client generation (`impl/sglang_client.py`).
- [x] Rollout pipeline (`tests/test_rollout.py`).
- [ ] FSDP2 engine with transformers models.
- [ ] SGLang update weights.
- [ ] Synchronous PPO training pipeline (`impl/trainer/ppo.py`).
- [ ] SGLang rollout interruption.
- [ ] Asynchronous RL system-wide utilities (e.g., `DataPipe`).
- [ ] Asynchronous PPO training pipeline (`impl/trainer/async_ppo.py`).
- [ ] Benchmark performance versus the original AReaL code.
- [ ] Various launching scripts: ray, torchrun, slurm.
- [ ] Allow external persistent SGLang servers for debugging purposes.