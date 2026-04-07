### Synchroush Rollout Design

Verl-Tool is designed to decouple the RL training and the tool calling processes. The RL training components (computation of logp, reward, advantages, RL algorithms, etc.) are handled by the verl framework, while the tool calling process is managed by the tool server **through an additional plugin to the verl rollout**.

We achieve this by inheriting from `ActorRolloutRefWorker` → `AgentActorRolloutRefWorker` in [verl_tool/worker/fsdp_workers.py](verl_tool/worker/fsdp_workers.py) and then overriding the original RefWorker in [verl_tool/trainer/main_ppo](verl_tool/trainer/main_ppo.py) by adding this line:

```python
from verl_tool.trainer.ppo.ray_trainer import AgentRayPPOTrainer as RayPPOTrainer
```

We only modify one function after inheriting the `ActorRolloutRefWorker` class: the `generate_sequences` function. We add a conditional statement to delegate the agent rollout process to `AgentActorManager.run_llm_loop` in [verl_tool/llm_agent/manager.py](verl_tool/llm_agent/manager.py):

```python
class AgentActorRolloutRefWorker(Worker, ActorRolloutRefWorker, metaclass=AgentActorRolloutRefWorkerMeta):
    def __agent_init__(self, config: DictConfig, role: str):
        # agent init
        ...
        self.manager = AgentActorManager(...)
        ...
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        ...
            if not self.agent_config.enable_agent:
                # old behavior
                output = self.rollout.generate_sequences(prompts=prompts)
            else:
                # agent behavior (delegated to AgentActorManager)
                output = self.manager.run_llm_loop(prompts) # our agent behavior
        ...
```

The `AgentActorManager` handles the multi-turn interaction between the model and the tool server, where the model can call tools and receive observations from the tool server. Please check the detailed design in [verl_tool/llm_agent/manager.py](verl_tool/llm_agent/manager.py).

Configuration parameters are defined in the `AgentActorConfig` class in [verl_tool/llm_agent/config.py](verl_tool/llm_agent/config.py). You can set these parameters by adding `actor_rollout_ref.agent.{param_name}=value` to the training command.