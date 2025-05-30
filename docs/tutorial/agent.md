# Agentic RL

This guide provides an example of training a multi-turn math agent using end-to-end RL. The math agent will continuously attempt to think through and solve math problems until it reaches the correct answer.

## Define Your Agent

Create a new file under `realhf/impl/agent/`, for example, `math_multi_turn_agent.py`. Your `Agent` must implement the interface defined in `realhf/api/core/agent.py`, which requires implementing a single method: `collect_trajectory`.

```python
class MathMultiTurnAgent(Agent):
    def __init__(
        self,
        ... # Any required configurations here
    ):
        ...
    
    async def collect_trajectory(
        self,
        prompt: SequenceSample,
        env: EnvironmentService,
        obs_queue: asyncio.Queue,
        act_queue: asyncio.Queue,
    ) -> List[SequenceSample]:
        ...
```

## Implement the `collect_trajectory` Logic

The `collect_trajectory` function takes a task prompt, an environment, and two queues as input, then produces several trajectories for the RL trainer. Within this function, you can create arbitrary data processing logic to produce the input prompt for the inference engine and extract the action from the generated tokens.

In this example, the initial observation is the math problem itself, which is already included in the `prompt` parameter. We put the token IDs and generation config into `obs_queue` and wait for the action produced by the inference engine from `act_queue`. After the inference engine returns, we extract the generated answers and send them to the environment.

```python
for turn in range(self.num_turns):
    await obs_queue.put((qid, token_ids, self.gconfig))
    act: BundledGenerationOutputs = await act_queue.get()
    ...
    _, success, *_ = await env.step((qid, answers))
```

The environment is similar to a [gym environment](https://github.com/Farama-Foundation/Gymnasium), which defines two methods: `reset` and `step`. However, to maintain efficiency, we use an asynchronous implementation to avoid mutual blocking across different environment instances.

Although the environment can be quite complex (e.g., for a SWE-agent), the implementation in this example is straightforward. The math environment is single-step and essentially serves as a wrapper around the reward function:

```python
class MathCodeSingleStepEnv(EnvironmentService):

    async def reset(self, seed=None, options=None):
        return None, {}

    async def step(self, action: Tuple[str, List[str]]):
        qid, answers = action
        group_size = len(answers)
        qid = qid.split("@")[0]
        cur_task = self.id2info[qid]["task"]

        format_rewards = await asyncio.to_thread(
            math_verify_call,
            self.id2info,
            answers,
            [qid for _ in range(group_size)],
        )

        return None, format_rewards, True, False, {}
```

After `env.step` returns the reward for the current step, we can check whether the answer is correct. If not, we can append a user prompt and send it to `obs_queue` again to enter the next round.

```python
for turn in range(self.num_turns):
    ...
    feedback = None
    if success[0]:
        feedback = "Congratulations! You are correct!"
    else:
        feedback = "Unfortunately your answer is wrong. Let's try again."
    
    feedback = "\n" + self.tokenizer.apply_chat_template(
        [dict(content=feedback, role="user")],
        add_generation_prompt=True,
        tokenize=False,
    )
    feedback = self.tokenizer(feedback)["input_ids"]
    token_ids.extend(feedback)
```

## Modify the Configuration

Finally, arrange all the data in the proper format and return it. You're now close to running the end-to-end RL loop. The final step is to register and import your implementation, then modify the experiment configuration.

```python
# in realhf/impl/agent/math_multi_turn_agent.py
register_agent("math-multi-turn", MathMultiTurnAgent)
```

```python
# in realhf/impl/agent/__init__.py
import realhf.impl.agent.math_multi_turn_agent
```

```diff
@dataclasses.dataclass
class AsyncPPOMATHConfig(AsyncRLExperimentConfig, PPOMATHConfig):
    # in realhf/experiments/async_exp/async_ppo_math_exp.py
    @property
    def agent(self) -> AgentAbstraction:
        return AgentAbstraction(
-           "math-single-step",
+           "math-multi-turn",  # Your registered name
            args=dict(
-                gconfig=self.generation_config,
-                tokenizer_path=self.actor.path,
-                success_rate_lb=self.success_rate_lb,
-                success_rate_ub=self.success_rate_ub,
-                reward_scaling=self.ppo.reward_output_scaling,
-                reward_bias=self.ppo.reward_output_bias,
+                # Any configurations for your agent
+                ...
            ),
        )

    @property
    def env(self) -> EnvServiceAbstraction:
-        return EnvServiceAbstraction(
-            "math-code-single-step", args=dict(dataset_path=self.dataset.path)
-        )
+        # Change to your customized environment if necessary
+        # The same registration and importing mechanism as Agents
+        return EnvServiceAbstraction(
+            "my-env", args=dict(...)
+        )
```

## Run Training

Please follow the guide in [training.md](training.md). Generally, create a YAML configuration called `math-multi-turn.yaml` under `training/configs/async-ppo` and run:

```bash
python3 training/main_async_ppo.py --config-name=math-multi-turn
```

Happy coding!