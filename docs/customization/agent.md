# Rollout and Agentic RL

This guide demonstrates how to customize rollout behavior for PPO training by
implementing a multi-turn math agent that uses end-to-end reinforcement learning. Our
example agent will continuously try to solve a math problem until it reaches the correct
answer.

## Approach: Using AReaLite (Recommended)

The complete implementation is placed at `arealite/workflow/multi_turn.py`.

### Step 1: Define Your Workflow

AReaLite takes a flexible approach to agent definition. Rather than using rigid `Agent`
classes that might limit your agentic capabilities, AReaLite captures all rollout
behavior in a `RolloutWorkflow` class. This design gives you complete freedom to
customize your agent's behavior.

```python
# arealite/api/workflow_api.py
class RolloutWorkflow:
    async def arun_episode(
        self, engine: InferenceEngine, data: Dict[str, Any]
    ) -> TensorDict:
        """Run a single episode of the workflow.

        See concrete example implementations under the `arealite/workflow` directory.
        """
        raise NotImplementedError()
```

The workflow exposes a single `arun_episode` method that runs and collects data from a
single episode. This method takes two key arguments:

1. **InferenceEngine**: Provides the `agenerate` method for generating responses to user
   inputs
1. **data**: The prompt data loaded from your RL dataset

Within this method, you have complete control over how your agent and environment
interact.

#### Setting Up the Multi-Turn Math Workflow

Let's build a multi-turn rollout workflow for solving math problems. First, we'll define
the `__init__` method to capture the utilities we need during rollout:

> **Note**: You have complete flexibility in defining the `__init__` method. Pass any
> arguments needed to construct your workflow. If you want to use tools, pass the
> corresponding environment here so your agent can call it in the `arun_episode` method.

```python
class MultiTurnWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,  # aka sampling_params
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int,
        turn_discount: float,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        # Discount rewards if the agent takes longer to find the correct answer
        self.turn_discount = turn_discount
```

#### Implementing the Episode Logic

Now let's implement the `arun_episode` method. We'll start by tokenizing the prompt data
and converting it into an `LLMRequest` object for the inference engine:

```python
class MultiTurnWorkflow(RolloutWorkflow):
    # ... __init__ method above ...

    async def arun_episode(self, engine: InferenceEngine, data) -> TensorDict:
        # Initialize result containers
        seq, logprobs, loss_mask, versions = [], [], [], []
        messages = data["messages"]
        # Run multi-turn rollout until we get the correct answer
        t = reward = 0
        discount = 1.0
        rid = uuid.uuid4().hex
        while reward == 0 and t < self.max_turns:
            # Convert the conversation into input tokens
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            # Generate response from the model
            req = LLMRequest(
                rid=rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
            )
            resp = await engine.agenerate(req)
            # ... continue processing ...
```

> **Note**: This example accesses the "messages" key from the prompt data to get
> OpenAI-compatible messages. This isn't mandatory—the key and prompt format depend
> entirely on your implementation. For instance, if your dataset stores prompt strings
> in a "prompt" column, you could get input token IDs via
> `self.tokenizer.encode(data["prompt"])`.

> **Note**: The `rid` field in `LLMRequest` is the request ID. Requests with the same ID
> will reuse the LLM inference server's KV caches for efficiency.

#### Handling Multi-Turn Conversations

Next, we'll evaluate whether the current answer is correct using our `reward_fn`. This
function should return 1 for correct answers and 0 otherwise. When the answer is wrong,
we'll apply a discount, add feedback to the conversation, and let the model try again:

```python
class MultiTurnWorkflow(RolloutWorkflow):
    # ... previous methods ...

    async def arun_episode(self, engine: InferenceEngine, data) -> TensorDict:
        # ... initialization code ...
        while reward == 0 and t < self.max_turns:
            # Add feedback if the previous answer was incorrect
            if t > 0:
                messages += [
                    {"role": "assistant", "content": completions_str},
                    {
                        "role": "user",
                        "content": "Your answer is not correct. Please try to answer it again."
                    },
                ]
            # Generate response (code from above)
            # ...
            # Evaluate the response
            prompt_str = self.tokenizer.decode(input_ids)
            completions_str = self.tokenizer.decode(resp.output_tokens)
            reward = self.reward_fn(
                prompt=prompt_str,
                completions=completions_str,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )
            # Update counters
            t += 1
            discount *= self.turn_discount
```

#### Reward Function Signature

For convenience when switching between different reward functions, we recommend
following this pre-defined signature:

```python
def reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: List[int],
    completion_ids: List[int],
    **kwargs,
):
    """Reward function for evaluating agent performance.

    This signature is recommended for compatibility with predefined workflows,
    but you can modify it freely in custom implementations.

    Args:
        prompt: The task description string
        completions: The agent's response string
        prompt_ids: Tokenized prompt
        completion_ids: Tokenized response
        **kwargs: Additional dataset attributes (solutions, input_outputs, etc.)

    Returns:
        float: Reward value (typically 1.0 for correct, 0.0 for incorrect)
    """
```

While this signature is convenient, there are no strict restrictions on reward functions
in custom workflows—modify them as needed for your specific use case.

#### Collecting Training Data

Finally, let's complete the implementation by collecting trajectories in the
`TensorDict` format:

```python
class MultiTurnWorkflow(RolloutWorkflow):
    # ... previous methods ...

    async def arun_episode(self, engine: InferenceEngine, data) -> TensorDict:
        # ... episode logic above ...

        while reward == 0 and t < self.max_turns:
            # ... generation and evaluation ...

            # Collect trajectory data
            input_len = len(resp.input_tokens) - len(seq)
            seq += resp.input_tokens[-input_len:] + resp.output_tokens
            logprobs += [0.0] * input_len + resp.output_logprobs
            loss_mask += [0] * input_len + [1] * resp.output_len
            versions += [-1] * input_len + resp.output_versions

        # Package results
        res = dict(
            input_ids=torch.tensor(seq),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask),
            versions=torch.tensor(versions),
            rewards=torch.tensor(float(reward * discount)),
            attention_mask=torch.ones(len(seq), dtype=torch.bool),
        )
        res = {k: v.unsqueeze(0) for k, v in res.items()}
        return concat_padded_tensors([res])
```

> **Important**: The returned `TensorDict` must follow HuggingFace's padded data format,
> where each tensor has shape `[batch_size, sequence_length, *]`. This allows AReaLite
> to automatically batch multiple trajectories for training. Since this example returns
> a single trajectory, we use `unsqueeze(0)` to create a batch of size 1.

> **Note**: You're not restricted to specific keys in your `TensorDict`—different
> algorithms need different keys. This example targets the GRPO algorithm, so we include
> `input_ids`, `loss_mask`, `attention_mask`, and `logprobs` (needed for computing
> importance ratios).

## Step 2: Training with Your Custom Workflow

Using your custom workflow is straightforward—just create it in your training script and
pass it to the `rollout_batch` or `prepare_batch` method:

```python
def main(args):
    # ... setup code ...

    # Create your custom workflow
    workflow = MultiTurnWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        turn_discount=0.9,
        max_turns=5,
    )

    # Run training—no other changes needed!
    data_generator = itertools.cycle(train_dataloader)
    for global_step in range(max_steps):
        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            else:
                batch = rollout.rollout_batch(next(data_generator), workflow=workflow)
        # ... continue with training loop ...
```

Then update your experiment configuration in
`realhf/experiments/async_exp/async_math_ppo.py`:

```python
@dataclasses.dataclass
class AsyncPPOMATHConfig(AsyncRLExperimentConfig, PPOMATHConfig):
    # Add any new CLI arguments your agent needs
    my_param: float = 1.0

    @property
    def agent(self) -> AgentAbstraction:
        return AgentAbstraction(
            "math-multi-turn",  # Your registered agent name
            args=dict(
                # Pass any arguments needed for your __init__ method
                my_param=self.my_param,
                # ... other configuration ...
            ),
        )

    @property
    def env(self) -> EnvServiceAbstraction:
        # Update to use your custom environment if needed
        return EnvServiceAbstraction(
            "math-code-single-step",
            args=dict(dataset_path=self.dataset.path)
        )
```

### Step 4: Run Training

Follow the standard training procedure outlined in the
[quickstart guide](../tutorial/quickstart.md). Launch your experiment with:

```bash
python3 training/main_async_ppo.py my_param=5.0  # plus any additional CLI arguments
```

### Training Results

Here's an example of the training reward curve from our multi-turn math agent:

![Multi-turn Training Rewards](multiturn_reward.png)

The agent successfully learns to solve math problems with improved accuracy over time,
demonstrating the effectiveness of the multi-turn approach.

______________________________________________________________________

**Note**: While this legacy approach works, we strongly recommend using the AReaLite
workflow system for new projects. It provides better flexibility, cleaner abstractions,
and easier maintenance. Consider migrating existing legacy agents to the workflow-based
approach when possible.

Happy coding!
