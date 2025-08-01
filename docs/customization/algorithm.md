# Training Algorithm

> **Note**: We recommend the user to first read the
> [agent customization guide](agent.md).

AReaL structures RL algorithms around two core components:

- **RolloutWorkflow**: Defines what data to generate during rollouts
- **TrainEngine**: Defines how to process the generated data for training

We'll demonstrate this by implementing an RL algorithm similar to ReMax.

## Step 1: Implementing the RolloutWorkflow

The rollout workflow generates both greedy and sampled completions, then uses the reward
difference as the final training signal:

```python
class ReMaxRLVRWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine: InferenceEngine, data):
        # Prepare input tokens from chat messages
        input_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        n_samples = self.gconfig.n_samples
        rid = uuid.uuid4().hex

        # Create requests for both sampled and greedy generation
        sample_req = LLMRequest(
            rid=rid,
            input_ids=input_ids,
            gconfig=self.gconfig,
        )
        greedy_req = LLMRequest(
            rid=rid,
            input_ids=input_ids,
            gconfig=self.gconfig.new(greedy=True),
        )

        # Generate both responses concurrently
        resp, greedy_resp = await asyncio.gather(
            engine.agenerate(sample_req),
            engine.agenerate(greedy_req),
        )

        # Calculate rewards for both completions
        prompt_str = self.tokenizer.decode(input_ids)
        completions_str = self.tokenizer.decode(resp.output_tokens)

        sample_reward = self.reward_fn(
            prompt=prompt_str,
            completions=completions_str,
            prompt_ids=resp.input_tokens,
            completion_ids=resp.output_tokens,
            **data,
        )

        greedy_completions = self.tokenizer.decode(greedy_resp.output_tokens)
        greedy_reward = self.reward_fn(
            prompt=prompt_str,
            completions=greedy_completions,
            prompt_ids=greedy_resp.input_tokens,
            completion_ids=greedy_resp.output_tokens,
            **data,
        )

        # Package results for training
        res = dict(
            # Add batch dimension
            input_ids=torch.tensor(resp.input_tokens + resp.output_tokens).unsqueeze(0),
            loss_mask=torch.tensor([0] * resp.input_len + [1] * resp.output_len).unsqueeze(0),
            versions=torch.tensor([-1] * resp.input_len + resp.output_versions).unsqueeze(0),
            attention_mask=torch.ones(resp.input_len + resp.output_len, dtype=torch.bool).unsqueeze(0),
            # Use reward difference across all tokens
            rewards=torch.tensor([float(sample_reward - greedy_reward)] * (resp.input_len + resp.output_len)),
        )

        return TensorDict(res, batch_size=[1])
```

> **Note**: For detailed guidance on customizing rollout workflows, see the
> [agent customization guide](agent.md).

## Step 2: Implementing the REINFORCE Training Algorithm

Training algorithms are implemented by subclassing `TrainEngine` and using its atomic
operations like `forward`, `train_batch`, and `eval_batch`.

First, let's define the REINFORCE loss function:

```python
def reinforce_loss_fn(logits, data):
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"].bool()
    rewards = data["rewards"]

    logprobs = gather_logprobs(
        logits, torch.roll(input_ids, shifts=-1, dims=-1)
    )
    loss = -logprobs * rewards
    loss = torch.where(loss_mask, loss, 0.0)

    return loss.sum() / loss_mask.count_nonzero()
```

```{note}
To decrease memory usage, AReaL automatically packs multiple sequences in an 1D tensor before forward passes. Hence, the loss function should assume handling 1D *packed* tensors instead of *padded* tensors.
```

Next, we implement the training engine. We use a two-class design to maintain backend
compatibility:

```python
class ReinforceActor:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    def train_reinforce(self, data: TensorDict):
        # Enable gradient checkpointing
        self.engine.train()
        return self.engine.train_batch(
            data,
            loss_fn=reinforce_loss_fn,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )

class FSDPReinforceActor(FSDPEngine):
    def __init__(self):
        self.actor = ReinforceActor(self)

    def train_reinforce(self, *args, **kwargs):
        return self.actor.train_reinforce(*args, **kwargs)
```

**Why two classes?** This design separates concerns:

1. **Backend Agnostic Logic**: `ReinforceActor` contains the core REINFORCE algorithm
   that works with any backend (FSDP, DeepSpeed, Megatron) since they share the same
   `train_batch` API.

1. **Backend-Specific Features**: `FSDPReinforceActor` inherits from `FSDPEngine` to
   provide backend-specific utilities like `save`, `load`, and `upload_weights`. For
   other backends, you'd create `MegatronReinforceActor`, etc.

> **Note**: This pattern is similar to interfaces in Go or traits in Rust, adapted for
> Python's object model.

## Step 3: Composing the Complete Training Loop

The main training loop brings everything together:

```python
def main(args):
    # Initialize inference engine for rollouts
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)

    # Initialize training engine
    actor = FSDPReinforceActor(config=config.actor)
    actor.initialize(None, ft_spec)

    # Create rollout workflow
    workflow = ReMaxRLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
    )

    # Main training loop
    data_generator = itertools.cycle(dataloader)
    for global_step in range(max_steps):
        # Generate training data
        with stats_tracker.record_timing("rollout"):
            batch = rollout.rollout_batch(next(data_generator), workflow=workflow)

        batch = batch.to(actor.device)

        # Synchronize all processes
        dist.barrier()
        torch.cuda.synchronize()

        # Training step
        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("actor"),
        ):
            stats = actor.train_reinforce(batch)
            actor.step_lr_scheduler()

        # Update model weights
        with stats_tracker.record_timing("update_weights"):
            # Weight update logic here
            ...
```
