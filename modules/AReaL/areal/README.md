# AReaL Design Document

## TL;DR

Follow our
[step-by-step code walkthrough](https://inclusionai.github.io/AReaL/lite/gsm8k_grpo.html)
to get started with AReaL immediately!

## Motivation

AReaL provides an *algorithm-first* design philosophy built around three core
principles:

- **Lightweight customization:** Implement algorithms and training workflows with
  minimal, focused code—often in a single file or just a few files.
- **Effortless scaling:** Scale experiments seamlessly without needing deep knowledge of
  underlying system or infrastructure details.
- **Ecosystem integration:** Freely integrate with code or APIs from other AI libraries,
  or plug AReaL APIs into other frameworks.

## Design Principles

To achieve an *algorithm-first* and *lightweight* design while maintaining efficiency,
AReaL is guided by seven core principles:

1. **Native asynchronous RL training** — Built from the ground up for decoupled
   generation and training
1. **System-abstracted design** — Minimize exposure to low-level system concepts like
   "PlacementGroup"
1. **PyTorch-centric approach** — Use native PyTorch types without unnecessary
   abstractions
1. **Transparent orchestration** — Make the flow of operations clear and understandable
1. **Developer-friendly navigation** — Enable easy access to implementation details via
   IDE features (Ctrl+click)
1. **Ecosystem compatibility** — Integrate seamlessly with existing ML/RL tools
1. **Single-file customization** — Support RL pipeline modifications within a single
   file

## Architecture

### Core Directory Structure

```
areal/
├── api/           # Abstract interfaces and dataclasses
├── engine/        # Training and inference engines
├── launcher/      # Launcher for different backends
├── tests/         # Standalone test scripts
└── workflow/      # Custom RL rollout workflows
```

### Component Overview

The AReaL codebase is structured into four distinct layers: API, backend, customization,
and entry point. As illustrated below, workflow and algorithm customization logic
resides in separate layers above the backend. This design keeps the entry point and
customization layers clean and intuitive by isolating them from complex backend
implementation details. Users can define custom agentic training workflows and
algorithms entirely within a single entry point file.

![areal-lite-layers](../assets/areal_lite_layers.png)

#### 1. API Layer (`api/`)

The API layer establishes clean contracts between components through abstract interfaces
and data classes:

- **`engine_api.py`**: Defines `TrainEngine` for SPMD-based distributed training
  backends and `InferenceEngine` for streaming LLM inference
- **`workflow.py`**: Defines `RolloutWorkflow` for RL data collection with a unified
  method interface
- **`cli_args.py`**: Configuration dataclasses for all system components

The workflow object invokes `InferenceEngine` to complete data collection following
customized patterns, providing flexibility while maintaining consistency.

AReaL's design philosophy **discourages** implementing base classes or
infrastructure/algorithm logic in the API layer. This layer should contain only API
interfaces and utility dataclass objects. AReaL prioritizes
[composition](https://www.youtube.com/watch?v=hxGOiiR9ZKg) and
[dependency injection](https://www.youtube.com/watch?v=DpMGEhwuuyA) patterns over
inheritance.

#### 2. Backend Layer (`engine/`)

The backend layer provides adapters for third-party libraries, ensuring they conform to
the APIs defined in `engine_api.py`. These components deliver core inference and
training capabilities:

- **`fsdp_engine.py`**: FSDP-based training engine using PyTorch FSDP2
- **`megatron_engine.py`**: Megatron-LM based training engine
- **`sglang_remote.py`**: Client interface for remote SGLang server generation
- **`vllm_remote.py`**: Client interface for remote vLLM server generation

We design APIs to ensure concrete algorithm implementations (discussed next) remain
backend-agnostic. This layer abstracts the complexity of training and inference
infrastructure, allowing system developers to focus on deep profiling and optimizations.

#### 3. Customization Layer (`engine/ppo/`, `workflow/`)

This layer leverages backend capabilities to implement specific reinforcement learning
pipelines. Algorithm and agentic workflow implementations are backend-agnostic thanks to
the composition pattern:

- **`engine/ppo/actor.py`**: PPO/GRPO algorithm leveraging `TrainEngine`
- **`engine/ppo/critic.py`**: PPO critic implementation
- **`engine/sft/model.py`**: Supervised fine-tuning implementation
- **`engine/rw/model.py`**: Reward model training implementation
- **`workflow/rlvr.py`**: RLVR workflow using `InferenceEngine` to sample multiple
  responses per prompt

New algorithms and application-level agents should be implemented at this layer. If you
are familiar with Rust or Go, the algorithm implementations in AReaL are actually
[traits](https://doc.rust-lang.org/book/ch10-02-traits.html) or
[interfaces](https://go.dev/tour/methods/9). We essentially attach the
algorithm-specific functionalities to a specific training backend, which is considered
to be scalable and easy-to-maintain.

#### 4. Entry Point Layer (`examples/`)

The entry point layer composes customization layer implementations into complete RL
training pipelines. While we provide reference examples, users have complete freedom to
adapt them to specific use cases.

Entry points are launched using launchers from `areal/launcher/`, similar to distributed
launch tools like `torchrun`:

```bash
python3 -m areal.launcher.ray entrypoint.py --config my-config.yaml
```

## Usage Examples

### Basic RL Training

A YAML configuration file is required, though configuration parameters can be overridden
for hyperparameter searches or experimental variations:

```bash
# Launch with Ray launcher: 4 nodes (4 GPUs each), 3 nodes for generation, 1 for training
python3 -m areal.launcher.ray examples/math/gsm8k_grpo.py \
    --config examples/math/gsm8k_grpo.yaml \
    experiment_name=<your_experiment_name> \
    trial_name=<your_trial_name> \
    allocation_mode=sglang:d12p1t1+d4p1t1 \
    cluster.n_nodes=4 \
    cluster.n_gpus_per_node=4

# Launch with Slurm launcher: 16 nodes (8 GPUs each), 12 for generation, 4 for training
python3 -m areal.launcher.slurm examples/math/gsm8k_grpo.py \
    --config examples/math/gsm8k_grpo.yaml \
    experiment_name=<your_experiment_name> \
    trial_name=<your_trial_name> \
    allocation_mode=sglang:d96p1t1+d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8
```

## Customization Guide

For detailed customization instructions, please refer to our documentation:

- [Adding new agents](https://inclusionai.github.io/AReaL/customization/agent.html)
- [Adding new datasets](https://inclusionai.github.io/AReaL/customization/dataset.html)
- [Adding new algorithms](https://inclusionai.github.io/AReaL/customization/algorithm.html)

## Implementation Details

### Entry Point Design Philosophy

We considered two primary design patterns for entry points, each with distinct
tradeoffs:

#### Single-Controller Pattern

The most modular approach uses a single-controller pattern where only one process in the
cluster executes the main coordination logic.

**Note**: The following code snippet represents a **conceptual design pattern**.
`RolloutController` and `TrainController` lack concrete implementations in the current
codebase. This example serves as an architectural reference for future extensibility.

```python
def my_reward_fn(prompt, completion, prompt_ids, completion_ids, **kwargs):
    return len(completion_ids)

class MyRolloutWorkflow:
    async def arun_episode(self, engine: InferenceEngine,
                           data: dict[str, Any]) -> dict[str, Any]:
        message = [
            {"role": "system", "message": ...},
            {"role": "user", "message": ...},
        ]

        for _ in range(self.config.num_turns):
            text = tokenizer.apply_chat_template(message, tools=self.env.list_tools())
            req = ModelRequest(text=text, ...)
            resp = await engine.agenerate(req)
            tool_name, tool_args = parse_tool(resp)
            cur_time = await self.env.aexecute(tool_name, tool_args)
            message += [{"role": "user", "message": f"The current time is {cur_time}"}]

        reward = my_reward_fn(None, None, None, req.input_ids, **data)
        return output

def main_grpo():
    config, _ = load_expr_config(args, GRPOConfig)

    # Create rollout workflow
    workflow = MyRolloutWorkflow()

    # Single-controller mode initialization
    scheduler = SlurmScheduler()
    rollout = RolloutController(SGLangEngine, config=config.rollout, scheduler=scheduler)
    actor = TrainController(MegatronPPOActor, config=config.actor, scheduler=scheduler)

    # Training loop
    dataloader = StatefulDataLoader(dataset)
    for _ in range(max_steps):
        # Collect trajectories using rollout workflow
        batch = rollout.prepare_batch(dataloader, workflow=workflow)
        batch: DistributedBatch  # For distributed coordination across processes

        # Prepare training inputs
        batch = actor.compute_advantages(batch)

        # Execute PPO update
        actor.ppo_update(batch)

        # Update inference engine weights (non-blocking to prevent NCCL blocking)
        actor.update_weights(wcfg)
```

**Advantages:**

- Maximum flexibility for device allocation, scheduling, and data arrangement

**Disadvantages:**

- Introduces multiple abstractions (`TrainController`, `Scheduler`, `DistributedBatch`)
  that increase script complexity

#### SPMD Pattern

Given AI researchers' familiarity with the SPMD (Single Program, Multiple Data) pattern
used in standard model training, we also provide entry points following this approach.
With N GPUs dedicated to training, N processes execute the following code:

**Note**: The following code snippet is based on the actual implementation in
`examples/math/gsm8k_grpo_megatron.py` but simplified for demonstration.

```python
def main_grpo():
    config, _ = load_expr_config(args, GRPOConfig)

    # Create rollout workflow
    workflow = MyRolloutWorkflow()

    # SPMD mode initialization
    rollout = RemoteSGLangEngine(config.rollout)
    actor = MegatronPPOActor(config.actor)

    # Training loop
    dataloader = StatefulDataLoader(dataset)
    for _ in range(max_steps):
        # Data collection using prepare_batch with distributed coordination
        batch = actor.prepare_batch(
            dataloader,
            granularity=actor.config.group_size,  # For GRPO grouping
            workflow=workflow,
        )
        batch: dict[str, Any]

        # Prepare training inputs
        batch = actor.compute_advantages(batch)

        # Execute PPO update
        actor.ppo_update(batch)

        # Update weights (coordinated across processes)
        actor.update_weights(wcfg)
```

Each SPMD process launches a CPU client connecting to inference servers
(`RemoteSGLangEngine`) and uses the train engine (`MegatronGRPOActor`) to run
distributed training on GPU.

**Advantages:**

- Uses only concepts familiar to AI researchers

**Disadvantages:**

- Requires some control flow branching based on parallelism strategy
- May incur data imbalance because prompts are evenly partitioned across processes
- Less flexible for allocating multiple models

### Training Engine Architecture

The training engine operates at two abstraction levels, balancing flexibility with ease
of use.

#### Basic Level: Backend Adapters

The foundational level provides unified interfaces for RL algorithms, handling
computation, parameter management, and weight updates for inference engines. Each RL
training experiment must use one of the implemented backends:

```python
class TrainEngine(abc.ABC):

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
        """Initialize distributed training environment and load models."""
        raise NotImplementedError()

    def destroy(self):
        """Clean up engine resources and release GPU memory."""
        pass

    def update_weights(self, meta: WeightUpdateMeta):
        """Update weights to inference engine (blocking operation)."""
        raise NotImplementedError()

    def connect_engine(self, engine: "InferenceEngine", meta: WeightUpdateMeta):
        """Connect to an inference engine for online training."""
        raise NotImplementedError()

    def save(self, meta: SaveLoadMeta):
        """Save model weights and optimizer states for checkpointing."""
        raise NotImplementedError()

    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from checkpoint."""
        raise NotImplementedError()

    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Update model parameters using provided batch and loss function."""
        raise NotImplementedError()

    @torch.no_grad()
    def forward(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        post_hook: Callable[[torch.Tensor, dict[str, Any]], Any] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Execute gradient-free forward pass for inference."""
        raise NotImplementedError()
```

#### Algorithm Level: Extended Engines

Extended engines like PPO Actor provide algorithm-specific organization and
computational interfaces. They leverage backend core methods (such as `forward`) to
generate algorithm-required tensors and execute specialized model updates. The produced
objects (e.g., `FSDPPPOActor`) are also instances of `TrainEngine`, but with methods
specifically designed for the algorithm (e.g., `ppo_update`).

```python
class PPOActor:

    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine
        self.temperature = config.temperature

    @torch.no_grad()
    def compute_logp(
        self,
        data: dict[str, Any],
    ) -> torch.Tensor | None:

        def calc_logprobs(logits, input_data):
            labels = torch.roll(input_data["input_ids"], shifts=-1, dims=-1)
            logprobs = gather_logprobs(logits, labels, self.temperature)
            return logprobs

        self.engine.eval()
        return self.engine.forward(
            input_=data,
            post_hook=calc_logprobs,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    def compute_advantages(self, data: dict[str, Any]) -> None:
        """Compute advantages for PPO training."""
        # Implementation details...
        pass

    def ppo_update(self, data: dict[str, Any]) -> list[dict[str, float]]:
        """Execute PPO policy update."""
        # Implementation details...
        pass

class FSDPPPOActor(FSDPEngine):
    """FSDP-backed PPO Actor implementation."""

    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> None:
        self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> list[dict[str, float]]:
        return self.actor.ppo_update(*args, **kwargs)
```

### Inference Engine Design

The inference engine's core functionality revolves around `generate` and
`update_weights` methods. These methods can interface with HTTP server APIs or invoke
local LLM engines:

```python
class InferenceEngine(abc.ABC):

    def initialize(self, addr: str | None, ft_spec):
        """Initialize distributed inference environment and load models."""
        raise NotImplementedError()

    def destroy(self):
        """Clean up engine resources and release GPU memory."""
        pass

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Generate response asynchronously for the given request."""
        raise NotImplementedError()

    def update_weights(self, meta: WeightUpdateMeta) -> Future:
        """Update inference engine weights asynchronously."""
        raise NotImplementedError()
```

#### Workflow Integration

User-defined rollout workflows utilize `InferenceEngine` to generate trajectories. The
workflow's `arun_episode` method produces one or more trajectories from a single prompt.
The generation process is streaming rather than batched, with each dataset item
processed independently. Here's a simplified RLVR example:

```python
class RLVRWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine: InferenceEngine, data: dict[str, Any]):
        input_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        n_samples = self.gconfig.n_samples
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
        )

        # Generate multiple responses concurrently
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])

        results = []
        for resp in resps:
            reward = self.reward_fn(
                prompt=prompt_str,
                completions=completions_str,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )

            results.append(res)

        return concat_padded_tensors(results)
```

#### Batch Processing and Asynchronous Operations

While individual trajectory collection is straightforward, batching and asynchronous
execution require additional infrastructure. `InferenceEngine` provides high-level
methods: `submit`, `wait`, and `prepare_batch`.

The `prepare_batch` method submits multiple `workflow.arun_episode` jobs to an
asynchronous thread pool using `submit`, then waits for completion using `wait`. This
enables controlled staleness and asynchronous rollout generation:

**Note**: The code below is simplified for clarity. See `core/workflow_executor.py` for
full implementations with staleness management, performance tracing, and result
filtering.

```python
@dataclass
class _RolloutTaskInput:
    """Internal wrapper for rollout-specific task input."""
    data: dict[str, Any]
    workflow: RolloutWorkflow
    should_accept_fn: Callable | None = None
    request_id: int | None = None  # For performance tracing

def submit(
    self,
    data: dict[str, Any],
    workflow: "RolloutWorkflow" | None = None,
    workflow_builder: Callable | None = None,
    should_accept_fn: Callable | None = None,
) -> None:
    """Submit a request to the workflow executor.

    See workflow_executor.py:513-546 for full implementation.
    """
    if workflow is None:
        workflow = workflow_builder()

    # Tasks are queued internally (not directly via queue.put_nowait)
    self._pending_inputs.append(
        _RolloutTaskInput(
            data=data,
            workflow=workflow,
            should_accept_fn=should_accept_fn,
        )
    )
    # Notify staleness manager of enqueued rollout tasks
    self.staleness_manager.on_rollout_enqueued()

def wait(
    self,
    count: int,
    timeout: float | None = None,
    raise_timeout: bool = True  # Allow quiet waiting when timeout occurs
) -> dict[str, Any]:
    """Wait for specified number of results with optional filtering.

    See workflow_executor.py:569-653 for full implementation including:
    - Capacity-based submission control (staleness management)
    - Result filtering for rejected trajectories
    - Performance tracing and result shuffling
    """
    # Simplified: actual implementation has staleness control,
    # result filtering, caching, and performance tracing
    pass

def prepare_batch(
    self,
    dataloader: StatefulDataLoader,
    workflow: "RolloutWorkflow" | None = None,
    workflow_builder: Callable | None = None,
    should_accept_fn: Callable | None = None,
):
    """Prepare batch for asynchronous processing with controlled staleness.

    See workflow_executor.py:655-693 for full implementation including:
    - Data generator creation/caching
    - Staleness control via staleness_manager
    - Queue size checking and loop-based submission
    """
    # Simplified: actual implementation orchestrates the entire
    # async rollout pipeline with staleness and capacity management
    pass
```

### RolloutWorkflow Interface

The `RolloutWorkflow` class provides the `arun_episode` method with a standardized
signature for collecting agent trajectories.

**Note**: The example below is pedagogical and demonstrates a tool-calling workflow
pattern. For production implementations, see `workflow/rlvr.py` for simple multi-sample
rollouts or `workflow/multi_turn.py` for complex multi-turn interactions with reward
feedback.

```python
class MyRolloutWorkflow:
    def __init__(self, config: Any):
        self.config = config
        self.tool_executor = ToolExecutor()
        self.tool_executor.register_tool(get_current_time)

    async def arun_episode(self, engine: InferenceEngine,
                           data: dict[str, Any]) -> dict[str, Tensor]:
        req = ModelRequest(input_ids=data['input_ids'], ...)

        for _ in range(self.config.num_turns):
            resp = await engine.agenerate(req)
            res = await self.tool_executor.aexecute_tool(resp.completion)
            req.input_ids += res

        reward = my_reward_fn(None, None, None, req.input_ids, **data)
        return output
```

### Controller Architecture

`RolloutController` and `TrainController` mirror the APIs of `InferenceEngine` and
`TrainEngine`, respectively. Controllers handle engine deployment across the cluster and
manage data distribution, invoking engine methods through remote procedure calls (RPCs).
This architecture enables distributed operation while maintaining familiar interfaces.
