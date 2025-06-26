# AReaLite Design Doc

## Motivation

AReaL is too heavy for AI researchers to use, understand, and develop with for several reasons. The most important issue is that its code architecture is *system-centric* rather than *AI-centric* — the RL algorithm workflow consists of multiple *workers* that run consecutive *model function calls*, neither of which are well-known concepts for AI researchers. As a result, users must first understand these concepts before they can develop workflows and algorithms for their own use cases.

Additionally, due to historical reasons, AReaL's code is not clean. There are large pieces of code inherited from previous projects that are not useful but significantly increase the burden on users and developers. Sometimes debugging is difficult even for core developers like myself.

Since the tools for building RL workflows are becoming increasingly mature, implementing a framework that achieves comparable efficiency requires much fewer lines of code. Now is the proper time to revisit the API design and distill the giant codebase into a neat and clean one. The distilled codebase does not need to be ultra-efficient. Instead, we want to deliver 90% functionality of the original AReaL while minimizing the lines of code and the burden on potential users. Our aim is to build an RL training framework that is fast to use, fast to read, and fast to execute. Here comes the lite version of AReaL — AReaLite.

AReaLite is the first step in AReaL's refactoring process. It is not only a standalone training library with shallow interfaces, but will also provide the core API definitions to be used by AReaL in the future. AReaL will essentially transform its current worker-based architecture into an AI-centric architecture like AReaLite. AReaL will **extend** AReaLite's APIs and implementations to support more backends for efficient large-scale training.

## Expectations of AReaLite

### Highlights

+ Fully asynchronous training with decoupled inference and training.
+ Elastic inference device scaling — users can shut down or launch more inference processes independently during training.
+ Full SFT/RL algorithmic functionality matching AReaL.
+ Arbitrary agentic rollout workflow customization in a single file.
+ Easy navigation to implementation references via Ctrl+click in VSCode.
+ Support for distributed launching with Ray/SLURM/torchrun.

### AReaLite's Scope

+ Not bound to Ray.
+ Only supports SGLang and PyTorch FSDP2 with SPMD launching.
+ No customized data structures like `SequenceSample`. All data are PyTorch tensors.
+ Uses HuggingFace (models, datasets) and PyTorch (FSDP, data structures) as much as possible.

## Architecture

### Core Components

```
arealite/
├── api/           # Abstract interfaces and data structures
├── impl/          # Concrete implementations
├── cli/           # Command-line interfaces
├── config/        # Configuration templates
└── tests/         # Standalone test scripts
```

#### 1. API Layer (`api/`)

The API layer defines abstract interfaces and data structures that provide a clean contract between different components:

- **`engine_api.py`**: Defines `SPMDWrapper` for SPMD-based training backends (FSDP) and `EngineFactory`
- **`trainer_api.py`**: Defines `Trainer` base class for different training algorithms and `TrainerFactory`
- **`rollout_api.py`**: Defines `RolloutWorkflow`, `Agent`, `Environment` for RL data collection and `RolloutWorkflowFactory`
- **`cli_args.py`**: Defines configuration dataclasses for all components

#### 2. Implementation Layer (`impl/`)

The implementation layer contains concrete implementations of the API interfaces:

- **`fsdp_wrapper.py`**: FSDP-based training engine using PyTorch FSDP2
- **`trainer/grpo.py`**: GRPO trainer implementation for reinforcement learning
- **`rollout_controller.py`**: Coordinates rollout data collection across workers
- **`rlvr/`**: RLVR (RL via Verification and Refinement) workflow implementations
- **`agentic/`**: Agentic workflow implementations (math, code tasks)

#### 3. CLI Layer (`cli/`)

The CLI layer provides user-facing commands:

- **`main.py`**: Main entry point for launching complete training pipelines
- **`launch_server.py`**: Utility for launching standalone LLM servers

### Data Flow Architecture

AReaLite uses an **async producer-consumer pattern**:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LLM Servers   │◄──►│ Rollout Workers  │───►│   Data Buffer   │
│   (SGLang)      │    │  (Async Batch)   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        ▲                                               │
        │                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Checkpoints   │◄───│  FSDP Trainer    │◄───│ Training Loop   │
│                 │    │  (Sync Batch)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Design Principles

#### 1. **AI-Centric API Design**
Unlike the original AReaL's system-centric approach with workers and model functions, AReaLite uses familiar ML concepts:
- `Agent` and `Environment` (from RL literature)
- `RolloutWorkflow` (combines multiple agents and the environment to generate rollout data)
- `Trainer` (from HuggingFace/PyTorch, fetches rollout data and updates model parameters)

#### 2. **Factory Pattern for Extensibility**
Each major component uses a factory pattern for easy customization:
- `EngineFactory` creates training backends
- `TrainerFactory` creates training algorithms  
- `RolloutWorkflowFactory` creates rollout workflows

#### 3. **Configuration-Driven Architecture**
All components are configured through dataclasses defined in `cli_args.py`, enabling:
- Type-safe configuration
- Easy CLI argument generation
- Clear documentation of available options


## Implementation Details

### Training Pipeline

1. **Initialization**: Factory classes create configured instances of engines, trainers, and rollout workflows
2. **Rollout Phase**: `RolloutController` coordinates async data collection across multiple `RolloutWorker` instances
3. **Training Phase**: `Trainer` performs synchronous gradient updates using collected data
4. **Weight Updates**: Updated model weights are pushed to LLM servers via `update_weights_to()`

### Rollout System

The rollout system supports arbitrary agentic rollout paradigms, implemented as `RolloutWorkflow` instances. `RolloutWorkflow` exposes a `run_episode` method for users to implement the logic of collecting a complete agentic trajectory. Users can implement gymnasium-compatible `Agent` and `Environment` interfaces first and combine them as a workflow as in normal RL literature (in `arealite/impl/agentic/`), or users can implement the workflow directly if the agent-environment interfaces are not compatible with the desired use cases (in `arealite/impl/rlvr/`).

## Expected Usage

### Basic RL Training
```bash
python3 arealite.cli.main \
    experiment_name=my-exp trial_name=my-trial \
    trainer.type=grpo \
    trainer.grpo.actor.path=Qwen/Qwen2-0.5B
```

### Rollout-Only Evaluation
```bash
python3 arealite.cli.main \
    trainer.type=null \
    valid_dataset.path=huggingface/dataset
```

### Distributed Training
```bash
python3 arealite.cli.main \
    mode=ray \
    allocation_mode=sglang.d16p1m1+d32p2m1 \
    trainer.type=grpo
```

## Customization Guide

### Adding New Trainers

1. **Implement trainer class** in `impl/trainer/`:
```python
from arealite.api.trainer_api import Trainer

class MyTrainer(Trainer):
    def train(self, resume_from_checkpoint=None):
        # Implementation here
        pass
```

2. **Add configuration** in `cli_args.py`:
```python
@dataclass  
class MyTrainerConfig:
    learning_rate: float = 1e-4
```

3. **Register in factory** in `trainer_api.py`:
```python
def make_trainer(self, config: TrainerConfig) -> Trainer:
    if config.type == "my_trainer":
        return MyTrainer(...)
```

### Adding New Rollout Workflows

1. **Implement workflow** in `impl/`:
```python
from arealite.api.rollout_api import RolloutWorkflow

class MyWorkflow(RolloutWorkflow):
    async def arun_episode(self, gconfig, env_option=None, seed=None):
        # Implementation here
        pass
```

2. **Register in factory** in `rollout_api.py`:
```python
def make_workflow(self, config: RolloutWorkflowConfig):
    if config.type == "my_workflow":
        return MyWorkflow(...)
```

## Roadmap

- [ ] Finalize API design. (In-progress)
- [x] Implement standalone SGLang server (`impl/sglang_server.py`).
- [x] Implement SGLang client generation (`impl/sglang_client.py`).
- [x] Rollout pipeline (`tests/test_rollout.py`).
- [x] SGLang rollout interruption.
- [x] Asynchronous RL system-wide utilities (e.g., `RolloutController`).
- [ ] Various launching scripts: ray, torchrun, slurm.
- [ ] FSDP2 engine with transformers models. (In-progress)
- [ ] SFT trainer. (In-progress)
- [ ] SGLang update weights. (In-progress)
- [x] GRPO trainer.
- [ ] Add benchmarking against original AReaL
- [ ] CI and unittests.
- [ ] Other RL algorithms (DPO, REINFORCE, etc.)
- [ ] Support for multi-modal models
- [ ] User guide for transitioning from v0.3.0.
- [ ] Advanced agentic workflows (tool use, planning)
- [ ] Examples of training GSM8K, TLDR, and a search agent.
- [ ] Allow external persistent SGLang servers for debugging purposes.
