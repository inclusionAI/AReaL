# AReaL's GRPO Implementation - Summary

## Overview

AReaL provides a **complete, production-ready GRPO (Group Relative Policy Optimization)** implementation designed for large-scale distributed training on GPU clusters. This is a fully functional RL framework, not just a proof of concept.

---

## What AReaL Has Implemented

### 1. **Core GRPO Algorithm** ✅

**Location**: `areal/engine/ppo/actor.py` - `PPOActor` class

**Features**:
- ✅ **Group-relative advantage normalization**: Normalizes rewards within groups
- ✅ **PPO clipped loss**: Standard PPO objective with importance sampling
- ✅ **KL divergence regularization**: Optional KL penalty against reference model
- ✅ **GAE (Generalized Advantage Estimation)**: For temporal credit assignment
- ✅ **Reward scaling/normalization**: Configurable reward processing
- ✅ **Dynamic sampling**: Adaptive sampling based on group performance

**Key Methods**:
- `compute_advantages()`: Computes group-relative advantages
- `ppo_update()`: Performs PPO policy update
- `compute_logp()`: Computes log probabilities for importance weights

### 2. **Rollout Workflow (RLVR)** ✅

**Location**: `areal/workflow/rlvr.py` - `RLVRWorkflow` class

**Features**:
- ✅ **Multiple completions per prompt**: Generates `n_samples` completions
- ✅ **Async reward computation**: Non-blocking reward evaluation
- ✅ **Version tracking**: Handles weight versioning for async training
- ✅ **Sequence formatting**: Properly formats input/output sequences
- ✅ **Logging/dumping**: Saves generated completions for analysis

**Key Methods**:
- `arun_episode()`: Generates completions and computes rewards for one prompt

### 3. **Training Engine** ✅

**Location**: `areal/engine/ppo/actor.py` - `FSDPPPOActor` class

**Features**:
- ✅ **FSDP (Fully Sharded Data Parallel)**: Distributed training backend
- ✅ **Megatron support**: Alternative training backend
- ✅ **Gradient checkpointing**: Memory optimization
- ✅ **Micro-batching**: Handles large batches efficiently
- ✅ **Weight synchronization**: Updates inference servers after training

### 4. **Inference Engine** ✅

**Location**: `areal/engine/sglang_remote.py` - `RemoteSGLangEngine` class

**Features**:
- ✅ **SGLang integration**: Fast inference server
- ✅ **Async generation**: Non-blocking model.generate()
- ✅ **Weight versioning**: Handles model updates during training
- ✅ **Load balancing**: Distributes requests across multiple servers
- ✅ **Caching**: Reuses KV cache for efficiency

### 5. **Complete Training Loop** ✅

**Location**: `examples/math/gsm8k_grpo.py`

**Features**:
- ✅ **End-to-end GRPO training**: Complete working example
- ✅ **Rollout → Reward → Advantage → PPO update** cycle
- ✅ **Evaluation**: Periodic model evaluation
- ✅ **Checkpointing**: Save/load model checkpoints
- ✅ **Recovery**: Resume from checkpoints
- ✅ **Logging**: W&B, tensorboard, stats tracking

---

## Architecture Components

### Training Flow

```
1. Dataset → Dataloader
2. RLVRWorkflow.arun_episode() → Generate N completions
3. Reward Function → Compute rewards (0/1 for correctness)
4. PPOActor.compute_advantages() → Group-relative normalization
5. PPOActor.ppo_update() → Apply PPO loss
6. FSDPPPOActor → Update model weights
7. RemoteSGLangEngine → Sync weights to inference servers
```

### Key Classes

| Class | Purpose | Location |
|-------|---------|----------|
| `RLVRWorkflow` | Rollout generation | `areal/workflow/rlvr.py` |
| `PPOActor` | PPO algorithm logic | `areal/engine/ppo/actor.py` |
| `FSDPPPOActor` | FSDP training engine | `areal/engine/ppo/actor.py` |
| `RemoteSGLangEngine` | Inference server client | `areal/engine/sglang_remote.py` |
| `GRPOConfig` | Configuration dataclass | `areal/api/cli_args.py` |

---

## Official OS Support (From Documentation)

### Operating System Requirements

**From `docs/tutorial/installation.md`**:

| Component | Official Support |
|-----------|------------------|
| **OS** | **CentOS 7 / Ubuntu 22.04** (or systems meeting requirements) |
| **NVIDIA Driver** | 550.127.08 |
| **CUDA** | 12.8 |
| **Python** | 3.10, 3.11, 3.12 (from `pyproject.toml`) |
| **Container** | Docker with `ghcr.io/inclusionai/areal-runtime:v0.3.4` |

**Key Points**:
- ✅ **Linux-based OS required** (CentOS/Ubuntu)
- ✅ **POSIX :: Linux** (from pyproject.toml classifiers)
- ❌ **No official macOS support**
- ❌ **No official Windows support** (except via Docker/WSL2)

### Hardware Requirements

**From `docs/tutorial/installation.md`**:

- **GPU**: 8x H800 per node (tested configuration)
- **CPU**: 64 cores per node
- **Memory**: 1TB per node
- **Network**: NVSwitch + RoCE 3.2 Tbps
- **Storage**: 1TB local (single node) / 10TB shared (multi-node)

### Software Stack

- **Docker** (recommended): Uses pre-built image with all dependencies
- **SGLang**: Required for inference server (Linux-only)
- **NCCL**: For distributed training (NVIDIA GPUs only)
- **FSDP/Megatron**: Training backends (GPU clusters)

---

## How to Run GRPO (Official Instructions)

### From `docs/algorithms/grpo.md`:

```bash
# Local launcher (single node)
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
    --config examples/math/gsm8k_grpo.yaml \
    experiment_name=<name> \
    trial_name=<trial>

# Ray launcher (distributed)
python3 -m areal.launcher.ray examples/math/gsm8k_grpo.py \
    --config examples/math/gsm8k_grpo.yaml \
    experiment_name=<name> \
    trial_name=<trial>

# Slurm launcher (HPC clusters)
python3 -m areal.launcher.slurm examples/math/gsm8k_grpo.yaml \
    --config examples/math/gsm8k_grpo.yaml \
    experiment_name=<name> \
    trial_name=<trial>
```

### From `docs/tutorial/quickstart.md`:

**Prerequisites**:
1. Docker environment with GPU access
2. Shared storage for multi-node experiments
3. Ray cluster setup (for distributed)

**Command**:
```bash
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py \
    --config examples/math/gsm8k_grpo.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name>
```

---

## Configuration Structure

### Key GRPO Parameters (from `examples/math/gsm8k_grpo.yaml`)

```yaml
gconfig:
  n_samples: 4              # Number of completions per prompt
  max_new_tokens: 1024       # Max generation length

actor:
  group_size: 4              # Matches n_samples (for grouping)
  eps_clip: 0.4             # PPO clipping epsilon
  reward_scaling: 10.0       # Reward multiplier
  kl_ctl: 0.0               # KL penalty (0 = no KL regularization)
  adv_norm:                  # Advantage normalization
    mean_level: batch       # Normalize within batch
    std_level: batch        # Normalize within batch
```

---

## What's Missing / Limitations

### For macOS Users:

1. ❌ **No official macOS support**
   - AReaL requires Linux (CentOS/Ubuntu)
   - SGLang doesn't support macOS natively
   - NCCL requires NVIDIA GPUs

2. ❌ **No MPS (Metal) support**
   - AReaL is designed for CUDA GPUs
   - FSDP/Megatron backends require CUDA

3. ❌ **Local launcher requires GPUs**
   - `LocalLauncher` checks for GPU devices
   - Doesn't support CPU-only mode

### Workarounds (Not Official):

1. **Docker with GPU passthrough**: Possible on macOS with external GPU
2. **Linux VM**: Run AReaL in a Linux virtual machine
3. **Remote cluster**: Use SSH to access Linux GPU cluster
4. **WSL2 (Windows)**: Can run AReaL with NVIDIA GPUs

---

## Summary

### What AReaL Provides:
- ✅ **Complete GRPO implementation** with all components
- ✅ **Production-ready** distributed training infrastructure
- ✅ **Full documentation** in `docs/` folder
- ✅ **Working examples** in `examples/math/`
- ✅ **Multiple backends** (FSDP, Megatron)
- ✅ **Multiple launchers** (Local, Ray, Slurm)

### OS Support:
- ✅ **Linux (CentOS/Ubuntu)** - Fully supported
- ❌ **macOS** - Not officially supported
- ❌ **Windows** - Not officially supported (except Docker/WSL2)

### For Your Use Case:
Since you're on macOS, AReaL's official infrastructure won't work out of the box. This is why:
- We created `train_grpo_hf.py` - Simplified version using HuggingFace Trainer
- Avoids SGLang, launchers, and other Linux-specific dependencies
- Works on macOS but currently only implements SFT (not full GRPO)

---

## References

- **Official GRPO Docs**: `docs/algorithms/grpo.md`
- **Installation Guide**: `docs/tutorial/installation.md`
- **Quickstart Guide**: `docs/tutorial/quickstart.md`
- **Code Walkthrough**: `docs/lite/gsm8k_grpo.md`
- **Example Script**: `examples/math/gsm8k_grpo.py`
- **Config Template**: `examples/math/gsm8k_grpo.yaml`

