# Tau2-Bench RL Training Example

This example demonstrates how to train an agent on [tau2-bench](https://github.com/sierra-research/tau2-bench) using AReaL's proxy server mode with the archon backend.

## Prerequisites

1. Install tau2-bench:
```bash
pip install tau2-bench
# Or install from local path
pip install /path/to/tau2-bench
```

2. Install additional dependencies:
```bash
pip install tenacity torch_memory_saver litellm==1.75 transformers==4.57.1
```

3. Set up tau2 data directory:
```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

## Quick Start

### Using SLURM Scheduler

```bash
# Set PYTHONPATH
export PYTHONPATH=/path/to/AReaL:$PYTHONPATH

# Run training
python3 -m areal.launcher.slurm \
    examples/tau2/tau2_train.py \
    --config examples/tau2/config.yaml \
    experiment_name=tau2-grpo \
    trial_name=run-airline \
    cluster.n_nodes=2 \
    allocation_mode=sglang:d4t2+archon:d4p2t1 \
    econfig.domain=airline \
    actor.path=/path/to/model
```

### Using Local Scheduler (for debugging)

```bash
python3 -m areal.launcher.local \
    examples/tau2/tau2_train.py \
    --config examples/tau2/config.yaml \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=2 \
    allocation_mode=sglang:d1+archon:d1 \
    train_dataset.batch_size=4 \
    gconfig.n_samples=4 \
    econfig.domain=airline
```

## Configuration

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `allocation_mode` | Resource allocation (e.g., `sglang:d4t2+archon:d4p2t1`) |
| `econfig.domain` | Tau2 domain: `retail`, `airline`, or `telecom` |
| `econfig.max_steps` | Maximum steps per episode |
| `econfig.solo_mode` | Whether to use solo mode (no user simulator) |
| `econfig.user_llm` | User simulator LLM model |
| `econfig.user_llm_base_url` | User simulator LLM endpoint |
| `gconfig.n_samples` | Number of samples per prompt (group size) |

### Allocation Mode Format

The allocation mode uses the format: `<gen_backend>:<gen_parallel>+<train_backend>:<train_parallel>`

- **Gen backend**: `sglang` or `vllm`
- **Train backend**: `archon`, `fsdp`, or `megatron`
- **Parallel spec**: `d<dp>p<pp>t<tp>c<cp>e<ep>` where:
  - `d` = data parallel size
  - `p` = pipeline parallel size
  - `t` = tensor parallel size
  - `c` = context parallel size
  - `e` = expert parallel size

Example: `sglang:d4t2+archon:d4p2t1` means:
- SGLang with 4 data parallel × 2 tensor parallel = 8 GPUs for inference
- Archon with 4 data parallel × 2 pipeline parallel × 1 tensor parallel = 8 GPUs for training

## Architecture

```
tau2_train.py      - Main training script using PPOTrainer
tau2_agent.py      - AgentWorkflow implementation for tau2
tau2_utils.py      - Utility classes (Tau2EnvConfig, Tau2RunInfo)
config.yaml        - Default configuration
tau2_rl.sh         - Launch script
```

## How It Works

1. **Dataset**: Task IDs are loaded from tau2-bench's registry based on domain and split
2. **Workflow**: `Tau2AgentWorkflow` implements `AgentWorkflow` interface, using litellm for LLM calls
3. **Proxy Server**: AReaL's proxy server intercepts OpenAI API calls to:
   - Route requests to the rollout inference engine
   - Track interactions for RL training
   - Collect rewards set by the workflow
4. **Training**: PPOTrainer handles the RL optimization loop with the archon backend

## Supported Domains

- `retail` - Retail customer service
- `airline` - Airline customer service
- `telecom` - Telecommunications customer service
