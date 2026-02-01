# Tau2 Training with Proxy Server and Archon Backend

This example demonstrates how to train tau2-bench agents using AReaL's
**single-controller mode** with proxy server and archon backend.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Single-Controller Mode                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Controller  │────▶│ Proxy Server │────▶│ SGLang/vLLM  │    │
│  │  (tau2_train)│     │              │     │  Inference   │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                    ▲                                   │
│         │                    │                                   │
│         ▼                    │                                   │
│  ┌──────────────┐     ┌──────┴──────┐                           │
│  │    Archon    │     │ AgentWorkflow │                          │
│  │   (Actor)    │     │ (tau2_agent)  │                          │
│  └──────────────┘     └──────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Controller**: Orchestrates the training loop, manages workers
2. **Proxy Server**: Translates OpenAI API calls to inference engine
3. **SGLang/vLLM**: Inference backend for model generation
4. **Archon**: Training backend handling PPO updates (supports DP, TP, PP)
5. **AgentWorkflow**: Tau2 agent logic using AsyncOpenAI client

## Differences from SPMD Mode

| Aspect | SPMD Mode (`tau2/`) | Single-Controller Mode (`tau2-proxy/`) |
|--------|---------------------|---------------------------------------|
| Workflow | `RolloutWorkflow` | `AgentWorkflow` |
| Inference | Direct engine access | Via proxy server |
| Entry Point | `engine` parameter | `base_url` + `http_client` |
| Client | `ArealOpenAI` | `AsyncOpenAI` |
| Scheduler | Any | `scheduler.type=slurm` |

## Usage

```bash
# Run with default config
./tau2_rl.sh

# Or directly with python
python3 tau2_train.py config.yaml scheduler.type=slurm

# Override specific parameters
python3 tau2_train.py config.yaml \
    scheduler.type=slurm \
    econfig.domain=telecom \
    econfig.solo_mode=true
```

## Configuration

Key configuration options in `config.yaml`:

```yaml
# Single-controller mode requires slurm scheduler
scheduler:
  type: slurm

# Allocation mode: inference backend + training backend
# sglang:d{dp}t{tp} + archon:d{dp}p{pp}t{tp}
allocation_mode: sglang:d4t2+archon:d4p1t2

# Tau2 environment config
econfig:
  domain: airline          # airline, retail, telecom
  max_steps: 200           # Max steps per episode
  solo_mode: false         # true = no user simulator
  user_llm_base_url: ...   # URL for user LLM
```

## Known Issues

1. **Pipeline Parallelism (PP > 1)**: May cause "Schedule1F1B object has no
   attribute 'eval'" error. Use `p1` in allocation_mode as workaround.

2. **Proxy Server + Ray**: Not supported. Use `scheduler.type=slurm`.

## Files

- `tau2_agent.py`: AgentWorkflow implementation
- `tau2_train.py`: Training entry point
- `tau2_utils.py`: Shared utilities and config
- `config.yaml`: Training configuration
- `tau2_rl.sh`: Launch script
