# Agent Service Examples

## Overview

This directory contains experimental examples built on top of AReaL's agent service.
The examples are grouped by scenario:

- `claude/` — a standalone Claude Agent SDK service demo
- `tau2/` — a tau2 customer-service rollout example that combines the agent service
  with the experimental inference service

The agent service exposes complete agent sessions through Router, DataProxy, Worker,
and Gateway microservices, and can be paired with the experimental inference service
for RL data collection.

## Example 1: Claude Agent SDK Service

This is the Claude Agent SDK example under the new `claude/` subdirectory.

### Prerequisites

```bash
uv pip install claude-agent-sdk
export ANTHROPIC_API_KEY=sk-...
```

### Run

```bash
python examples/experimental/agent_service/claude/run_agent_service.py
python examples/experimental/agent_service/claude/run_agent_service.py --num-pairs 4
```

The script creates a `LocalScheduler`, launches Guard workers, then forks Router,
Worker+DataProxy pairs, and Gateway. An interactive prompt lets you chat with the
Claude agent through `POST /v1/responses`.

Files:

- `claude/agent.py` — Claude Agent SDK worker implementation
- `claude/run_agent_service.py` — interactive launcher for the Claude example

## Example 2: Tau2 Agent Service Rollout

This example runs the tau2 customer-service agent inside the experimental agent service
while the experimental inference service collects RL trajectories. Unlike the reference
inference-service example, this script initializes `RolloutControllerV2` but does not
use `rollout_batch()`. It directly runs the tau2 workflow and returns exported
trajectories from the inference service.

### Additional Prerequisites

```bash
pip install git+https://github.com/dhh1995/tau2-bench.git@dhh/async-and-custom-completion
pip install pydantic-ai
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

If `econfig.solo_mode=false`, also start a user simulator model and set
`econfig.user_llm_base_url` in `tau2/config.yaml`.

### Run

```bash
python examples/experimental/agent_service/tau2/run_rollout.py \
    --config examples/experimental/agent_service/tau2/config.yaml \
    cluster.fileroot=<EXPERIMENT_ROOT> \
    cluster.name_resolve.nfs_record_root=<NAME_RESOLVE_ROOT>
```

### What it does

1. Starts the experimental inference service with `RolloutControllerV2`.
2. Starts the experimental agent service with `AgentController`.
3. For each tau2 task, the workflow:
   - calls `AgentController.start_session()` (which grants capacity and starts the RL
     session),
   - drives the tau2 conversation through `AgentController.step()`,
   - calls `AgentController.set_reward()`,
   - calls `AgentController.export_trajectory()` and returns the exported interactions.

### Files

| File | Description |
| --- | --- |
| `claude/agent.py` | Claude Agent SDK example agent |
| `claude/run_agent_service.py` | Interactive launcher for the Claude agent service |
| `tau2/agent.py` | Tau2 agent-service worker agent |
| `tau2/workflow.py` | Tau2 rollout workflow using async controller APIs |
| `tau2/run_rollout.py` | Direct rollout driver for the tau2 workflow |
| `tau2/config.yaml` | Example config for the tau2 rollout driver |
