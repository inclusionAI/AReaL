# Offline Tau2 Bench Rollout with AReaL Inference Service

This example runs rollout-only data generation on the
[$\\tau^2$-Bench](https://github.com/sierra-research/tau2-bench) using the AReaL
Inference Service (`GatewayInferenceController`). Unlike the full training pipeline in
`examples/tau2/`, this script performs rollouts without a training step — useful for
evaluation, data collection, or debugging agent behaviour.

## Installation

### AReaL

Follow the
[AReaL installation guide](https://inclusionai.github.io/AReaL/en/tutorial/installation.html).

### Tau2

Install the (forked) tau2-bench package:

```bash
pip install git+https://github.com/dhh1995/tau2-bench.git@dhh/async-and-custom-completion
```

Set the `TAU2_DATA_DIR` environment variable:

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

## Running

All commands should be executed from the **repository root**.

```bash
python3 examples/experimental/inference_service/tau2_rollout.py \
    --config examples/experimental/inference_service/tau2_rollout.yaml \
    econfig.user_llm_base_url=<USER_LLM_BASE_URL> \
    cluster.fileroot=<EXPERIMENT_ROOT> \
    cluster.name_resolve.nfs_record_root=<NAME_RESOLVE_ROOT>
```

| Placeholder           | Description                                             | Example                     |
| --------------------- | ------------------------------------------------------- | --------------------------- |
| `<USER_LLM_BASE_URL>` | OpenAI-compatible base URL of the user simulator LLM    | `http://localhost:8000/v1/` |
| `<EXPERIMENT_ROOT>`   | Directory for experiment artifacts (logs, trajectories) | `/tmp/areal/experiments`    |
| `<NAME_RESOLVE_ROOT>` | Shared path for name-resolve records                    | `/tmp/areal/name_resolve`   |

## Result

A successful rollout prints per-batch statistics after every batch:

```
(AReaL) 20260319-14:18:25.768 Tau2GatewayRollout INFO: Batch 2: n_trajs=16, rewards=tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.]), avg_reward=0.1250
```

Each line reports the batch index, number of trajectories, individual rewards, and the
batch-level average reward.
