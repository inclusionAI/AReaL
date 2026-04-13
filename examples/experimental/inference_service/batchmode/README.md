# Inference Service Benchmark (Target Experiment)

Measures AReaL inference service full-stack overhead on TAU²-bench agent tasks.

```
Request path:
  OpenClaw CLI → IS Gateway (:30098) → Router (:8081) → DataProxy (:8082) → ArealOpenAI Client → SGLang /generate (:30000)

Two SGLang instances:
  Agent SGLang (port 30000) — benchmark target, --disable-radix-cache, TP=8
  User  SGLang (port 30001) — simulates user, NOT measured, TP=8
```

## Prerequisites

| Dependency            | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| Singularity container | AReaL dev image with SGLang, PyTorch, etc.                     |
| Model weights         | Qwen3-235B-A22B-Instruct-2507 (local path)                     |
| tau2-bench            | pip-installable TAU²-bench source                              |
| openclaw-benchmark    | pip-installable OpenClaw TAU² integration package              |
| Slurm cluster         | 2 nodes × 8 GPUs (Agent + User SGLang) + 1 node for IS + sweep |

## Step 1: Start SGLang Servers

Edit `start_servers.sh` to set your container image, model path, and log directory,
then:

```bash
bash start_servers.sh
```

This submits two Slurm jobs (Agent + User SGLang). Wait for both to start:

```bash
squeue -u $(whoami)
# Note the node names from the NODELIST column, e.g.:
#   Agent → node-A  (port 30000)
#   User  → node-B  (port 30001)
```

Verify servers are healthy:

```bash
curl -sf http://<agent-node>:30000/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"
curl -sf http://<user-node>:30001/v1/models  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"
```

## Step 2: Configure sweep.sh

Edit the top of `sweep.sh` to match your environment:

```bash
# ── Paths (MUST update) ──
CONTAINER="<path-to-singularity-image>"
PROJECT="<path-to-openclaw-benchmark-repo>"
AREAL_PATCH="<path-to-areal-repo-with-is-patches>"
MODEL_PATH="<path-to-qwen3-235b-model>"

# ── Endpoints (MUST update) ──
SGLANG_PORT=30000                              # Agent SGLang, must be on same node as IS
USER_ENDPOINT="http://<user-node>:30001/v1"    # User SGLang node from Step 1
```

## Step 3: Run the Sweep

SSH into the **Agent SGLang node** (IS processes must co-locate with Agent SGLang on
localhost):

```bash
ssh <agent-node>
```

Run the sweep:

```bash
bash sweep.sh \
    "5,10,15,20,25,30" \   # concurrency levels
    50 \                    # tasks per trial
    4 \                     # trials per concurrency
    reproduce               # output tag
```

| Argument | Default            | Description                            |
| -------- | ------------------ | -------------------------------------- |
| `$1`     | `5,10,15,20,25,30` | Comma-separated concurrency levels     |
| `$2`     | `50`               | Number of TAU²-bench tasks per trial   |
| `$3`     | `4`                | Number of trials per concurrency level |
| `$4`     | `<timestamp>`      | Tag for output directory               |

The script automatically:

1. Enters Singularity container
1. Installs dependencies (openclaw-benchmark, tau2-bench)
1. Patches IS code into container's AReaL installation
1. Starts Router → DataProxy → Gateway (registers DataProxy with Router)
1. Runs `collect_trajectories.py` for each (concurrency, trial) combination
1. Prints summary table on completion

## What Happens Inside

```
┌─ Singularity Container ──────────────────────────────────────────────┐
│                                                                      │
│  Router (:8081)  ←─ register ─  DataProxy (:8082)  ←─  SGLang (:30000)
│       ↑                              ↑                    (localhost)
│       │                              │
│  Gateway (:30098)                    │
│       ↑                         ArealOpenAI
│       │                    (tokenize → /generate)
│  collect_trajectories.py
│       │
│  N × OpenClaw CLI (subprocess)
│       │
│  worker.py (per task)
│       │
│  tau2-bench orchestrator + evaluator
│       │
│  User SGLang (remote, :30001) ────────────────────────── External Node
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

Per-task flow:

1. `collect_trajectories.py` calls `POST /grant_capacity` then `POST /rl/start_session`
   → gets session API key
1. Spawns `worker.py` subprocess with OpenClaw CLI pointed at Gateway
1. OpenClaw runs TAU²-bench task (multi-turn: agent calls tools via Gateway, user sim
   via remote SGLang)
1. On completion, calls `POST /rl/set_reward` with task reward
1. Calls `POST /export_trajectories` → saves trajectory JSON to disk

## Output

Results are saved to `$PROJECT/trajectories/sweep_<tag>/`:

```
sweep_<tag>/
├── c5/
│   ├── trial_1/
│   │   ├── collection_summary.json    # pass rate, wall clock, tasks/min
│   │   ├── task_0_session_0-0.json    # per-task trajectory
│   │   └── ...
│   ├── trial_2/
│   └── ...
├── c10/
└── ...
```

`collection_summary.json` fields:

| Field               | Description                 |
| ------------------- | --------------------------- |
| `completed`         | Total tasks finished        |
| `passed` / `failed` | Tasks with reward > 0 / = 0 |
| `errors`            | Tasks that hit errors       |
| `pass_rate`         | passed / completed          |
| `total_time_s`      | Wall clock seconds          |
| `tasks_per_min`     | Throughput                  |

## Configuration Reference

### Benchmark Parameters (in collect_trajectories.py)

| Parameter            | Value     | Description                   |
| -------------------- | --------- | ----------------------------- |
| `--domain`           | `airline` | TAU²-bench domain             |
| `--num-tasks`        | `50`      | Tasks per trial               |
| `--max-steps`        | `200`     | Max agent turns per task      |
| `--max-errors`       | `10`      | Max errors before abort       |
| `--seed`             | `300`     | Random seed for task ordering |
| `--openclaw-timeout` | `3000`    | Subprocess timeout (seconds)  |

### IS Component Ports

| Component    | Port  | Flag                 |
| ------------ | ----- | -------------------- |
| Router       | 8081  | `--port`             |
| DataProxy    | 8082  | `--port`             |
| Gateway      | 30098 | `--port`             |
| Agent SGLang | 30000 | Must be on localhost |

### SGLang Flags

| Flag                      | Agent | User | Reason                                       |
| ------------------------- | ----- | ---- | -------------------------------------------- |
| `--disable-radix-cache`   | ✅    | ❌   | Consistent no-cache for IS benchmark         |
| `--tool-call-parser`      | ✅    | ✅   | Model-specific, e.g. `qwen25` for Qwen3      |
| `--enable-metrics`        | ✅    | ✅   | Prometheus endpoint for `collect_metrics.py` |
| `--context-length 262144` | ✅    | ✅   | Qwen3-235B max context                       |
| `--tp 8`                  | ✅    | ✅   | Tensor parallelism across 8 GPUs             |

## Reference Results: Qwen3-235B-A22B-Instruct-2507

Tested on TAU²-bench airline domain, 50 tasks × 4 trials per concurrency, 2 nodes ×
8×H200 GPUs.

> Results below are from a single experiment run. Exact numbers may vary slightly across
> runs due to non-determinism in concurrent GPU inference and system scheduling.

### Baseline B (OpenClaw → SGLang direct) vs Target (OpenClaw → IS → SGLang)

Each cell: `Baseline B / Target (Δ)`.

| Metric      | c=5                   | c=10                  | c=15                  | c=20                  | c=25                  | c=30                  |
| ----------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| Pass@1      | 38% / 30% (-8pp)      | 38% / 38% (+1pp)      | 34% / 32% (-2pp)      | 36% / 34% (-2pp)      | 34% / 34% (0pp)       | 36% / 38% (+3pp)      |
| Avg E2E (s) | 4.82 / 4.57 (-5%)     | 8.99 / 8.57 (-5%)     | 13.05 / 12.58 (-4%)   | 17.03 / 16.33 (-4%)   | 20.73 / 20.16 (-3%)   | 24.72 / 23.41 (-5%)   |
| Input Tok/s | 15,207 / 16,017 (+5%) | 18,204 / 19,204 (+5%) | 18,820 / 20,138 (+7%) | 19,281 / 19,388 (+1%) | 19,433 / 20,474 (+5%) | 19,480 / 20,780 (+7%) |
| Req/s       | 0.69 / 0.72 (+4%)     | 0.82 / 0.87 (+6%)     | 0.85 / 0.90 (+6%)     | 0.87 / 0.87 (0%)      | 0.87 / 0.93 (+7%)     | 0.89 / 0.95 (+7%)     |
| Tasks/min   | 2.5 / 2.6 (+6%)       | 3.0 / 3.2 (+5%)       | 3.1 / 3.1 (+1%)       | 3.0 / 3.0 (0%)        | 3.0 / 3.4 (+13%)      | 3.4 / 3.7 (+8%)       |

### Target SGLang Metrics (per concurrency)

| Metric        | c=5    | c=10   | c=15   | c=20   | c=25   | c=30   |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Input Tok/s   | 16,017 | 19,204 | 20,138 | 19,388 | 20,474 | 20,780 |
| Output Tok/s  | 75     | 88     | 89     | 88     | 97     | 93     |
| Avg E2E (s)   | 4.57   | 8.57   | 12.58  | 16.33  | 20.16  | 23.41  |
| Avg TTFT (s)  | 2.76   | 4.85   | 7.08   | 9.13   | 11.41  | 14.18  |
| Avg Queue (s) | 0.31   | 0.77   | 1.34   | 1.92   | 2.70   | 3.58   |
| Total Reqs    | 3,318  | 3,332  | 3,492  | 3,539  | 3,309  | 3,162  |
| Avg InTok/Req | 22,099 | 22,040 | 22,283 | 22,324 | 22,132 | 21,766 |
