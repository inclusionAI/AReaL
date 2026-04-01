# AReaL Inference Service Examples

This directory contains two examples that use the AReaL Inference Service
(`GatewayInferenceController`) — an experimental rollout backend that exposes an
OpenAI-compatible proxy gateway so any external agent runtime can submit chat requests
and receive RL training data.

---

## Example 1: Offline τ²-Bench Rollout

This example runs rollout-only data generation on the
[$\tau^2$-Bench](https://github.com/sierra-research/tau2-bench) using the AReaL
Inference Service. Unlike the full training pipeline in `examples/tau2/`, this script
performs rollouts without a training step — useful for evaluation, data collection, or
debugging agent behaviour.

### Installation

#### AReaL

Follow the
[AReaL installation guide](https://inclusionai.github.io/AReaL/en/tutorial/installation.html).

#### Tau2

Install the (forked) tau2-bench package:

```bash
pip install git+https://github.com/dhh1995/tau2-bench.git@dhh/async-and-custom-completion
```

Set the `TAU2_DATA_DIR` environment variable:

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

### Running

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

### Result

A successful rollout prints per-batch statistics after every batch:

```
(AReaL) 20260319-14:18:25.768 Tau2GatewayRollout INFO: Batch 2: n_trajs=16, rewards=tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.]), avg_reward=0.1250
```

Each line reports the batch index, number of trajectories, individual rewards, and the
batch-level average reward.

---

## Example 2: Human-in-the-Loop Online RL Demo

This example demonstrates **human-in-the-loop (HITL) online RL**: a human (or an
automated script acting as one) chats with the model through any OpenAI-compatible
client, provides feedback after each conversation, and the gateway accumulates
trajectories into a training batch. It is the simplest end-to-end illustration of how
AReaL's inference service enables closed-loop RL without modifying the training code.

The automated demo script is `run_online_demo.sh`. It uses
[zeroclaw](https://github.com/dhh1995/zeroclaw) as the chat client and exercises the
following procedure:

1. **Launch `online_rollout.py`** — starts the SGLang inference engine and the proxy
   gateway, then waits until the gateway address is printed to the log.
2. **Patch `~/.zeroclaw/config.toml`** — redirects zeroclaw's default provider to the
   local gateway and injects the admin API key so all requests are attributed to a single
   RL session. The original config is restored on exit.
3. **Run four HITL rounds** — for each round the script:
   - Asks the model *"how many r's are in the word strawberry?"*.
   - If the answer is wrong, provides a corrective turn and asks once more.
   - Calls `set_reward.py` to push a scalar reward (`1.0` for correct, `0.0` for
     wrong after two attempts) back to the gateway via `POST /rl/set_reward`.
4. **Verify the batch** — waits for `online_rollout.py` to emit a `Rollout complete`
   log line confirming that all four trajectories (= `batch_size`) were collected and
   processed.

### Prerequisites

- **AReaL installed** — follow the
  [installation guide](https://inclusionai.github.io/AReaL/en/tutorial/installation.html).
- **zeroclaw installed** — any OpenAI-compatible CLI that supports
  `--session-state-file` can be substituted; the demo uses zeroclaw for convenience.
- **A zeroclaw config at `~/.zeroclaw/config.toml`** with at least a
  `default_provider` key — the script will patch it temporarily.
- **One GPU** — the default YAML (`online_rollout.yaml`) requests 1 GPU with SGLang.

### Running the Automated Demo

All commands should be executed from the **repository root**.

```bash
bash examples/experimental/inference_service/run_online_demo.sh
```

Key tunables at the top of the script:

| Variable             | Default                         | Description                                                               |
| -------------------- | ------------------------------- | ------------------------------------------------------------------------- |
| `ACTOR_PATH`         | `/storage/.../Qwen__Qwen3-0.6B` | Path to the HuggingFace model weights                                     |
| `ADMIN_KEY`          | `sk-test123456`                 | Admin API key (must match `rollout.openai.admin_api_key` in the YAML)     |
| `REQUEST_TIMEOUT`    | `3600`                          | Per-request timeout in seconds                                            |
| `GATEWAY_WAIT_SECS`  | `600`                           | Seconds to wait for the gateway to become ready                           |
| `QUESTION`           | *strawberry question*           | Question posed in every HITL round                                        |

You can override the model path without editing the script:

```bash
ACTOR_PATH=/path/to/your/model \
bash examples/experimental/inference_service/run_online_demo.sh
```

### Running a Manual HITL Session

To drive the rollout interactively instead of using the automated script:

**Step 1 — Start the online rollout server:**

```bash
python3 examples/experimental/inference_service/online_rollout.py \
    --config examples/experimental/inference_service/online_rollout.yaml \
    actor.path=<MODEL_PATH>
```

Wait until the log prints:

```
Proxy gateway available at http://127.0.0.1:<PORT>
```

**Step 2 — Chat with the model** using any OpenAI-compatible client, pointing it at
`http://127.0.0.1:<PORT>/v1` with `Authorization: Bearer sk-test123456`.

**Step 3 — Submit a reward** after each conversation turn:

```bash
python3 examples/experimental/inference_service/set_reward.py \
    http://127.0.0.1:<PORT> \
    --api-key sk-test123456 \
    --reward 1.0
```

Repeat Steps 2–3 until `batch_size` (default: 4) trajectories are complete. The server
will log `Rollout complete` and exit.

### Expected Output

When the demo finishes successfully you should see:

```
════════════════════════════════════════════════════════════════
  Step 5: Check online_rollout output for databatch
════════════════════════════════════════════════════════════════
  ── Rollout log (last 40 lines) ──
  ...
  ✔ Databatch detected in online_rollout output:
  (AReaL) ... InferenceServiceOnlineTrain INFO: Rollout complete (4 trajectories), avg_reward=X.XXXX
```

Each of the four HITL rounds also prints whether the model answered correctly on the
first or second try, for example:

```
  ── Trajectory 0 ──
  Q: how many r's are in the word strawberry?
  A: There are 3 r's in the word "strawberry".
  ✔ Correct on first try.
```
