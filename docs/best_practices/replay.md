# Replay Rollout Submissions

## Step 1: Run a real experiment with perf_tracer

Any experiment ***with `perf_tracer` enabled*** could be fine. The experiment can be
local or distributed.

We suggest using the `PPOTrainer` in your training script.

```bash
python3 -m areal.launcher.slurm examples/math/gsm8k_grpo.py \
    --config examples/math/gsm8k_grpo.yaml \
    allocation_mode=sglang:d4+fsdp:d2c2 \
    experiment_name=my-exp \
    trial_name=trial0 total_train_epochs=1
```

## Step 2: Launch the inference servers for replay

Use the same command above but with a ***inference-only allocation mode***:

```bash
python3 -m areal.launcher.slurm examples/math/gsm8k_grpo.py \
    --config examples/math/gsm8k_grpo.yaml \
    allocation_mode=sglang:d4 \
    experiment_name=my-exp \
    trial_name=trial0
```

Note that we recommend use the same experiment and trial name.

## (Optional) Step 3: Connect all server addresses to the proxy

The log of inference servers will prompt the server addresses, e.g.,

```
20251115-21:32:39.544 SGLangServer Wrapper INFO: SGLang server launched at: http://33.180.160.150:20908
20251115-21:32:40.567 SGLangServer Wrapper INFO: SGLang server launched at: http://33.180.160.150:11988
20251115-21:32:40.567 SGLangServer Wrapper INFO: SGLang server launched at: http://33.180.160.150:25651
20251115-21:32:40.567 SGLangServer Wrapper INFO: SGLang server launched at: http://33.180.160.150:15442
20251115-21:32:40.881 Launcher Utils INFO: Found 4 rollout servers: 33.180.160.150:25651, 33.180.160.150:15442, 33.180.160.150:20908, 33.180.160.150:11988
20251115-21:32:40.881 Local Scheduler INFO: LLM inference server launched at: AREAL_LLM_SERVER_ADDRS=33.180.160.150:25651,33.180.160.150:15442,33.180.160.150:20908,33.180.160.150:11988
```

You can then launch the HTTP proxy with the following server addresses.

## Step 4: Run the replay job

In your training script, replace `PPOTrainer` with `MockPPOTrainer`. Launch the job
with:

```bash
torchrun --nproc-per-node ${train_world_size} \
    examples/math/gsm8k_grpo_replay.py \
    --config examples/math/gsm8k_grpo.yaml \
    allocation_mode=sglang:d4+fsdp:d2c2 \
    experiment_name=my-exp \
    trial_name=trial0 total_train_epochs=1
```

Note that:

- No matter which launcher you used to run the real job, you should directly use
  `torchrun` to launch replay processes.
- You should replace the training script with the one that uses `MockPPOTrainer`.
- `${train_world_size}` is the number of GPUs you used for training in your real job
- You must use the same configuration as the real job, importantly, `experiment_name`
  and `trial_name`.
- If you launch a proxy, you should overwrite the server address with an additional
  environment variable `AREAL_LLM_SERVER_ADDRS=${proxy_ip}:${proxy_port}`
