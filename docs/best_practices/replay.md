# Replay Rollout Submissions

## Step 0: Validate your training script

Your training script should be able to run with `PPOTrainer` and `MockPPOTrainer`. Which
trainer to run depends on the configuration `is_replay=${True|False}`.

An example is `examples/replay/gsm8k_grpo.py`. Note that `is_replay` should be available
in the yaml configuration, e.g., `examples/replay/gsm8k_grpo.yaml`

## Step 1: Run a real experiment with perf_tracer

Any experiment ***with `perf_tracer` enabled*** will work. The experiment can be either
local or distributed.

```bash
python3 -m areal.launcher.${local|slurm} examples/replay/gsm8k_grpo.py \
    --config examples/replay/gsm8k_grpo.yaml \
    allocation_mode=sglang:d4+fsdp:d2c2 \
    experiment_name=my-exp \
    trial_name=trial0 total_train_epochs=1
```

## Step 2: Run the replay job

Use the same command but with `is_replay=True`:

```bash
python3 -m areal.launcher.${local|slurm} examples/replay/gsm8k_grpo.py \
    --config examples/replay/gsm8k_grpo.yaml \
    allocation_mode=sglang:d4+fsdp:d2c2 \
    experiment_name=my-exp \
    trial_name=trial0 total_train_epochs=1 is_replay=True
```

Note that you must use the same configuration as the real job, especially
`experiment_name` and `trial_name`.

### How it works

The replay job is launched by the local/slurm/ray launcher. The launcher will see if
this is a replay job. If so, the launcher will launch the router (binary placed at
`area/router/router`) and add all inference servers to the router. Then, the launcher
replaces the `AREAL_LLM_SERVER_ADDRSS` environment variable to the router's address, and
pass that environment variable to the trainer.

In the replay mode, the trainer job will not occupy GPUs.
