# Quickstart

Welcome to the **AReaLite** Quickstart Guide!
This guide demonstrates how to run an AReaLite experiment training an LLM on the GSM8K dataset using the GRPO algorithm with function-based rewards.
Ensure you've completed [the installation and environment setup](installation.md) before proceeding.

## Running the Experiment (on a single node)

To run the experiment, you will need:
- Training script: [examples/arealite/gsm8k_grpo.py](../../examples/arealite/gsm8k_grpo.py)
- Config YAML: [examples/arealite/configs/gsm8k_grpo.yaml](../../examples/arealite/configs/gsm8k_grpo.yaml)

Our training scripts will automatically download the dataset (openai/gsm8k) and model (Qwen/Qwen2-1.5B-Instruct).
To run the example with default configuration, execute from the repository directory:
```
python3 -m arealite.launcher.local examples/arealite/gsm8k_grpo.py --config examples/arealite/configs/gsm8k_grpo.yaml experiment_name=<your experiment name> trial_name=<your trial name>
```

> **Note**: The command above uses `LocalLauncher`, which only works for a single node (`cluster.n_nodes == 1`). For  distributed experiments, see [Distributed Experiments with Ray or Slurm](quickstart_arealite.md#distributed-experiments-with-ray-or-slurm).

## Modifying configuration

All available configuration options are listed in [arealite/api/cli_args.py](https://github.com/inclusionAI/AReaL/blob/main/arealite/api/cli_args.py). 
To customize the experiment (models, resources, algorithm options), you can:
1. Edit the YAML file directly at [examples/arealite/configs/gsm8k_grpo.yaml](../../examples/arealite/configs/gsm8k_grpo.yaml).
2. Add command-line options:
   - For existing options in the YAML file, directly add the option: `actor.path=Qwen/Qwen3-1.7B`.
   - For other options in `cli_args.py`, but not in the YAML file, add with a prefix "+": `+sglang.attention_backend=triton`.

<!--
1. Adding command line options. For entries that exist in the config YAML, you could directly add the options after your command. For example: `actor.path=Qwen/Qwen3-1.7B`. For other options in `cli_args.py` but not in YAML, you could add these options with a prefix "+". For example: `+sglang.attention_backend=triton`. 
-->

For example, here is the command to launch a customized configuration, based on our GSM8K GRPO example:
```
python3 -m arealite.launcher.local examples/arealite/gsm8k_grpo.py \
    --config examples/arealite/configs/gsm8k_grpo.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    allocation_mode=sglang.d2p1t1+d2p1t1 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=4 \
    gconfig.max_new_tokens=2048 \
    train_dataset.batch_size=1024 \
    +sglang.attention_backend=triton
```

::::{important}
We're currently refactoring from legacy AReaL to AReaLite, which introduces some configuration differences.  We provide a **config converter** to transfer old AReaL config into AReaLite YAML file for users' convenience. [Click here](xxx) to learn how to use the **config converter**. 
::::

## Distributed Experiments with Ray or Slurm

AReaLite provides standalone launchers for distributed experiments. After setting up your Ray or Slurm cluster, launch experiments similarly to `LocalLauncher`:

```
# Launch with Ray launcher. 4 nodes (4 GPUs each), 3 nodes for generation, 1 node for training.
python3 -m arealite.launcher.ray examples/arealite/gsm8k_grpo.py \
    --config examples/arealite/configs/gsm8k_grpo.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    allocation_mode=sglang.d12p1t1+d4p1t1 \
    cluster.n_nodes=4 \
    cluster.n_gpus_per_node=4 \
    ...

# Launch with Slurm launcher. 16 nodes (8 GPUs each), 12 nodes for generation, 4 nodes for training
python3 -m arealite.launcher.slurm examples/arealite/gsm8k_grpo.py \
    --config examples/arealite/configs/gsm8k_grpo.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    allocation_mode=sglang.d96p1t1+d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8 \
    ...
```

Additional references:
- For more options for launchers, check `LauncherConfig` in [arealite/api/cli_args.py](https://github.com/inclusionAI/AReaL/blob/main/arealite/api/cli_args.py).
- [Ray cluster setup guide](installation.md#optional-launch-ray-cluster-for-distributed-training) for a guide on how to set up a ray cluster. 

> **Important Notes**:
> 1. Ensure `allocation_mode` matches your cluster configuration (`#GPUs == cluster.n_nodes * cluster.n_gpus_per_node`)
> 2. Ray/Slurm launchers only works for more than 1 node (`cluster.n_nodes > 1`). For single node scenario, please use `LocalLauncher`.
> 3. In Ray/Slurm launchers, GPUs are allocated at node granularity, which means #GPUs for generation or training must be integer multiples of `cluster.n_gpus_per_node`.

<!--
> **Notes**: Before launching distributed experiments, please check if your `allocation_mode` matches your cluster configuration. Make sure #GPUs allocated by `allocation_mode` equals to `cluster.n_nodes * cluster.n_gpus_per_node`. 
> **Note**: Ray and Slurm launchers only work for distributed experiments with more than 1 node (`cluster.n_nodes > 1`). They allocate GPUs for training and generation at the granularity of **nodes**, which means the number of GPUs allocated for generation and training must be integer multiples of `cluster.n_gpus_per_node`.
-->

## Next Steps

1. Check [Getting Started with AReaLite](../arealite/gsm8k_grpo.md) for a complete code walkthrough on the GRPO GSM8K Example.
2. Customization guides:
- [Custom dataset](../customization/dataset.md)
- [Custom agentic/RVLR rollout workflows](../customization/agent.md)
- [Custom algorithms](../customization/algorithm.md)