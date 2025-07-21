# Quickstart

Welcome to AReaLite quickstart guide!
In this guide, we provide an example that runs an AReaLite experiment that trains an LLM on GSM8K dataset with GRPO algorithm and function-based rewards. 
Please ensure you have properly [installed dependencies and set up the runtime environment](installation.md) before proceeding.

# Running the Experiment (on a single node)

To run the experiment, you need a training script and a config YAML file.
- Training script: [examples/arealite/gsm8k_grpo.py](../../examples/arealite/gsm8k_grpo.py)
- Config YAML: [examples/arealite/configs/gsm8k_grpo.yaml](../../examples/arealite/configs/gsm8k_grpo.yaml)

Our training scripts will automatically download the dataset (openai/gsm8k) and model (Qwen/Qwen3-1.7B) for you. 
You do not need to prepare any dataset or model files before running the experiment.
To run the example with the default configuration, execute following command from the repository directory: 
```
python3 -m arealite.launcher.local examples/arealite/gsm8k_grpo.py --config examples/arealite/configs/gsm8k_grpo.yaml experiment_name=<your experiment name> trial_name=<your trial name>
```

> **Note**: The command above uses `LocalLauncher`, which only works for a single node (`cluster.n_nodes == 1`). For launching distributed experiments, please check out [Distributed Experiments with Ray or Slurm](quickstart.md#distributed-experiments-with-ray-or-slurm).

### Modifying configuration

All available options for experiment configuration are listed in [arealite/api/cli_args.py](https://github.com/inclusionAI/AReaL/blob/main/arealite/api/cli_args.py). 
To change the experiment configuration, including models, resource allocation, and algorithm options, you can:
1. Directly modifying the config YAML file at [examples/arealite/configs/gsm8k_grpo.yaml](../../examples/arealite/configs/gsm8k_grpo.yaml).
2. Adding command line options. For entries that exist in the config YAML, you could directly add the options after your command. For example: `actor.path=Qwen/Qwen3-1.7B`. For other options in `cli_args.py` but not in YAML, you could add these options with a prefix "+". For example: `+sglang.attention_backend=triton`. 

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
Since we are working on a refactor from legacy AReaL to AReaLite, there are some changes that makes available options for AReaLite slightly different from legacy AReaL. We provide a **config converter** to transfer old AReaL config into AReaLite YAML file for users' convenience. [Click here](xxx) for the usage of **config converter**. 
::::

### Distributed Experiments with Ray or Slurm

AReaLite also provide standalone Ray or Slurm launchers for distributed experiments. Once you have properly setup your Ray or Slurm cluster, you could launch your experiment with `arealite.launcher.ray` and `arealite.launcher.slurm`, similar to the `LocalLauncher`:

```
# Launch with Ray launcher. 4 nodes with 4 GPUs each node, 3 nodes for generation, 1 node for training.
python3 -m arealite.launcher.ray examples/arealite/gsm8k_grpo.py \
    --config examples/arealite/configs/gsm8k_grpo.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    allocation_mode=sglang.d12p1t1+d4p1t1 \
    cluster.n_nodes=4 \
    cluster.n_gpus_per_node=4 \
    ...

# Launch with Slurm launcher. 16 nodes with 8 GPUs each node, 12 nodes for generation, 4 nodes for training
python3 -m arealite.launcher.slurm examples/arealite/gsm8k_grpo.py \
    --config examples/arealite/configs/gsm8k_grpo.yaml \
    experiment_name=<your experiment name> \
    trial_name=<your trial name> \
    allocation_mode=sglang.d96p1t1+d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8 \
    ...
```

[Click here](installation.md#optional-launch-ray-cluster-for-distributed-training) for a guide on how to set up a ray cluster. For more options for launchers, check `LauncherConfig` in [arealite/cli_args.py](quickstart.md#distributed-experiments-with-ray-or-slurm).

> **Note**: Before launching distributed experiments, please check if your `allocation_mode` matches your cluster configuration. Make sure #GPUs allocated by `allocation_mode` equals to `cluster.n_nodes * cluster.n_gpus_per_node`. 
> **Note**: Ray and Slurm launchers only work for distributed experiments with more than 1 node (`cluster.n_nodes > 1`). They allocate GPUs for training and generation at the granularity of **nodes**, which means the number of GPUs allocated for generation and training must be integer multiples of `cluster.n_gpus_per_node`.

## Next Steps

Check [Getting Started with AReaLite](../arealite/gsm8k_grpo.md) for a complete code walkthrough for the GRPO GSM8K Example.