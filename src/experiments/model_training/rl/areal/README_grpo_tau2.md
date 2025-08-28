# GRPO on $\tau^2$-bench

The [code](./tau2_grpo.py) is adopted from [GSM8k GRPO](https://github.com/inclusionAI/AReaL/blob/main/examples/lite/gsm8k_grpo.py).

## Example

```bash
python3 -m areal.launcher.local examples/lite/tau2_grpo.py \
    --config examples/lite/configs/tau2_grpo.yaml \
    trial_name=small_8gpus_trial2 \
    actor.path=Qwen/Qwen2.5-3B-Instruct \
    actor.ppo_n_minibatches=4 \
    actor.ds_auto_tp.autotp_size=4 \
    ref.ds_auto_tp.autotp_size=4 \
    allocation_mode=sglang.d4p1t1+d4p1t1 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=8 \
    econfig.domain=telecom \
    econfig.solo_mode=True \
    train_dataset.batch_size=4 \
    train_dataset.path=tau2/small \
    valid_dataset.batch_size=16 \
    valid_dataset.path=tau2/test \
    stats_logger.wandb.mode=online \
    +sglang.attention_backend=triton
```

In the above example command, we use 8 GPUs to train a 3B model on the $\tau^2$-bench.

Now let's explain other arguments in the command:
* `actor.path`: the huggingface path to the actor model, here we use the 3B model from Qwen2.5.
* `actor.ppo_n_minibatches`: the number of minibatches to use for PPO training, we choose it to make each minibatch contains one data point.
* `actor.ds_auto_tp.autotp_size`: the tensor parallelism size for the actor model.
* `ref.ds_auto_tp.autotp_size`: the tensor parallelism size for the reference model.
* `allocation_mode`: the allocation mode (data parallelism + pipeline parallelism + tensor parallelism) for the training, `sglang.d4p1t1+d4p1t1` means 4 GPUs for inference (with sglang) and 4 GPU for training.
* `cluster.n_nodes`: the number of nodes to use for the training.
* `cluster.n_gpus_per_node`: the number of GPUs per node to use for the training.
* `econfig.domain`: the domain of the $\tau^2$-bench, here we use the telecom domain.
* `econfig.solo_mode`: whether to use solo mode, here we use solo mode.
* `train_dataset.batch_size`: the batch size for the training set.
* `train_dataset.path`: we use this to select the split of the training set, here we use the small split, which contains tasks with 1 subtasks.
* `valid_dataset.batch_size`: the batch size for the validation set.
* `valid_dataset.path`: we use this to select the split of the validation set, here we use the test split.
* `stats_logger.wandb.mode`: the mode for the wandb logger, here we use online mode to enable wandb online logging.
* `+sglang.attention_backend=triton`: the attention backend for the training, here we use triton.

To enable LoRA training, we need to add the following arguments (work with the `tau` branch):
```bash
+actor.peft.enable=True \
+actor.peft.r=8 \
+actor.peft.lora_alpha=64 \
+actor.peft.lora_dropout=0.1 \
+actor.peft.inference_mode=False
```

* `+actor.peft.enable=True`: enable LoRA training.
* `+actor.peft.r=8`: the rank of the LoRA matrix.
* `+actor.peft.lora_alpha=64`: the alpha of the LoRA matrix.
* `+actor.peft.lora_dropout=0.1`: the dropout of the LoRA matrix.
* `+actor.peft.inference_mode=False`: whether to use the LoRA matrix for inference.

## logs
The log for runs in default is located in `/tmp/areal/experiments/logs`.

# GPU memory calculation

Rollout: TODO

Training: TODO

# TODOs
1. [Done] calculate GPU memory, make 3B training working
2. add partial reward (based on progress)
