# Reproduction Guide

The experiment configurations used to train our released models can be found in `examples/configs`. The user can overwrite `training/config/async_ppo.yaml` with our provided config (e.g., by copy-pasting) and then run `python3 training/main_async_ppo.py` as illustrated in the [quickstart section](../tutorial/quickstart.md).

**Available Configurations:**

+ `examples/configs/v0.2-qwen2-math`: Configuration for reproducing the boba math models based on R1-Distilled-Qwen models
  + Dataset: [AReaL-boba-106k](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/blob/main/AReaL-boba-106k.jsonl)

+ `examples/configs/v0.3-qwen3-code`: Configuration for reproducing the boba² coding model based on Qwen3
  + Dataset: [DeepCoder Dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) (duplicated problems and problems with incorrect answers have been filtered out)

## Adjusting Computation Resources

We recommend using the provided number of GPUs and corresponding parallelization strategy for optimal reproducibility. When resources are limited, try decreasing the `n_nodes` and reducing the data parallelism degree in `allocation_mode`.

**Example Resource Adjustment:**

When reproducing the Qwen3 8B coding model, the original configuration is:
+ `n_nodes=24` 
+ `allocation_mode=sglang.d80m2p1+d4m2p4`

If only 6 nodes are available, you can adjust the configuration to:
+ `n_nodes=6` 
+ `allocation_mode=sglang.d20m2p1+d1m2p4`

If you encounter out-of-memory (OOM) errors, please refer to the [troubleshooting section](../tutorial/troubleshooting.md) or raise an issue on GitHub.

```{note}
We acknowledge that our provided device allocation and parallelism strategies could be further optimized. However, since this guide focuses on reproducibility, we do not recommend changing them arbitrarily due to data loading randomness and potential precision issues. We hope our provided configurations can serve as a starting point for further development and optimization.
```