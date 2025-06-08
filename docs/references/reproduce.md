# Reproduction Guide

The experiment configurations can be found [here](https://github.com/inclusionAI/AReaL/tree/main/examples/configs/v0.3-qwen3-code).

Users should overwrite [training/config/async_ppo.yaml](https://github.com/inclusionAI/AReaL/blob/main/training/configs/async-ppo.yaml) and then run:

```bash
python3 training/main_async_ppo.py
```

More information can be found in the [quickstart section](../tutorial/quickstart.md).

**Math:**

+ [`examples/configs/v0.2-qwen2-math`](https://github.com/inclusionAI/AReaL/tree/main/examples/configs/v0.2-qwen2-math): Configuration for reproducing the boba math models based on R1-Distilled-Qwen models
  + Dataset: [AReaL-boba-106k](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/blob/main/AReaL-boba-106k.jsonl)

**Code:**

+ [`examples/configs/v0.3-qwen3-code`](https://github.com/inclusionAI/AReaL/tree/main/examples/configs/v0.3-qwen3-code): Configuration for reproducing the boba² coding model based on Qwen3
  + Dataset: [AReaL-boba-2-code](https://huggingface.co/datasets/inclusionAI/AReaL-boba-2-RL-Code). We perform deduplication and filtering on the [DeepCoder dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset). After deduplication and quality filtering, approximately 7,600 coding problems remained. Additionally, we have around 10,000 internal data entries that can further improve the model's performance. We are actively working on open-sourcing this portion of the data.

## Adjusting Computational Resources

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