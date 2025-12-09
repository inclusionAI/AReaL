<div align="center">
<h1><img src="../assets/multiverse-logo.png" height="40px" align="top"/> Multiverse Training
</h1>
</div>

# Description

This training setup is adapted from the [s1: Simple test-time scaling](https://github.com/simplescaling/s1) repository. In the training section we provide the code to train:
1. `Multiverse-32B`: the Multiverse reasoning model applying Multiverse Attention mechanism which enables its splist/merge ability during inference, trained by `Multiverse 1K` dataset.
2. `Autoregressive-32B`: the AR reasoning model trained by `Multiverse 1K` dataset.

## Environment

To train `Multiverse-32B`/`Autoregressive-32B`, we recommend 16 A/H100 GPUs (i.e., 2 nodes with 8 each) or 8 B100/200 GPUs.

## Setup and Training

### Quick Start

1.  With prepared training data in Data section, you can train `Multiverse-32B` with following command.
    ```bash
    bash sft_multiverse.sh
    ```
2.  With prepared training data in Data section, you can train `Autoregressive-32B` with following command.
    ```bash
    bash sft_autoregressive.sh
    ```
3. We provide the original slurm/multinode support for the training in `sft_multinode.sh` and `sft_slurm.sh`.

### Notes
*   If you encounter an out-of-memory (OOM) issue with 8 GPUs, consider enabling gradient checkpointing by adding `--gradient_checkpointing=True` to `train/sft.sh`.
*   For `s1.1`, the original authors set the block size to 20000 to avoid OOM. This is configured in `train/sft.sh`.
