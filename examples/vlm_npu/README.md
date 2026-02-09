# Training VLMs with GRPO on NPU:

In this instruction, we will introduce how to train VLMs with GRPO on Ascend NPU.

## Examples

This directory contains examples for training vision-language models on NPU with GRPO:

1. **Qwen2.5-VL-3B** - `qwen2_5_vl_3b_geometry3k_grpo.*` - GRPO training on Geometry3K
   with Qwen2.5-VL-2B model dataset
1. **Qwen3-VL-2B** - `qwen3_vl_2b_geometry3k_grpo.*` - GRPO training on Geometry3K
   dataset with Qwen3-VL-2B model

## Running the Examples

### Qwen2.5-VL-3B

```bash
bash examples/vlm_npu/qwen2_5_vl_3b_geometry3k_grpo.sh
```

### Qwen3-VL-2B

```bash
bash examples/vlm_npu/qwen3_vl_2b_geometry3k_grpo.sh
```

Both examples use the same Geometry3K dataset and GRPO training configuration, but with
different model architectures. The Qwen3-VL-2B model is smaller and may be more suitable
for environments with limited resources.

## Testing Qwen2.5-VL-3B

### Hardware

The following hardware configuration has been extensively tested:

- **NPU**: 16x NPU per node
- **CPU**: 64 cores per node
- **Memory**: 1TB per node
- **Network**: RoCE 3.2 Tbps
- **Storage**:
  - 1TB local storage for single-node experiments
  - 10TB shared storage (NAS) for distributed experiments

### Key Contributions

- Trained Qwen2.5VL-3B-instruct model upto 70 epochs with (4 cards+ 4 cards) train-infer
  configuration. Took around 19hr to finish full training.
- Trained model is tested with more than one benchmark using VLMEvalKit.

### Results:

We trained Qwen2.5-VL-3B for 70 epochs on Geometry3K and evaluated the checkpoints using
VLMEvalKit on out of distribution tasks such as MathVision, MathVista, and LogicVista.
The training was performed on both NPU and GPU and results are as follows:

| Method     | LogicVista | MathVision_mini | MathVista_mini | Avg.     |
| ---------- | ---------- | --------------- | -------------- | -------- |
| Base Model | 31.0       | 18.3            | 52.3           | 33.8     |
| GRPO-GPU   | 35.4       | 20.9            | 55.9           | **37.4** |
| GRPO-NPU   | 35.3       | 20.5            | 54.7           | **36.8** |
