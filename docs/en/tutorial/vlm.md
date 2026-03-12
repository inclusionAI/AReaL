# Vision-Language Model (VLM) Training

This guide explains how to train Vision-Language Models (VLMs) with AReaL using GRPO
reinforcement learning and SFT. AReaL supports VLM training on both NVIDIA GPUs and
Ascend NPUs.

## Overview

AReaL provides VLM training examples under two directories:

- `examples/vlm/` — GPU-based VLM training (SFT and GRPO)
- `examples/vlm_npu/` — Ascend NPU-based VLM training (GRPO only)

All VLM GRPO examples use the `VisionRLVRWorkflow`
(`areal.workflow.vision_rlvr.VisionRLVRWorkflow`), which extends the standard RLVR
workflow with multimodal (image + text) input support.

## GPU Examples (`examples/vlm/`)

### Supported Tasks

| Example                   | Model                      | Dataset                       | Task            | Inference Backend |
| ------------------------- | -------------------------- | ----------------------------- | --------------- | ----------------- |
| `clevr_count_70k_grpo`    | Qwen2.5-VL-3B-Instruct    | BUAADreamer/clevr_count_70k   | GRPO RL         | SGLang            |
| `clevr_count_70k_sft`     | Qwen2-VL-7B               | BUAADreamer/clevr_count_70k   | SFT             | —                 |
| `geometry3k_grpo`         | Qwen2.5-VL-3B-Instruct    | hiyouga/geometry3k            | GRPO RL         | vLLM              |

### CLEVR Count 70K — GRPO Training

This example trains Qwen2.5-VL-3B-Instruct on the CLEVR counting task using GRPO.

**Run:**

```bash
python examples/vlm/clevr_count_70k_grpo.py \
    --config examples/vlm/clevr_count_70k_grpo.yaml \
    scheduler.type=local
```

**Key configuration** (`clevr_count_70k_grpo.yaml`):

- **Allocation**: `sglang:d1p1t1+d7p1t1` — 1 GPU for SGLang inference, 7 GPUs for
  training (actor + ref colocation)
- **Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Generation**: 4 samples per prompt, max 512 new tokens, temperature 1.0
- **GRPO settings**: `eps_clip=0.4`, `reward_scaling=10.0`, `reward_bias=-0.5`,
  `kl_ctl=0.0`, `decoupled_loss=true`
- **SGLang**: `enable_multimodal=true`, `context_length=32768`,
  `mem_fraction_static=0.8`

**Reward function**: Extracts a numeric answer from brackets `[answer]` in the model
output and compares it with the ground truth (exact string match).

```python
def clevr_count_70k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    sol = extract_answer(completions, data_name="")  # extracts from [number]
    if sol is None or answer is None:
        return 0
    return float(sol.strip() == answer.strip())
```

### CLEVR Count 70K — SFT Training

This example runs supervised fine-tuning on the same CLEVR counting dataset.

**Run:**

```bash
python examples/vlm/clevr_count_70k_sft.py \
    --config examples/vlm/clevr_count_70k_sft.yaml \
    scheduler.type=local
```

**Key configuration** (`clevr_count_70k_sft.yaml`):

- **Allocation**: `d8p1t1` — all 8 GPUs for training (no inference engine needed)
- **Model**: `Qwen/Qwen2-VL-7B`
- **Optimizer**: Adam with `lr=2e-5`, cosine LR scheduler, `weight_decay=0.05`
- **Dataset type**: `sft` (not `rl`)
- **Batch size**: 128

**Training script**: Uses `SFTTrainer` instead of `PPOTrainer`, and loads the processor
and tokenizer for multimodal data preprocessing.

```python
with SFTTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
    trainer.train()
```

### Geometry3K — GRPO Training

This example trains Qwen2.5-VL-3B-Instruct on geometry problem solving.

**Run:**

```bash
python examples/vlm/geometry3k_grpo.py \
    --config examples/vlm/geometry3k_grpo.yaml \
    scheduler.type=local
```

**Key configuration** (`geometry3k_grpo.yaml`):

- **Allocation**: `vllm:d4p1t1+d4p1t1` — 4 GPUs for vLLM inference, 4 GPUs for
  training
- **Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Dataset**: `hiyouga/geometry3k`
- **vLLM**: `max_model_len=32768`, `gpu_memory_utilization=0.9`

**Reward function**: Combines a format reward (10%) and an accuracy reward (90%). The
format reward checks that the output follows the `<think>...</think>...\boxed{...}`
pattern. The accuracy reward uses `mathruler` to extract and grade the boxed answer.

```python
def geometry3k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    format_reward_val = format_reward(completions)   # checks <think>...\boxed{} format
    acc_reward_val = acc_reward(completions, answer)  # grades extracted answer
    format_score = 0.1
    return (1.0 - format_score) * acc_reward_val + format_score * format_reward_val
```

## NPU Examples (`examples/vlm_npu/`)

These examples demonstrate VLM GRPO training on **Ascend NPU** hardware. The training
scripts and YAML configs share the same structure as GPU examples, with NPU-specific
adjustments.

```{important}
All NPU examples set `USE_OPTIMIZED_MODEL=0` to disable vLLM-Ascend model optimizations
that may be incompatible with RLHF training weight syncing.
```

### Supported Tasks

| Example                                  | Model                   | Dataset              | Nodes | GPUs/Node |
| ---------------------------------------- | ----------------------- | -------------------- | ----- | --------- |
| `qwen2_5_vl_3b_geometry3k_grpo`          | Qwen2.5-VL-3B-Instruct | hiyouga/geometry3k   | 1     | 8         |
| `qwen3_vl_2b_geometry3k_grpo`            | Qwen3-VL-2B-Instruct   | hiyouga/geometry3k   | 1     | 8         |
| `qwen2_5_vl_3b_virl39k_grpo_multinode`   | Qwen2.5-VL-3B-Instruct | TIGER-Lab/ViRL39K    | 3     | 16        |

### Qwen2.5-VL-3B on Geometry3K (Single Node)

**Run:**

```bash
bash examples/vlm_npu/qwen2_5_vl_3b_geometry3k_grpo.sh
```

The shell script sets `USE_OPTIMIZED_MODEL=0` and launches:

```bash
export USE_OPTIMIZED_MODEL=0
python examples/vlm_npu/geometry3k_grpo.py \
    --config examples/vlm_npu/qwen2_5_vl_3b_geometry3k_grpo.yaml \
    scheduler.type=local
```

**Configuration**: Same structure as the GPU Geometry3K example — `vllm:d4p1t1+d4p1t1`
allocation, identical GRPO hyperparameters.

### Qwen3-VL-2B on Geometry3K (Single Node)

A smaller model variant suitable for resource-limited environments.

**Run:**

```bash
bash examples/vlm_npu/qwen3_vl_2b_geometry3k_grpo.sh
```

**Differences from Qwen2.5-VL-3B**:

- **Model**: `Qwen/Qwen3-VL-2B-Instruct` (smaller, 2B parameters)
- **Scheduling spec**: Explicitly requests `cpu: 4` per worker (in addition to 1 GPU)
- All other hyperparameters remain identical

### Qwen2.5-VL-3B on ViRL39K (Multi-Node)

Large-scale multi-node training with long-context generation.

**Run:**

```bash
bash examples/vlm_npu/qwen2_5_vl_3b_virl39k_grpo_multinode.sh
```

**Key overrides applied via command line**:

| Parameter                          | Value                |
| ---------------------------------- | -------------------- |
| `scheduler.type`                   | `ray`                |
| `allocation_mode`                  | `vllm:d32+d16`       |
| `cluster.n_nodes`                  | `3`                  |
| `cluster.n_gpus_per_node`          | `16`                 |
| `train_dataset.path`               | `data/ViRL39K`       |
| `gconfig.max_new_tokens`           | `16384`              |
| `gconfig.n_samples`                | `5`                  |
| `actor.mb_spec.max_tokens_per_mb`  | `16384`              |
| `train_dataset.batch_size`         | `528`                |
| `total_train_epochs`               | `5`                  |

This configuration uses **3 nodes with 16 NPUs each** (48 NPUs total), allocating 32 for
vLLM inference and 16 for training. The Ray scheduler manages cross-node resource
allocation.

**Reward function** (`virl39k_grpo.py`): Same format + accuracy reward as Geometry3K. The
training script additionally passes `img_folder_path=None` to `get_custom_dataset` for
the train split, indicating images are embedded in the dataset.

## NPU Benchmark Results

### Single-Node: Qwen2.5-VL-3B on Geometry3K

Trained for 70 epochs with 4+4 (train+infer) card configuration. Total training time:
~19 hours.

**Hardware tested**: 16x NPU per node, 64 CPU cores, 1TB memory, RoCE 3.2 Tbps network.

**Out-of-distribution evaluation** (VLMEvalKit):

| Method     | LogicVista | MathVision_mini | MathVista_mini | Avg.     |
| ---------- | ---------- | --------------- | -------------- | -------- |
| Base Model | 31.0       | 18.3            | 52.3           | 33.8     |
| GRPO-GPU   | 35.4       | 20.9            | 55.9           | **37.4** |
| GRPO-NPU   | 35.3       | 20.5            | 54.7           | **36.8** |

NPU results closely match GPU results, demonstrating training parity across platforms.

### Multi-Node: AReaL vs. verl on ViRL39K

Comparison of AReaL asynchronous training vs. verl synchronous training on 3 nodes
(48 NPUs total), with `max_new_tokens=16384`.

| Framework | Checkpoint | Training Time | LogicVista | MathVision_mini | WeMath   | DynaMath | MathVerse | MMMU_Pro_v | Avg. |
| --------- | ---------- | ------------- | ---------- | --------------- | -------- | -------- | --------- | ---------- | ---- |
| verl      | Epoch 1    | 6.8 hours     | 33.0       | 18.8            | **19.6** | 32.9     | **31.3**  | **23.5**   | 26.5 |
| AReaL     | Epoch 2    | **4.3 hours** | 33.8       | 20.1            | 17.4     | 34.9     | 29.6      | 22.3       | 26.3 |
| AReaL     | Epoch 3    | **6.6 hours** | **34.1**   | **20.3**        | 18.9     | **35.7** | 30.2      | 22.5       | 27.0 |

AReaL completes 2 epochs in 4.3 hours (vs. verl's 6.8 hours for 1 epoch) with comparable
accuracy. By epoch 3, AReaL surpasses verl in overall accuracy (27.0 vs. 26.5) while
still using less total time (6.6 hours).

## Common Patterns Across VLM Examples

### Training Script Structure

All VLM GRPO training scripts follow the same pattern:

1. **Load config**: `load_expr_config(args, GRPOConfig)`
2. **Load processor + tokenizer**: `load_hf_processor_and_tokenizer(config.tokenizer_path)`
3. **Create datasets**: `get_custom_dataset(split, dataset_config, tokenizer, processor)`
4. **Define workflow kwargs**: Specify `reward_fn`, `gconfig`, `tokenizer`, `processor`,
   and `enable_thinking`
5. **Launch training**: `PPOTrainer` with `VisionRLVRWorkflow`

```python
workflow_kwargs = dict(
    reward_fn=my_reward_fn,
    gconfig=config.gconfig,
    tokenizer=config.tokenizer_path,
    processor=config.tokenizer_path,
    enable_thinking=False,
)

with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
    trainer.train(
        workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
        workflow_kwargs=workflow_kwargs,
        eval_workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
        eval_workflow_kwargs=eval_workflow_kwargs,
    )
```

### Key Differences: GPU vs. NPU

| Aspect                     | GPU                                    | NPU                                               |
| -------------------------- | -------------------------------------- | -------------------------------------------------- |
| Environment variable       | Not needed                             | `export USE_OPTIMIZED_MODEL=0`                     |
| Inference backend          | SGLang or vLLM                         | vLLM (via vllm-ascend)                             |
| Reward function reference  | String path (e.g., `"examples.vlm..."`)| Direct function reference                          |
| Launch method              | Direct Python                          | Shell script wrapping Python                       |
| Multi-node scheduler       | —                                      | Ray (`scheduler.type=ray`)                         |

### YAML Configuration Reference

Key sections in the YAML config files:

- **`allocation_mode`**: Defines GPU split between inference and training
  - `sglang:d1p1t1+d7p1t1` — 1 GPU SGLang + 7 GPU training
  - `vllm:d4p1t1+d4p1t1` — 4 GPU vLLM + 4 GPU training
  - `vllm:d32+d16` — 32 GPU vLLM + 16 GPU training (multi-node)
- **`sglang`/`vllm`**: Inference engine config with `enable_multimodal: true`
- **`gconfig`**: Generation config (`n_samples`, `max_new_tokens`, `temperature`)
- **`actor`**: Training model config with GRPO hyperparameters
- **`ref`**: Reference model (colocated with actor via `scheduling_strategy`)
