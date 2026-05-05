# Training Report — `qwen3vl8b-thinking-5ds-v2-0419-ct65536/checkpoint-280`

> SFT checkpoint under LLaMA-Factory, trained on 2 nodes × 8 GPU with DeepSpeed Zero-2.
>
> **Full checkpoint path**:
> `/storage/openpsi/models/lcy_image_edit/sft_workspace/qwen3vl8b-thinking-5ds-v2-0419-ct65536/checkpoint-280`

## 1. Base Model & Objective

- **Base model**: `/storage/openpsi/models/Qwen3-VL-8B-Thinking`
  - Arch: `Qwen3VLForConditionalGeneration`
  - Text config: 36 layers, 32 attention heads, 8 KV heads, hidden 4096
  - RoPE θ = 5,000,000, `max_position_embeddings = 262,144`
  - Vocab 151,936
- **Stage**: `sft` (supervised fine-tuning)
- **Finetuning type**: `full` (all LM params trainable)
- **Vision tower**: **frozen** (`freeze_vision_tower: true`)
- **Multimodal projector**: **frozen** (`freeze_multi_modal_projector: true`)
- **Language model**: **trained** (`freeze_language_model: false`)

## 2. Data Mix — `merged_5ds_v2_0419` (10,661 examples)

| Source dataset | Records | % |
| --- | ---: | ---: |
| MapTrace (Qwen3-VL-235B augmented SFT data) | 3,698 | 34.7% |
| Mini-o3-Coldstart-Dataset (incl. `deepeyes_chart` etc.) | 3,000 | 28.1% |
| reasonmap_plus_train_0413 (Qwen3-VL-235B augmented) | 1,979 | 18.6% |
| mm_mapqa (Qwen3-VL-235B augmented) | 1,716 | 16.1% |
| reasonmap_train_0413 (Qwen3-VL-235B augmented) | 268 | 2.5% |
| **Total** | **10,661** | 100% |

- **Format**: ShareGPT (`conversations`, `images`, `system`)
- **Template**: `qwen3_vl`
- **All records share the same tool-calling system prompt** (reasoning + tool usage protocol)

## 3. Training Hyperparameters

| Field | Value |
| --- | --- |
| `num_train_epochs` | 1.0 (reached 0.838 at step 280 out of 334 total) |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 1 |
| Effective global batch size | 16 GPUs × 1 × 1 = **16** |
| `learning_rate` | 1.0 × 10⁻⁵ |
| `lr_scheduler_type` | cosine |
| `warmup_ratio` | 0.1 |
| `weight_decay` | 0.01 |
| `max_grad_norm` | 1.0 |
| Optimizer | AdamW (torch), β = (0.9, 0.999), ε = 1e-8 |
| Precision | bf16 mixed (`bf16_full_eval = false`) |
| `gradient_checkpointing` | true |
| `seed` | 42 |
| `ddp_timeout` | 180,000,000 |

## 4. Sequence / Image Limits

| Field | Value |
| --- | --- |
| `cutoff_len` (max text tokens) | 65,536 |
| `image_max_pixels` | 6,291,456 (≈ 2508² — note: README banner said 8,388,608; YAML has 6,291,456) |
| `video_max_pixels` | 16,384 |
| `generation_max_length` | 65,536 |

## 5. Parallelism / Infra

| Field | Value |
| --- | --- |
| Hardware | 2 nodes × 8 L20X = **16 GPUs** |
| `hostfile` | `33.180.163.26 slots=8`, `33.180.160.92 slots=8` |
| Master addr / port | `33.180.163.26` / `45661` |
| Launcher | `deepspeed --hostfile ...` |
| Distributed type | `DeepSpeed` |
| NCCL backend | default |
| DeepSpeed Zero stage | **Stage 2** |
| Zero offload | none (optimizer & param **on-device**) |
| Allgather / reduce bucket | 5 × 10⁸ |
| `overlap_comm` | false |
| `reduce_scatter` | true |
| `round_robin_gradients` | true |

### `ds_z2_config.json`

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {"enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000,
           "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1},
  "bf16": {"enabled": "auto"},
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "round_robin_gradients": true
  }
}
```

## 6. Checkpointing

| Field | Value |
| --- | --- |
| `save_strategy` | steps |
| `save_steps` | 20 |
| `save_total_limit` | 3 |
| `save_only_model` | false (full trainer state saved) |
| `output_dir` | `/storage/openpsi/models/lcy_image_edit/sft_workspace/qwen3vl8b-thinking-5ds-v2-0419-ct65536` |
| Resume-from-checkpoint | `.../checkpoint-280` (as set in YAML — this checkpoint was used as a mid-training resume point) |

## 7. Loss / LR Curve (every 10 steps, step 10 → 280)

| Step | Loss | LR | Grad-norm | Epoch |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 1.5288 | 2.65e-06 | 5.422 | 0.030 |
| 20 | 1.1340 | 5.59e-06 | 2.322 | 0.060 |
| 30 | 0.9333 | 8.53e-06 | 1.943 | 0.090 |
| 40 | 0.8622 | 9.99e-06 (peak) | 1.525 | 0.120 |
| 50 | 0.8122 | 9.94e-06 | 1.729 | 0.150 |
| 60 | 0.7895 | 9.83e-06 | 1.675 | 0.180 |
| 70 | 0.7720 | 9.67e-06 | 1.670 | 0.210 |
| 80 | 0.7694 | 9.46e-06 | 1.519 | 0.240 |
| 90 | 0.7555 | 9.19e-06 | 1.545 | 0.270 |
| 100 | 0.7500 | 8.89e-06 | 1.356 | 0.299 |
| 110 | 0.7462 | 8.54e-06 | 1.653 | 0.329 |
| 120 | 0.7301 | 8.15e-06 | 1.336 | 0.359 |
| 130 | 0.7342 | 7.72e-06 | 1.621 | 0.389 |
| 140 | 0.7247 | 7.27e-06 | 1.494 | 0.419 |
| 150 | 0.7005 | 6.79e-06 | 1.602 | 0.449 |
| 160 | 0.7176 | 6.29e-06 | 1.539 | 0.479 |
| 170 | 0.7056 | 5.78e-06 | 1.411 | 0.509 |
| 180 | 0.6807 | 5.26e-06 | 1.322 | 0.539 |
| 190 | 0.7031 | 4.74e-06 | 1.273 | 0.569 |
| 200 | 0.6774 | 4.22e-06 | 1.447 | 0.599 |
| 210 | 0.6865 | 3.71e-06 | 1.496 | 0.629 |
| 220 | 0.6777 | 3.21e-06 | 1.619 | 0.659 |
| 230 | 0.6751 | 2.73e-06 | 1.364 | 0.689 |
| 240 | 0.6719 | 2.28e-06 | 1.519 | 0.719 |
| 250 | 0.6868 | 1.85e-06 | 1.361 | 0.749 |
| 260 | 0.6537 | 1.46e-06 | 1.332 | 0.778 |
| 270 | 0.6538 | 1.11e-06 | 1.310 | 0.808 |
| **280** | **0.6619** | **8.07e-07** | **1.521** | **0.838** |

- Loss 1.53 → 0.66 (56% reduction)
- LR peaked at step ≈ 40 (≈ 1e-5, matches warmup ratio 0.1 × 334 ≈ 33)
- Cosine-annealed to ~8e-7 by step 280
- Grad-norm stable around 1.3 – 1.7 after warmup
- `total_flos` at step 280: 3.43 × 10¹⁸

## 8. Reporting / Other

| Field | Value |
| --- | --- |
| `report_to` | `tensorboard` |
| `logging_steps` | 10 |
| `logging_strategy` | steps |
| `plot_loss` | true |
| `project` | huggingface |
| `train_sampling_strategy` | random |
| `use_cache` | false (during train) |
| `trust_remote_code` | true |
| `preprocessing_num_workers` | 32 |
| `dataloader_num_workers` | 32 |

## 9. Related files

- **Multi-node launch**: [`batch_0414/run_qwen3vl8b_5ds_v2_0419_multinode.sh`](/storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/run_qwen3vl8b_5ds_v2_0419_multinode.sh)
- **Single-node launch**: [`batch_0414/run_qwen3vl8b_5ds_v2_0419.sh`](/storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/run_qwen3vl8b_5ds_v2_0419.sh)
- **LLaMA-Factory YAML**: [`configs/qwen3vl8b-thinking-5ds-v2-0419.yaml`](/storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/configs/qwen3vl8b-thinking-5ds-v2-0419.yaml)
- **DeepSpeed config**: [`configs/ds_z2_config.json`](/storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/configs/ds_z2_config.json)
- **Hostfile**: [`configs/hostfile`](/storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/configs/hostfile)
- **Training script**: `batch_0414/train_no_pil_limit.py` (LLaMA-Factory fork with PIL image-size safety removed)
- **Trainer state**: `$CKPT/trainer_state.json` (28 log entries, every 10 steps)
- **Raw args**: `$CKPT/training_args.bin` (pickled LLaMA-Factory `Seq2SeqTrainingArguments`; requires `llamafactory` module to unpickle natively — decoded via custom unpickler in this report)

## 10. Key observations

1. **Trained under LLaMA-Factory** (`train_no_pil_limit.py` + `training_args.bin` carries `llamafactory` module references).
2. **Step 280 ≠ final** — training was planned for 334 total steps (1 epoch). Step 280 is a resume-from checkpoint mark set in the YAML.
3. **Loss trajectory healthy** — smooth cosine decay, no spikes after step 10.
4. **Vision tower frozen** — only the LM + (implicit) projector input is updated; this preserves Qwen3-VL-8B-Thinking's visual encoder unchanged.
5. **Effective batch size 16 is small**; gradient accumulation is kept at 1 (compute-efficient but noisy gradient signal per step — hence the Grad-norm ~1.5 range).
6. **Mini-o3-Coldstart-Dataset is 28% of the data** — suggests this SFT is partly a cold-start for tool-using Thinking-style reasoning (matches the system prompt "You are an advanced AI agent capable of complex reasoning and tool usage").
