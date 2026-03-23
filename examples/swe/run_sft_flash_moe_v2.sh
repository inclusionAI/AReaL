#!/bin/bash
set -euo pipefail

# v2: Fixed optimizer config, recovery, and lr
# Changes from v1:
#   - lr: 1e-5 -> 5e-6, lr_scheduler: cosine -> constant, warmup: 0.001 -> 0.0
#   - grad_reduce_in_fp32: false -> true
#   - overlap_grad_reduce: false -> true (performance)
#   - weight_decay: 0.05 -> 0.01 (align with internal config)
#   - beta2: 0.95 -> 0.999 (align with internal config)
#   - total_train_epochs: 10 -> 7 (~1099 steps)
#   - recover.mode: auto, no_save_optim: true (skip optimizer to avoid flattened_range error)

IMAGE_PATH=/storage/openpsi/images/areal-latest.sif
INFER_IMAGE_PATH=$IMAGE_PATH
MODEL_PATH=/storage/openpsi/models/Ling2.5/swe_ring25_flash_tokenizer
FLASH_LINEAR_ATTENTION=/storage/openpsi/users/chucai.dzq/codes/flash-linear-attention

PYTHONPATH=/storage/openpsi/users/chucai.dzq/codes/AReaL
WANDB_API_KEY=local-b8ad8e6a05487e8245c05b20d58c32fadecd8952 \
WANDB_BASE_URL=http://8.150.1.98:8080 \
python -m areal.infra.launcher.slurm examples/swe/train_sft.py \
      --config examples/swe/swe_sft_flash_moe_v2.yaml \
      scheduler.type=null \
      stats_logger.wandb.mode=online \
      experiment_name=dzq-swe-sft \
      trial_name=0323_flash_moe_v4 \
      train_dataset.batch_size=128 \
      swe.num_proc=16 \
      swe.filter_errors=true \
      swe.strip_all_thinking=true \
      train_dataset.path=/storage/openpsi/users/shenxujie.sxj/datasets/swe_data/sft_test_dataset_filter_openai_all_last.jsonl \
      actor.path=$MODEL_PATH \
      cluster.fileroot=/storage/openpsi/experiments \
      cluster.name_resolve.nfs_record_root=/storage/openpsi/experiments/name_resolve/lite-grpo \
      +cluster.name_resolve.etcd3_addr=etcd-client.openpsi-etcd.svc.sigma-su18-01.hn01.su18-hn.local:2379 \
      cluster.n_nodes=8 \
      cluster.n_gpus_per_node=8
