#!/bin/bash
set -euo pipefail

# v7: cherry-pick thinking tag normalize from sxj, 4 epochs with HF save each epoch
# Changes from v6:
#   - normalize <thinking> -> <think> (single special token)
#   - 4 epochs (was 200), save every epoch
#   - num_proc=16 (was 8)

MODEL_PATH=/storage/openpsi/models/Ling2.5/swe_ring25_flash_tokenizer

PYTHONPATH=/storage/openpsi/users/chucai.dzq/codes/AReaL \
WANDB_API_KEY=local-b8ad8e6a05487e8245c05b20d58c32fadecd8952 \
WANDB_BASE_URL=http://8.150.1.98:8080 \
python -m areal.infra.launcher.slurm examples/swe/train_sft.py \
      --config examples/swe/sft_ring_flash.yaml \
      scheduler.type=null \
      stats_logger.wandb.mode=online \
      experiment_name=swe-sft \
      trial_name=flash_moe_bs256_lr5e-6_0 \
      total_train_epochs=4 \
      train_dataset.batch_size=256 \
      swe.num_proc=16 \
      swe.filter_errors=true \
      swe.strip_all_thinking=false \
      saver.freq_epochs=1 \
      train_dataset.path=/storage/openpsi/users/shenxujie.sxj/datasets/swe_data/sft_test_dataset_filter_openai_all_last.jsonl \
      actor.path=$MODEL_PATH \
      cluster.fileroot=/storage/openpsi/experiments \
      cluster.name_resolve.nfs_record_root=/storage/openpsi/experiments/name_resolve/lite-grpo \
      +cluster.name_resolve.etcd3_addr=etcd-client.openpsi-etcd.svc.sigma-su18-01.hn01.su18-hn.local:2379 \
      cluster.n_nodes=16 \
      cluster.n_gpus_per_node=8
