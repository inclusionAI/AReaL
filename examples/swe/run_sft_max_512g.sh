#!/bin/bash
set -euo pipefail

# v8: PP=16,TP=8,CP=2,EP=16,DP=1 — model is 80 layers (not 32), need PP=16 to fit
# 80 expert-layers/rank (was 160 with PP=8, all OOM/hang)

MODEL_PATH=/storage/openpsi/users/fenghui/ring_2_5_validation_release24_128k_0215_1344_iter2961/sglang_release/

PYTHONPATH=/storage/openpsi/users/chucai.dzq/codes/AReaL \
WANDB_API_KEY=local-3bca3d5f00a980f3075b3e8ff2e16adc4ef43ffe \
WANDB_BASE_URL=http://8.150.1.98:8080 \
python -m areal.infra.launcher.slurm examples/swe/train_sft.py \
      --config examples/swe/swe_ring_max_256g.yaml \
      scheduler.type=null \
      allocation_mode="megatron:(attn:d2p16t8c2|ffn:d2p16e16)" \ 
      stats_logger.wandb.mode=online \
      experiment_name=swe-ring-max-sft \
      trial_name=bs512_lr5e-6_p16t8c2_traindata_0327_0 \
      total_train_epochs=5 \
      train_dataset.batch_size=256 \
      swe.num_proc=1 \
      swe.filter_errors=true \
      swe.strip_all_thinking=false \
      saver.freq_epochs=1 \
      train_dataset.path=/storage/openpsi/users/chucai.dzq/datasets/swe_sft_tokenized_max_v1 \
      actor.path=$MODEL_PATH \
      cluster.fileroot=/storage/openpsi/experiments \
      cluster.name_resolve.nfs_record_root=/storage/openpsi/experiments/name_resolve/lite-grpo \
      +cluster.name_resolve.etcd3_addr=etcd-client.openpsi-etcd.svc.sigma-su18-01.hn01.su18-hn.local:2379 \
      cluster.n_nodes=64 \
      cluster.n_gpus_per_node=8
