#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1
N_GPU=2
EXP_NAME=greso-dapo-clip-delta
TRIAL_NAME=trial0
FILE_ROOT=/data/yanglu/AReaL/tmp/areal/experiments
ACTOR_PATH=/data/yanglu/model/Qwen/Qwen2.5-Math-7B
PRM_PATH=/data/yanglu/model/Qwen/Qwen2.5-Math-PRM-7B
TRAIN_DATASET_PATH=/data/yanglu/dataset/greso
VALID_DATASET_PATH=/data/yanglu/dataset/greso

TOTAL_TRAIN_EPOCHS=1

python3 -m areal.launcher.local \
    examples/prm/greso_dapo_prm.py \
    --config examples/experimental/dapo/gsm8k_dapo.yaml \
    experiment_name="$EXP_NAME" \
    trial_name="$TRIAL_NAME" \
    +prm_path="$PRM_PATH" \
    +prmconfig.reward_shaping_alpha=0.02 \
    total_train_epochs="$TOTAL_TRAIN_EPOCHS" \
    allocation_mode=sglang.d1p1t1+d1p1t1 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node="$N_GPU" \
    cluster.fileroot="$FILE_ROOT" \
    +gconfig.top_p=0.7 \
    actor.path="$ACTOR_PATH" \
    actor.optimizer.lr=1e-6 \
    actor.optimizer.weight_decay=0.01 \
    actor.overlong_reward_penalty=false \
    actor.ppo_n_minibatches=64 \
    +actor.c_clip=10.0 \
    train_dataset.path="$TRAIN_DATASET_PATH" \
    valid_dataset.path="$VALID_DATASET_PATH"