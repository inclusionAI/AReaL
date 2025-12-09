#!/bin/bash
export http_proxy=""
export https_proxy=""


export CUDA_VISIBLE_DEVICES=0,1
allocation_mode="vllm:d1p1t1+d1p1t1"
N_GPUS=2
# export CUDA_VISIBLE_DEVICES=0,1
# allocation_mode="sglang:d1p1t1+d1p1t1"
# N_GPUS=2


SAVE_DIR="/home/ma-user/work/mohsen/areal_logs"
CHECKPOINT_SAVE_DIR="/data/mohsen/areal/checkpoints"

## TRAIN HYPERPARAMETERS:
EXP_NAME="Qwen2.5-VL-3B-Instruct-geometry3k"

# TRIAL_NAME="test"
# rm -rf /home/ma-user/work/ahmad/vlm/areal-fresh/main-Nov17/experiments/logs/ma-user/Qwen2.5-VL-3B-Instruct-geometry3k/test

# TRIAL_NAME="GRPO_exp16_nov17"  ### default settings as clever with MAX_TOKENS_PER_MB=2048, max_concurrent_rollouts=128, Batch_size=16 , epoch=15, max_new_tokens=512, MB_SPEC_N_MBS=1, without reshaping

# TRIAL_NAME="GRPO_exp18_nov17"  ### default settings as clever with MAX_TOKENS_PER_MB=2048, max_concurrent_rollouts=128, Batch_size=32 , epoch=40, max_new_tokens=1024, MB_SPEC_N_MBS=1, without reshaping, max_head_offpolicyness = 4
# TRIAL_NAME="GRPO_exp19_nov17"  ### default settings as clever with MAX_TOKENS_PER_MB=2048, max_concurrent_rollouts=128, Batch_size=32 , epoch=40, max_new_tokens=1024, MB_SPEC_N_MBS=1, without reshaping, max_head_offpolicyness = 4
# TRIAL_NAME="GRPO_exp20_nov17"  ### default settings as clever with MAX_TOKENS_PER_MB=2048, max_concurrent_rollouts=128, Batch_size=32 , epoch=70, max_new_tokens=1024, MB_SPEC_N_MBS=1, original reshaping, original code with bug, max_head_offpolicyness = 4
TRIAL_NAME="GRPO_exp0_dec4"  ### default settings as clever with MAX_TOKENS_PER_MB=2048, max_concurrent_rollouts=128, Batch_size=32 , epoch=70, max_new_tokens=1024, MB_SPEC_N_MBS=1, without reshaping, max_head_offpolicyness = 4
# TRIAL_NAME="GRPO_exp22_nov17"  ### default settings as clever with MAX_TOKENS_PER_MB=2048, max_concurrent_rollouts=128, Batch_size=32 , epoch=70, max_new_tokens=1024, MB_SPEC_N_MBS=1, without reshaping, max_head_offpolicyness = 0


### MODEL and DATA:
MODEL_PATH="/home/ma-user/work/pretrained_models/Qwen2.5-VL-3B-Instruct"
TRAIN_DATASET="/home/ma-user/work/datasets/geometry3k"
VALID_DATASET="/home/ma-user/work/datasets/geometry3k"

# TRAIN_DATASET="/home/ma-user/work/datasets/ViRL39K/data_forVerl"
# TRAIN_DATASET="/home/ma-user/work/datasets/clevr_count_70k"
# VALID_DATASET="/home/ma-user/work/datasets/clevr_count_70k"

DATASET_TYPE="rl"

BATCH_SIZE=32
VALID_BATCH_SIZE=32

EPOCHS=70
SEED=1  ### original =1


### KL CONTROL
KL_CTL=0.0
# KL_LOSS_COEF=0.01

# ### LOSS CONTROL
# RECOMPUTE_LOGPROB=true
# USE_DECOUPLED_LOSS=false

## Mini Batch settings:
PPO_N_MINIBATCHES=1
MB_SPEC_N_MBS=1
MAX_TOKENS_PER_MB=2048 # 1000000000000 #
PAD_TO_MAXIMUM=false


# ### REWARD:
# reward_scaling=1.0
# reward_bias=0.0

### RESUME TRAINING:
RECOVER_MODE=auto

### ROLLOUT CONFIG:
ASYNC_TRAINING=false
max_concurrent_rollouts=128
max_head_offpolicyness=4 # 4

### GENERATION CONFIG:
GROUP_SIZE=4
MAX_NEW_TOKENS=1024 # 4096

### SAVING:
SAVE_FREQ_EPOCHS=1


USE_VISION_CSL_LOSS=False

# VLLM_DISABLE_CUDA_GRAPH=1 NCCL_DEBUG=INFO VLLM_LOGGING_LEVEL=DEBUG 
#  NCCL_P2P_IGNORE_DISABLED=1
#NCCL_P2P_DISABLE=1 
python -m areal.launcher.local \
    examples/vlm/geometry3k_grpo.py --config examples/vlm/geometry3k_grpo.yaml \
    experiment_name=${EXP_NAME} \
    trial_name=${TRIAL_NAME} \
    allocation_mode=${allocation_mode} \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=${N_GPUS} \
    cluster.fileroot=${SAVE_DIR} \
    train_dataset.path=${TRAIN_DATASET} \
    valid_dataset.path=${VALID_DATASET} \
    actor.path=${MODEL_PATH} \
    actor.mb_spec.max_tokens_per_mb=${MAX_TOKENS_PER_MB} \
    rollout.max_concurrent_rollouts=${max_concurrent_rollouts} \
    train_dataset.batch_size=${BATCH_SIZE} \
    valid_dataset.batch_size=${VALID_BATCH_SIZE} \
    total_train_epochs=${EPOCHS} \
    seed=${SEED} \
    gconfig.max_new_tokens=${MAX_NEW_TOKENS} \
    saver.fileroot=${CHECKPOINT_SAVE_DIR} \
    rollout.max_head_offpolicyness=${max_head_offpolicyness} \

    

