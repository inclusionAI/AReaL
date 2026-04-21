#!/bin/bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-"/storage/openpsi/data/lcy_image_edit/mixed_rl"}
EVAL_DIR="${WORKSPACE}/full_eval"
MODEL_PATH=${MODEL_PATH:?'MODEL_PATH is required'}
RUN_NAME=${RUN_NAME:-$(echo "$MODEL_PATH" | grep -oP '[^/]+/global_step_\d+' | tr '/' '_' || basename "$MODEL_PATH")}

ulysses_sequence_parallel_size=1
max_prompt_length=16384
max_response_length=32768
n_gpus_per_node=${N_GPUS_PER_NODE:-8}
n_nodes=${N_NODES:-1}

val_data="${EVAL_DIR}/visual_probe_easy.parquet"
val_data="${val_data},${EVAL_DIR}/visual_probe_medium.parquet"
val_data="${val_data},${EVAL_DIR}/visual_probe_hard.parquet"
val_data="${val_data},${EVAL_DIR}/map_trace.parquet"
val_data="${val_data},${EVAL_DIR}/reason_map.parquet"
val_data="${val_data},${EVAL_DIR}/reason_map_plus.parquet"
val_data="${val_data},${EVAL_DIR}/mm_mapqa.parquet"

python3 -m verl.trainer.main_ppo \
    data.train_files="${WORKSPACE}/new_train.parquet" \
    data.val_files="[${val_data}]" \
    data.val_batch_size=512 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.agent.max_turns=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.logger='[console,wandb]' \
    trainer.project_name=mixed_rl_eval \
    trainer.experiment_name="eval_only_${RUN_NAME}" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=-1 \
    trainer.validation_data_dir="${EVAL_DIR}/results/${RUN_NAME}"
