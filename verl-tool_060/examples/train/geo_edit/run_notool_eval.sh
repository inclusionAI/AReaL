#!/bin/bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-"/storage/openpsi/data/lcy_image_edit/mixed_rl"}
EVAL_DIR="${WORKSPACE}/full_eval"
MODEL_PATH=${MODEL_PATH:?'MODEL_PATH is required'}
RUN_NAME=${RUN_NAME:-$(echo "$MODEL_PATH" | grep -oP '[^/]+/global_step_\d+' | tr '/' '_' || basename "$MODEL_PATH")}

n_gpus_per_node=${N_GPUS_PER_NODE:-8}
n_nodes=${N_NODES:-1}

max_prompt_length=32768
max_response_length=32768
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)
max_num_batched_tokens=$(expr $max_prompt_length + $max_response_length)

gpu_memory_utilization=0.8
strategy=fsdp2
use_dynamic_bsz=True
fsdp_size=-1
rollout_n=${ROLLOUT_N:-1}
if [ "$rollout_n" -gt 1 ]; then
    val_do_sample=True
    val_temperature=1.0
else
    val_do_sample=False
    val_temperature=0
fi

unset ROCR_VISIBLE_DEVICES
mkdir -p $WORKSPACE/logs/notool_eval_$RUN_NAME

declare -A EVAL_GROUPS
EVAL_GROUPS=(
    ["visual_probe"]="${EVAL_DIR}/visual_probe_easy-notool.parquet,${EVAL_DIR}/visual_probe_medium-notool.parquet,${EVAL_DIR}/visual_probe_hard-notool.parquet"
    ["reasonmap"]="${EVAL_DIR}/reason_map_dedup-notool.parquet,${EVAL_DIR}/reason_map_plus_dedup-notool.parquet"
    ["map_trace"]="${EVAL_DIR}/map_trace-notool.parquet"
    ["mm_mapqa"]="${EVAL_DIR}/mm_mapqa-notool.parquet"
)


run_eval_group() {
    local group_name=$1
    local val_data=$2

    echo ""
    echo "========================================"
    echo "  Evaluating (no-tool): $group_name"
    echo "========================================"

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="${EVAL_DIR}/visual_probe_hard-notool.parquet" \
        data.val_files="[${val_data}]" \
        data.train_batch_size=64 \
        data.val_batch_size=128 \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.filter_overlong_prompts=False \
        data.truncation='right' \
        reward_model.reward_manager=geo_vision_qa \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.strategy=$strategy \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
        actor_rollout_ref.rollout.data_parallel_size=$(expr $n_nodes \* $n_gpus_per_node) \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
        actor_rollout_ref.rollout.temperature=0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.n=$rollout_n \
        actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
        actor_rollout_ref.rollout.val_kwargs.do_sample=$val_do_sample \
        actor_rollout_ref.rollout.val_kwargs.n=$rollout_n \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.rollout.max_num_seqs=16 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.test_freq=1 \
        trainer.total_epochs=1 \
        trainer.logger='[console,wandb]' \
        trainer.project_name=mixed_rl_notool_eval \
        trainer.experiment_name="notool_eval_${RUN_NAME}_${group_name}" \
        trainer.n_gpus_per_node=$n_gpus_per_node \
        trainer.nnodes=$n_nodes \
        trainer.save_freq=-1 \
        trainer.validation_data_dir="${EVAL_DIR}/results/${RUN_NAME}/notool_${group_name}" \
        2>&1 | tee $WORKSPACE/logs/notool_eval_$RUN_NAME/eval_${group_name}.log

    echo "  Done: $group_name -> ${EVAL_DIR}/results/${RUN_NAME}/notool_${group_name}"
}

for group in visual_probe reasonmap map_trace; do
    run_eval_group "$group" "${EVAL_GROUPS[$group]}"
done

echo ""
echo "All no-tool eval groups finished. Results: ${EVAL_DIR}/results/${RUN_NAME}/"
