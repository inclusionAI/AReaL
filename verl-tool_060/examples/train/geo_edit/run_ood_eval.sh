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
max_action_length=8192 
max_obs_length=8192
max_obs_length_image=8192
max_obs_length_text=6144
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)
max_num_batched_tokens=$(expr $max_prompt_length + $max_response_length)

max_turns=10
enable_agent=True
action_stop_tokens='</action>'
mask_observations=True
enable_mtrl=True
additional_eos_token_ids=[151645]
reward_manager=geo_vision_qa

log_prob_micro_batch_size_per_gpu=16
ppo_micro_batch_size_per_gpu=4
gpu_memory_utilization=0.8
ulysses_sequence_parallel_size=1
tensor_model_parallel_size=1
strategy=fsdp2
use_dynamic_bsz=True
do_offload=False
fsdp_size=-1
rollout_n=${ROLLOUT_N:-1}
if [ "$rollout_n" -gt 1 ]; then
    val_do_sample=True
    val_temperature=1.0
else
    val_do_sample=False
    val_temperature=0
fi

mkdir -p $WORKSPACE/logs/ood_eval_$RUN_NAME
action_stop_tokens_file="$WORKSPACE/logs/ood_eval_$RUN_NAME/action_stop_tokens.txt"
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file

if [ -n "${TOOL_SERVER_URL:-}" ]; then
    tool_server_url=$TOOL_SERVER_URL
elif [ -n "${TOOL_SERVER_IP:-}" ]; then
    tool_server_url=http://$TOOL_SERVER_IP:30888/get_observation
else
    WORKER_IP=$(python3 -c "
import ray; ray.init(address='auto',ignore_reinit_error=True)
for n in ray.nodes():
    if n['Resources'].get('tool_agent',0)>0 and n['Alive']:
        print(n['NodeManagerAddress']); break
")
    tool_server_url=http://$WORKER_IP:30888/get_observation
fi
echo "Using tool server at $tool_server_url"

declare -A EVAL_GROUPS
EVAL_GROUPS=(
    ["mapeval_visual"]="${EVAL_DIR}/mapeval_visual-tool.parquet"
    ["carto_mfs"]="${EVAL_DIR}/carto_mfs-tool.parquet"
    ["carto_mml"]="${EVAL_DIR}/carto_mml-tool.parquet"
    ["carto_mtmf"]="${EVAL_DIR}/carto_mtmf-tool.parquet"
    ["carto_rle"]="${EVAL_DIR}/carto_rle-tool.parquet"
    ["carto_srn"]="${EVAL_DIR}/carto_srn-tool.parquet"
    ["carto_stmf_counting"]="${EVAL_DIR}/carto_stmf_counting-tool.parquet"
    ["carto_stmf_name_listing"]="${EVAL_DIR}/carto_stmf_name_listing-tool.parquet"
    ["carto_stmf_presence"]="${EVAL_DIR}/carto_stmf_presence-tool.parquet"
)


run_eval_group() {
    local group_name=$1
    local val_data=$2

    echo ""
    echo "========================================"
    echo "  Evaluating: $group_name"
    echo "========================================"

    PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
        algorithm.adv_estimator=gigpo \
        data.train_files="${EVAL_DIR}/visual_probe_hard.parquet" \
        data.val_files="[${val_data}]" \
        data.train_batch_size=64 \
        data.val_batch_size=128 \
        data.dataloader_num_workers=128 \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.filter_overlong_prompts=False \
        data.truncation='right' \
        reward_model.reward_manager=$reward_manager \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
        actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.strategy=$strategy \
        actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
        actor_rollout_ref.agent.enable_agent=$enable_agent \
        actor_rollout_ref.agent.tool_server_url=$tool_server_url \
        actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
        actor_rollout_ref.agent.max_response_length=$max_response_length \
        actor_rollout_ref.agent.max_start_length=$max_prompt_length \
        actor_rollout_ref.agent.max_obs_length=$max_obs_length \
        +actor_rollout_ref.agent.max_obs_length_image=$max_obs_length_image \
        +actor_rollout_ref.agent.max_obs_length_text=$max_obs_length_text \
        actor_rollout_ref.agent.max_turns=$max_turns \
        actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
        actor_rollout_ref.agent.mask_observations=$mask_observations \
        actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
        actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
        actor_rollout_ref.agent.max_action_length=$max_action_length \
        actor_rollout_ref.agent.tool_call_timeout=600 \
        actor_rollout_ref.agent.max_concurrent_trajectories=128 \
        +actor_rollout_ref.agent.dispatch_mode=work_queue \
        actor_rollout_ref.rollout.agent.num_workers=$(expr $n_nodes \* $n_gpus_per_node) \
        actor_rollout_ref.rollout.data_parallel_size=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
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
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.mm-processor-cache-gb=8 \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
        critic.optim.lr=1e-5 \
        critic.strategy=$strategy \
        critic.model.path=$MODEL_PATH \
        critic.model.fsdp_config.fsdp_size=$fsdp_size \
        critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
        critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.test_freq=1 \
        trainer.total_epochs=1 \
        trainer.logger='[console,wandb]' \
        trainer.project_name=mixed_rl_ood_eval \
        trainer.experiment_name="ood_eval_${RUN_NAME}_${group_name}" \
        trainer.n_gpus_per_node=$n_gpus_per_node \
        trainer.nnodes=$n_nodes \
        trainer.save_freq=-1 \
        trainer.validation_data_dir="${EVAL_DIR}/results/${RUN_NAME}/ood_${group_name}" \
        2>&1 | tee $WORKSPACE/logs/ood_eval_$RUN_NAME/eval_${group_name}.log

    echo "  Done: $group_name -> ${EVAL_DIR}/results/${RUN_NAME}/ood_${group_name}"
}

# for group in mapeval_visual carto_mfs carto_mml carto_mtmf carto_rle carto_srn carto_stmf_counting carto_stmf_name_listing carto_stmf_presence; do
for group in  carto_mml carto_stmf_counting carto_stmf_name_listing carto_stmf_presence; do
    run_eval_group "$group" "${EVAL_GROUPS[$group]}"
done

has_carto=false 
for group in mapeval_visual carto_mfs carto_mml carto_mtmf carto_rle carto_srn carto_stmf_counting carto_stmf_name_listing carto_stmf_presence; do
    if [[ "$group" == carto_* ]] || [[ "$group" == mapeval_* ]]; then
        if [ -f "${EVAL_DIR}/results/${RUN_NAME}/ood_${group}/0.jsonl" ]; then
            has_carto=true
            break
        fi
    fi
done

if $has_carto; then
    echo ""
    echo "Running post-evaluation for CartoMapQA..."
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    python3 "${SCRIPT_DIR}/post_eval_cartomapqa.py" \
        --results_dir "${EVAL_DIR}/results/${RUN_NAME}" \
        --output "${EVAL_DIR}/results/${RUN_NAME}/post_eval_cartomapqa.json" \
        2>&1 | tee -a $WORKSPACE/logs/ood_eval_$RUN_NAME/post_eval.log
fi

echo ""
echo "All OOD eval groups finished. Results: ${EVAL_DIR}/results/${RUN_NAME}/"
