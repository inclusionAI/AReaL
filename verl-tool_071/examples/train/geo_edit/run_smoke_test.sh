#!/usr/bin/env bash
set -x

WORKSPACE=/storage/openpsi/models/lcy_image_edit/rl_workspace

train_data="[$WORKSPACE/data/chartqa_rl_train.parquet]"
val_data="[$WORKSPACE/data/chartqa_rl_val.parquet]"
model_name=/storage/openpsi/models/lcy_image_edit/sft_workspace/qwen3vl8b-instruct-chartqa-1third
run_name="chartqa-rl-smoke-test"
rl_alg=grpo
n_gpus_per_node=8
n_nodes=1
n=2
batch_size=16
ppo_mini_batch_size=16
max_prompt_length=4096
max_response_length=4096
max_action_length=2048
max_obs_length=4096
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)
temperature=1.0
top_p=1.0
enable_agent=True
strategy="fsdp2"
action_stop_tokens='</action>'
max_turns=3
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=geo_vision_qa
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=1
gpu_memory_utilization=0.6
do_offload=False
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=True
max_num_batched_tokens=10000
rollout_mode='async'
total_epochs=1
save_freq=100
test_freq=100

export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_TOOL_ROOT="${VERL_TOOL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export PYTHONPATH=$VERL_TOOL_ROOT/verl:$VERL_TOOL_ROOT:${PYTHONPATH:-}
echo "VERL_TOOL_ROOT=$VERL_TOOL_ROOT"
echo "PYTHONPATH=$PYTHONPATH"

mkdir -p $WORKSPACE/logs/$run_name

action_stop_tokens_file="$(mktemp)"
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file

export GEOEDIT_ENABLE_TOOLS="general,chart"
WORKER_IP=${TOOL_SERVER_IP:-$(python3 -c "
import ray; ray.init(address='auto',ignore_reinit_error=True)
for n in ray.nodes():
    if n['Resources'].get('tool_agent',0)>0 and n['Alive']:
        print(n['NodeManagerAddress']); break
")}
tool_server_url=http://$WORKER_IP:30888/get_observation
echo "Using tool server on worker at $tool_server_url"

unset ROCR_VISIBLE_DEVICES

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=16 \
    data.dataloader_num_workers=4 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=GeoVisionQARewardManager \
    reward.reward_manager.module.path=pkg://verl_tool.workers.reward_manager.geo_vision_qa \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
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
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console'] \
    trainer.project_name=chartqa_rl \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.rollout_data_dir=$WORKSPACE/logs/$run_name/step_records \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=2 \
    2>&1 | tee $WORKSPACE/logs/$run_name/train.log

echo "Smoke test finished"
