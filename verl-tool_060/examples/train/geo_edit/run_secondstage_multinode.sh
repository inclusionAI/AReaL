#!/usr/bin/env bash
set -x

# ============================================================
# Multi-node (4×8 GPU) second-stage training for geo_edit
# Requires an existing Ray cluster spanning all 4 nodes.
#
# Environment variables:
#   WORKSPACE        – default: /storage/openpsi/data/reasonmap_rl
#   MODEL_PATH       – path to SFT checkpoint
#   TOOL_SERVER_URL  – tool server URL (must be reachable from all nodes)
#   TOOL_SERVER_IP   – tool server IP (port defaults to 30888)
#   JUDGE_API_KEY / JUDGE_API_BASE / JUDGE_MODEL – LLM judge config
#   WANDB_API_KEY / WANDB_BASE_URL – wandb config
# ============================================================

WORKSPACE=${WORKSPACE:-/storage/openpsi/data/reasonmap_rl}
model_name=${MODEL_PATH:-/storage/openpsi/models/lcy_image_edit/sft_workspace/qwen3vl8b-thinking-mixed-reasonmap-0413-lr1e-5}

train_data="[$WORKSPACE/combined_train_rl_only.parquet]"
val_data="[$WORKSPACE/combined_test_10pct.parquet]"
run_name="reasonmap-rl-secondstage-4node"
rl_alg=grpo

# ---- Cluster topology ----
n_gpus_per_node=8
n_nodes=4

# ---- Batch sizes (scaled for 4 nodes) ----
n=2
batch_size=64
ppo_mini_batch_size=8

# ---- Sequence lengths ----
max_prompt_length=16384
max_response_length=16384
max_action_length=4096
max_obs_length=4096
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# ---- Sampling ----
temperature=1.0
top_p=1.0

# ---- Agent / tool ----
enable_agent=True
action_stop_tokens='</action>'
max_turns=5
mask_observations=True
enable_mtrl=True
additional_eos_token_ids=[151645]
reward_manager=geo_vision_qa

# ---- Training ----
strategy="fsdp2"
lr=1e-6
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl

# ---- Per-GPU micro batches ----
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1

# ---- Parallelism ----
tensor_model_parallel_size=1
ulysses_sequence_parallel_size=1
fsdp_size=-1

# ---- Memory ----
gpu_memory_utilization=0.5
do_offload=True
use_dynamic_bsz=True

# ---- Rollout ----
max_num_batched_tokens=32768
rollout_mode='async'

# ---- Schedule ----
total_epochs=3
save_freq=10
test_freq=10

# ============================================================
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
mkdir -p $WORKSPACE/logs/$run_name

action_stop_tokens_file="$WORKSPACE/logs/$run_name/action_stop_tokens.txt"
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file

# ---- Resolve tool server URL ----
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

# ---- Verify Ray cluster has enough nodes ----
python3 -c "
import ray, sys
ray.init(address='auto', ignore_reinit_error=True)
alive = [n for n in ray.nodes() if n['Alive']]
total_gpus = sum(n['Resources'].get('GPU', 0) for n in alive)
print(f'Ray cluster: {len(alive)} nodes, {int(total_gpus)} GPUs')
expected = $n_nodes * $n_gpus_per_node
if total_gpus < expected:
    print(f'ERROR: need {expected} GPUs but only {int(total_gpus)} available')
    sys.exit(1)
print('Cluster OK')
ray.shutdown()
"

# When using a pre-started Ray cluster, env vars must be set on each node
# BEFORE running `ray start`. Example for each worker node:
#   export JUDGE_API_KEY=xxx JUDGE_API_BASE=xxx JUDGE_MODEL=xxx
#   export WANDB_API_KEY=xxx WANDB_BASE_URL=xxx
#   ray start --address=<head_ip>:6379
#
# Alternatively, start Ray workers with --runtime-env-json:
#   ray start --address=<head_ip>:6379 \
#       --runtime-env-json='{"env_vars":{"JUDGE_API_KEY":"xxx","WANDB_API_KEY":"xxx"}}'

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=128 \
    data.dataloader_num_workers=32 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=8 \
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
    actor_rollout_ref.agent.tool_call_timeout=600 \
    actor_rollout_ref.agent.max_concurrent_trajectories=64 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
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
    actor_rollout_ref.rollout.max_num_seqs=16 \
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
    trainer.logger=['console','wandb'] \
    trainer.project_name=reasonmap_rl \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$WORKSPACE/checkpoints/$run_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.rollout_data_dir=$WORKSPACE/logs/$run_name/step_records \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    2>&1 | tee $WORKSPACE/logs/$run_name/train.log

echo "Training finished"
