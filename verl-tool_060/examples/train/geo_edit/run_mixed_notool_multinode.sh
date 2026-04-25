#!/usr/bin/env bash
set -x

# ============================================================
# Multi-node (4×8 GPU) no-tool RL training for geo_edit
# Same data/model as run_mixed_rl_multinode.sh but with
# agent/tool-calling disabled (pure VLM reasoning).
#
# Requires an existing Ray cluster spanning all 4 nodes.
#
# Environment variables:
#   WORKSPACE        – default: /storage/openpsi/data/lcy_image_edit/mixed_rl
#   MODEL_PATH       – path to SFT checkpoint
#   JUDGE_API_KEY / JUDGE_API_BASE / JUDGE_MODEL – LLM judge config
#   WANDB_API_KEY / WANDB_BASE_URL – wandb config
# ============================================================

WORKSPACE=${WORKSPACE:-/storage/openpsi/data/lcy_image_edit/mixed_rl}
model_name=${MODEL_PATH:-/storage/openpsi/models/Qwen3-VL-8B-Thinking}

train_data="[/storage/openpsi/data/reasonmap_rl/combined_train_rl_only_notool.parquet,$WORKSPACE/new_train_notool.parquet]"
val_data="[/storage/openpsi/data/reasonmap_rl/combined_test_10pct_notool.parquet,$WORKSPACE/new_val_notool.parquet,$WORKSPACE/mapqa_val_200_notool.parquet]"
run_name="mixed-grpo-notool-4node"
rl_alg=grpo
 
# ---- Cluster topology ----
n_gpus_per_node=8
n_nodes=4

# ---- Batch sizes (scaled for 4 nodes) ----
n=4
batch_size=64
ppo_mini_batch_size=64

# ---- Sequence lengths ----
max_prompt_length=16384
max_response_length=16384
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# ---- Sampling ----
temperature=1.0
top_p=1.0

# ---- No agent/tool ----
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
ppo_micro_batch_size_per_gpu=4
log_prob_micro_batch_size_per_gpu=16

# ---- Parallelism ----
tensor_model_parallel_size=1
ulysses_sequence_parallel_size=1
fsdp_size=-1

# ---- Memory ----
gpu_memory_utilization=0.8
do_offload=False
use_dynamic_bsz=True

# ---- Rollout ----
max_num_batched_tokens=$(expr $max_prompt_length + $max_response_length)

# ---- Schedule ----
total_epochs=3
save_freq=10
test_freq=20

# ============================================================
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
export WANDB_DIR=$WORKSPACE/logs/$run_name
export WANDB_RESUME=allow
export WANDB_RUN_ID=$run_name
unset ROCR_VISIBLE_DEVICES
unset JUDGE_API_KEY
mkdir -p $WORKSPACE/logs/$run_name

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

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=256 \
    data.dataloader_num_workers=64 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.shuffle=True \
    reward_model.reward_manager=$reward_manager \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=32 \
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
    trainer.project_name=mixed_rl \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$WORKSPACE/checkpoints/$run_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.rollout_data_dir=$WORKSPACE/logs/$run_name/step_records \
    trainer.nnodes=$n_nodes \
    +trainer.max_actor_ckpt_to_keep=20 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.resume_mode=auto \
    2>&1 | tee $WORKSPACE/logs/$run_name/train.log

echo "Training finished"
