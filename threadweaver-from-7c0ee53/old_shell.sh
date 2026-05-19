#!/usr/bin/env bash
set -euo pipefail

export MODELSCOPE_IMPORT=False
export VLLM_USE_V1=1

RL_CODE_DIR=/storage/openpsi/users/zzy/threadweaver/threadweaver_rl
MODEL_PATH=/storage/openpsi/models/zzy/augment/Qwen3-8B-131072-sft-tw8x

cd "$RL_CODE_DIR"
export PYTHONPATH="$RL_CODE_DIR:${PYTHONPATH:-}"

python3 -c 'import verl; print("Driver verl:", verl.__file__)'

parse_positive_int() {
  local name="$1"
  local value="$2"

  if [[ "$value" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$value"
  elif [[ "$value" =~ :([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  elif [[ "$value" =~ ^([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    echo "Could not parse integer $name from: $value" >&2
    exit 1
  fi
}

TRAINER_GPUS_PER_NODE="$(parse_positive_int "GPUs per node" "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}")"
TRAINER_NNODES="$(parse_positive_int "number of nodes" "${SLURM_NNODES:-6}")"

echo "trainer.n_gpus_per_node=$TRAINER_GPUS_PER_NODE"
echo "trainer.nnodes=$TRAINER_NNODES"

PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 python3 -m verl.trainer.main_ppo \ 
  algorithm.adv_estimator=grpo \
  data.train_files="/storage/openpsi/users/zzy/sync/polaris/polaris-data-53K-tw.parquet" \
  data.val_files="/storage/openpsi/users/zzy/sync/AIME24_converted_copy.parquet" \
  data.filter_overlong_prompts=True \
  data.train_batch_size=128 \
  data.val_batch_size=512 \
  data.max_prompt_length=9216 \
  data.max_response_length=40960 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
  actor_rollout_ref.actor.ppo_mini_batch_size=null \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
  actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=False \
  trainer.critic_warmup=0 \
  'trainer.logger=["console","tensorboard","wandb"]' \
  trainer.project_name='deepscaler' \
  trainer.experiment_name="${TRAINER_NNODES}n-tw-reporduce" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node="$TRAINER_GPUS_PER_NODE" \
  trainer.nnodes="$TRAINER_NNODES" \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.default_local_dir=/storage/openpsi/users/zzy/checkpoints/deepscaler/TW-reproduce \
  trainer.default_hdfs_dir=null \
  trainer.rollout_data_dir=/storage/openpsi/users/zzy/zzy_tw_rollback/TW-reproduce \
  trainer.total_epochs=30 \
  actor_rollout_ref.rollout.max_model_len=40960 \
  reward_model.config.acceleration_ratio_reward=1.0 \
  reward_model.config.acceleration_ratio_reward_factor=0.5 \
  reward_model.config.acceleration_ratio_clip_max=0.2 \
  reward_model.config.version=v2 \
  reward_model.config.require_think_end=False \
  reward_model.reward_manager_type=reward_manager_with_server \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.agent.enable_parallel_branching=True \
  actor_rollout_ref.rollout.agent_return_expanded_sequences=True \
  actor_rollout_ref.rollout.agent.no_conclusion=true \
  algorithm.broadcast_from_last=True \
  reward_model.config.strip_comma_from_answer=True \
  data.return_raw_chat=True \
  actor_rollout_ref.rollout.mode=async \
  trainer.max_actor_ckpt_to_keep=10 \
  actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']"
