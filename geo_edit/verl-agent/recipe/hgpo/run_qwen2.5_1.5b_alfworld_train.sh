set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HOME=${HF_HOME} # hugging face home directory
export WANDB_API_KEY=${WANDB_API_KEY} # wandb api key
export WANDB_DIR=${WANDB_DIR} # wandb directory
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} # cuda visible devices

project_name="qwen2.5_1.5b_alfworld_train"
history_length=2 # history length 2 or 4
num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=128
group_size=8    
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
weight_type="length"
length_weight_alpha=1.0  # weight is L^alpha, alpha=0 is uniform weight
base_group=False  # add episode_advantages as initial group to aggregate weight computation

experiment_name="k${history_length}_hgpo_${weight_type}_alpha${length_weight_alpha}_baseGroup_${base_group}"
CHECKPOINTS_DIR=${CHECKPOINTS_DIR} # checkpoints directory

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m recipe.hgpo.main_hgpo \
    algorithm.adv_estimator='hgpo' \
    algorithm.hgpo.weight_type=$weight_type \
    algorithm.hgpo.mode=$mode \
    algorithm.hgpo.length_weight_alpha=$length_weight_alpha \
    algorithm.hgpo.base_group=$base_group \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=alfworld/AlfredTWEnv \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.seed=0 \
    env.history_length=$history_length \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=5 \
    trainer.total_epochs=160 \
    trainer.default_local_dir="${CHECKPOINTS_DIR}/${project_name}/${experiment_name}" \
    trainer.val_only=False \
    trainer.val_before_train=False $@